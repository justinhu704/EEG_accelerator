// Standalone complete CNN + GRU accelerator top.
// This module does not instantiate cnn_stage1/2/3_top.
//
// RAM A -> Conv1/BN1/ReLU1                   -> Conv1 even/odd banks
// Conv1 banks -> Conv2/BN2/ReLU2/Pool1        -> RAM A
// RAM A -> Conv3/BN3/ReLU3/streaming Pool2   -> reused Conv1 banks
// Reused Conv1 banks -> GRU                  -> RAM A
module cnn_gru_top #(
    parameter INPUT_FILE   = "mem/golden/q_in_act.mem",
    parameter CONV1_W_FILE = "mem/weights/conv1_W.mem",
    parameter CONV1_B_FILE = "mem/weights/conv1_b.mem",
    parameter CONV1_PACKED_W_FILE = "mem/weights/conv1_W_x3.mem",
    parameter CONV1_PACKED_B_FILE = "mem/weights/conv1_b_x3.mem",
    parameter BN1_A_FILE   = "mem/weights/bn1_A.mem",
    parameter BN1_B_FILE   = "mem/weights/bn1_B.mem",
    parameter CONV2_W_FILE = "mem/weights/conv2_W.mem",
    parameter CONV2_B_FILE = "mem/weights/conv2_b.mem",
    parameter CONV2_PACKED_W_FILE = "mem/weights/conv2_W_x5.mem",
    parameter CONV2_KH2_PACKED_W_FILE = "mem/weights/conv2_W_x5_kh2.mem",
    parameter CONV2_PACKED_B_FILE = "mem/weights/conv2_b_x5.mem",
    parameter BN2_A_FILE   = "mem/weights/bn2_A.mem",
    parameter BN2_B_FILE   = "mem/weights/bn2_B.mem",
    parameter CONV3_W_FILE = "mem/weights/conv3_W.mem",
    parameter CONV3_B_FILE = "mem/weights/conv3_b.mem",
    parameter CONV3_PACKED_W_FILE = "mem/weights/conv3_W_x3.mem",
    parameter CONV3_PACKED_B_FILE = "mem/weights/conv3_b_x3.mem",
    parameter BN3_A_FILE   = "mem/weights/bn3_A.mem",
    parameter BN3_B_FILE   = "mem/weights/bn3_B.mem",
    parameter GRU_WR_FILE  = "mem/weights/gru_Wr.mem",
    parameter GRU_WZ_FILE  = "mem/weights/gru_Wz.mem",
    parameter GRU_WH_FILE  = "mem/weights/gru_Wh.mem",
    parameter GRU_UR_FILE  = "mem/weights/gru_Ur.mem",
    parameter GRU_UZ_FILE  = "mem/weights/gru_Uz.mem",
    parameter GRU_UH_FILE  = "mem/weights/gru_Uh.mem",
    parameter GRU_BR_FILE  = "mem/weights/gru_br.mem",
    parameter GRU_BZ_FILE  = "mem/weights/gru_bz.mem",
    parameter GRU_BH_FILE  = "mem/weights/gru_bh.mem",
    parameter SIGMOID_FILE = "mem/lut/sigmoid_half_lut_q15.mem",
    parameter TANH_FILE    = "mem/lut/tanh_half_lut_q15.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // External loader for one 21x160 EEG sample. Writes are accepted only
    // while this CNN/GRU block is idle. Addresses 0..3359 hold one sample.
    input  logic               input_write_en,
    input  logic [11:0]        input_write_addr,
    input  logic signed [15:0] input_write_data,
    output logic               input_ready,

    // Live final GRU stream: 9 hidden units x 18 time steps.
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data,

    // Final GRU results occupy result RAM (RAM A) addresses 0..161.
    input  logic [15:0]        result_read_addr,
    output logic signed [15:0] result_read_data
);
    localparam int RAM_A_DEPTH = 19 * 18 * 20;  // Pool1 is RAM A's largest tensor.
    localparam int RAM_ADDR_W = 16;
    localparam int POOL1_ADDR_W = 13;
    localparam int CONV1_SIZE = 20 * 156 * 21;

    typedef enum logic [3:0] {
        S_IDLE,
        S_RUN_CONV1,
        S_START_CONV2,
        S_RUN_CONV2,
        S_START_CONV3,
        S_RUN_CONV3,
        S_START_GRU,
        S_RUN_GRU,
        S_DONE
    } state_t;
    state_t state;

    logic conv1_start, conv1_busy, conv1_valid;
    logic [31:0] conv1_input_addr, conv1_addr;
    logic signed [15:0] conv1_data;
    logic conv2_start, conv2_busy, conv2_valid;
    logic [31:0] conv2_input_addr, conv2_input_addr_kh1, conv2_addr;
    logic signed [15:0] conv2_data;
    logic conv3_start, conv3_busy, conv3_valid;
    logic [31:0] conv3_input_addr, conv3_addr;
    logic signed [15:0] conv3_data;

    logic pool1_start, pool1_busy, pool1_done, pool1_valid;
    logic [12:0] pool1_addr;
    logic signed [15:0] pool1_data;
    logic pool2_start, pool2_busy, pool2_done, pool2_valid;
    logic [12:0] pool2_addr;
    logic signed [15:0] pool2_data;

    logic gru_start, gru_busy, gru_done, gru_valid;
    logic [31:0] gru_input_addr, gru_addr;
    logic signed [15:0] gru_data;

    logic [RAM_ADDR_W-1:0] ram_a_internal_read_addr;
    logic signed [15:0] ram_a_read_data;
    logic signed [15:0] conv1_bank_data_kh0, conv1_bank_data_kh1;
    logic signed [15:0] shared_pool2_gru_data;
    logic ram_a_write_en;
    logic [RAM_ADDR_W-1:0] ram_a_write_addr;
    logic signed [15:0] ram_a_write_data;

    always_comb begin
        conv1_start = (state == S_IDLE) && start;
        conv2_start = (state == S_START_CONV2);
        // Conv2 與 maxpool1 同步
        pool1_start = conv2_start;
        conv3_start = (state == S_START_CONV3);
        // Conv3 and Pool2 start together so Pool2 can consume the live
        // ReLU3 stream. Starting Pool2 after Conv3 would lose that stream.
        pool2_start = conv3_start;
        gru_start = (state == S_START_GRU);
    end

    // Select which engine owns each synchronous RAM read port.
    always_comb begin
        if ((state == S_START_CONV3) || (state == S_RUN_CONV3))
            ram_a_internal_read_addr = conv3_input_addr[15:0];
        else if (state == S_RUN_CONV1)
            ram_a_internal_read_addr = conv1_input_addr[15:0];
        else
            ram_a_internal_read_addr = result_read_addr;

    end

    // =======================================
    // RAM A destinations: external EEG loader, streaming Pool1, and final GRU results
    // =======================================
    always_comb begin
        ram_a_write_en = (input_ready && input_write_en) ||
                         pool1_valid || gru_valid;
        if (gru_valid) begin
            ram_a_write_addr = gru_addr[15:0];
            ram_a_write_data = gru_data;
        end else if (input_ready && input_write_en) begin
            ram_a_write_addr = {{(RAM_ADDR_W-12){1'b0}}, input_write_addr};
            ram_a_write_data = input_write_data;
        end else begin
            ram_a_write_addr = {
                {(RAM_ADDR_W - POOL1_ADDR_W){1'b0}}, pool1_addr
            };
            ram_a_write_data = pool1_data;
        end
    end

    activation_ram #(
        .DATA_W(16), .DEPTH(RAM_A_DEPTH), .ADDR_W(RAM_ADDR_W),
        .MEM_FILE(INPUT_FILE)
    ) u_ram_a (
        .clk(clk),
        .write_en(ram_a_write_en),
        .write_addr(ram_a_write_addr), .write_data(ram_a_write_data),
        .read_addr(ram_a_internal_read_addr), .read_data(ram_a_read_data)
    );

    // =======================================
    // Conv1 banked RAM: ReLU1 is quantized to UQ5 and separated by height
    // parity. Conv2 reads one even and one odd activation every clock.
    // =======================================
    conv1_banked_ram #(
        .INPUT_F(11), .STORED_F(5), .LOG_ADDR_W(RAM_ADDR_W),
        .BANK_DEPTH(CONV1_SIZE / 2)
    ) u_conv1_ram (
        .clk(clk), .rst_n(rst_n),
        .write_en(conv1_valid),
        .write_logical_addr(conv1_addr[RAM_ADDR_W-1:0]),
        .write_q11_data(conv1_data),
        .pool2_write_en(pool2_valid),
        .pool2_write_addr(pool2_addr[8:0]),
        .pool2_write_data(pool2_data),
        .gru_read_mode((state == S_START_GRU) || (state == S_RUN_GRU)),
        .gru_read_addr(gru_input_addr[8:0]),
        .gru_read_data(shared_pool2_gru_data),
        .read_logical_addr_kh0(conv2_input_addr[RAM_ADDR_W-1:0]),
        .read_logical_addr_kh1(conv2_input_addr_kh1[RAM_ADDR_W-1:0]),
        .read_q11_data_kh0(conv1_bank_data_kh0),
        .read_q11_data_kh1(conv1_bank_data_kh1)
    );

    assign result_read_data = ram_a_read_data;

    conv_bn_relu_parallel_block #(
        .IN_H(21), .IN_W(160), .IN_CH(1),
        .K_H(2), .K_W(5), .OUT_CH(21), .LANES(3),
        .CONV_BIAS_SHIFT(12), .CONV_OUTPUT_SHIFT(14),
        .BN_BIAS_SHIFT(11), .BN_OUTPUT_SHIFT(13),
        .RELU_LEFT_SHIFT(0),
        .PACKED_WEIGHT_FILE(CONV1_PACKED_W_FILE),
        .PACKED_BIAS_FILE(CONV1_PACKED_B_FILE),
        .BN_A_FILE(BN1_A_FILE), .BN_B_FILE(BN1_B_FILE)
    ) u_conv1_bn_relu (
        .clk(clk), .rst_n(rst_n), .start(conv1_start),
        .busy(conv1_busy),
        .input_addr(conv1_input_addr), .input_data(ram_a_read_data),
        .output_valid(conv1_valid), .output_addr(conv1_addr),
        .output_data(conv1_data)
    );

    conv_bn_relu_parallel_kh2_block #(
        .IN_H(20), .IN_W(156), .IN_CH(21),
        .K_H(2), .K_W(5), .OUT_CH(20), .LANES(5),
        .CONV_BIAS_SHIFT(10), .CONV_OUTPUT_SHIFT(15),
        .BN_BIAS_SHIFT(11), .BN_OUTPUT_SHIFT(14),
        .RELU_LEFT_SHIFT(1),
        .PACKED_WEIGHT_FILE(CONV2_KH2_PACKED_W_FILE),
        .PACKED_BIAS_FILE(CONV2_PACKED_B_FILE),
        .BN_A_FILE(BN2_A_FILE), .BN_B_FILE(BN2_B_FILE)
    ) u_conv2_bn_relu (
        .clk(clk), .rst_n(rst_n), .start(conv2_start),
        .busy(conv2_busy),
        .input_addr_kh0(conv2_input_addr),
        .input_addr_kh1(conv2_input_addr_kh1),
        .input_data_kh0(conv1_bank_data_kh0),
        .input_data_kh1(conv1_bank_data_kh1),
        .output_valid(conv2_valid), .output_addr(conv2_addr),
        .output_data(conv2_data)
    );

    streaming_maxpool #(
        .IN_H(19), .IN_W(152), .IN_CH(20),
        .POOL_W(10), .STRIDE_W(8), .LANES(5),
        .INPUT_F(13), .OUTPUT_F(13)
    ) u_pool1 (
        .clk(clk), .rst_n(rst_n), .start(pool1_start),
        .busy(pool1_busy), .done(pool1_done),
        .input_valid(conv2_valid), .input_data(conv2_data),
        .output_valid(pool1_valid), .output_addr(pool1_addr),
        .output_data(pool1_data)
    );

    conv_bn_relu_parallel_block #(
        .IN_H(19), .IN_W(18), .IN_CH(20),
        .K_H(2), .K_W(5), .OUT_CH(15), .LANES(3),
        .CONV_BIAS_SHIFT(12), .CONV_OUTPUT_SHIFT(17),
        .BN_BIAS_SHIFT(11), .BN_OUTPUT_SHIFT(13),
        .RELU_LEFT_SHIFT(0),
        .PACKED_WEIGHT_FILE(CONV3_PACKED_W_FILE),
        .PACKED_BIAS_FILE(CONV3_PACKED_B_FILE),
        .BN_A_FILE(BN3_A_FILE), .BN_B_FILE(BN3_B_FILE)
    ) u_conv3_bn_relu (
        .clk(clk), .rst_n(rst_n), .start(conv3_start),
        .busy(conv3_busy),
        .input_addr(conv3_input_addr), .input_data(ram_a_read_data),
        .output_valid(conv3_valid), .output_addr(conv3_addr),
        .output_data(conv3_data)
    );

    streaming_maxpool #(
        .IN_H(18), .IN_W(14), .IN_CH(15),
        .POOL_W(10), .STRIDE_W(8), .LANES(3),
        .INPUT_F(12), .OUTPUT_F(13)
    ) u_pool2 (
        .clk(clk), .rst_n(rst_n), .start(pool2_start),
        .busy(pool2_busy), .done(pool2_done),
        .input_valid(conv3_valid), .input_data(conv3_data),
        .output_valid(pool2_valid), .output_addr(pool2_addr),
        .output_data(pool2_data)
    );

    gru_engine_pipeline #(
        .WR_FILE(GRU_WR_FILE), .WZ_FILE(GRU_WZ_FILE),
        .WH_FILE(GRU_WH_FILE), .UR_FILE(GRU_UR_FILE),
        .UZ_FILE(GRU_UZ_FILE), .UH_FILE(GRU_UH_FILE),
        .BR_FILE(GRU_BR_FILE), .BZ_FILE(GRU_BZ_FILE),
        .BH_FILE(GRU_BH_FILE),
        .SIGMOID_FILE(SIGMOID_FILE), .TANH_FILE(TANH_FILE)
    ) u_gru (
        .clk(clk), .rst_n(rst_n), .start(gru_start),
        .busy(gru_busy), .done(gru_done),
        .input_addr(gru_input_addr), .input_data(shared_pool2_gru_data),
        .output_valid(gru_valid), .output_addr(gru_addr),
        .output_data(gru_data)
    );

    // Move to the next layer only after the final value has been written.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE:
                    if (start)
                        state <= S_RUN_CONV1;
                S_RUN_CONV1:
                    if (conv1_valid && (conv1_addr == CONV1_SIZE-1))
                        state <= S_START_CONV2;
                S_START_CONV2:
                    state <= S_RUN_CONV2;
                S_RUN_CONV2:
                    // pool1_done is asserted only after all 57,760 ReLU2
                    // stream values have passed through the pool reducer.
                    if (pool1_done)
                        state <= S_START_CONV3;
                S_START_CONV3:
                    state <= S_RUN_CONV3;
                S_RUN_CONV3:
                    // pool2_done is asserted only after the complete ReLU3
                    // stream has passed through the synchronous max buffer.
                    if (pool2_done)
                        state <= S_START_GRU;
                S_START_GRU:
                    state <= S_RUN_GRU;
                S_RUN_GRU:
                    if (gru_done)
                        state <= S_DONE;
                S_DONE:
                    state <= S_IDLE;
                default:
                    state <= S_IDLE;
            endcase
        end
    end

    always_comb begin
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
        // Do not accept a write in the same cycle as start.
        input_ready = (state == S_IDLE) && !start;
        output_valid = gru_valid;
        output_addr = gru_addr;
        output_data = gru_data;
    end
endmodule

// Reusable Conv -> BN -> ReLU block with external activation RAM input.
// It is a datapath block, not a Stage1/2/3 top-level module.
module conv_bn_relu_block #(
    parameter int IN_H = 21,
    parameter int IN_W = 160,
    parameter int IN_CH = 1,
    parameter int K_H = 2,
    parameter int K_W = 5,
    parameter int OUT_CH = 21,
    parameter int OUT_H = IN_H - K_H + 1,
    parameter int OUT_W = IN_W - K_W + 1,
    parameter int CONV_BIAS_SHIFT = 12,
    parameter int CONV_OUTPUT_SHIFT = 14,
    parameter int BN_BIAS_SHIFT = 11,
    parameter int BN_OUTPUT_SHIFT = 13,
    parameter int RELU_LEFT_SHIFT = 0,
    parameter WEIGHT_FILE = "mem/weights/conv1_W.mem",
    parameter BIAS_FILE = "mem/weights/conv1_b.mem",
    parameter BN_A_FILE = "mem/weights/bn1_A.mem",
    parameter BN_B_FILE = "mem/weights/bn1_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic [31:0]        input_addr,
    input  logic signed [15:0] input_data,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int WEIGHT_DEPTH = K_H * K_W * IN_CH * OUT_CH;
    localparam int BIAS_DEPTH = OUT_CH;
    localparam int WEIGHT_ADDR_W = $clog2(WEIGHT_DEPTH);
    localparam int BIAS_ADDR_W = $clog2(BIAS_DEPTH);

    logic conv_done, conv_valid;
    logic [31:0] weight_addr, bias_addr, conv_addr, conv_channel;
    logic signed [15:0] weight_data, bias_data, conv_data;
    logic bn_valid;
    logic signed [15:0] bn_data;
    logic relu_valid;
    logic signed [15:0] relu_data;
    logic valid_d1;
    logic [31:0] addr_d1, addr_d2;

    weight_rom #(
        .DATA_W(16), .DEPTH(WEIGHT_DEPTH), .ADDR_W(WEIGHT_ADDR_W),
        .MEM_FILE(WEIGHT_FILE)
    ) u_weight_rom (
        .clk(clk), .addr(weight_addr[WEIGHT_ADDR_W-1:0]),
        .data(weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(BIAS_DEPTH), .ADDR_W(BIAS_ADDR_W),
        .MEM_FILE(BIAS_FILE)
    ) u_bias_rom (
        .clk(clk), .addr(bias_addr[BIAS_ADDR_W-1:0]),
        .data(bias_data)
    );

    conv_engine #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .BIAS_SHIFT(CONV_BIAS_SHIFT),
        .OUTPUT_SHIFT(CONV_OUTPUT_SHIFT)
    ) u_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(conv_done),
        .input_addr(input_addr), .input_data(input_data),
        .weight_addr(weight_addr), .weight_data(weight_data),
        .bias_addr(bias_addr), .bias_data(bias_data),
        .output_valid(conv_valid), .output_addr(conv_addr),
        .output_ch_idx(conv_channel), .output_data(conv_data)
    );

    bn_affine #(
        .CHANNELS(OUT_CH),
        .BIAS_SHIFT(BN_BIAS_SHIFT), .OUTPUT_SHIFT(BN_OUTPUT_SHIFT),
        .A_FILE(BN_A_FILE), .B_FILE(BN_B_FILE)
    ) u_bn (
        .clk(clk), .rst_n(rst_n),
        .in_valid(conv_valid), .in_data(conv_data),
        .in_ch_idx(conv_channel),
        .out_valid(bn_valid), .out_data(bn_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(RELU_LEFT_SHIFT)) u_relu (
        .in_valid(bn_valid), .in_data(bn_data),
        .out_valid(relu_valid), .out_data(relu_data)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            addr_d1 <= '0;
            addr_d2 <= '0;
        end else begin
            valid_d1 <= conv_valid;
            if (conv_valid)
                addr_d1 <= conv_addr;
            if (valid_d1)
                addr_d2 <= addr_d1;
        end
    end

    always_comb begin
        output_valid = relu_valid;
        output_addr = addr_d2;
        output_data = relu_data;
    end
endmodule


// Conv -> BN -> ReLU wrapper for the output-channel-parallel engine. The
// parallel convolution emits its lane results serially, so the existing
// scalar BN/ReLU pipeline and one-write-port activation RAM remain unchanged.
module conv_bn_relu_parallel_block #(
    parameter int IN_H   = 20,
    parameter int IN_W   = 156,
    parameter int IN_CH  = 21,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 20,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int LANES  = 4,
    parameter int CONV_BIAS_SHIFT   = 10,
    parameter int CONV_OUTPUT_SHIFT = 15,
    parameter int BN_BIAS_SHIFT     = 11,
    parameter int BN_OUTPUT_SHIFT   = 14,
    parameter int RELU_LEFT_SHIFT   = 1,
    parameter PACKED_WEIGHT_FILE = "mem/weights/conv2_W_x5.mem",
    parameter PACKED_BIAS_FILE   = "mem/weights/conv2_b_x5.mem",
    parameter BN_A_FILE          = "mem/weights/bn2_A.mem",
    parameter BN_B_FILE          = "mem/weights/bn2_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic [31:0]        input_addr,
    input  logic signed [15:0] input_data,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int OUT_GROUPS = (OUT_CH + LANES - 1) / LANES;
    localparam int PACKED_WEIGHT_DEPTH = K_H * K_W * IN_CH * OUT_GROUPS;
    localparam int PACKED_WEIGHT_ADDR_W = $clog2(PACKED_WEIGHT_DEPTH);
    localparam int PACKED_BIAS_ADDR_W = $clog2(OUT_GROUPS);

    logic conv_done_unused;
    logic conv_valid;
    logic [31:0] weight_addr, bias_addr, conv_addr, conv_channel;
    logic signed [(16*LANES)-1:0] packed_weight_data;
    logic signed [(16*LANES)-1:0] packed_bias_data;
    logic signed [15:0] conv_data;
    logic bn_valid;
    logic signed [15:0] bn_data;
    logic relu_valid;
    logic signed [15:0] relu_data;
    logic valid_d1;
    logic [31:0] addr_d1, addr_d2;

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(PACKED_WEIGHT_DEPTH),
        .ADDR_W(PACKED_WEIGHT_ADDR_W), .MEM_FILE(PACKED_WEIGHT_FILE)
    ) u_packed_weight_rom (
        .clk(clk), .addr(weight_addr[PACKED_WEIGHT_ADDR_W-1:0]),
        .data(packed_weight_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OUT_GROUPS),
        .ADDR_W(PACKED_BIAS_ADDR_W), .MEM_FILE(PACKED_BIAS_FILE)
    ) u_packed_bias_rom (
        .clk(clk), .addr(bias_addr[PACKED_BIAS_ADDR_W-1:0]),
        .data(packed_bias_data)
    );

    conv_engine_parallel_counter #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W), .LANES(LANES),
        .BIAS_SHIFT(CONV_BIAS_SHIFT),
        .OUTPUT_SHIFT(CONV_OUTPUT_SHIFT)
    ) u_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(conv_done_unused),
        .input_addr(input_addr), .input_data(input_data),
        .weight_addr(weight_addr), .weight_data(packed_weight_data),
        .bias_addr(bias_addr), .bias_data(packed_bias_data),
        .output_valid(conv_valid), .output_addr(conv_addr),
        .output_ch_idx(conv_channel), .output_data(conv_data)
    );

    bn_affine #(
        .CHANNELS(OUT_CH),
        .BIAS_SHIFT(BN_BIAS_SHIFT), .OUTPUT_SHIFT(BN_OUTPUT_SHIFT),
        .A_FILE(BN_A_FILE), .B_FILE(BN_B_FILE)
    ) u_bn (
        .clk(clk), .rst_n(rst_n),
        .in_valid(conv_valid), .in_data(conv_data),
        .in_ch_idx(conv_channel),
        .out_valid(bn_valid), .out_data(bn_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(RELU_LEFT_SHIFT)) u_relu (
        .in_valid(bn_valid), .in_data(bn_data),
        .out_valid(relu_valid), .out_data(relu_data)
    );

    // BN has two registered stages. Delay each convolution address by the
    // same two clocks so data and destination remain aligned.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            addr_d1 <= '0;
            addr_d2 <= '0;
        end else begin
            valid_d1 <= conv_valid;
            if (conv_valid)
                addr_d1 <= conv_addr;
            if (valid_d1)
                addr_d2 <= addr_d1;
        end
    end

    always_comb begin
        output_valid = relu_valid;
        output_addr  = addr_d2;
        output_data  = relu_data;
    end
endmodule
