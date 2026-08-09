// Two convolution stages followed by Pool1 using ping-pong buffers:
//   RAM A -> Conv1 -> BN1 -> ReLU1 -> RAM B
//   RAM B -> Conv2 -> BN2 -> ReLU2 -> RAM A
//   RAM A -> MaxPool1 -> RAM B
module cnn_stage2_top #(
    parameter bit RUN_FULL_CNN = 1'b0,
    parameter INPUT_FILE   = "mem/golden/q_in_act.mem",
    parameter CONV1_W_FILE = "mem/weights/conv1_W.mem",
    parameter CONV1_B_FILE = "mem/weights/conv1_b.mem",
    parameter BN1_A_FILE   = "mem/weights/bn1_A.mem",
    parameter BN1_B_FILE   = "mem/weights/bn1_B.mem",
    parameter CONV2_W_FILE = "mem/weights/conv2_W.mem",
    parameter CONV2_B_FILE = "mem/weights/conv2_b.mem",
    parameter BN2_A_FILE   = "mem/weights/bn2_A.mem",
    parameter BN2_B_FILE   = "mem/weights/bn2_B.mem",
    parameter CONV3_W_FILE = "mem/weights/conv3_W.mem",
    parameter CONV3_B_FILE = "mem/weights/conv3_b.mem",
    parameter BN3_A_FILE   = "mem/weights/bn3_A.mem",
    parameter BN3_B_FILE   = "mem/weights/bn3_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // Final ReLU2 stream while it is written back to RAM A.
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data,

    // Stage-1 stream is kept visible for integration debugging.
    output logic               stage1_output_valid,
    output logic [31:0]        stage1_output_addr,
    output logic signed [15:0] stage1_output_data,

    // Pool1 is the final output of this integration milestone.
    output logic               pool1_output_valid,
    output logic [31:0]        pool1_output_addr,
    output logic signed [15:0] pool1_output_data,

    // Pool2 is the final output when RUN_FULL_CNN is enabled.
    output logic               pool2_output_valid,
    output logic [31:0]        pool2_output_addr,
    output logic signed [15:0] pool2_output_data,

    // Readback ports are available while the corresponding RAM is idle.
    input  logic [15:0]        ram_a_read_addr,
    output logic signed [15:0] ram_a_read_data,
    input  logic [15:0]        ram_b_read_addr,
    output logic signed [15:0] ram_b_read_data
);
    localparam int RAM_DEPTH = 65520;
    localparam int RAM_ADDR_W = 16;
    localparam int STAGE1_SIZE = 20 * 156 * 21;
    localparam int STAGE2_SIZE = 19 * 152 * 20;
    localparam int POOL1_SIZE = 19 * 18 * 20;
    localparam int STAGE3_SIZE = 18 * 14 * 15;
    localparam int POOL2_SIZE = 18 * 1 * 15;
    localparam int CONV1_W_DEPTH = 2 * 5 * 1 * 21;
    localparam int CONV1_B_DEPTH = 21;
    localparam int CONV2_W_DEPTH = 2 * 5 * 21 * 20;
    localparam int CONV2_B_DEPTH = 20;
    localparam int CONV3_W_DEPTH = 2 * 5 * 20 * 15;
    localparam int CONV3_B_DEPTH = 15;

    typedef enum logic [3:0] {
        S_IDLE,
        S_RUN_STAGE1,
        S_START_STAGE2,
        S_RUN_STAGE2,
        S_START_POOL1,
        S_RUN_POOL1,
        S_START_STAGE3,
        S_RUN_STAGE3,
        S_START_POOL2,
        S_RUN_POOL2,
        S_DONE
    } state_t;
    state_t state;

    logic conv1_start;
    logic conv1_busy;
    logic conv1_done;
    logic conv1_valid;
    logic [31:0] conv1_input_addr;
    logic [31:0] conv1_weight_addr;
    logic [31:0] conv1_bias_addr;
    logic [31:0] conv1_addr;
    logic [31:0] conv1_channel;
    logic signed [15:0] conv1_input_data;
    logic signed [15:0] conv1_weight_data;
    logic signed [15:0] conv1_bias_data;
    logic signed [15:0] conv1_data;
    logic bn1_valid;
    logic signed [15:0] bn1_data;
    logic relu1_valid;
    logic signed [15:0] relu1_data;
    logic conv1_valid_d1;
    logic conv1_valid_d2;
    logic [31:0] conv1_addr_d1;
    logic [31:0] conv1_addr_d2;

    logic conv2_start;
    logic conv2_busy;
    logic conv2_done;
    logic conv2_valid;
    logic [31:0] conv2_input_addr;
    logic [31:0] conv2_weight_addr;
    logic [31:0] conv2_bias_addr;
    logic [31:0] conv2_addr;
    logic [31:0] conv2_channel;
    logic signed [15:0] conv2_input_data;
    logic signed [15:0] conv2_weight_data;
    logic signed [15:0] conv2_bias_data;
    logic signed [15:0] conv2_data;
    logic bn2_valid;
    logic signed [15:0] bn2_data;
    logic relu2_valid;
    logic signed [15:0] relu2_data;
    logic conv2_valid_d1;
    logic conv2_valid_d2;
    logic [31:0] conv2_addr_d1;
    logic [31:0] conv2_addr_d2;

    logic pool1_start;
    logic pool1_busy;
    logic pool1_done;
    logic pool1_valid;
    logic [31:0] pool1_input_addr;
    logic [31:0] pool1_addr;
    logic signed [15:0] pool1_input_data;
    logic signed [15:0] pool1_data;

    logic conv3_start;
    logic conv3_busy;
    logic conv3_done;
    logic conv3_valid;
    logic [31:0] conv3_input_addr;
    logic [31:0] conv3_weight_addr;
    logic [31:0] conv3_bias_addr;
    logic [31:0] conv3_addr;
    logic [31:0] conv3_channel;
    logic signed [15:0] conv3_input_data;
    logic signed [15:0] conv3_weight_data;
    logic signed [15:0] conv3_bias_data;
    logic signed [15:0] conv3_data;
    logic bn3_valid;
    logic signed [15:0] bn3_data;
    logic relu3_valid;
    logic signed [15:0] relu3_data;
    logic conv3_valid_d1;
    logic conv3_valid_d2;
    logic [31:0] conv3_addr_d1;
    logic [31:0] conv3_addr_d2;

    logic pool2_start;
    logic pool2_busy;
    logic pool2_done;
    logic pool2_valid;
    logic [31:0] pool2_input_addr;
    logic [31:0] pool2_addr;
    logic signed [15:0] pool2_input_data;
    logic signed [15:0] pool2_data;

    logic ram_a_write_en;
    logic [RAM_ADDR_W-1:0] ram_a_write_addr;
    logic signed [15:0] ram_a_write_data;

    logic ram_b_write_en;
    logic [RAM_ADDR_W-1:0] ram_b_write_addr;
    logic signed [15:0] ram_b_write_data;

    logic [RAM_ADDR_W-1:0] ram_a_internal_read_addr;
    logic [RAM_ADDR_W-1:0] ram_b_internal_read_addr;

    // A one-clock start pulse is generated for each convolution engine.
    always_comb begin
        conv1_start = (state == S_IDLE) && start;
        conv2_start = (state == S_START_STAGE2);
        pool1_start = (state == S_START_POOL1);
        conv3_start = (state == S_START_STAGE3);
        pool2_start = (state == S_START_POOL2);
    end

    // During processing, the active convolution owns the source RAM read
    // port. Outside that stage the port is available to the testbench.
    always_comb begin
        if ((state == S_START_POOL2) || (state == S_RUN_POOL2))
            ram_a_internal_read_addr = pool2_input_addr[RAM_ADDR_W-1:0];
        else if ((state == S_START_POOL1) || (state == S_RUN_POOL1))
            ram_a_internal_read_addr = pool1_input_addr[RAM_ADDR_W-1:0];
        else if (state == S_RUN_STAGE1)
            ram_a_internal_read_addr = conv1_input_addr[RAM_ADDR_W-1:0];
        else
            ram_a_internal_read_addr = ram_a_read_addr;

        if ((state == S_START_STAGE3) || (state == S_RUN_STAGE3))
            ram_b_internal_read_addr = conv3_input_addr[RAM_ADDR_W-1:0];
        else if ((state == S_START_STAGE2) || (state == S_RUN_STAGE2))
            ram_b_internal_read_addr = conv2_input_addr[RAM_ADDR_W-1:0];
        else
            ram_b_internal_read_addr = ram_b_read_addr;
    end

    // RAM A receives ReLU2 and later ReLU3. The stages are sequential.
    always_comb begin
        ram_a_write_en = relu2_valid || relu3_valid;
        if (relu3_valid) begin
            ram_a_write_addr = conv3_addr_d2[RAM_ADDR_W-1:0];
            ram_a_write_data = relu3_data;
        end else begin
            ram_a_write_addr = conv2_addr_d2[RAM_ADDR_W-1:0];
            ram_a_write_data = relu2_data;
        end
    end

    // RAM B receives ReLU1 during stage 1 and Pool1 during the final stage.
    // These stages never overlap, so one write port is sufficient.
    always_comb begin
        // RAM B 寫入邏輯
        ram_b_write_en = relu1_valid || pool1_valid || pool2_valid;
        if (pool2_valid) begin
            ram_b_write_addr = pool2_addr[RAM_ADDR_W-1:0];
            ram_b_write_data = pool2_data;
        end else if (pool1_valid) begin
            ram_b_write_addr = pool1_addr[RAM_ADDR_W-1:0];
            ram_b_write_data = pool1_data;
        end else begin
            ram_b_write_addr = conv1_addr_d2[RAM_ADDR_W-1:0];
            ram_b_write_data = relu1_data;
        end
    end

    // RAM A: 初始存入EEG資料
    activation_ram #(
        .DATA_W(16), .DEPTH(RAM_DEPTH), .ADDR_W(RAM_ADDR_W),
        .MEM_FILE(INPUT_FILE)
    ) u_ram_a (
        .clk(clk),
        .write_en(ram_a_write_en),
        .write_addr(ram_a_write_addr),
        .write_data(ram_a_write_data),
        .read_addr(ram_a_internal_read_addr),
        .read_data(ram_a_read_data)
    );

    // RAM B: 存放ReLU1 & Pool1 -> Conv2
    activation_ram #(
        .DATA_W(16), .DEPTH(RAM_DEPTH), .ADDR_W(RAM_ADDR_W),
        .MEM_FILE("")
    ) u_ram_b (
        .clk(clk),
        .write_en(ram_b_write_en),
        .write_addr(ram_b_write_addr),
        .write_data(ram_b_write_data),
        .read_addr(ram_b_internal_read_addr),
        .read_data(ram_b_read_data)
    );

// *************************************************

    // Conv1 weight rom
    weight_rom #(
        .DATA_W(16), .DEPTH(CONV1_W_DEPTH),
        .ADDR_W($clog2(CONV1_W_DEPTH)), .MEM_FILE(CONV1_W_FILE)
    ) u_conv1_weight_rom (
        .clk(clk),
        .addr(conv1_weight_addr[$clog2(CONV1_W_DEPTH)-1:0]),
        .data(conv1_weight_data)
    );

    // Conv1 Bias rom
    weight_rom #(
        .DATA_W(16), .DEPTH(CONV1_B_DEPTH),
        .ADDR_W($clog2(CONV1_B_DEPTH)), .MEM_FILE(CONV1_B_FILE)
    ) u_conv1_bias_rom (
        .clk(clk),
        .addr(conv1_bias_addr[$clog2(CONV1_B_DEPTH)-1:0]),
        .data(conv1_bias_data)
    );
    // ==================================================
    // Conv1
    // ==================================================
    conv_engine #(
        .IN_H(21), .IN_W(160), .IN_CH(1),
        .K_H(2), .K_W(5), .OUT_CH(21),
        .BIAS_SHIFT(12), .OUTPUT_SHIFT(14)
    ) u_conv1 (
        .clk(clk), .rst_n(rst_n), .start(conv1_start),
        .busy(conv1_busy), .done(conv1_done),
        .input_addr(conv1_input_addr), .input_data(conv1_input_data),
        .weight_addr(conv1_weight_addr), .weight_data(conv1_weight_data),
        .bias_addr(conv1_bias_addr), .bias_data(conv1_bias_data),
        .output_valid(conv1_valid), .output_addr(conv1_addr),
        .output_ch_idx(conv1_channel), .output_data(conv1_data)
    );

    // Conv1 輸入資料為 RAM A
    always_comb conv1_input_data = ram_a_read_data;

    // ==================================================
    // BN1
    // ==================================================
    bn_affine #(
        .CHANNELS(21), .BIAS_SHIFT(11), .OUTPUT_SHIFT(13),
        .A_FILE(BN1_A_FILE), .B_FILE(BN1_B_FILE)
    ) u_bn1 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(conv1_valid), .in_data(conv1_data),
        .in_ch_idx(conv1_channel),
        .out_valid(bn1_valid), .out_data(bn1_data)
    );

    // ==================================================
    // ReLU1
    // ==================================================
    relu #(.OUTPUT_LEFT_SHIFT(0)) u_relu1 (
        .in_valid(bn1_valid), .in_data(bn1_data),
        .out_valid(relu1_valid), .out_data(relu1_data)
    );
    
// *************************************************

    weight_rom #(
        .DATA_W(16), .DEPTH(CONV2_W_DEPTH),
        .ADDR_W($clog2(CONV2_W_DEPTH)), .MEM_FILE(CONV2_W_FILE)
    ) u_conv2_weight_rom (
        .clk(clk),
        .addr(conv2_weight_addr[$clog2(CONV2_W_DEPTH)-1:0]),
        .data(conv2_weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(CONV2_B_DEPTH),
        .ADDR_W($clog2(CONV2_B_DEPTH)), .MEM_FILE(CONV2_B_FILE)
    ) u_conv2_bias_rom (
        .clk(clk),
        .addr(conv2_bias_addr[$clog2(CONV2_B_DEPTH)-1:0]),
        .data(conv2_bias_data)
    );

    // ==================================================
    // Conv2
    // ==================================================
    conv_engine #(
        .IN_H(20), .IN_W(156), .IN_CH(21),
        .K_H(2), .K_W(5), .OUT_CH(20),
        .BIAS_SHIFT(10), .OUTPUT_SHIFT(15)
    ) u_conv2 (
        .clk(clk), .rst_n(rst_n), .start(conv2_start),
        .busy(conv2_busy), .done(conv2_done),
        .input_addr(conv2_input_addr), .input_data(conv2_input_data),
        .weight_addr(conv2_weight_addr), .weight_data(conv2_weight_data),
        .bias_addr(conv2_bias_addr), .bias_data(conv2_bias_data),
        .output_valid(conv2_valid), .output_addr(conv2_addr),
        .output_ch_idx(conv2_channel), .output_data(conv2_data)
    );

    // RAM B's registered read data is Conv2's activation input.
    always_comb conv2_input_data = ram_b_read_data;

    bn_affine #(
        .CHANNELS(20), .BIAS_SHIFT(11), .OUTPUT_SHIFT(14),
        .A_FILE(BN2_A_FILE), .B_FILE(BN2_B_FILE)
    ) u_bn2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(conv2_valid), .in_data(conv2_data),
        .in_ch_idx(conv2_channel),
        .out_valid(bn2_valid), .out_data(bn2_data)
    );

    // BN2 produces F11 data; ReLU2 converts it to the F12 RAM format.
    relu #(.OUTPUT_LEFT_SHIFT(1)) u_relu2 (
        .in_valid(bn2_valid), .in_data(bn2_data),
        .out_valid(relu2_valid), .out_data(relu2_data)
    );

    // Pool1 reads the completed ReLU2 tensor from RAM A and writes its
    // 19x18x20 result into the beginning of RAM B.
    maxpool_engine #(
        .IN_H(19), .IN_W(152), .IN_CH(20),
        .POOL_H(1), .POOL_W(10),
        .STRIDE_H(1), .STRIDE_W(8),
        .INPUT_F(12), .OUTPUT_F(12)
    ) u_pool1 (
        .clk(clk), .rst_n(rst_n), .start(pool1_start),
        .busy(pool1_busy), .done(pool1_done),
        .input_addr(pool1_input_addr),
        .input_data(pool1_input_data),
        .output_valid(pool1_valid),
        .output_addr(pool1_addr),
        .output_data(pool1_data)
    );

    always_comb pool1_input_data = ram_a_read_data;

    weight_rom #(
        .DATA_W(16), .DEPTH(CONV3_W_DEPTH),
        .ADDR_W($clog2(CONV3_W_DEPTH)), .MEM_FILE(CONV3_W_FILE)
    ) u_conv3_weight_rom (
        .clk(clk),
        .addr(conv3_weight_addr[$clog2(CONV3_W_DEPTH)-1:0]),
        .data(conv3_weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(CONV3_B_DEPTH),
        .ADDR_W($clog2(CONV3_B_DEPTH)), .MEM_FILE(CONV3_B_FILE)
    ) u_conv3_bias_rom (
        .clk(clk),
        .addr(conv3_bias_addr[$clog2(CONV3_B_DEPTH)-1:0]),
        .data(conv3_bias_data)
    );

    conv_engine #(
        .IN_H(19), .IN_W(18), .IN_CH(20),
        .K_H(2), .K_W(5), .OUT_CH(15),
        .BIAS_SHIFT(12), .OUTPUT_SHIFT(17)
    ) u_conv3 (
        .clk(clk), .rst_n(rst_n), .start(conv3_start),
        .busy(conv3_busy), .done(conv3_done),
        .input_addr(conv3_input_addr), .input_data(conv3_input_data),
        .weight_addr(conv3_weight_addr), .weight_data(conv3_weight_data),
        .bias_addr(conv3_bias_addr), .bias_data(conv3_bias_data),
        .output_valid(conv3_valid), .output_addr(conv3_addr),
        .output_ch_idx(conv3_channel), .output_data(conv3_data)
    );

    always_comb conv3_input_data = ram_b_read_data;

    bn_affine #(
        .CHANNELS(15), .BIAS_SHIFT(11), .OUTPUT_SHIFT(13),
        .A_FILE(BN3_A_FILE), .B_FILE(BN3_B_FILE)
    ) u_bn3 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(conv3_valid), .in_data(conv3_data),
        .in_ch_idx(conv3_channel),
        .out_valid(bn3_valid), .out_data(bn3_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(0)) u_relu3 (
        .in_valid(bn3_valid), .in_data(bn3_data),
        .out_valid(relu3_valid), .out_data(relu3_data)
    );

    // Pool2 reads ReLU3 from RAM A. F13 is shifted left once to F14,
    // then the 18x1x15 result is written into RAM B.
    maxpool_engine #(
        .IN_H(18), .IN_W(14), .IN_CH(15),
        .POOL_H(1), .POOL_W(10),
        .STRIDE_H(1), .STRIDE_W(8),
        .INPUT_F(13), .OUTPUT_F(14)
    ) u_pool2 (
        .clk(clk), .rst_n(rst_n), .start(pool2_start),
        .busy(pool2_busy), .done(pool2_done),
        .input_addr(pool2_input_addr),
        .input_data(pool2_input_data),
        .output_valid(pool2_valid),
        .output_addr(pool2_addr),
        .output_data(pool2_data)
    );

    always_comb pool2_input_data = ram_a_read_data;

    // Each BN has two registered stages, so delay each convolution address
    // by two clocks to keep its RAM write address aligned with ReLU data.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv1_valid_d1 <= 1'b0;
            conv1_valid_d2 <= 1'b0;
            conv1_addr_d1 <= '0;
            conv1_addr_d2 <= '0;
            conv2_valid_d1 <= 1'b0;
            conv2_valid_d2 <= 1'b0;
            conv2_addr_d1 <= '0;
            conv2_addr_d2 <= '0;
            conv3_valid_d1 <= 1'b0;
            conv3_valid_d2 <= 1'b0;
            conv3_addr_d1 <= '0;
            conv3_addr_d2 <= '0;
        end else begin
            conv1_valid_d1 <= conv1_valid;
            conv1_valid_d2 <= conv1_valid_d1;
            if (conv1_valid)
                conv1_addr_d1 <= conv1_addr;
            if (conv1_valid_d1)
                conv1_addr_d2 <= conv1_addr_d1;

            conv2_valid_d1 <= conv2_valid;
            conv2_valid_d2 <= conv2_valid_d1;
            if (conv2_valid)
                conv2_addr_d1 <= conv2_addr;
            if (conv2_valid_d1)
                conv2_addr_d2 <= conv2_addr_d1;

            conv3_valid_d1 <= conv3_valid;
            conv3_valid_d2 <= conv3_valid_d1;
            if (conv3_valid)
                conv3_addr_d1 <= conv3_addr;
            if (conv3_valid_d1)
                conv3_addr_d2 <= conv3_addr_d1;
        end
    end

    // The next stage starts only after the previous stage's final ReLU value
    // has reached and been written into its destination RAM.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE:
                    if (start)
                        state <= S_RUN_STAGE1;
                S_RUN_STAGE1:
                    if (relu1_valid && (conv1_addr_d2 == STAGE1_SIZE-1))
                        state <= S_START_STAGE2;
                S_START_STAGE2:
                    state <= S_RUN_STAGE2;
                S_RUN_STAGE2:
                    if (relu2_valid && (conv2_addr_d2 == STAGE2_SIZE-1))
                        state <= S_START_POOL1;
                S_START_POOL1:
                    state <= S_RUN_POOL1;
                S_RUN_POOL1:
                    if (pool1_valid && (pool1_addr == POOL1_SIZE-1)) begin
                        if (RUN_FULL_CNN)
                            state <= S_START_STAGE3;
                        else
                            state <= S_DONE;
                    end
                S_START_STAGE3:
                    state <= S_RUN_STAGE3;
                S_RUN_STAGE3:
                    if (relu3_valid && (conv3_addr_d2 == STAGE3_SIZE-1))
                        state <= S_START_POOL2;
                S_START_POOL2:
                    state <= S_RUN_POOL2;
                S_RUN_POOL2:
                    if (pool2_valid && (pool2_addr == POOL2_SIZE-1))
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

        stage1_output_valid = relu1_valid;
        stage1_output_addr = conv1_addr_d2;
        stage1_output_data = relu1_data;

        output_valid = relu2_valid;
        output_addr = conv2_addr_d2;
        output_data = relu2_data;

        pool1_output_valid = pool1_valid;
        pool1_output_addr = pool1_addr;
        pool1_output_data = pool1_data;

        pool2_output_valid = pool2_valid;
        pool2_output_addr = pool2_addr;
        pool2_output_data = pool2_data;
    end

    // conv1_done/conv2_done occur before their BN/ReLU pipelines drain;
    // completion therefore uses the final aligned ReLU write instead.
endmodule
