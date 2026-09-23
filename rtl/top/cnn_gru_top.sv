// Standalone complete CNN + GRU accelerator top.
// DS-Conv1 and DS-Conv2 exchange data through a six-column rolling buffer.
module cnn_gru_top #(
    parameter INPUT_FILE = "mem/dsconv1_dsconv2/board/ram_a_sample0_q12.mem",
    parameter INPUT_EVEN_FILE = "mem/dsconv1_dsconv2/board/sample0_q12_even.mem",
    parameter INPUT_ODD_FILE = "mem/dsconv1_dsconv2/board/sample0_q12_odd.mem",
    parameter CONV1_DW_W_FILE = "mem/dsconv1_dsconv2/weights/conv1_depthwise_W_kh2.mem",
    parameter CONV1_DW_B_FILE = "mem/dsconv1_dsconv2/weights/conv1_depthwise_b.mem",
    parameter CONV1_PW_W_FILE = "mem/dsconv1_dsconv2/weights/conv1_pointwise_W_x3.mem",
    parameter CONV1_PW_B_FILE = "mem/dsconv1_dsconv2/weights/conv1_pointwise_b_x3.mem",
    parameter BN1_A_FILE = "mem/dsconv1_dsconv2/weights/bn1_A.mem",
    parameter BN1_B_FILE = "mem/dsconv1_dsconv2/weights/bn1_B.mem",
    parameter CONV2_DW_W_FILE = "mem/dsconv1_dsconv2/weights/conv2_depthwise_W_kh2.mem",
    parameter CONV2_DW_B_FILE = "mem/dsconv1_dsconv2/weights/conv2_depthwise_b.mem",
    parameter CONV2_PW_W_FILE = "mem/dsconv1_dsconv2/weights/conv2_pointwise_W_x5.mem",
    parameter CONV2_PW_B_FILE = "mem/dsconv1_dsconv2/weights/conv2_pointwise_b_x5.mem",
    parameter BN2_A_FILE = "mem/dsconv1_dsconv2/weights/bn2_A.mem",
    parameter BN2_B_FILE = "mem/dsconv1_dsconv2/weights/bn2_B.mem",
    parameter CONV3_W_FILE = "mem/dsconv1_dsconv2/weights/conv3_W.mem",
    parameter CONV3_B_FILE = "mem/dsconv1_dsconv2/weights/conv3_b.mem",
    parameter CONV3_PACKED_W_FILE = "mem/dsconv1_dsconv2/weights/conv3_W_x3.mem",
    parameter CONV3_PACKED_B_FILE = "mem/dsconv1_dsconv2/weights/conv3_b_x3.mem",
    parameter BN3_A_FILE = "mem/dsconv1_dsconv2/weights/bn3_A.mem",
    parameter BN3_B_FILE = "mem/dsconv1_dsconv2/weights/bn3_B.mem",
    parameter GRU_WR_FILE = "mem/dsconv1_dsconv2/weights/gru_Wr.mem",
    parameter GRU_WZ_FILE = "mem/dsconv1_dsconv2/weights/gru_Wz.mem",
    parameter GRU_WH_FILE = "mem/dsconv1_dsconv2/weights/gru_Wh.mem",
    parameter GRU_UR_FILE = "mem/dsconv1_dsconv2/weights/gru_Ur.mem",
    parameter GRU_UZ_FILE = "mem/dsconv1_dsconv2/weights/gru_Uz.mem",
    parameter GRU_UH_FILE = "mem/dsconv1_dsconv2/weights/gru_Uh.mem",
    parameter GRU_BR_FILE = "mem/dsconv1_dsconv2/weights/gru_br.mem",
    parameter GRU_BZ_FILE = "mem/dsconv1_dsconv2/weights/gru_bz.mem",
    parameter GRU_BH_FILE = "mem/dsconv1_dsconv2/weights/gru_bh.mem",
    parameter ACTIVATION_LUT_FILE = "mem/lut/gru_activation_lut_q15.mem"
) (
    input logic clk, input logic rst_n, input logic start,
    output logic busy, output logic done,
    input logic input_write_en,
    input logic [11:0] input_write_addr,
    input logic signed [15:0] input_write_data,
    output logic input_ready,
    output logic output_valid,
    output logic [31:0] output_addr,
    output logic signed [15:0] output_data,
    input logic result_read_en,
    input logic [15:0] result_read_addr,
    output logic signed [15:0] result_read_data
);
    localparam int RAM_A_DEPTH = 19 * 18 * 20;
    localparam int RAM_ADDR_W = 16;
    localparam int POOL1_ADDR_W = 13;
    localparam int POOL1_OUT_H = 19;
    localparam int POOL1_OUT_W = 18;
    localparam int POOL1_OUT_CH = 20;
    localparam int POOL1_FIRST_COLUMN_LAST_ADDR =
                   (POOL1_OUT_H - 1)
                 + POOL1_OUT_H * POOL1_OUT_W * (POOL1_OUT_CH - 1);

    typedef enum logic [2:0] {
        S_IDLE, S_START_CNN, S_RUN_CNN,
        S_START_GRU, S_RUN_GRU, S_DONE
    } state_t;
    state_t state;

    logic conv1_start, conv1_busy, conv1_valid, conv1_last;
    logic [31:0] conv1_input_addr_kh0, conv1_input_addr_kh1;
    logic [4:0] conv1_input_h_unused;
    logic [2:0] conv1_input_kw_unused;
    logic conv1_input_channel_unused;
    logic [31:0] conv1_local_addr, conv1_addr;
    logic [4:0] conv1_h, conv1_channel;
    logic conv1_w_unused;
    logic signed [15:0] conv1_data;
    logic [7:0] conv1_column;
    logic [2:0] conv1_write_slot;

    logic conv2_start, conv2_busy, conv2_valid, conv2_last;
    logic conv2_global_last;
    logic [31:0] conv2_input_addr, conv2_input_addr_kh1;
    logic [4:0] conv2_input_h, conv2_input_channel;
    logic [2:0] conv2_input_kw;
    logic [31:0] conv2_local_addr, conv2_addr;
    logic [4:0] conv2_h, conv2_channel;
    logic conv2_w_unused;
    logic signed [15:0] conv2_data;
    logic [7:0] conv2_column;
    logic [2:0] conv2_window_slot;
    logic conv12_start, conv12_busy, conv12_done;
    logic conv12_finished;

    logic conv3_start, conv3_busy, conv3_valid;
    logic [31:0] conv3_input_addr, conv3_addr;
    logic [31:0] conv3_required_w;
    logic conv3_input_ready;
    logic conv3_started;
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
    logic ram_a_read_en, ram_a_write_en;
    logic [RAM_ADDR_W-1:0] ram_a_write_addr;
    logic signed [15:0] ram_a_read_data, ram_a_write_data;
    logic signed [15:0] input_shadow_data_kh0, input_shadow_data_kh1;
    logic signed [15:0] conv1_buffer_kh0, conv1_buffer_kh1;
    logic signed [15:0] shared_pool2_gru_data;
    logic pool1_finished, pool2_finished;
    logic [4:0] pool1_columns_ready;
    logic [12:0] pool1_column_last_addr;

    always_comb begin
        conv12_start = (state == S_START_CNN);
        pool1_start = conv2_start && (conv2_column == 0);
        // Pool1 完成前 5 欄後即可啟動第一個 Conv3 視窗。
        conv3_start = (state == S_RUN_CNN) && !conv3_started
                    && (pool1_columns_ready >= 5);
        pool2_start = conv3_start;
        gru_start = (state == S_START_GRU);
        conv2_global_last = conv2_last && (conv2_column == 151);
        conv3_input_ready = pool1_finished
                         || (conv3_required_w < pool1_columns_ready);

        conv1_addr = conv1_h + 20 * (conv1_column + 156 * conv1_channel);
        conv2_addr = conv2_h + 19 * (conv2_column + 152 * conv2_channel);
    end

    // RAM A keeps the input, Pool1 output and final GRU result.
    always_comb begin
        ram_a_read_en = result_read_en;
        if (conv3_start || (conv3_started && !pool2_finished)) begin
            ram_a_internal_read_addr = conv3_input_addr[15:0];
            ram_a_read_en = 1'b1;
        end
        else begin
            ram_a_internal_read_addr = result_read_addr;
        end

        ram_a_write_en = (input_ready && input_write_en) || pool1_valid || gru_valid;
        if (gru_valid) begin
            ram_a_write_addr = gru_addr[15:0];
            ram_a_write_data = gru_data;
        end else if (input_ready && input_write_en) begin
            ram_a_write_addr = {{(RAM_ADDR_W-12){1'b0}}, input_write_addr};
            ram_a_write_data = input_write_data;
        end else begin
            ram_a_write_addr = {{(RAM_ADDR_W-POOL1_ADDR_W){1'b0}}, pool1_addr};
            ram_a_write_data = pool1_data;
        end
    end

    activation_ram #(
        .DATA_W(16), .DEPTH(RAM_A_DEPTH), .ADDR_W(RAM_ADDR_W),
        .MEM_FILE(INPUT_FILE), .USE_READ_ENABLE(1'b1)
    ) u_ram_a (
        .clk(clk), .write_en(ram_a_write_en),
        .write_addr(ram_a_write_addr), .write_data(ram_a_write_data),
        .read_en(ram_a_read_en), .read_addr(ram_a_internal_read_addr),
        .read_data(ram_a_read_data)
    );

    // 原始 EEG 依線性位址奇偶分 bank，Conv1 每拍讀取 kh0/kh1。
    input_banked_ram #(
        .DATA_W(16), .DEPTH(21*160), .ADDR_W(12),
        .EVEN_MEM_FILE(INPUT_EVEN_FILE), .ODD_MEM_FILE(INPUT_ODD_FILE)
    ) u_input_banks (
        .clk(clk), .rst_n(rst_n),
        .write_en(input_ready && input_write_en),
        .write_addr(input_write_addr), .write_data(input_write_data),
        .read_en(conv1_start || conv1_busy),
        .read_addr_kh0(conv1_input_addr_kh0[11:0]),
        .read_addr_kh1(conv1_input_addr_kh1[11:0]),
        .read_data_kh0(input_shadow_data_kh0),
        .read_data_kh1(input_shadow_data_kh1)
    );

    ds_conv2_bn_relu_block #(
        .IN_H(21), .IN_W(5), .IN_CH(1),
        .K_H(2), .K_W(5), .OUT_CH(21), .LANES(3),
        .DW_BIAS_SHIFT(12), .DW_OUTPUT_SHIFT(14),
        .PW_BIAS_SHIFT(13), .PW_OUTPUT_SHIFT(13),
        .BN_BIAS_SHIFT(13), .BN_OUTPUT_SHIFT(13),
        .DW_WEIGHT_FILE(CONV1_DW_W_FILE), .DW_BIAS_FILE(CONV1_DW_B_FILE),
        .PW_WEIGHT_FILE(CONV1_PW_W_FILE), .PW_BIAS_FILE(CONV1_PW_B_FILE),
        .BN_A_FILE(BN1_A_FILE), .BN_B_FILE(BN1_B_FILE)
    ) u_conv1_bn_relu (
        .clk(clk), .rst_n(rst_n), .start(conv1_start),
        .input_base_addr(conv1_column * 21), .busy(conv1_busy),
        .input_addr_kh0(conv1_input_addr_kh0),
        .input_addr_kh1(conv1_input_addr_kh1),
        .input_issue_h(conv1_input_h_unused),
        .input_issue_kw(conv1_input_kw_unused),
        .input_issue_channel(conv1_input_channel_unused),
        .input_data_kh0(input_shadow_data_kh0),
        .input_data_kh1(input_shadow_data_kh1),
        .output_valid(conv1_valid), .output_last(conv1_last),
        .output_addr(conv1_local_addr), .output_h(conv1_h),
        .output_w(conv1_w_unused), .output_channel(conv1_channel),
        .output_data(conv1_data)
    );

    // 重疊排程器
    // 6 個 slot 讓 Conv1 寫入下一欄時，Conv2 同時讀取目前 5 欄。
    dsconv12_overlap_scheduler u_conv12_scheduler (
        .clk(clk), .rst_n(rst_n), .start(conv12_start),
        .conv1_column_done(conv1_valid && conv1_last),
        .conv2_window_done(conv2_valid && conv2_last),
        .busy(conv12_busy), .done(conv12_done),
        .conv1_start(conv1_start), .conv1_column(conv1_column),
        .conv1_write_slot(conv1_write_slot),
        .conv2_start(conv2_start), .conv2_column(conv2_column),
        .conv2_window_slot(conv2_window_slot)
    );

    dsconv12_window_buffer #(.COLS(6)) u_conv12_buffer (
        .clk(clk), .rst_n(rst_n),
        .write_en(conv1_valid), .write_slot(conv1_write_slot),
        .write_h(conv1_h), .write_channel(conv1_channel),
        .write_data(conv1_data),
        .read_en(conv2_start || conv2_busy),
        .window_base_slot(conv2_window_slot),
        .read_h(conv2_input_h),
        .read_kw(conv2_input_kw),
        .read_channel(conv2_input_channel),
        .read_data_kh0(conv1_buffer_kh0),
        .read_data_kh1(conv1_buffer_kh1)
    );

    ds_conv2_bn_relu_block #(
        .IN_H(20), .IN_W(5), .IN_CH(21),
        .K_H(2), .K_W(5), .OUT_CH(20), .LANES(5),
        .DW_BIAS_SHIFT(11), .DW_OUTPUT_SHIFT(15),
        .PW_BIAS_SHIFT(11), .PW_OUTPUT_SHIFT(15),
        .BN_BIAS_SHIFT(11), .BN_OUTPUT_SHIFT(14),
        .DW_WEIGHT_FILE(CONV2_DW_W_FILE), .DW_BIAS_FILE(CONV2_DW_B_FILE),
        .PW_WEIGHT_FILE(CONV2_PW_W_FILE), .PW_BIAS_FILE(CONV2_PW_B_FILE),
        .BN_A_FILE(BN2_A_FILE), .BN_B_FILE(BN2_B_FILE)
    ) u_conv2_bn_relu (
        .clk(clk), .rst_n(rst_n), .start(conv2_start),
        .input_base_addr(32'd0), .busy(conv2_busy),
        .input_addr_kh0(conv2_input_addr),
        .input_addr_kh1(conv2_input_addr_kh1),
        .input_issue_h(conv2_input_h),
        .input_issue_kw(conv2_input_kw),
        .input_issue_channel(conv2_input_channel),
        .input_data_kh0(conv1_buffer_kh0),
        .input_data_kh1(conv1_buffer_kh1),
        .output_valid(conv2_valid), .output_last(conv2_last),
        .output_addr(conv2_local_addr), .output_h(conv2_h),
        .output_w(conv2_w_unused), .output_channel(conv2_channel),
        .output_data(conv2_data)
    );

    dsconv_streaming_pool #(
        .IN_H(19), .IN_W(152), .IN_CH(20),
        .POOL_W(10), .STRIDE_W(8), .INPUT_F(11), .OUTPUT_F(11)
    ) u_pool1 (
        .clk(clk), .rst_n(rst_n), .start(pool1_start),
        .busy(pool1_busy), .done(pool1_done),
        .input_valid(conv2_valid), .input_last(conv2_global_last),
        .input_data(conv2_data), .input_h(conv2_h),
        .input_w(conv2_column), .input_channel(conv2_channel),
        .output_valid(pool1_valid), .output_addr(pool1_addr),
        .output_data(pool1_data)
    );

    conv_bn_relu_parallel_block #(
        .IN_H(19), .IN_W(18), .IN_CH(20),
        .K_H(2), .K_W(5), .OUT_CH(15), .LANES(3),
        .REGISTER_MAC_INPUTS(1'b1),
        .CONV_BIAS_SHIFT(11), .CONV_OUTPUT_SHIFT(17),
        .BN_BIAS_SHIFT(10), .BN_OUTPUT_SHIFT(12),
        .PACKED_WEIGHT_FILE(CONV3_PACKED_W_FILE),
        .PACKED_BIAS_FILE(CONV3_PACKED_B_FILE),
        .BN_A_FILE(BN3_A_FILE), .BN_B_FILE(BN3_B_FILE)
    ) u_conv3_bn_relu (
        .clk(clk), .rst_n(rst_n), .start(conv3_start), .busy(conv3_busy),
        .input_addr(conv3_input_addr), .input_required_w(conv3_required_w),
        .input_ready(conv3_input_ready), .input_data(ram_a_read_data),
        .output_valid(conv3_valid), .output_addr(conv3_addr),
        .output_data(conv3_data)
    );

    streaming_maxpool #(
        .IN_H(18), .IN_W(14), .IN_CH(15),
        .POOL_W(10), .STRIDE_W(8), .LANES(3),
        .INPUT_F(12), .OUTPUT_F(12)
    ) u_pool2 (
        .clk(clk), .rst_n(rst_n), .start(pool2_start),
        .busy(pool2_busy), .done(pool2_done),
        .input_valid(conv3_valid), .input_data(conv3_data),
        .output_valid(pool2_valid), .output_addr(pool2_addr),
        .output_data(pool2_data)
    );

    pool2_gru_ram u_pool2_gru_ram (
        .clk(clk), .write_en(pool2_valid), .write_addr(pool2_addr[8:0]),
        .write_data(pool2_data),
        .read_en((state == S_START_GRU) || (state == S_RUN_GRU)),
        .read_addr(gru_input_addr[8:0]), .read_data(shared_pool2_gru_data)
    );

    gru_engine_pipeline #(
        .WR_FILE(GRU_WR_FILE), .WZ_FILE(GRU_WZ_FILE), .WH_FILE(GRU_WH_FILE),
        .UR_FILE(GRU_UR_FILE), .UZ_FILE(GRU_UZ_FILE), .UH_FILE(GRU_UH_FILE),
        .BR_FILE(GRU_BR_FILE), .BZ_FILE(GRU_BZ_FILE), .BH_FILE(GRU_BH_FILE),
        .ACTIVATION_LUT_FILE(ACTIVATION_LUT_FILE)
    ) u_gru (
        .clk(clk), .rst_n(rst_n), .start(gru_start),
        .busy(gru_busy), .done(gru_done),
        .input_addr(gru_input_addr), .input_data(shared_pool2_gru_data),
        .output_valid(gru_valid), .output_addr(gru_addr), .output_data(gru_data)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            conv12_finished <= 1'b0;
            pool1_finished <= 1'b0;
            pool2_finished <= 1'b0;
            conv3_started <= 1'b0;
            pool1_columns_ready <= '0;
            pool1_column_last_addr <= POOL1_FIRST_COLUMN_LAST_ADDR;
        end else begin
            if (conv12_done)
                conv12_finished <= 1'b1;
            if (pool1_done)
                pool1_finished <= 1'b1;
            if (pool2_done)
                pool2_finished <= 1'b1;
            if (conv3_start)
                conv3_started <= 1'b1;

            // 每當一個 Pool1 欄位的最後一筆寫入 RAM，才公開該欄給 Conv3。
            if (pool1_valid && (pool1_addr == pool1_column_last_addr)) begin
                pool1_columns_ready <= pool1_columns_ready + 1'b1;
                pool1_column_last_addr <= pool1_column_last_addr
                                        + POOL1_OUT_H;
            end

            case (state)
                S_IDLE: if (start) begin
                    conv12_finished <= 1'b0;
                    pool1_finished <= 1'b0;
                    pool2_finished <= 1'b0;
                    conv3_started <= 1'b0;
                    pool1_columns_ready <= '0;
                    pool1_column_last_addr <= POOL1_FIRST_COLUMN_LAST_ADDR;
                    state <= S_START_CNN;
                end
                S_START_CNN: state <= S_RUN_CNN;
                S_RUN_CNN: begin
                    if ((conv12_finished || conv12_done) &&
                        (pool1_finished || pool1_done) &&
                        (pool2_finished || pool2_done))
                        state <= S_START_GRU;
                end
                S_START_GRU: state <= S_RUN_GRU;
                S_RUN_GRU: if (gru_done) state <= S_DONE;
                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

    assign result_read_data = ram_a_read_data;
    always_comb begin
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
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
        .MEM_FILE(WEIGHT_FILE), .USE_READ_ENABLE(1'b1)
    ) u_weight_rom (
        .clk(clk), .read_en(start || busy),
        .addr(weight_addr[WEIGHT_ADDR_W-1:0]),
        .data(weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(BIAS_DEPTH), .ADDR_W(BIAS_ADDR_W),
        .MEM_FILE(BIAS_FILE), .USE_READ_ENABLE(1'b1)
    ) u_bias_rom (
        .clk(clk), .read_en(start || busy),
        .addr(bias_addr[BIAS_ADDR_W-1:0]),
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
    parameter bit REGISTER_MAC_INPUTS = 1'b0,
    parameter bit REGISTER_OUTPUT = 1'b0,
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
    output logic [31:0]        input_required_w,
    input  logic               input_ready,
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
    logic output_valid_d;
    logic [31:0] output_addr_d;
    logic signed [15:0] output_data_d;

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(PACKED_WEIGHT_DEPTH),
        .ADDR_W(PACKED_WEIGHT_ADDR_W), .MEM_FILE(PACKED_WEIGHT_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_packed_weight_rom (
        .clk(clk), .read_en(start || busy),
        .addr(weight_addr[PACKED_WEIGHT_ADDR_W-1:0]),
        .data(packed_weight_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OUT_GROUPS),
        .ADDR_W(PACKED_BIAS_ADDR_W), .MEM_FILE(PACKED_BIAS_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_packed_bias_rom (
        .clk(clk), .read_en(start || busy),
        .addr(bias_addr[PACKED_BIAS_ADDR_W-1:0]),
        .data(packed_bias_data)
    );

    conv_engine_parallel_counter #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W), .LANES(LANES),
        .BIAS_SHIFT(CONV_BIAS_SHIFT),
        .OUTPUT_SHIFT(CONV_OUTPUT_SHIFT),
        .REGISTER_MAC_INPUTS(REGISTER_MAC_INPUTS)
    ) u_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(conv_done_unused),
        .input_addr(input_addr), .input_required_w(input_required_w),
        .input_ready(input_ready), .input_data(input_data),
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
            output_valid_d <= 1'b0;
            output_addr_d <= '0;
            output_data_d <= '0;
        end else begin
            valid_d1 <= conv_valid;
            if (conv_valid)
                addr_d1 <= conv_addr;
            if (valid_d1)
                addr_d2 <= addr_d1;

            // Conv1 可選擇暫存 ReLU 輸出，切開 BN/ReLU 到 RAM 的長路徑。
            output_valid_d <= relu_valid;
            if (relu_valid) begin
                output_addr_d <= addr_d2;
                output_data_d <= relu_data;
            end
        end
    end

    always_comb begin
        if (REGISTER_OUTPUT) begin
            output_valid = output_valid_d;
            output_addr  = output_addr_d;
            output_data  = output_data_d;
        end else begin
            output_valid = relu_valid;
            output_addr  = addr_d2;
            output_data  = relu_data;
        end
    end
endmodule
