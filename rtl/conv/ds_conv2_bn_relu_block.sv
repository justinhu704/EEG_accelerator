// DS-Conv2 -> BN2 -> ReLU2 wrapper.
module ds_conv2_bn_relu_block #(
    parameter int IN_H   = 20,
    parameter int IN_W   = 156,
    parameter int IN_CH  = 21,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 20,
    parameter int LANES  = 5,
    parameter int DW_BIAS_SHIFT = 10,
    parameter int DW_OUTPUT_SHIFT = 15,
    parameter int PW_BIAS_SHIFT = 10,
    parameter int PW_OUTPUT_SHIFT = 15,
    parameter int BN_BIAS_SHIFT = 10,
    parameter int BN_OUTPUT_SHIFT = 13,
    parameter int RELU_LEFT_SHIFT = 0,
    parameter DW_WEIGHT_FILE = "mem/dsconv2/weights/conv2_depthwise_W_kh2.mem",
    parameter DW_BIAS_FILE = "mem/dsconv2/weights/conv2_depthwise_b.mem",
    parameter PW_WEIGHT_FILE = "mem/dsconv2/weights/conv2_pointwise_W_x5.mem",
    parameter PW_BIAS_FILE = "mem/dsconv2/weights/conv2_pointwise_b_x5.mem",
    parameter BN_A_FILE = "mem/dsconv2/weights/bn2_A.mem",
    parameter BN_B_FILE = "mem/dsconv2/weights/bn2_B.mem"
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic busy,
    output logic [31:0] input_addr_kh0,
    output logic [31:0] input_addr_kh1,
    input  logic signed [15:0] input_data_kh0,
    input  logic signed [15:0] input_data_kh1,
    output logic output_valid,
    output logic output_last,
    output logic [31:0] output_addr,
    output logic [$clog2(IN_H-K_H+1)-1:0] output_h,
    output logic [$clog2(IN_W-K_W+1)-1:0] output_w,
    output logic [$clog2(OUT_CH)-1:0] output_channel,
    output logic signed [15:0] output_data
);
    localparam int OUT_H = IN_H - K_H + 1;
    localparam int OUT_W = IN_W - K_W + 1;
    localparam int OUT_GROUPS = OUT_CH / LANES;
    localparam int DW_WEIGHT_DEPTH = IN_CH * K_W;
    localparam int PW_WEIGHT_DEPTH = IN_CH * OUT_GROUPS;

    logic [$clog2(DW_WEIGHT_DEPTH)-1:0] dw_weight_addr;
    logic [$clog2(IN_CH)-1:0] dw_bias_addr;
    logic [$clog2(PW_WEIGHT_DEPTH)-1:0] pw_weight_addr;
    logic [$clog2(OUT_GROUPS)-1:0] pw_bias_addr;
    logic signed [31:0] dw_weight_data;
    logic signed [15:0] dw_bias_data;
    logic signed [(16*LANES)-1:0] pw_weight_data;
    logic signed [(16*LANES)-1:0] pw_bias_data;

    logic conv_done_unused;
    logic conv_valid, conv_last;
    logic [31:0] conv_addr;
    logic [$clog2(OUT_H)-1:0] conv_h;
    logic [$clog2(OUT_W)-1:0] conv_w;
    logic [$clog2(OUT_CH)-1:0] conv_channel;
    logic signed [15:0] conv_data;

    logic bn_valid, relu_valid;
    logic signed [15:0] bn_data, relu_data;
    logic metadata_valid_d1;
    logic [31:0] addr_d1, addr_d2;
    logic [$clog2(OUT_H)-1:0] h_d1, h_d2;
    logic [$clog2(OUT_W)-1:0] w_d1, w_d2;
    logic [$clog2(OUT_CH)-1:0] channel_d1, channel_d2;
    logic last_d1, last_d2;

    weight_rom #(
        .DATA_W(32), .DEPTH(DW_WEIGHT_DEPTH),
        .ADDR_W($clog2(DW_WEIGHT_DEPTH)), .MEM_FILE(DW_WEIGHT_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_dw_weight_rom (
        .clk(clk), .read_en(start || busy),
        .addr(dw_weight_addr), .data(dw_weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(IN_CH), .ADDR_W($clog2(IN_CH)),
        .MEM_FILE(DW_BIAS_FILE), .USE_READ_ENABLE(1'b1)
    ) u_dw_bias_rom (
        .clk(clk), .read_en(start || busy),
        .addr(dw_bias_addr), .data(dw_bias_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(PW_WEIGHT_DEPTH),
        .ADDR_W($clog2(PW_WEIGHT_DEPTH)), .MEM_FILE(PW_WEIGHT_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_pw_weight_rom (
        .clk(clk), .read_en(start || busy),
        .addr(pw_weight_addr), .data(pw_weight_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OUT_GROUPS),
        .ADDR_W($clog2(OUT_GROUPS)), .MEM_FILE(PW_BIAS_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_pw_bias_rom (
        .clk(clk), .read_en(start || busy),
        .addr(pw_bias_addr), .data(pw_bias_data)
    );

    ds_conv2_engine #(
        .INPUT_ADDR_WIDTH(32), .OUTPUT_ADDR_WIDTH(32),
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH), .LANES(LANES),
        .DW_BIAS_SHIFT(DW_BIAS_SHIFT),
        .DW_OUT_SHIFT(DW_OUTPUT_SHIFT),
        .PW_BIAS_SHIFT(PW_BIAS_SHIFT),
        .PW_OUT_SHIFT(PW_OUTPUT_SHIFT)
    ) u_ds_conv2 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(conv_done_unused),
        .input_addr_kh0(input_addr_kh0),
        .input_addr_kh1(input_addr_kh1),
        .input_data_kh0(input_data_kh0),
        .input_data_kh1(input_data_kh1),
        .dw_weight_addr(dw_weight_addr), .dw_weight_data(dw_weight_data),
        .dw_bias_addr(dw_bias_addr), .dw_bias_data(dw_bias_data),
        .pw_weight_addr(pw_weight_addr), .pw_weight_data(pw_weight_data),
        .pw_bias_addr(pw_bias_addr), .pw_bias_data(pw_bias_data),
        .output_valid(conv_valid), .output_last(conv_last),
        .output_data(conv_data), .output_addr(conv_addr),
        .output_h(conv_h), .output_w(conv_w),
        .output_channel(conv_channel)
    );

    bn_affine #(
        .CHANNELS(OUT_CH),
        .BIAS_SHIFT(BN_BIAS_SHIFT), .OUTPUT_SHIFT(BN_OUTPUT_SHIFT),
        .A_FILE(BN_A_FILE), .B_FILE(BN_B_FILE)
    ) u_bn2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(conv_valid), .in_data(conv_data),
        .in_ch_idx({{(32-$clog2(OUT_CH)){1'b0}}, conv_channel}),
        .out_valid(bn_valid), .out_data(bn_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(RELU_LEFT_SHIFT)) u_relu2 (
        .in_valid(bn_valid), .in_data(bn_data),
        .out_valid(relu_valid), .out_data(relu_data)
    );

    // BN 有兩級暫存，空間座標與 last 必須同步延遲。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            metadata_valid_d1 <= 1'b0;
            addr_d1 <= '0;
            addr_d2 <= '0;
            h_d1 <= '0;
            h_d2 <= '0;
            w_d1 <= '0;
            w_d2 <= '0;
            channel_d1 <= '0;
            channel_d2 <= '0;
            last_d1 <= 1'b0;
            last_d2 <= 1'b0;
        end else begin
            metadata_valid_d1 <= conv_valid;
            if (conv_valid) begin
                addr_d1 <= conv_addr;
                h_d1 <= conv_h;
                w_d1 <= conv_w;
                channel_d1 <= conv_channel;
                last_d1 <= conv_last;
            end
            if (metadata_valid_d1) begin
                addr_d2 <= addr_d1;
                h_d2 <= h_d1;
                w_d2 <= w_d1;
                channel_d2 <= channel_d1;
                last_d2 <= last_d1;
            end
        end
    end

    always_comb begin
        output_valid = relu_valid;
        output_last = last_d2;
        output_addr = addr_d2;
        output_h = h_d2;
        output_w = w_d2;
        output_channel = channel_d2;
        output_data = relu_data;
    end
endmodule
