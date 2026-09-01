// Conv2 block，將 conv_engine_parallel_kh2 的輸出加上 BN 和 ReLU

module conv_bn_relu_parallel_kh2_block #(
    parameter int IN_H   = 20,
    parameter int IN_W   = 156,
    parameter int IN_CH  = 21,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 20,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int LANES  = 5,
    parameter int CONV_BIAS_SHIFT   = 10,
    parameter int CONV_OUTPUT_SHIFT = 15,
    parameter int BN_BIAS_SHIFT     = 11,
    parameter int BN_OUTPUT_SHIFT   = 14,
    parameter int RELU_LEFT_SHIFT   = 1,
    parameter PACKED_WEIGHT_FILE = "mem/weights/conv2_W_x5_kh2.mem",
    parameter PACKED_BIAS_FILE   = "mem/weights/conv2_b_x5.mem",
    parameter BN_A_FILE          = "mem/weights/bn2_A.mem",
    parameter BN_B_FILE          = "mem/weights/bn2_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic [31:0]        input_addr_kh0,
    output logic [31:0]        input_addr_kh1,
    input  logic signed [15:0] input_data_kh0,
    input  logic signed [15:0] input_data_kh1,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int OUT_GROUPS = (OUT_CH + LANES - 1) / LANES;
    localparam int PACKED_WEIGHT_DEPTH = K_W * IN_CH * OUT_GROUPS;
    localparam int PACKED_WEIGHT_ADDR_W = $clog2(PACKED_WEIGHT_DEPTH);
    localparam int PACKED_BIAS_ADDR_W = $clog2(OUT_GROUPS);

    logic conv_done_unused, conv_valid;
    logic [31:0] weight_addr, bias_addr, conv_addr, conv_channel;
    logic signed [(32*LANES)-1:0] packed_weight_data;
    logic signed [(16*LANES)-1:0] packed_bias_data;
    logic signed [15:0] conv_data;
    logic bn_valid, relu_valid, valid_d1;
    logic signed [15:0] bn_data, relu_data;
    logic [31:0] addr_d1, addr_d2;

    weight_rom #(
        .DATA_W(32*LANES), .DEPTH(PACKED_WEIGHT_DEPTH),
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

    conv_engine_parallel_kh2 #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W), .LANES(LANES),
        .BIAS_SHIFT(CONV_BIAS_SHIFT),
        .OUTPUT_SHIFT(CONV_OUTPUT_SHIFT)
    ) u_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(conv_done_unused),
        .input_addr_kh0(input_addr_kh0),
        .input_addr_kh1(input_addr_kh1),
        .input_data_kh0(input_data_kh0),
        .input_data_kh1(input_data_kh1),
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

    // BN has two registered stages. Delay the convolution address equally.
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
