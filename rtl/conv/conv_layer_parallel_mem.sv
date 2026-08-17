// testing

// Standalone memory wrapper used to verify the parallel engine before it is
// connected to cnn_gru_top. Packed weight lines contain lane 3..lane 0 from
// MSB to LSB; lane 0 therefore occupies bits [15:0].
module conv_layer_parallel_mem #(
    parameter int IN_H   = 20,
    parameter int IN_W   = 156,
    parameter int IN_CH  = 21,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 20,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int LANES  = 4,
    parameter int BIAS_SHIFT   = 10,
    parameter int OUTPUT_SHIFT = 15,
    parameter INPUT_FILE         = "mem/golden/q_relu1_act.mem",
    parameter PACKED_WEIGHT_FILE = "mem/weights/conv2_W_x4.mem",
    parameter PACKED_BIAS_FILE   = "mem/weights/conv2_b_x4.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic [31:0]        output_ch_idx,
    output logic signed [15:0] output_data
);
    localparam int INPUT_DEPTH = IN_H * IN_W * IN_CH;
    localparam int INPUT_ADDR_W = $clog2(INPUT_DEPTH);
    localparam int OUT_GROUPS = (OUT_CH + LANES - 1) / LANES;
    localparam int PACKED_WEIGHT_DEPTH = K_H * K_W * IN_CH * OUT_GROUPS;
    localparam int PACKED_WEIGHT_ADDR_W = $clog2(PACKED_WEIGHT_DEPTH);
    localparam int PACKED_BIAS_ADDR_W = $clog2(OUT_GROUPS);

    logic [31:0] input_addr;
    logic [31:0] weight_addr;
    logic [31:0] bias_addr;
    logic signed [15:0] input_data;
    logic signed [(16*LANES)-1:0] packed_weight_data;
    logic signed [(16*LANES)-1:0] packed_bias_data;

    activation_ram #(
        .DATA_W(16), .DEPTH(INPUT_DEPTH), .ADDR_W(INPUT_ADDR_W),
        .MEM_FILE(INPUT_FILE)
    ) u_input_ram (
        .clk(clk), .write_en(1'b0),
        .write_addr('0), .write_data('0),
        .read_addr(input_addr[INPUT_ADDR_W-1:0]),
        .read_data(input_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(PACKED_WEIGHT_DEPTH),
        .ADDR_W(PACKED_WEIGHT_ADDR_W), .MEM_FILE(PACKED_WEIGHT_FILE)
    ) u_packed_weight_rom (
        .clk(clk),
        .addr(weight_addr[PACKED_WEIGHT_ADDR_W-1:0]),
        .data(packed_weight_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OUT_GROUPS),
        .ADDR_W(PACKED_BIAS_ADDR_W), .MEM_FILE(PACKED_BIAS_FILE)
    ) u_packed_bias_rom (
        .clk(clk),
        .addr(bias_addr[PACKED_BIAS_ADDR_W-1:0]),
        .data(packed_bias_data)
    );

    conv_engine_parallel #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W), .LANES(LANES),
        .BIAS_SHIFT(BIAS_SHIFT), .OUTPUT_SHIFT(OUTPUT_SHIFT)
    ) u_parallel_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_addr(input_addr), .input_data(input_data),
        .weight_addr(weight_addr), .weight_data(packed_weight_data),
        .bias_addr(bias_addr), .bias_data(packed_bias_data),
        .output_valid(output_valid), .output_addr(output_addr),
        .output_ch_idx(output_ch_idx), .output_data(output_data)
    );
endmodule