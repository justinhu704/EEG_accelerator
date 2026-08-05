// Standalone convolution layer wrapper with synchronous activation, weight,
// and bias memories. This keeps conv_engine reusable for later CNN integration.
module conv_layer_mem #(
    parameter int IN_H   = 21,
    parameter int IN_W   = 160,
    parameter int IN_CH  = 1,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 21,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int INPUT_DEPTH  = IN_H * IN_W * IN_CH,
    parameter int WEIGHT_DEPTH = K_H * K_W * IN_CH * OUT_CH,
    parameter int BIAS_DEPTH   = OUT_CH,
    parameter int BIAS_SHIFT   = 12,
    parameter int OUTPUT_SHIFT = 14,
    parameter INPUT_FILE  = "mem/golden/q_in_act.mem",
    parameter WEIGHT_FILE = "mem/weights/conv1_W.mem",
    parameter BIAS_FILE   = "mem/weights/conv1_b.mem"
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
    localparam int INPUT_ADDR_W  = $clog2(INPUT_DEPTH);
    localparam int WEIGHT_ADDR_W = $clog2(WEIGHT_DEPTH);
    localparam int BIAS_ADDR_W   = $clog2(BIAS_DEPTH);

    logic [31:0] input_addr;
    logic [31:0] weight_addr;
    logic [31:0] bias_addr;
    logic signed [15:0] input_data;
    logic signed [15:0] weight_data;
    logic signed [15:0] bias_data;

    activation_ram #(
        .DATA_W(16), .DEPTH(INPUT_DEPTH), .ADDR_W(INPUT_ADDR_W),
        .MEM_FILE(INPUT_FILE)
    ) u_input_ram (
        .clk(clk),
        .write_en(1'b0), .write_addr('0), .write_data('0),
        .read_addr(input_addr[INPUT_ADDR_W-1:0]),
        .read_data(input_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(WEIGHT_DEPTH), .ADDR_W(WEIGHT_ADDR_W),
        .MEM_FILE(WEIGHT_FILE)
    ) u_weight_rom (
        .clk(clk),
        .addr(weight_addr[WEIGHT_ADDR_W-1:0]),
        .data(weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(BIAS_DEPTH), .ADDR_W(BIAS_ADDR_W),
        .MEM_FILE(BIAS_FILE)
    ) u_bias_rom (
        .clk(clk),
        .addr(bias_addr[BIAS_ADDR_W-1:0]),
        .data(bias_data)
    );

    conv_engine #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W),
        .BIAS_SHIFT(BIAS_SHIFT), .OUTPUT_SHIFT(OUTPUT_SHIFT)
    ) u_engine (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_addr(input_addr), .input_data(input_data),
        .weight_addr(weight_addr), .weight_data(weight_data),
        .bias_addr(bias_addr), .bias_data(bias_data),
        .output_valid(output_valid),
        .output_addr(output_addr),
        .output_ch_idx(output_ch_idx),
        .output_data(output_data)
    );
endmodule
