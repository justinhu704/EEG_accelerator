// Convolution arithmetic engine with external synchronous memory interfaces.
module conv_engine #(
    parameter int IN_H   = 21,
    parameter int IN_W   = 160,
    parameter int IN_CH  = 1,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 21,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int BIAS_SHIFT   = 12,
    parameter int OUTPUT_SHIFT = 14
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    output logic [31:0]        input_addr,
    input  logic signed [15:0] input_data,
    output logic [31:0]        weight_addr,
    input  logic signed [15:0] weight_data,
    output logic [31:0]        bias_addr,
    input  logic signed [15:0] bias_data,

    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic [31:0]        output_ch_idx,
    output logic signed [15:0] output_data
);
    logic clear_acc;
    logic mac_en;
    logic signed [47:0] accumulator;
    logic signed [47:0] bias_extended;
    logic signed [47:0] bias_aligned;
    logic signed [47:0] sum_with_bias;
    logic signed [47:0] scaled_result;

    conv_controller #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W)
    ) u_controller (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .clear_acc(clear_acc), .mac_en(mac_en),
        .output_valid(output_valid),
        .input_addr(input_addr),
        .weight_addr(weight_addr),
        .bias_addr(bias_addr),
        .output_addr(output_addr),
        .output_ch_idx(output_ch_idx)
    );

    pe_mac u_mac (
        .clk(clk), .rst_n(rst_n),
        .clear_acc(clear_acc), .mac_en(mac_en),
        .data_in(input_data), .weight_in(weight_data),
        .accumulator(accumulator)
    );

    always_comb begin
        bias_extended = {{32{bias_data[15]}}, bias_data};
        bias_aligned  = bias_extended <<< BIAS_SHIFT;
        sum_with_bias = accumulator + bias_aligned;
        scaled_result = sum_with_bias >>> OUTPUT_SHIFT;
    end

    sat16 u_sat16 (
        .value_in(scaled_result),
        .value_out(output_data)
    );
endmodule
