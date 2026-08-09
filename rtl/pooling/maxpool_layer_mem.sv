// Standalone simulation/unit-test wrapper around maxpool_engine.
// Integrated CNN designs connect maxpool_engine directly to activation RAM.
module maxpool_layer_mem #(
    parameter int IN_H        = 19,
    parameter int IN_W        = 152,
    parameter int IN_CH       = 20,
    parameter int POOL_H      = 1,
    parameter int POOL_W      = 10,
    parameter int STRIDE_H    = 1,
    parameter int STRIDE_W    = 8,
    parameter int OUT_H       = ((IN_H - POOL_H) / STRIDE_H) + 1,
    parameter int OUT_W       = ((IN_W - POOL_W) / STRIDE_W) + 1,
    parameter int INPUT_F     = 12,
    parameter int OUTPUT_F    = 12,
    parameter int INPUT_DEPTH = IN_H * IN_W * IN_CH,
    parameter     INPUT_FILE  = "../mem/golden/q_relu2_act.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int INPUT_ADDR_W = (INPUT_DEPTH <= 1)
                                ? 1 : $clog2(INPUT_DEPTH);
    logic [31:0] input_addr;
    logic signed [15:0] input_data;

    activation_ram #(
        .DATA_W(16),
        .DEPTH(INPUT_DEPTH),
        .ADDR_W(INPUT_ADDR_W),
        .MEM_FILE(INPUT_FILE)
    ) u_input_ram (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0),
        .write_data('0),
        .read_addr(input_addr[INPUT_ADDR_W-1:0]),
        .read_data(input_data)
    );

    maxpool_engine #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .POOL_H(POOL_H), .POOL_W(POOL_W),
        .STRIDE_H(STRIDE_H), .STRIDE_W(STRIDE_W),
        .OUT_H(OUT_H), .OUT_W(OUT_W),
        .INPUT_F(INPUT_F), .OUTPUT_F(OUTPUT_F)
    ) u_engine (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_addr(input_addr), .input_data(input_data),
        .output_valid(output_valid),
        .output_addr(output_addr), .output_data(output_data)
    );
endmodule
