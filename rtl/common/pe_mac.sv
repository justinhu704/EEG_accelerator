// multiply-accumulate

module pe_mac #(
    parameter int DATA_W = 16,
    parameter int ACC_W  = 48
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         clear_acc,
    input  logic                         mac_en,
    input  logic signed [DATA_W-1:0]     data_in,
    input  logic signed [DATA_W-1:0]     weight_in,
    output logic signed [ACC_W-1:0]      accumulator
);
    // double width for product
    logic signed [(2*DATA_W)-1:0] product;

    always_comb product = data_in * weight_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            accumulator <= '0;
        else if (clear_acc)
            accumulator <= '0;
        else if (mac_en)
            accumulator <= accumulator + product;
    end
endmodule
