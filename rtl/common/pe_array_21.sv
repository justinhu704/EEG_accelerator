module pe_array_21 #(
    parameter int DATA_W = 16,
    parameter int ACC_W  = 48,
    parameter int PES    = 21
) (
    input  logic                              clk,
    input  logic                              rst_n,
    input  logic                              clear_acc,
    input  logic                              mac_en,
    input  logic signed [DATA_W-1:0]          data_in,
    input  logic signed [PES-1:0][DATA_W-1:0] weights,
    output logic signed [PES-1:0][ACC_W-1:0]  accumulators
);
    genvar i;
    generate
        for (i = 0; i < PES; i = i + 1) begin : gen_pe
            pe_mac #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
                .clk         (clk),
                .rst_n       (rst_n),
                .clear_acc   (clear_acc),
                .mac_en      (mac_en),
                .data_in     (data_in),
                .weight_in   (weights[i]),
                .accumulator (accumulators[i])
            );
        end
    endgenerate
endmodule
