// Small RAM used only for Pool2 -> GRU after the Conv1 full-frame buffer is removed.
module pool2_gru_ram #(
    parameter int DEPTH = 270,
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  logic clk,
    input  logic write_en,
    input  logic [ADDR_W-1:0] write_addr,
    input  logic signed [15:0] write_data,
    input  logic read_en,
    input  logic [ADDR_W-1:0] read_addr,
    output logic signed [15:0] read_data
);
    (* ramstyle = "M10K" *) logic signed [15:0] mem [0:DEPTH-1];
    always_ff @(posedge clk) begin
        if (write_en)
            mem[write_addr] <= write_data;
        if (read_en)
            read_data <= mem[read_addr];
    end
endmodule