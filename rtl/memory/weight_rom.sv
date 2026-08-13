// Synchronous ROM for weights or biases.
// data is available one clock after addr is presented.

module weight_rom #(
    parameter int DATA_W = 16,
    parameter int DEPTH  = 210,

    // DEPTH向上取整數
    parameter int ADDR_W = $clog2(DEPTH),

    parameter     MEM_FILE = "mem/weights/conv1_W.mem"
) (
    input  logic                         clk,
    input  logic [ADDR_W-1:0]            addr,
    output logic signed [DATA_W-1:0]     data
);
    logic signed [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, mem, 0, DEPTH-1);
    end

    always_ff @(posedge clk) begin
        data <= mem[addr];
    end
endmodule
