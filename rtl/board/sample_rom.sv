// Synchronous ROM containing one fixed 21x160 quantized EEG sample.

// 將.mem資料存入ROM
module sample_rom #(
    parameter int DATA_W = 16,
    parameter int DEPTH  = 3360,
    parameter int ADDR_W = 12,
    parameter MEM_FILE   = "../mem/board/sample0_q12.mem"
) (
    input  logic                      clk,
    input  logic [ADDR_W-1:0]         addr,
    output logic signed [DATA_W-1:0]  data
);
    (* romstyle = "M10K" *)
    logic signed [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, mem);
    end

    always_ff @(posedge clk) begin
        data <= mem[addr];
    end
endmodule
