// Synchronous ROM for weights or biases.
// data is available one clock after addr is presented.

module weight_rom #(
    parameter int DATA_W = 16,
    parameter int DEPTH  = 210,

    // DEPTH向上取整數
    parameter int ADDR_W = $clog2(DEPTH),

    parameter     MEM_FILE = "mem/weights/conv1_W.mem",
    parameter bit USE_READ_ENABLE = 1'b0
) (
    input  logic                         clk,
    input  logic                         read_en,
    input  logic [ADDR_W-1:0]            addr,
    output logic signed [DATA_W-1:0]     data
);
    logic signed [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, mem, 0, DEPTH-1);
    end

    // 保留原本一個 clock 的 ROM latency，只在引擎使用時更新輸出。
    generate
        if (USE_READ_ENABLE) begin : gen_read_enable
            always_ff @(posedge clk) begin
                if (read_en)
                    data <= mem[addr];
            end
        end else begin : gen_legacy_read
            always_ff @(posedge clk) begin
                data <= mem[addr];
            end
        end
    endgenerate
endmodule
