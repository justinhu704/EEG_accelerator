module activation_ram #(
    parameter int DATA_W = 16,
    parameter int DEPTH  = 65520,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter     MEM_FILE = "",
    parameter bit USE_READ_ENABLE = 1'b0
) (
    input  logic                        clk,

    input  logic                        write_en,
    input  logic [ADDR_W-1:0]           write_addr,
    input  logic signed [DATA_W-1:0]    write_data,

    input  logic                        read_en,
    input  logic [ADDR_W-1:0]           read_addr,
    output logic signed [DATA_W-1:0]    read_data
);
    // memory blocks
    (* ramstyle = "M10K" *)
    logic signed [DATA_W-1:0] mem [0:DEPTH-1];

    // optional initialization
    initial begin
        if (MEM_FILE != "")
            $readmemh(MEM_FILE, mem, 0, DEPTH-1);
    end

    // independent synchronous write port
    always_ff @(posedge clk) begin
        if (write_en)
            mem[write_addr] <= write_data;
    end

    // independent synchronous read port with one-clock read latency
    // 啟用 read enable 時，閒置期間保持輸出，避免不必要的 M10K 切換。
    generate
        if (USE_READ_ENABLE) begin : gen_read_enable
            always_ff @(posedge clk) begin
                if (read_en)
                    read_data <= mem[read_addr];
            end
        end else begin : gen_legacy_read
            always_ff @(posedge clk) begin
                read_data <= mem[read_addr];
            end
        end
    endgenerate
endmodule
