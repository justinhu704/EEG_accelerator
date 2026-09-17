// 原始 EEG 輸入依線性位址奇偶分成兩個 RAM。
// kh0/kh1 為相鄰位址，因此每拍可同時讀出且不需複製完整記憶體。
module input_banked_ram #(
    parameter int DATA_W = 16,
    parameter int DEPTH = 21 * 160,
    parameter int ADDR_W = $clog2(DEPTH),
    parameter int BANK_DEPTH = (DEPTH + 1) / 2,
    parameter int BANK_ADDR_W = $clog2(BANK_DEPTH),
    parameter EVEN_MEM_FILE = "",
    parameter ODD_MEM_FILE = ""
) (
    input  logic clk,
    input  logic rst_n,

    input  logic write_en,
    input  logic [ADDR_W-1:0] write_addr,
    input  logic signed [DATA_W-1:0] write_data,

    input  logic read_en,
    input  logic [ADDR_W-1:0] read_addr_kh0,
    input  logic [ADDR_W-1:0] read_addr_kh1,
    output logic signed [DATA_W-1:0] read_data_kh0,
    output logic signed [DATA_W-1:0] read_data_kh1
);
    (* ramstyle = "M10K" *)
    logic signed [DATA_W-1:0] even_mem [0:BANK_DEPTH-1];
    (* ramstyle = "M10K" *)
    logic signed [DATA_W-1:0] odd_mem [0:BANK_DEPTH-1];

    logic [BANK_ADDR_W-1:0] even_read_addr, odd_read_addr;
    logic signed [DATA_W-1:0] even_read_data, odd_read_data;
    logic kh0_is_odd_d1;

    initial begin
        if (EVEN_MEM_FILE != "")
            $readmemh(EVEN_MEM_FILE, even_mem, 0, BANK_DEPTH-1);
        if (ODD_MEM_FILE != "")
            $readmemh(ODD_MEM_FILE, odd_mem, 0, BANK_DEPTH-1);
    end

    // 只取位址最低位與右移，避免除法或餘數形成 critical path。
    always_comb begin
        if (!read_addr_kh0[0]) begin
            even_read_addr = read_addr_kh0[ADDR_W-1:1];
            odd_read_addr  = read_addr_kh1[ADDR_W-1:1];
        end else begin
            even_read_addr = read_addr_kh1[ADDR_W-1:1];
            odd_read_addr  = read_addr_kh0[ADDR_W-1:1];
        end

        if (kh0_is_odd_d1) begin
            read_data_kh0 = odd_read_data;
            read_data_kh1 = even_read_data;
        end else begin
            read_data_kh0 = even_read_data;
            read_data_kh1 = odd_read_data;
        end
    end

    always_ff @(posedge clk) begin
        if (write_en && !write_addr[0])
            even_mem[write_addr[ADDR_W-1:1]] <= write_data;
        if (write_en && write_addr[0])
            odd_mem[write_addr[ADDR_W-1:1]] <= write_data;

        if (read_en) begin
            even_read_data <= even_mem[even_read_addr];
            odd_read_data  <= odd_mem[odd_read_addr];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            kh0_is_odd_d1 <= 1'b0;
        else if (read_en)
            kh0_is_odd_d1 <= read_addr_kh0[0];
    end

`ifndef SYNTHESIS
    initial begin
        if ((1 << BANK_ADDR_W) < BANK_DEPTH)
            $error("input_banked_ram BANK_ADDR_W is too small");
    end
`endif
endmodule
