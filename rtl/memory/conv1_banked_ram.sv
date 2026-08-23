// Conv1 activation buffer with even/odd height RAM banking.

module conv1_banked_ram #(
    parameter int INPUT_F    = 11,
    parameter int STORED_F   = 5,
    parameter int LOG_ADDR_W = 16,
    parameter int BANK_DEPTH = (20 * 156 * 21) / 2,
    parameter int BANK_ADDR_W = (BANK_DEPTH <= 1)
                                  ? 1 : $clog2(BANK_DEPTH) // 15
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       write_en,
    input  logic [LOG_ADDR_W-1:0]      write_logical_addr,
    input  logic signed [15:0]         write_q11_data,

    input  logic [LOG_ADDR_W-1:0]      read_logical_addr_kh0,
    input  logic [LOG_ADDR_W-1:0]      read_logical_addr_kh1,
    output logic signed [15:0]         read_q11_data_kh0,
    output logic signed [15:0]         read_q11_data_kh1
);
    localparam int SHIFT = INPUT_F - STORED_F;
    localparam int ROUNDING = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;

    (* ramstyle = "M10K" *)
    logic [7:0] even_height_mem [0:BANK_DEPTH-1];
    (* ramstyle = "M10K" *)
    logic [7:0] odd_height_mem [0:BANK_DEPTH-1];

    logic [7:0] quantized_write_data;
    logic [BANK_ADDR_W-1:0] write_bank_addr;
    logic [BANK_ADDR_W-1:0] even_read_addr, odd_read_addr;
    logic [7:0] even_read_data, odd_read_data;
    logic kh0_is_odd_d1;

    // =======================================
    // 量化成 UQ5
    // =======================================
    function automatic logic [7:0] quantize_u8(
        input logic signed [15:0] value
    );
        logic signed [16:0] rounded;
        logic signed [16:0] scaled;
        begin
            if (value <= 0) begin
                quantize_u8 = 8'd0;
            end else begin
                rounded = {1'b0, value} + ROUNDING; // 加上 32 進行四捨五入
                scaled = rounded >>> SHIFT; // 右移 6-bit
                // 飽和截斷
                if (scaled > 17'sd255)
                    quantize_u8 = 8'hff;
                else
                    quantize_u8 = scaled[7:0];
            end
        end
    endfunction

    // =======================================
    // 還原 Q11 ，給 Conv2 使用
    // =======================================
    function automatic logic signed [15:0] restore_q11(
        input logic [7:0] value
    );
        logic [15:0] extended;
        begin
            extended = {8'd0, value};
            restore_q11 = $signed(extended << SHIFT);
        end
    endfunction

    // =======================================
    // 計算記憶體位址
    // =======================================
    always_comb begin
        // 輸入資料量化與位址計算
        quantized_write_data = quantize_u8(write_q11_data);
        write_bank_addr = write_logical_addr[LOG_ADDR_W-1:1];

        // 判斷 kh0、kh1 的高度是否為奇數，將偶數存在 even_mem ，奇數存在 odd_mem
        if (!read_logical_addr_kh0[0]) begin
            // 除以二存入
            even_read_addr = read_logical_addr_kh0[LOG_ADDR_W-1:1];
            odd_read_addr  = read_logical_addr_kh1[LOG_ADDR_W-1:1];
        end else begin
            even_read_addr = read_logical_addr_kh1[LOG_ADDR_W-1:1];
            odd_read_addr  = read_logical_addr_kh0[LOG_ADDR_W-1:1];
        end

        // 將偶數和奇數的 Q5 還原成 Q11
        if (!kh0_is_odd_d1) begin
            read_q11_data_kh0 = restore_q11(even_read_data);
            read_q11_data_kh1 = restore_q11(odd_read_data);
        end else begin
            read_q11_data_kh0 = restore_q11(odd_read_data);
            read_q11_data_kh1 = restore_q11(even_read_data);
        end
    end

    always_ff @(posedge clk) begin
        // 寫入偶數
        if (write_en && !write_logical_addr[0])
            even_height_mem[write_bank_addr] <= quantized_write_data;
        // 寫入奇數
        if (write_en && write_logical_addr[0])
            odd_height_mem[write_bank_addr] <= quantized_write_data;

        even_read_data <= even_height_mem[even_read_addr];
        odd_read_data  <= odd_height_mem[odd_read_addr];
    end

    // 與 RAM 同一個時脈延遲
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            kh0_is_odd_d1 <= 1'b0;
        else
            kh0_is_odd_d1 <= read_logical_addr_kh0[0];
    end

    initial begin
        if (INPUT_F < STORED_F)
            $error("conv1_banked_ram requires INPUT_F >= STORED_F");
        if ((1 << BANK_ADDR_W) < BANK_DEPTH)
            $error("conv1_banked_ram BANK_ADDR_W is too small");
    end
endmodule
