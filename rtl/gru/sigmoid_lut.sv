// sigmoid_lut
// 儲存 0~3193 的 sigmoid 值， range[-6.23, +6.23]
module sigmoid_lut #(
    parameter MEM_FILE = "mem/lut/sigmoid_half_lut_q15.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic signed [15:0] in_data,
    output logic               out_valid,
    output logic signed [15:0] out_data
);
    localparam int LUT_DEPTH = 256;
    localparam int RANGE_RAW = 3193;  // 6.236 * 512 (趨近於1)

    // 輸入映射給 256 個 index
    localparam int INDEX_GAIN  = 83741; // 255 / 3193 * 2^20
    localparam int INDEX_SHIFT = 20;

    logic signed [15:0] lut [0:LUT_DEPTH-1];
    logic [11:0] magnitude;  // 3193 => 12 bits
    logic [28:0] index_product;  // 3193 * 83741 => 29 bits
    logic [7:0]  lut_addr;
    logic signed [15:0] rom_data;
    logic negative_d1;
    logic signed [16:0] reflected_value;

    initial begin
        $readmemh(MEM_FILE, lut, 0, LUT_DEPTH-1);
    end

    // 計算出 in_data 轉換的 lut_addr
    always_comb begin
        magnitude = '0;
        index_product = '0;
        lut_addr = '0;

        // 若超出範圍，直接給邊界值
        if (($signed(in_data) <= -RANGE_RAW) ||
            ($signed(in_data) >=  RANGE_RAW)) begin
            magnitude = RANGE_RAW;
            lut_addr = 8'd255;
        end else begin
            if (in_data[15])
                // 負數取絕對值
                magnitude = -$signed(in_data);
            else
                magnitude = in_data;
            // 12-bit x 17-bit = 29-bit
            index_product = magnitude * INDEX_GAIN;
            // 取高 8 bits 當作 ROM 地址
            lut_addr = index_product >> INDEX_SHIFT;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            rom_data <= '0;
            negative_d1 <= 1'b0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                // 讀取 ROM
                rom_data <= lut[lut_addr];
                // 鎖存上一拍的正負號
                negative_d1 <= in_data[15];
            end
        end
    end

    always_comb begin
        // 1.0 in Q15 is 32768, so use 17 bits for the subtraction.
        reflected_value = 17'sd32768 - $signed({1'b0, rom_data});
        if (!out_valid)
            out_data = '0;
        else if (negative_d1)
            out_data = reflected_value[15:0];
        else
            out_data = rom_data;
    end
endmodule
