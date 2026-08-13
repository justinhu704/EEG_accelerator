// 檢查錯誤
// sample_id + word_count + EEG payload

module uart_crc16_ccitt (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear,
    input  logic        enable,
    input  logic [7:0]  data,
    output logic [15:0] crc
);
    function automatic logic [15:0] next_crc16(
        input logic [15:0] current_crc,
        input logic [7:0]  next_byte
    );
        logic [15:0] value;
        integer bit_index;
        begin
            value = current_crc;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                if (value[15] ^ next_byte[7-bit_index])
                    value = {value[14:0], 1'b0} ^ 16'h1021;
                else
                    value = {value[14:0], 1'b0};
            end
            next_crc16 = value;
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            crc <= 16'hFFFF;
        else if (clear)
            crc <= 16'hFFFF;
        else if (enable)
            crc <= next_crc16(crc, data);
    end
endmodule
