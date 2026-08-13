// 合成 uart_rx 傳來的封包，並寫入 eeg_top 的 RAM A 中
// Request packet (little-endian fields):
//   A5 5A (2 bytes: 封包開頭) | 
//   sample_id[31:0] (4 bytes: 編號，哪一筆測試資料) | 
//   word_count[15:0] (2 bytes: 測試資料長度，共3360個int16) | 
//   signed int16 payload (6720 bytes: 3360 * 2 個 int16) | 
//   CRC16 (2 bytes: 檢查碼)
//   total 6730 bytes
// CRC-16/CCITT-FALSE covers sample_id, word_count, and payload (not magic).

module uart_sample_loader #(
    parameter integer SAMPLE_WORDS = 3360,
    parameter integer ADDR_W       = 12
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [7:0]         rx_data,
    input  logic               rx_valid,
    input  logic               rx_framing_error,

    input  logic               input_ready,
    output logic               input_write_en,
    output logic [ADDR_W-1:0]  input_write_addr,
    output logic signed [15:0] input_write_data,
    output logic               inference_start,
    input  logic               inference_done,

    output logic [31:0]        sample_id,
    output logic               loader_ready,
    output logic               loader_busy,
    output logic               packet_error,
    output logic               packet_loaded
);
    typedef enum logic [3:0] {
        WAIT_MAGIC_A5,
        WAIT_MAGIC_5A,
        READ_ID0,
        READ_ID1,
        READ_ID2,
        READ_ID3,
        READ_LEN0,
        READ_LEN1,
        READ_DATA_LOW,
        READ_DATA_HIGH,
        READ_CRC_LOW,
        READ_CRC_HIGH,
        WAIT_INFERENCE
    } loader_state_t;

    loader_state_t state;
    logic [15:0] received_length;
    logic [15:0] word_index;
    logic [7:0] low_byte;
    logic [7:0] crc_low_byte;
    logic [15:0] crc_value;
    logic crc_clear, crc_enable;

    assign crc_clear = rx_valid && (state == WAIT_MAGIC_5A) &&
                       (rx_data == 8'h5A);
    assign crc_enable = rx_valid && (
                        (state == READ_ID0) || (state == READ_ID1) ||
                        (state == READ_ID2) || (state == READ_ID3) ||
                        (state == READ_LEN0) || (state == READ_LEN1) ||
                        (state == READ_DATA_LOW) ||
                        (state == READ_DATA_HIGH));

    uart_crc16_ccitt u_request_crc (
        .clk(clk), .rst_n(rst_n),
        .clear(crc_clear), .enable(crc_enable),
        .data(rx_data), .crc(crc_value)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= WAIT_MAGIC_A5;
            received_length   <= '0;
            word_index        <= '0;
            low_byte          <= '0;
            crc_low_byte      <= '0;
            sample_id         <= '0;
            input_write_en    <= 1'b0;
            input_write_addr  <= '0;
            input_write_data  <= '0;
            inference_start   <= 1'b0;
            packet_error      <= 1'b0;
            packet_loaded     <= 1'b0;
        end else begin
            input_write_en  <= 1'b0;
            inference_start <= 1'b0;
            packet_error    <= 1'b0;
            packet_loaded   <= 1'b0;

            if (rx_framing_error) begin
                state        <= WAIT_MAGIC_A5;
                packet_error <= 1'b1;
            end else if (rx_valid) begin
                case (state)
                    WAIT_MAGIC_A5: begin
                        if (input_ready && (rx_data == 8'hA5))
                            state <= WAIT_MAGIC_5A;
                    end

                    WAIT_MAGIC_5A: begin
                        if (rx_data == 8'h5A) begin
                            state <= READ_ID0;
                        end else if (rx_data != 8'hA5) begin
                            state <= WAIT_MAGIC_A5;
                        end
                    end

                    READ_ID0: begin
                        sample_id[7:0] <= rx_data;
                        state <= READ_ID1;
                    end
                    READ_ID1: begin
                        sample_id[15:8] <= rx_data;
                        state <= READ_ID2;
                    end
                    READ_ID2: begin
                        sample_id[23:16] <= rx_data;
                        state <= READ_ID3;
                    end
                    READ_ID3: begin
                        sample_id[31:24] <= rx_data;
                        state <= READ_LEN0;
                    end

                    READ_LEN0: begin
                        received_length[7:0] <= rx_data;
                        state <= READ_LEN1;
                    end
                    READ_LEN1: begin
                        received_length[15:8] <= rx_data;
                        word_index <= 16'd0;
                        if ({rx_data, received_length[7:0]} == SAMPLE_WORDS)
                            state <= READ_DATA_LOW;
                        else begin
                            packet_error <= 1'b1;
                            state <= WAIT_MAGIC_A5;
                        end
                    end

                    READ_DATA_LOW: begin
                        low_byte <= rx_data;
                        state <= READ_DATA_HIGH;
                    end

                    READ_DATA_HIGH: begin
                        input_write_en   <= 1'b1;
                        input_write_addr <= word_index[ADDR_W-1:0];
                        input_write_data <= $signed({rx_data, low_byte});
                        if (word_index == SAMPLE_WORDS-1) begin
                            state <= READ_CRC_LOW;
                        end else begin
                            word_index <= word_index + 1'b1;
                            state <= READ_DATA_LOW;
                        end
                    end

                    READ_CRC_LOW: begin
                        crc_low_byte <= rx_data;
                        state <= READ_CRC_HIGH;
                    end

                    READ_CRC_HIGH: begin
                        if ({rx_data, crc_low_byte} == crc_value) begin
                            inference_start <= 1'b1;
                            packet_loaded   <= 1'b1;
                            state <= WAIT_INFERENCE;
                        end else begin
                            packet_error <= 1'b1;
                            state <= WAIT_MAGIC_A5;
                        end
                    end

                    default: state <= WAIT_MAGIC_A5;
                endcase
            end else if ((state == WAIT_INFERENCE) && inference_done) begin
                state <= WAIT_MAGIC_A5;
            end
        end
    end

    assign loader_ready = (state == WAIT_MAGIC_A5) && input_ready;
    assign loader_busy  = (state != WAIT_MAGIC_A5);
endmodule
