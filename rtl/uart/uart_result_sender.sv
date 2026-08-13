// Sends one inference response:
//   5A A5 (2 bytes: 封包開頭) | 
//   sample_id[31:0] (4 bytes: 編號，哪一筆測試資料) | 
//   status (1 byte: 錯誤碼：1:有錯誤, 0:正確) | 
//   class (1 byte: 分類結果: 0~104) | 
//   winning_logit[15:0] (2 bytes: 最大FC值) | 
//   CRC16 (2 bytes: 檢查碼)
//   total 12 bytes
// CRC-16/CCITT-FALSE covers sample_id through winning_logit (not magic).

module uart_result_sender (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic [31:0]        sample_id,
    input  logic [7:0]         status,
    input  logic [6:0]         class_index,
    input  logic signed [15:0] winning_logit,

    output logic               tx_start,
    output logic [7:0]         tx_data,
    input  logic               tx_busy,
    input  logic               tx_done,

    output logic               busy,
    output logic               done
);
    typedef enum logic [1:0] {
        SEND_IDLE,
        SEND_BYTE,
        WAIT_BYTE
    } sender_state_t;

    sender_state_t state;
    logic [3:0] byte_index;
    logic [31:0] saved_sample_id;
    logic [7:0] saved_status;
    logic [7:0] saved_class;
    logic [15:0] saved_logit;
    logic [15:0] crc_value;
    logic crc_clear, crc_enable;

    assign crc_clear  = (state == SEND_IDLE) && start;
    assign crc_enable = tx_start && (byte_index >= 4'd2) &&
                        (byte_index <= 4'd9);

    uart_crc16_ccitt u_response_crc (
        .clk(clk), .rst_n(rst_n),
        .clear(crc_clear), .enable(crc_enable),
        .data(tx_data), .crc(crc_value)
    );

    always_comb begin
        case (byte_index)
            4'd0:  tx_data = 8'h5A;
            4'd1:  tx_data = 8'hA5;
            4'd2:  tx_data = saved_sample_id[7:0];
            4'd3:  tx_data = saved_sample_id[15:8];
            4'd4:  tx_data = saved_sample_id[23:16];
            4'd5:  tx_data = saved_sample_id[31:24];
            4'd6:  tx_data = saved_status;
            4'd7:  tx_data = saved_class;
            4'd8:  tx_data = saved_logit[7:0];
            4'd9:  tx_data = saved_logit[15:8];
            4'd10: tx_data = crc_value[7:0];
            4'd11: tx_data = crc_value[15:8];
            default: tx_data = 8'h00;
        endcase
    end

    assign tx_start = (state == SEND_BYTE) && !tx_busy;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= SEND_IDLE;
            byte_index      <= '0;
            saved_sample_id <= '0;
            saved_status    <= '0;
            saved_class     <= '0;
            saved_logit     <= '0;
            done            <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                SEND_IDLE: begin
                    if (start) begin
                        saved_sample_id <= sample_id;
                        saved_status    <= status;
                        saved_class     <= {1'b0, class_index};
                        saved_logit     <= winning_logit;
                        byte_index      <= 4'd0;
                        state           <= SEND_BYTE;
                    end
                end

                SEND_BYTE: begin
                    if (!tx_busy)
                        state <= WAIT_BYTE;
                end

                WAIT_BYTE: begin
                    if (tx_done) begin
                        if (byte_index == 4'd11) begin
                            done  <= 1'b1;
                            state <= SEND_IDLE;
                        end else begin
                            byte_index <= byte_index + 1'b1;
                            state <= SEND_BYTE;
                        end
                    end
                end

                default: state <= SEND_IDLE;
            endcase
        end
    end

    assign busy = (state != SEND_IDLE);
endmodule
