// DS-Conv1/DS-Conv2 欄位重疊排程器。
// 先填滿 5 欄，再使用第 6 個 slot 同時產生下一欄與計算目前視窗。
module dsconv12_overlap_scheduler (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic       conv1_column_done,
    input  logic       conv2_window_done,
    output logic       busy,
    output logic       done,
    output logic       conv1_start,
    output logic [7:0] conv1_column,
    output logic [2:0] conv1_write_slot,
    output logic       conv2_start,
    output logic [7:0] conv2_column,
    output logic [2:0] conv2_window_slot
);
    typedef enum logic [3:0] {
        S_IDLE,
        S_START_FILL,
        S_WAIT_FILL,
        S_START_PAIR,
        S_WAIT_PAIR,
        S_START_LAST,
        S_WAIT_LAST,
        S_DONE
    } state_t;

    state_t state;
    logic conv1_done_seen;
    logic conv2_done_seen;

    always_comb begin
        conv1_start = (state == S_START_FILL) || (state == S_START_PAIR);
        conv2_start = (state == S_START_PAIR) || (state == S_START_LAST);
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            conv1_column <= '0;
            conv1_write_slot <= '0;
            conv2_column <= '0;
            conv2_window_slot <= '0;
            conv1_done_seen <= 1'b0;
            conv2_done_seen <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        conv1_column <= 0;
                        conv1_write_slot <= 0;
                        conv2_column <= 0;
                        conv2_window_slot <= 0;
                        conv1_done_seen <= 1'b0;
                        conv2_done_seen <= 1'b0;
                        state <= S_START_FILL;
                    end
                end

                S_START_FILL:
                    state <= S_WAIT_FILL;

                S_WAIT_FILL: begin
                    if (conv1_column_done) begin
                        if (conv1_column < 4) begin
                            conv1_column <= conv1_column + 1'b1;
                            conv1_write_slot <= conv1_write_slot + 1'b1;
                            state <= S_START_FILL;
                        end else begin
                            conv1_column <= 5;
                            conv1_write_slot <= 5;
                            conv2_column <= 0;
                            conv2_window_slot <= 0;
                            state <= S_START_PAIR;
                        end
                    end
                end

                S_START_PAIR: begin
                    conv1_done_seen <= 1'b0;
                    conv2_done_seen <= 1'b0;
                    state <= S_WAIT_PAIR;
                end

                S_WAIT_PAIR: begin
                    if (conv1_column_done)
                        conv1_done_seen <= 1'b1;
                    if (conv2_window_done)
                        conv2_done_seen <= 1'b1;

                    if ((conv1_done_seen || conv1_column_done) &&
                        (conv2_done_seen || conv2_window_done)) begin
                        conv2_column <= conv2_column + 1'b1;
                        conv2_window_slot <= (conv2_window_slot == 5)
                                           ? 0 : conv2_window_slot + 1'b1;
                        if (conv2_column == 150) begin
                            // W151 只消耗最後 5 欄，不再產生新的 Conv1 欄。
                            state <= S_START_LAST;
                        end else begin
                            conv1_column <= conv1_column + 1'b1;
                            conv1_write_slot <= (conv1_write_slot == 5)
                                              ? 0 : conv1_write_slot + 1'b1;
                            state <= S_START_PAIR;
                        end
                    end
                end

                S_START_LAST:
                    state <= S_WAIT_LAST;

                S_WAIT_LAST: begin
                    if (conv2_window_done)
                        state <= S_DONE;
                end

                S_DONE:
                    state <= S_IDLE;

                default:
                    state <= S_IDLE;
            endcase
        end
    end
endmodule
