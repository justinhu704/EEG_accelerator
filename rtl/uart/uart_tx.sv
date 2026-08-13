// 傳送 uart byte

module uart_tx #(
    parameter integer CLKS_PER_BIT = 434
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic [7:0] data,
    output logic       serial_tx,
    output logic       busy,
    output logic       done
);
    localparam integer COUNT_W = $clog2(CLKS_PER_BIT + 1);

    typedef enum logic [1:0] {
        TX_IDLE,
        TX_START,
        TX_DATA,
        TX_STOP
    } tx_state_t;

    tx_state_t state;
    logic [COUNT_W-1:0] clock_count;
    logic [2:0] bit_index;
    logic [7:0] shift_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= TX_IDLE;
            clock_count <= '0;
            bit_index   <= '0;
            shift_reg   <= '0;
            serial_tx   <= 1'b1;
            done        <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                TX_IDLE: begin
                    serial_tx   <= 1'b1;
                    clock_count <= '0;
                    bit_index   <= '0;
                    if (start) begin
                        shift_reg <= data;
                        serial_tx <= 1'b0;
                        state     <= TX_START;
                    end
                end

                TX_START: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count <= '0;
                        serial_tx   <= shift_reg[0];
                        state       <= TX_DATA;
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                TX_DATA: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count <= '0;
                        if (bit_index == 3'd7) begin
                            bit_index <= '0;
                            serial_tx <= 1'b1;
                            state     <= TX_STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                            serial_tx <= shift_reg[bit_index + 1'b1];
                        end
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                TX_STOP: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count <= '0;
                        done        <= 1'b1;
                        state       <= TX_IDLE;
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                default: state <= TX_IDLE;
            endcase
        end
    end

    assign busy = (state != TX_IDLE);
endmodule
