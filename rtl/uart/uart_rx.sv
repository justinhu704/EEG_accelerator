// 接收 uart byte

module uart_rx #(
    parameter integer CLKS_PER_BIT = 434
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       serial_rx,
    output logic [7:0] data,
    output logic       valid,
    output logic       framing_error,
    output logic       busy
);
    localparam integer COUNT_W = $clog2(CLKS_PER_BIT + 1);

    typedef enum logic [1:0] {
        RX_IDLE,
        RX_START,
        RX_DATA,
        RX_STOP
    } rx_state_t;

    rx_state_t state;
    logic rx_meta, rx_sync;
    logic [COUNT_W-1:0] clock_count;
    logic [2:0] bit_index;
    logic [7:0] shift_reg;

    // The external UART input is asynchronous to CLOCK_50.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= serial_rx;
            rx_sync <= rx_meta;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= RX_IDLE;
            clock_count   <= '0;
            bit_index     <= '0;
            shift_reg     <= '0;
            data          <= '0;
            valid         <= 1'b0;
            framing_error <= 1'b0;
        end else begin
            valid         <= 1'b0;
            framing_error <= 1'b0;

            case (state)
                RX_IDLE: begin
                    clock_count <= '0;
                    bit_index   <= '0;
                    if (!rx_sync)
                        state <= RX_START;
                end

                RX_START: begin
                    if (clock_count == (CLKS_PER_BIT-1)/2) begin
                        clock_count <= '0;
                        if (!rx_sync)
                            state <= RX_DATA;
                        else
                            state <= RX_IDLE;
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                RX_DATA: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count         <= '0;
                        shift_reg[bit_index] <= rx_sync;
                        if (bit_index == 3'd7) begin
                            bit_index <= '0;
                            state     <= RX_STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                        end
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                RX_STOP: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count <= '0;
                        state       <= RX_IDLE;
                        if (rx_sync) begin
                            data  <= shift_reg;
                            valid <= 1'b1;
                        end else begin
                            framing_error <= 1'b1;
                        end
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                default: state <= RX_IDLE;
            endcase
        end
    end

    assign busy = (state != RX_IDLE);
endmodule
