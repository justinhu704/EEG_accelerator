// Parameterized max-pooling engine for column-major tensors [H, W, C].
// The activation input is supplied through a synchronous external RAM port.
module maxpool_engine #(
    parameter int IN_H     = 19,
    parameter int IN_W     = 152,
    parameter int IN_CH    = 20,
    parameter int POOL_H   = 1,
    parameter int POOL_W   = 10,
    parameter int STRIDE_H = 1,
    parameter int STRIDE_W = 8,
    parameter int OUT_H    = ((IN_H - POOL_H) / STRIDE_H) + 1,
    parameter int OUT_W    = ((IN_W - POOL_W) / STRIDE_W) + 1,
    parameter int INPUT_F  = 12,
    parameter int OUTPUT_F = 12
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // send address to RAM, wait for result to return
    output logic [31:0]        input_addr,
    input  logic signed [15:0] input_data,

    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_CLEAR,
        S_SCAN,
        S_DRAIN,
        S_OUTPUT,
        S_DONE
    } state_t;

    state_t state;
    integer out_h_count;
    integer out_w_count;
    integer out_ch_count;
    integer pool_count;
    integer kh;
    integer kw;
    logic data_valid;

    logic signed [15:0] max_value;
    logic signed [47:0] max_extended;
    logic signed [47:0] max_rescaled;
    logic signed [15:0] max_saturated;

    // column-major address mapping
    // address = h + IN_H * (w + IN_W * channel)
    always_comb begin
        kh = pool_count % POOL_H;
        kw = pool_count / POOL_H;

        input_addr = (out_h_count * STRIDE_H + kh)
                   + IN_H * ((out_w_count * STRIDE_W + kw)
                   + IN_W * out_ch_count);

        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    // Convert the selected maximum from INPUT_F to OUTPUT_F.
    // 小數位截斷 & overflow protection
    always_comb begin
        max_extended = {{32{max_value[15]}}, max_value};
        if (OUTPUT_F >= INPUT_F)
            max_rescaled = max_extended <<< (OUTPUT_F - INPUT_F);
        else
            max_rescaled = max_extended >>> (INPUT_F - OUTPUT_F);
    end

    sat16 u_sat16 (
        .value_in(max_rescaled),
        .value_out(max_saturated)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            out_h_count  <= 0;
            out_w_count  <= 0;
            out_ch_count <= 0;
            pool_count   <= 0;
            data_valid   <= 1'b0;
            max_value    <= 16'sh8000;
            output_valid <= 1'b0;
            output_addr  <= '0;
            output_data  <= '0;
        end else begin
            output_valid <= 1'b0;
            // input_data corresponds to the address issued one clock earlier.
            data_valid <= (state == S_SCAN);

            case (state)
                S_IDLE: begin
                    data_valid <= 1'b0;
                    if (start) begin
                        out_h_count  <= 0;
                        out_w_count  <= 0;
                        out_ch_count <= 0;
                        pool_count   <= 0;
                        state        <= S_CLEAR;
                    end
                end

                S_CLEAR: begin
                    pool_count <= 0;
                    max_value  <= 16'sh8000;
                    state      <= S_SCAN;
                end

                // Issue one RAM address per clock and compare the value that
                // returned from the preceding address.
                S_SCAN: begin
                    if (data_valid && (input_data > max_value))
                        max_value <= input_data;

                    if (pool_count == POOL_H * POOL_W - 1) begin
                        state <= S_DRAIN;
                    end else begin
                        pool_count <= pool_count + 1;
                    end
                end

                // Consume the value returned for the final issued address.
                S_DRAIN: begin
                    if (data_valid && (input_data > max_value))
                        max_value <= input_data;
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    output_valid <= 1'b1;
                    output_data  <= max_saturated;
                    output_addr  <= out_h_count
                                  + OUT_H * (out_w_count
                                  + OUT_W * out_ch_count);

                    if (out_h_count < OUT_H - 1) begin
                        out_h_count <= out_h_count + 1;
                        state <= S_CLEAR;
                    end else begin
                        out_h_count <= 0;
                        if (out_w_count < OUT_W - 1) begin
                            out_w_count <= out_w_count + 1;
                            state <= S_CLEAR;
                        end else begin
                            out_w_count <= 0;
                            if (out_ch_count < IN_CH - 1) begin
                                out_ch_count <= out_ch_count + 1;
                                state <= S_CLEAR;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                S_DONE:
                    state <= S_IDLE;

                default:
                    state <= S_IDLE;
            endcase
        end
    end
endmodule
