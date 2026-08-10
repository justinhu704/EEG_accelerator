// Pipelined multi-channel 2-D convolution controller.
// After one pipeline-fill clock, each S_STREAM clock issues the next memory
// address and accumulates the value returned for the previous address.
module conv_controller #(
    parameter int IN_H   = 21,
    parameter int IN_W   = 160,
    parameter int IN_CH  = 1,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 21,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        busy,
    output logic        done,
    output logic        clear_acc,
    output logic        mac_en,
    output logic        output_valid,
    output logic [31:0] input_addr,
    output logic [31:0] weight_addr,
    output logic [31:0] bias_addr,
    output logic [31:0] output_addr,
    output logic [31:0] output_ch_idx
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_CLEAR,
        S_STREAM,
        S_DRAIN,
        S_OUTPUT,
        S_DONE
    } state_t;

    state_t state;
    integer out_h_count;
    integer out_w_count;
    integer out_ch_count;
    // Explicit nested kernel coordinates avoid synthesizing division and
    // modulo logic from a flattened kernel_count.
    integer kh_count;
    integer kw_count;
    integer in_ch_count;
    logic data_valid;

    always_comb begin
        input_addr =
              (out_h_count + kh_count)
            + IN_H * ((out_w_count + kw_count) + IN_W * in_ch_count);

        weight_addr =
              kh_count
            + K_H * (kw_count + K_W *
              (in_ch_count + IN_CH * out_ch_count));

        bias_addr = out_ch_count;
        output_addr =
              out_h_count
            + OUT_H * (out_w_count + OUT_W * out_ch_count);
        output_ch_idx = out_ch_count;

        busy         = (state != S_IDLE) && (state != S_DONE);
        done         = (state == S_DONE);
        clear_acc    = (state == S_CLEAR);
        mac_en       = data_valid;
        output_valid = (state == S_OUTPUT);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            out_h_count  <= 0;
            out_w_count  <= 0;
            out_ch_count <= 0;
            kh_count     <= 0;
            kw_count     <= 0;
            in_ch_count  <= 0;
            data_valid   <= 1'b0;
        end else begin
            // This is the valid bit for data returned by synchronous memory.
            // It is delayed one clock from the address-issue phase.
            data_valid <= (state == S_STREAM);

            case (state)
                S_IDLE: begin
                    if (start) begin
                        out_h_count  <= 0;
                        out_w_count  <= 0;
                        out_ch_count <= 0;
                        kh_count     <= 0;
                        kw_count     <= 0;
                        in_ch_count  <= 0;
                        state        <= S_CLEAR;
                    end
                end

                S_CLEAR: begin
                    kh_count    <= 0;
                    kw_count    <= 0;
                    in_ch_count <= 0;
                    state <= S_STREAM;
                end

                // Issue one address every clock. At the same edge, pe_mac
                // consumes the value returned for the previous address.
                S_STREAM: begin
                    if (kh_count < K_H-1) begin
                        kh_count <= kh_count + 1;
                    end else begin
                        kh_count <= 0;
                        if (kw_count < K_W-1) begin
                            kw_count <= kw_count + 1;
                        end else begin
                            kw_count <= 0;
                            if (in_ch_count < IN_CH-1) begin
                                in_ch_count <= in_ch_count + 1;
                            end else begin
                                // The final (kh, kw, in_ch) address was
                                // issued in this clock. Drain its RAM result.
                                state <= S_DRAIN;
                            end
                        end
                    end
                end

                // Accumulate the final value returned after the last address.
                S_DRAIN: state <= S_OUTPUT;

                S_OUTPUT: begin
                    if (out_h_count < OUT_H-1) begin
                        out_h_count <= out_h_count + 1;
                        state <= S_CLEAR;
                    end else begin
                        out_h_count <= 0;
                        if (out_w_count < OUT_W-1) begin
                            out_w_count <= out_w_count + 1;
                            state <= S_CLEAR;
                        end else begin
                            out_w_count <= 0;
                            if (out_ch_count < OUT_CH-1) begin
                                out_ch_count <= out_ch_count + 1;
                                state <= S_CLEAR;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
