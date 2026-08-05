// Parameterized max-pooling engine for column-major tensors [H, W, C].
//
// The comparison is performed in INPUT_F. After the maximum is found, the
// result is rescaled to OUTPUT_F and saturated to signed 16-bit.
module maxpool_engine #(
    parameter int IN_H        = 19,
    parameter int IN_W        = 152,
    parameter int IN_CH       = 20,
    parameter int POOL_H      = 1,
    parameter int POOL_W      = 10,
    parameter int STRIDE_H    = 1,
    parameter int STRIDE_W    = 8,
    parameter int OUT_H       = ((IN_H - POOL_H) / STRIDE_H) + 1,
    parameter int OUT_W       = ((IN_W - POOL_W) / STRIDE_W) + 1,
    parameter int INPUT_F     = 12,
    parameter int OUTPUT_F    = 12,
    parameter int INPUT_DEPTH = IN_H * IN_W * IN_CH,
    parameter     INPUT_FILE  = "../mem/golden/q_relu2_act.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    logic signed [15:0] input_mem [0:INPUT_DEPTH-1];

    typedef enum logic [1:0] {
        S_IDLE,
        S_SCAN,
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

    logic [31:0] read_addr;
    logic signed [15:0] current_value;
    logic signed [15:0] max_value;
    logic signed [47:0] max_extended;
    logic signed [47:0] max_rescaled;
    logic signed [15:0] max_saturated;

    initial begin
        $readmemh(INPUT_FILE, input_mem, 0, INPUT_DEPTH-1);
    end

    // MATLAB/Verilog column-major address:
    // address = h + IN_H * (w + IN_W * channel)
    always_comb begin
        kh = pool_count % POOL_H;
        kw = pool_count / POOL_H;

        read_addr = (out_h_count * STRIDE_H + kh)
                  + IN_H * ((out_w_count * STRIDE_W + kw)
                  + IN_W * out_ch_count);
        current_value = input_mem[read_addr];

        busy = (state == S_SCAN) || (state == S_OUTPUT);
        done = (state == S_DONE);
    end

    // Convert the selected maximum from INPUT_F to OUTPUT_F.
    always_comb begin
        max_extended = {{32{max_value[15]}}, max_value};
        if (OUTPUT_F >= INPUT_F)
            max_rescaled = max_extended <<< (OUTPUT_F - INPUT_F);
        else
            max_rescaled = max_extended >>> (INPUT_F - OUTPUT_F);
    end

    sat16 u_sat16 (
        .value_in  (max_rescaled),
        .value_out (max_saturated)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            out_h_count   <= 0;
            out_w_count   <= 0;
            out_ch_count  <= 0;
            pool_count    <= 0;
            max_value     <= 16'sh8000;
            output_valid  <= 1'b0;
            output_addr   <= '0;
            output_data   <= '0;
        end else begin
            output_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        out_h_count  <= 0;
                        out_w_count  <= 0;
                        out_ch_count <= 0;
                        pool_count   <= 0;
                        max_value    <= 16'sh8000;
                        state        <= S_SCAN;
                    end
                end

                S_SCAN: begin
                    if (current_value > max_value)
                        max_value <= current_value;

                    if (pool_count == POOL_H * POOL_W - 1) begin
                        pool_count <= 0;
                        state      <= S_OUTPUT;
                    end else begin
                        pool_count <= pool_count + 1;
                    end
                end

                S_OUTPUT: begin
                    output_valid <= 1'b1;
                    output_data  <= max_saturated;
                    output_addr  <= out_h_count
                                  + OUT_H * (out_w_count
                                  + OUT_W * out_ch_count);
                    max_value <= 16'sh8000;

                    if (out_h_count < OUT_H - 1) begin
                        out_h_count <= out_h_count + 1;
                        state       <= S_SCAN;
                    end else begin
                        out_h_count <= 0;
                        if (out_w_count < OUT_W - 1) begin
                            out_w_count <= out_w_count + 1;
                            state       <= S_SCAN;
                        end else begin
                            out_w_count <= 0;
                            if (out_ch_count < IN_CH - 1) begin
                                out_ch_count <= out_ch_count + 1;
                                state        <= S_SCAN;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
