// Sequential fully-connected inference engine with an external synchronous
// activation RAM and internal synchronous weight/bias ROMs.
//
// MATLAB stores an [OUTPUT_SIZE, INPUT_SIZE] weight matrix in column-major
// order, therefore weight_addr = output + OUTPUT_SIZE * input.
module fc_engine #(
    parameter int INPUT_SIZE  = 162,
    parameter int OUTPUT_SIZE = 40,
    parameter int BIAS_SHIFT  = 16,
    parameter int OUTPUT_SHIFT = 17,
    parameter WEIGHT_FILE = "mem/weights/fc_1_W.mem",
    parameter BIAS_FILE   = "mem/weights/fc_1_b.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    output logic [31:0]        input_addr,
    input  logic signed [15:0] input_data,

    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int WEIGHT_DEPTH = INPUT_SIZE * OUTPUT_SIZE;
    localparam int WEIGHT_ADDR_W = (WEIGHT_DEPTH <= 1)
                                   ? 1 : $clog2(WEIGHT_DEPTH);
    localparam int BIAS_ADDR_W = (OUTPUT_SIZE <= 1)
                                 ? 1 : $clog2(OUTPUT_SIZE);

    typedef enum logic [2:0] {
        S_IDLE,
        S_CLEAR,
        S_STREAM,
        S_DRAIN,
        S_OUTPUT,
        S_DONE
    } state_t;
    state_t state;

    integer input_index;
    integer output_index;
    logic data_valid;
    logic clear_acc;

    logic [31:0] weight_addr_full;
    logic signed [15:0] weight_data;
    logic signed [15:0] bias_data;
    logic signed [47:0] accumulator;
    logic signed [47:0] bias_extended;
    logic signed [47:0] bias_aligned;
    logic signed [47:0] sum_with_bias;
    logic signed [47:0] scaled_result;
    logic signed [15:0] saturated_result;

    always_comb begin
        input_addr = input_index;
        weight_addr_full = output_index + OUTPUT_SIZE * input_index;
        output_addr = output_index;

        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
        clear_acc = (state == S_CLEAR);
        output_valid = (state == S_OUTPUT);
    end

    weight_rom #(
        .DATA_W(16),
        .DEPTH(WEIGHT_DEPTH),
        .ADDR_W(WEIGHT_ADDR_W),
        .MEM_FILE(WEIGHT_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_weight_rom (
        .clk(clk),
        .read_en(start || busy),
        .addr(weight_addr_full[WEIGHT_ADDR_W-1:0]),
        .data(weight_data)
    );

    weight_rom #(
        .DATA_W(16),
        .DEPTH(OUTPUT_SIZE),
        .ADDR_W(BIAS_ADDR_W),
        .MEM_FILE(BIAS_FILE),
        .USE_READ_ENABLE(1'b1)
    ) u_bias_rom (
        .clk(clk),
        .read_en(start || busy),
        .addr(output_index[BIAS_ADDR_W-1:0]),
        .data(bias_data)
    );

    pe_mac u_mac (
        .clk(clk), .rst_n(rst_n),
        .clear_acc(clear_acc),
        .mac_en(data_valid),
        .data_in(input_data),
        .weight_in(weight_data),
        .accumulator(accumulator)
    );

    // 加上 bias, 位移截斷
    always_comb begin
        bias_extended = {{32{bias_data[15]}}, bias_data};
        bias_aligned = bias_extended <<< BIAS_SHIFT;
        sum_with_bias = accumulator + bias_aligned;
        scaled_result = sum_with_bias >>> OUTPUT_SHIFT;
    end

    sat16 u_sat16 (
        .value_in(scaled_result),
        .value_out(saturated_result)
    );

    always_comb begin
        output_data = output_valid ? saturated_result : 16'sd0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            input_index <= 0;
            output_index <= 0;
            data_valid <= 1'b0;
        // IDLE 等待期間保持 FC1 暫存器，不改變任何運算 state/cycle。
        end else if ((state != S_IDLE) || start) begin
            data_valid <= (state == S_STREAM);

            case (state)
                S_IDLE: begin
                    data_valid <= 1'b0;
                    if (start) begin
                        input_index <= 0;
                        output_index <= 0;
                        state <= S_CLEAR;
                    end
                end

                S_CLEAR: begin
                    input_index <= 0;
                    state <= S_STREAM;
                end

                S_STREAM: begin
                    // 162個輸入
                    if (input_index == INPUT_SIZE-1)
                        state <= S_DRAIN;
                    else
                        input_index <= input_index + 1;
                end

                // 累積最後一個輸入地址返回的值。
                S_DRAIN:
                    state <= S_OUTPUT;

                S_OUTPUT: begin
                    // 40個輸出
                    if (output_index == OUTPUT_SIZE-1)
                        state <= S_DONE;
                    else begin
                        output_index <= output_index + 1;
                        state <= S_CLEAR;
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
