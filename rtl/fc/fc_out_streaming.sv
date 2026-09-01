// Streaming fully-connected output layer.
//
// Each valid input is retained while one multiplier updates all OUTPUT_SIZE
// class accumulators. The upstream FC1/BN interval is long enough to finish
// these updates before the next input arrives, so the 40-word activation RAM
// between FC1/BN and FC_out is no longer required.
module fc_out_streaming #(
    parameter int INPUT_SIZE   = 40,
    parameter int OUTPUT_SIZE  = 105,
    parameter int BIAS_SHIFT   = 12,
    parameter int OUTPUT_SHIFT = 17,
    parameter WEIGHT_FILE = "mem/weights/fc_out_W.mem",
    parameter BIAS_FILE   = "mem/weights/fc_out_b.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    input  logic               input_valid,
    input  logic [$clog2(INPUT_SIZE)-1:0] input_index,
    input  logic signed [15:0] input_data,

    output logic               output_valid,
    output logic [$clog2(OUTPUT_SIZE)-1:0] output_addr,
    output logic signed [15:0] output_data
);
    localparam int INPUT_INDEX_W = (INPUT_SIZE <= 1)
                                   ? 1 : $clog2(INPUT_SIZE);
    localparam int OUTPUT_INDEX_W = (OUTPUT_SIZE <= 1)
                                    ? 1 : $clog2(OUTPUT_SIZE);
    localparam int WEIGHT_DEPTH = INPUT_SIZE * OUTPUT_SIZE;
    localparam int WEIGHT_ADDR_W = (WEIGHT_DEPTH <= 1)
                                   ? 1 : $clog2(WEIGHT_DEPTH);
    localparam int BIAS_ADDR_W = (OUTPUT_SIZE <= 1)
                                 ? 1 : $clog2(OUTPUT_SIZE);

    typedef enum logic [2:0] {
        S_IDLE,
        S_WAIT_INPUT,
        S_MAC,
        S_MAC_DRAIN,
        S_OUTPUT,
        S_OUTPUT_DRAIN,
        S_DONE
    } state_t;
    state_t state;

    logic signed [47:0] accumulators [0:OUTPUT_SIZE-1];
    logic signed [15:0] current_input;
    logic [INPUT_INDEX_W-1:0] current_input_index;

    logic [OUTPUT_INDEX_W-1:0] mac_issue_index;
    logic [OUTPUT_INDEX_W-1:0] mac_index_d1;
    logic                       mac_valid_d1;
    logic [WEIGHT_ADDR_W-1:0]   weight_addr;
    logic signed [15:0]         weight_data;
    logic signed [31:0]         product;

    logic [OUTPUT_INDEX_W-1:0] output_issue_index;
    logic [OUTPUT_INDEX_W-1:0] output_index_d1;
    logic                       output_issue_valid_d1;
    logic signed [15:0]         bias_data;
    logic signed [47:0]         bias_extended;
    logic signed [47:0]         bias_aligned;
    logic signed [47:0]         output_sum;
    logic signed [47:0]         scaled_result;
    logic signed [15:0]         saturated_result;

    integer clear_index;

    always_comb begin
        weight_addr = mac_issue_index
                    + OUTPUT_SIZE * current_input_index;
        product = current_input * weight_data;

        bias_extended = {{32{bias_data[15]}}, bias_data};
        bias_aligned = bias_extended <<< BIAS_SHIFT;
        output_sum = accumulators[output_index_d1] + bias_aligned;
        scaled_result = output_sum >>> OUTPUT_SHIFT;

        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    weight_rom #(
        .DATA_W(16), .DEPTH(WEIGHT_DEPTH), .ADDR_W(WEIGHT_ADDR_W),
        .MEM_FILE(WEIGHT_FILE), .USE_READ_ENABLE(1'b1)
    ) u_weight_rom (
        .clk(clk), .read_en(start || busy),
        .addr(weight_addr), .data(weight_data)
    );

    weight_rom #(
        .DATA_W(16), .DEPTH(OUTPUT_SIZE), .ADDR_W(BIAS_ADDR_W),
        .MEM_FILE(BIAS_FILE), .USE_READ_ENABLE(1'b1)
    ) u_bias_rom (
        .clk(clk),
        .read_en(start || busy),
        .addr(output_issue_index[BIAS_ADDR_W-1:0]),
        .data(bias_data)
    );

    sat16 u_sat16 (
        .value_in(scaled_result), .value_out(saturated_result)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            current_input <= '0;
            current_input_index <= '0;
            mac_issue_index <= '0;
            mac_index_d1 <= '0;
            mac_valid_d1 <= 1'b0;
            output_issue_index <= '0;
            output_index_d1 <= '0;
            output_issue_valid_d1 <= 1'b0;
            output_valid <= 1'b0;
            output_addr <= '0;
            output_data <= '0;
            for (clear_index = 0; clear_index < OUTPUT_SIZE;
                 clear_index = clear_index + 1)
                accumulators[clear_index] <= '0;
        // IDLE 等待期間保持 FC-out pipeline 與 accumulator。
        end else if ((state != S_IDLE) || start) begin
            mac_valid_d1 <= (state == S_MAC);
            output_issue_valid_d1 <= (state == S_OUTPUT);
            output_valid <= 1'b0;

            // Accumulate the weight returned for the preceding issued class.
            if (mac_valid_d1)
                accumulators[mac_index_d1]
                    <= accumulators[mac_index_d1] + product;

            // Bias ROM is also synchronous; emit the preceding issued class.
            if (output_issue_valid_d1) begin
                output_valid <= 1'b1;
                output_addr <= output_index_d1;
                output_data <= saturated_result;
            end

            case (state)
                S_IDLE: begin
                    mac_valid_d1 <= 1'b0;
                    output_issue_valid_d1 <= 1'b0;
                    if (start) begin
                        // accumulators 清零
                        for (clear_index = 0; clear_index < OUTPUT_SIZE;
                             clear_index = clear_index + 1)
                            accumulators[clear_index] <= '0;
                        current_input_index <= '0;
                        mac_issue_index <= '0;
                        output_issue_index <= '0;
                        state <= S_WAIT_INPUT;
                    end
                end

                // 等待 FC1/BN 的输出，接收 index 0 到 39
                S_WAIT_INPUT: begin
                    if (input_valid) begin
                        current_input <= input_data;
                        current_input_index <= input_index;
                        mac_issue_index <= '0;
                        state <= S_MAC;
                    end
                end

                // 對當前的所有輸入做 mac
                S_MAC: begin
                    // 等待 ROM 輸出延遲
                    mac_index_d1 <= mac_issue_index;
                    if (mac_issue_index == OUTPUT_SIZE-1) begin
                        state <= S_MAC_DRAIN;
                    end else begin
                        mac_issue_index <= mac_issue_index + 1'b1;
                    end
                end

                // 消耗 mac 的最後一個輸出
                S_MAC_DRAIN: begin
                    mac_issue_index <= '0;
                    // 判斷是否讀取完 40 個輸入
                    if (current_input_index == INPUT_SIZE-1) begin
                        output_issue_index <= '0;
                        state <= S_OUTPUT;
                    end else begin
                        state <= S_WAIT_INPUT;
                    end
                end

                // 105 clock cycle，依次加上 Bias
                S_OUTPUT: begin
                    // 等待 ROM 輸出延遲
                    output_index_d1 <= output_issue_index;
                    if (output_issue_index == OUTPUT_SIZE-1) begin
                        state <= S_OUTPUT_DRAIN;
                    end else begin
                        output_issue_index <= output_issue_index + 1'b1;
                    end
                end

                // ROM 輸出延遲，吐出第 105 個輸出結果
                S_OUTPUT_DRAIN:
                    state <= S_DONE;

                S_DONE:
                    state <= S_IDLE;

                default:
                    state <= S_IDLE;
            endcase
        end
    end
endmodule
