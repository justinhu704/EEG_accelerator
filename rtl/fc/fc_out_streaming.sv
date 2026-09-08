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
    // ---------------------------------------------------------------------
    // 參數與狀態
    // ---------------------------------------------------------------------
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

    // ---------------------------------------------------------------------
    // Accumulator RAM：取代 105 組 register 與大型動態索引 mux
    // ---------------------------------------------------------------------
    (* ramstyle = "M10K, no_rw_check" *)
    logic signed [47:0] accumulator_ram [0:OUTPUT_SIZE-1];

    logic                       accumulator_read_enable;
    logic [OUTPUT_INDEX_W-1:0]  accumulator_read_addr;
    logic signed [47:0]         accumulator_read_data;
    logic                       accumulator_write_enable;
    logic [OUTPUT_INDEX_W-1:0]  accumulator_write_addr;
    logic signed [47:0]         accumulator_write_data;

    // 單讀單寫同步RAM，可由Quartus推導成M10K。
    always_ff @(posedge clk) begin
        if (accumulator_read_enable)
            accumulator_read_data
                <= accumulator_ram[accumulator_read_addr];

        if (accumulator_write_enable)
            accumulator_ram[accumulator_write_addr]
                <= accumulator_write_data;
    end

    // ---------------------------------------------------------------------
    // MAC pipeline：同步讀Weight/RAM -> 乘法 -> 加法並寫回
    // ---------------------------------------------------------------------
    logic signed [15:0] current_input;
    logic [INPUT_INDEX_W-1:0] current_input_index;

    logic [OUTPUT_INDEX_W-1:0] mac_issue_index;
    logic [WEIGHT_ADDR_W-1:0]  weight_issue_addr;
    logic signed [15:0]        weight_data;

    logic [OUTPUT_INDEX_W-1:0] mac_index_d1;
    logic                      mac_first_input_d1;
    logic                      mac_valid_d1;

    logic [OUTPUT_INDEX_W-1:0] mac_index_d2;
    logic                      mac_first_input_d2;
    logic                      mac_valid_d2;
    logic signed [31:0]        product_d2;
    logic signed [47:0]        accumulator_old_d2;
    logic signed [47:0]        product_extended;

    weight_rom #(
        .DATA_W(16), .DEPTH(WEIGHT_DEPTH), .ADDR_W(WEIGHT_ADDR_W),
        .MEM_FILE(WEIGHT_FILE), .USE_READ_ENABLE(1'b1)
    ) u_weight_rom (
        .clk(clk),
        .read_en(state == S_MAC),
        .addr(weight_issue_addr),
        .data(weight_data)
    );

    always_comb begin
        product_extended = {{16{product_d2[31]}}, product_d2};

        accumulator_write_enable = mac_valid_d2;
        accumulator_write_addr = mac_index_d2;

        // 第一個input直接覆寫，不需要額外花105 cycles清除RAM。
        if (mac_first_input_d2)
            accumulator_write_data = product_extended;
        else
            accumulator_write_data = accumulator_old_d2
                                   + product_extended;
    end

    // ---------------------------------------------------------------------
    // Output pipeline：讀Accumulator/Bias -> 加Bias -> Shift/Saturation
    // ---------------------------------------------------------------------
    logic [OUTPUT_INDEX_W-1:0] output_issue_index;
    logic [OUTPUT_INDEX_W-1:0] output_index_d1;
    logic                      output_valid_d1;
    logic [OUTPUT_INDEX_W-1:0] output_index_d2;
    logic                      output_valid_d2;
    logic signed [47:0]        output_sum_d2;
    logic [OUTPUT_INDEX_W-1:0] output_index_d3;
    logic                      output_valid_d3;
    logic signed [47:0]        scaled_result_d3;

    logic signed [15:0] bias_data;
    logic signed [47:0] bias_extended;
    logic signed [47:0] bias_aligned;
    logic signed [15:0] saturated_result;

    weight_rom #(
        .DATA_W(16), .DEPTH(OUTPUT_SIZE), .ADDR_W(BIAS_ADDR_W),
        .MEM_FILE(BIAS_FILE), .USE_READ_ENABLE(1'b1)
    ) u_bias_rom (
        .clk(clk),
        .read_en(state == S_OUTPUT),
        .addr(output_issue_index[BIAS_ADDR_W-1:0]),
        .data(bias_data)
    );

    sat16 u_sat16 (
        .value_in(scaled_result_d3),
        .value_out(saturated_result)
    );

    always_comb begin
        bias_extended = {{32{bias_data[15]}}, bias_data};
        bias_aligned = bias_extended <<< BIAS_SHIFT;

        // MAC與輸出共用Accumulator RAM的同步讀取port。
        accumulator_read_enable = (state == S_MAC)
                               || (state == S_OUTPUT);
        if (state == S_OUTPUT)
            accumulator_read_addr = output_issue_index;
        else
            accumulator_read_addr = mac_issue_index;

        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    // ---------------------------------------------------------------------
    // Pipeline控制與狀態機
    // ---------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            current_input <= '0;
            current_input_index <= '0;
            mac_issue_index <= '0;
            weight_issue_addr <= '0;
            mac_index_d1 <= '0;
            mac_first_input_d1 <= 1'b0;
            mac_valid_d1 <= 1'b0;
            mac_index_d2 <= '0;
            mac_first_input_d2 <= 1'b0;
            mac_valid_d2 <= 1'b0;
            product_d2 <= '0;
            accumulator_old_d2 <= '0;
            output_issue_index <= '0;
            output_index_d1 <= '0;
            output_valid_d1 <= 1'b0;
            output_index_d2 <= '0;
            output_valid_d2 <= 1'b0;
            output_sum_d2 <= '0;
            output_index_d3 <= '0;
            output_valid_d3 <= 1'b0;
            scaled_result_d3 <= '0;
            output_valid <= 1'b0;
            output_addr <= '0;
            output_data <= '0;
        // IDLE等待期間保持FC-out pipeline與Accumulator RAM。
        end else if ((state != S_IDLE) || start) begin
            mac_valid_d1 <= (state == S_MAC);
            mac_valid_d2 <= mac_valid_d1;
            output_valid_d1 <= (state == S_OUTPUT);
            output_valid_d2 <= output_valid_d1;
            output_valid_d3 <= output_valid_d2;
            output_valid <= 1'b0;

            // Weight ROM與Accumulator RAM均為同步讀取。
            if (mac_valid_d1) begin
                mac_index_d2 <= mac_index_d1;
                mac_first_input_d2 <= mac_first_input_d1;
                product_d2 <= current_input * weight_data;
                accumulator_old_d2 <= accumulator_read_data;
            end

            // 將Bias加法與最後的shift/saturation分成不同pipeline級。
            if (output_valid_d1) begin
                output_index_d2 <= output_index_d1;
                output_sum_d2 <= accumulator_read_data + bias_aligned;
            end
            if (output_valid_d2) begin
                output_index_d3 <= output_index_d2;
                scaled_result_d3 <= output_sum_d2 >>> OUTPUT_SHIFT;
            end
            if (output_valid_d3) begin
                output_valid <= 1'b1;
                output_addr <= output_index_d3;
                output_data <= saturated_result;
            end

            case (state)
                S_IDLE: begin
                    mac_valid_d1 <= 1'b0;
                    mac_valid_d2 <= 1'b0;
                    output_valid_d1 <= 1'b0;
                    output_valid_d2 <= 1'b0;
                    output_valid_d3 <= 1'b0;
                    if (start) begin
                        current_input_index <= '0;
                        mac_issue_index <= '0;
                        weight_issue_addr <= '0;
                        output_issue_index <= '0;
                        state <= S_WAIT_INPUT;
                    end
                end

                // 等待FC1/BN輸出，接收index 0到39。
                S_WAIT_INPUT: begin
                    if (input_valid) begin
                        current_input <= input_data;
                        current_input_index <= input_index;
                        mac_issue_index <= '0;
                        state <= S_MAC;
                    end
                end

                // 依序送出105個class的Weight與Accumulator地址。
                S_MAC: begin
                    mac_index_d1 <= mac_issue_index;
                    mac_first_input_d1 <= (current_input_index == '0);
                    weight_issue_addr <= weight_issue_addr + 1'b1;

                    if (mac_issue_index == OUTPUT_SIZE-1) begin
                        state <= S_MAC_DRAIN;
                    end else begin
                        mac_issue_index <= mac_issue_index + 1'b1;
                    end
                end

                // 等待最後一筆乘法與RAM寫回完成。
                S_MAC_DRAIN: begin
                    mac_issue_index <= '0;
                    if (mac_valid_d2
                            && (mac_index_d2 == OUTPUT_SIZE-1)) begin
                        if (current_input_index == INPUT_SIZE-1) begin
                            output_issue_index <= '0;
                            state <= S_OUTPUT;
                        end else begin
                            state <= S_WAIT_INPUT;
                        end
                    end
                end

                // 依序讀出105個Accumulator與Bias。
                S_OUTPUT: begin
                    output_index_d1 <= output_issue_index;
                    if (output_issue_index == OUTPUT_SIZE-1) begin
                        state <= S_OUTPUT_DRAIN;
                    end else begin
                        output_issue_index <= output_issue_index + 1'b1;
                    end
                end

                // 等待最後一筆logit通過輸出pipeline。
                S_OUTPUT_DRAIN: begin
                    if (output_valid_d3
                            && (output_index_d3 == OUTPUT_SIZE-1))
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
