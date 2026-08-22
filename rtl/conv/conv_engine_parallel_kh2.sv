// Conv2 
// Kernel height is 2
// 同 clk 計算兩個 kh ，並用 pipeline 串接 

module conv_engine_parallel_kh2 #(
    parameter int IN_H   = 20,
    parameter int IN_W   = 156,
    parameter int IN_CH  = 21,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 20,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int LANES  = 5,
    parameter int BIAS_SHIFT   = 10,
    parameter int OUTPUT_SHIFT = 15
) (
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           start,
    output logic                           busy,
    output logic                           done,

    output logic [31:0]                    input_addr_kh0,
    output logic [31:0]                    input_addr_kh1,
    input  logic signed [15:0]             input_data_kh0,
    input  logic signed [15:0]             input_data_kh1,

    output logic [31:0]                    weight_addr,
    input  logic signed [(32*LANES)-1:0]   weight_data,
    output logic [31:0]                    bias_addr,
    input  logic signed [(16*LANES)-1:0]   bias_data,

    output logic                           output_valid,
    output logic [31:0]                    output_addr,
    output logic [31:0]                    output_ch_idx,
    output logic signed [15:0]             output_data
);
    localparam int OUT_GROUPS = (OUT_CH + LANES - 1) / LANES;
    localparam int INPUT_DEPTH = IN_H * IN_W * IN_CH;
    localparam int KH2_WORDS_PER_GROUP = K_W * IN_CH;
    localparam int PACKED_WEIGHT_DEPTH = KH2_WORDS_PER_GROUP * OUT_GROUPS;

    localparam int OUT_H_W = (OUT_H <= 1) ? 1 : $clog2(OUT_H);
    localparam int OUT_W_W = (OUT_W <= 1) ? 1 : $clog2(OUT_W);
    localparam int OUT_GROUP_W = (OUT_GROUPS <= 1) ? 1 : $clog2(OUT_GROUPS);
    localparam int K_W_W = (K_W <= 1) ? 1 : $clog2(K_W);
    localparam int IN_CH_W = (IN_CH <= 1) ? 1 : $clog2(IN_CH);
    localparam int LANE_W = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam int OUTPUT_CHANNEL_W = ((OUT_GROUPS * LANES) <= 1)
                                    ? 1 : $clog2(OUT_GROUPS * LANES);
    localparam int INPUT_ADDR_W = (INPUT_DEPTH <= 1)
                                ? 1 : $clog2(INPUT_DEPTH);
    localparam int WEIGHT_ADDR_W = (PACKED_WEIGHT_DEPTH <= 1)
                                 ? 1 : $clog2(PACKED_WEIGHT_DEPTH);

    // Address order after kh is removed from the loop: kw -> input channel.
    localparam int INPUT_KW_STEP = IN_H;
    localparam int INPUT_CH_STEP = IN_H * IN_W - (K_W - 1) * IN_H;
    localparam int INPUT_NEXT_W_STEP = IN_H - (OUT_H - 1);

    typedef enum logic [2:0] {
        S_IDLE,
        S_CLEAR,
        S_STREAM,
        S_DRAIN,
        S_OUTPUT,
        S_DONE
    } state_t;
    state_t state;

    logic [OUT_H_W-1:0] out_h_count;
    logic [OUT_W_W-1:0] out_w_count;
    logic [OUT_GROUP_W-1:0] out_group_count;
    logic [K_W_W-1:0] kw_count;
    logic [IN_CH_W-1:0] in_ch_count;
    logic [LANE_W-1:0] output_lane_count;
    logic [OUTPUT_CHANNEL_W-1:0] output_channel;

    logic [INPUT_ADDR_W-1:0] input_window_base;
    logic [INPUT_ADDR_W-1:0] input_addr_count;
    logic [WEIGHT_ADDR_W-1:0] weight_group_base;
    logic [WEIGHT_ADDR_W-1:0] weight_addr_count;

    logic read_valid, product_valid, pair_valid;
    logic lane_is_valid;

    logic signed [15:0] weight_kh0 [0:LANES-1];
    logic signed [15:0] weight_kh1 [0:LANES-1];
    logic signed [15:0] bias_lane [0:LANES-1];
    logic signed [31:0] product_kh0 [0:LANES-1];
    logic signed [31:0] product_kh1 [0:LANES-1];
    logic signed [32:0] pair_sum [0:LANES-1];
    logic signed [47:0] accumulators [0:LANES-1];

    logic signed [47:0] selected_accumulator;
    logic signed [15:0] selected_bias;
    logic signed [47:0] bias_extended;
    logic signed [47:0] bias_aligned;
    logic signed [47:0] sum_with_bias;
    logic signed [47:0] scaled_result;
    logic signed [15:0] saturated_result;

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : gen_lanes
            assign weight_kh0[lane] = weight_data[(16*lane) +: 16];
            assign weight_kh1[lane] =
                weight_data[(16*(LANES + lane)) +: 16];
            assign bias_lane[lane] = bias_data[(16*lane) +: 16];
        end
    endgenerate

    always_comb begin
        // input feature map address
        input_addr_kh0 = '0;
        input_addr_kh0[INPUT_ADDR_W-1:0] = input_addr_count;
        input_addr_kh1 = '0;
        input_addr_kh1[INPUT_ADDR_W-1:0] = input_addr_count + 1'b1;

        // weight address
        weight_addr = '0;
        weight_addr[WEIGHT_ADDR_W-1:0] = weight_addr_count;

        // include LANES lane data
        bias_addr = '0;
        bias_addr[OUT_GROUP_W-1:0] = out_group_count;

        output_channel = out_group_count * LANES + output_lane_count;
        lane_is_valid = (output_channel < OUT_CH);
        output_addr = out_h_count
                    + OUT_H * (out_w_count + OUT_W * output_channel);
        output_ch_idx = output_channel;

        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
        output_valid = (state == S_OUTPUT) && lane_is_valid;

        selected_accumulator = accumulators[output_lane_count];
        selected_bias = bias_lane[output_lane_count];
        bias_extended = {{32{selected_bias[15]}}, selected_bias};
        bias_aligned = bias_extended <<< BIAS_SHIFT;
        // add bias
        sum_with_bias = selected_accumulator + bias_aligned;
        scaled_result = sum_with_bias >>> OUTPUT_SHIFT;
        output_data = output_valid ? saturated_result : 16'sd0;
    end

    sat16 u_sat16 (
        .value_in(scaled_result),
        .value_out(saturated_result)
    );

    integer lane_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            out_h_count <= '0;
            out_w_count <= '0;
            out_group_count <= '0;
            kw_count <= '0;
            in_ch_count <= '0;
            output_lane_count <= '0;
            input_window_base <= '0;
            input_addr_count <= '0;
            weight_group_base <= '0;
            weight_addr_count <= '0;
            read_valid <= 1'b0;
            product_valid <= 1'b0;
            pair_valid <= 1'b0;
            for (lane_index = 0; lane_index < LANES;
                 lane_index = lane_index + 1) begin
                product_kh0[lane_index] <= '0;
                product_kh1[lane_index] <= '0;
                pair_sum[lane_index] <= '0;
                accumulators[lane_index] <= '0;
            end
        end else begin
            // Valid bits follow the synchronous RAM output through all three
            // arithmetic stages. Once filled, one kh-pair is accepted/clock.
            read_valid <= (state == S_STREAM);
            product_valid <= read_valid;
            pair_valid <= product_valid;

            if (read_valid) begin
                for (lane_index = 0; lane_index < LANES;
                     lane_index = lane_index + 1) begin
                    product_kh0[lane_index]
                        <= input_data_kh0 * weight_kh0[lane_index];
                    product_kh1[lane_index]
                        <= input_data_kh1 * weight_kh1[lane_index];
                end
            end

            if (product_valid) begin
                for (lane_index = 0; lane_index < LANES;
                     lane_index = lane_index + 1)
                    pair_sum[lane_index]
                        <= product_kh0[lane_index] + product_kh1[lane_index];
            end

            if (pair_valid) begin
                for (lane_index = 0; lane_index < LANES;
                     lane_index = lane_index + 1)
                    accumulators[lane_index] <= accumulators[lane_index]
                        + {{15{pair_sum[lane_index][32]}}, pair_sum[lane_index]};
            end

            case (state)
                S_IDLE: begin
                    read_valid <= 1'b0;
                    product_valid <= 1'b0;
                    pair_valid <= 1'b0;
                    if (start) begin
                        out_h_count <= '0;
                        out_w_count <= '0;
                        out_group_count <= '0;
                        kw_count <= '0;
                        in_ch_count <= '0;
                        output_lane_count <= '0;
                        input_window_base <= '0;
                        input_addr_count <= '0;
                        weight_group_base <= '0;
                        weight_addr_count <= '0;
                        state <= S_CLEAR;
                    end
                end

                S_CLEAR: begin
                    kw_count <= '0;
                    in_ch_count <= '0;
                    output_lane_count <= '0;
                    input_addr_count <= input_window_base;
                    weight_addr_count <= weight_group_base;
                    for (lane_index = 0; lane_index < LANES;
                         lane_index = lane_index + 1)
                        accumulators[lane_index] <= '0;
                    state <= S_STREAM;
                end

                // start mac operation
                S_STREAM: begin
                    if (kw_count < K_W-1) begin
                        kw_count <= kw_count + 1'b1;
                        input_addr_count <= input_addr_count + INPUT_KW_STEP;
                        weight_addr_count <= weight_addr_count + 1'b1;
                    end else begin
                        kw_count <= '0;
                        if (in_ch_count < IN_CH-1) begin
                            in_ch_count <= in_ch_count + 1'b1;
                            input_addr_count
                                <= input_addr_count + INPUT_CH_STEP;
                            weight_addr_count <= weight_addr_count + 1'b1;
                        end else begin
                            state <= S_DRAIN;
                        end
                    end
                end

                // Wait until the final issued read has passed products,
                // pair_sum and the accumulator register.
                S_DRAIN: begin
                    if (!read_valid && !product_valid && !pair_valid) begin
                        output_lane_count <= '0;
                        state <= S_OUTPUT;
                    end
                end

                // K_W*IN_CH 組雙 kh mac 計算完畢，依次輸出
                S_OUTPUT: begin
                    if ((output_lane_count < LANES-1) &&
                        (output_channel + 1 < OUT_CH)) begin
                        output_lane_count <= output_lane_count + 1'b1;
                    // 所有 lane 結束
                    end else begin
                        // 依次輸出 LANES 筆 output_channel_addr
                        output_lane_count <= '0;
                        if (out_h_count < OUT_H-1) begin
                            out_h_count <= out_h_count + 1'b1;
                            input_window_base <= input_window_base + 1'b1;
                            state <= S_CLEAR;
                        end else begin
                            // height 跑完，進行 width 輸出
                            out_h_count <= '0;
                            if (out_w_count < OUT_W-1) begin
                                out_w_count <= out_w_count + 1'b1;
                                input_window_base <= input_window_base
                                                   + INPUT_NEXT_W_STEP;
                                state <= S_CLEAR;
                            end else begin
                                // width 跑完，進行 group 輸出
                                out_w_count <= '0;
                                input_window_base <= '0;
                                if (out_group_count < OUT_GROUPS-1) begin
                                    out_group_count
                                        <= out_group_count + 1'b1;
                                    weight_group_base <= weight_group_base
                                                       + KH2_WORDS_PER_GROUP;
                                    state <= S_CLEAR;
                                end else begin
                                    state <= S_DONE;
                                end
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
