// 將地址計算改為使用 counter，以縮短地址到 RAM 距離

module conv_engine_parallel_counter #(
    parameter int IN_H   = 20,
    parameter int IN_W   = 156,
    parameter int IN_CH  = 21,
    parameter int K_H    = 2,
    parameter int K_W    = 5,
    parameter int OUT_CH = 20,
    parameter int OUT_H  = IN_H - K_H + 1,
    parameter int OUT_W  = IN_W - K_W + 1,
    parameter int LANES  = 4,
    parameter int BIAS_SHIFT   = 10,
    parameter int OUTPUT_SHIFT = 15
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start,
    output logic                         busy,
    output logic                         done,

    output logic [31:0]                  input_addr,
    input  logic signed [15:0]           input_data,

    output logic [31:0]                  weight_addr,
    input  logic signed [(16*LANES)-1:0] weight_data,
    output logic [31:0]                  bias_addr,
    input  logic signed [(16*LANES)-1:0] bias_data,

    output logic                         output_valid,
    output logic [31:0]                  output_addr,
    output logic [31:0]                  output_ch_idx,
    output logic signed [15:0]           output_data
);
    localparam int OUT_GROUPS = (OUT_CH + LANES - 1) / LANES;
    localparam int INPUT_DEPTH = IN_H * IN_W * IN_CH;
    localparam int KERNEL_WORDS = K_H * K_W * IN_CH;
    localparam int PACKED_WEIGHT_DEPTH = KERNEL_WORDS * OUT_GROUPS;

    localparam int OUT_H_W = (OUT_H <= 1) ? 1 : $clog2(OUT_H);
    localparam int OUT_W_W = (OUT_W <= 1) ? 1 : $clog2(OUT_W);
    localparam int OUT_GROUP_W = (OUT_GROUPS <= 1)
                               ? 1 : $clog2(OUT_GROUPS);
    localparam int K_H_W = (K_H <= 1) ? 1 : $clog2(K_H);
    localparam int K_W_W = (K_W <= 1) ? 1 : $clog2(K_W);
    localparam int IN_CH_W = (IN_CH <= 1) ? 1 : $clog2(IN_CH);
    localparam int LANE_W = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam int OUTPUT_CHANNEL_W = ((OUT_GROUPS * LANES) <= 1)
                                    ? 1 : $clog2(OUT_GROUPS * LANES);
    localparam int INPUT_ADDR_W = (INPUT_DEPTH <= 1)
                                ? 1 : $clog2(INPUT_DEPTH);
    localparam int WEIGHT_ADDR_W = (PACKED_WEIGHT_DEPTH <= 1)
                                 ? 1 : $clog2(PACKED_WEIGHT_DEPTH);

    // Constant input-address strides for the current loop order:
    // kh (fastest) -> kw -> input channel.
    localparam int INPUT_KH_STEP = 1;
    localparam int INPUT_KW_STEP = IN_H - (K_H - 1);
    localparam int INPUT_CH_STEP = IN_H * IN_W
                                 - (K_W - 1) * IN_H
                                 - (K_H - 1);
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
    logic [K_H_W-1:0] kh_count;
    logic [K_W_W-1:0] kw_count;
    logic [IN_CH_W-1:0] in_ch_count;
    logic [LANE_W-1:0] output_lane_count;
    logic [OUTPUT_CHANNEL_W-1:0] output_channel;

    // Base registers change only when the output window/group changes.
    // Active address counters then use only one constant-width addition.
    logic [INPUT_ADDR_W-1:0] input_window_base;
    logic [INPUT_ADDR_W-1:0] input_addr_count;
    logic [WEIGHT_ADDR_W-1:0] weight_group_base;
    logic [WEIGHT_ADDR_W-1:0] weight_addr_count;

    logic data_valid;
    logic clear_acc;
    logic mac_en;
    logic lane_is_valid;

    logic signed [15:0] weight_lane [0:LANES-1];
    logic signed [15:0] bias_lane [0:LANES-1];
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
            assign weight_lane[lane] = weight_data[(16*lane) +: 16];
            assign bias_lane[lane] = bias_data[(16*lane) +: 16];

            pe_mac u_mac (
                .clk(clk), .rst_n(rst_n),
                .clear_acc(clear_acc), .mac_en(mac_en),
                .data_in(input_data), .weight_in(weight_lane[lane]),
                .accumulator(accumulators[lane])
            );
        end
    endgenerate

    always_comb begin
        // input feature map address
        input_addr = '0;
        input_addr[INPUT_ADDR_W-1:0] = input_addr_count;

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
        clear_acc = (state == S_CLEAR);
        mac_en = data_valid;
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            out_h_count <= '0;
            out_w_count <= '0;
            out_group_count <= '0;
            kh_count <= '0;
            kw_count <= '0;
            in_ch_count <= '0;
            output_lane_count <= '0;
            input_window_base <= '0;
            input_addr_count <= '0;
            weight_group_base <= '0;
            weight_addr_count <= '0;
            data_valid <= 1'b0;
        end else begin
            // Activation RAM and packed weight ROM both have one-clock latency.
            data_valid <= (state == S_STREAM);

            case (state)
                S_IDLE: begin
                    data_valid <= 1'b0;
                    if (start) begin
                        out_h_count <= '0;
                        out_w_count <= '0;
                        out_group_count <= '0;
                        kh_count <= '0;
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

                // Reset the two active counters to the registered bases. This
                // also clears all four MAC lanes for the next output group.
                S_CLEAR: begin
                    kh_count <= '0;
                    kw_count <= '0;
                    in_ch_count <= '0;
                    output_lane_count <= '0;
                    input_addr_count <= input_window_base;
                    weight_addr_count <= weight_group_base;
                    state <= S_STREAM;
                end

                // The address presented during this cycle belongs to the
                // current counters. Prepare the next address for the next edge.
                // start mac operation
                S_STREAM: begin
                    if (kh_count < K_H-1) begin
                        kh_count <= kh_count + 1'b1;
                        input_addr_count
                            <= input_addr_count + INPUT_KH_STEP;
                        weight_addr_count <= weight_addr_count + 1'b1;
                    end else begin
                        kh_count <= '0;
                        if (kw_count < K_W-1) begin
                            kw_count <= kw_count + 1'b1;
                            input_addr_count
                                <= input_addr_count + INPUT_KW_STEP;
                            weight_addr_count <= weight_addr_count + 1'b1;
                        end else begin
                            kw_count <= '0;
                            if (in_ch_count < IN_CH-1) begin
                                in_ch_count <= in_ch_count + 1'b1;
                                input_addr_count
                                    <= input_addr_count + INPUT_CH_STEP;
                                weight_addr_count
                                    <= weight_addr_count + 1'b1;
                            end else begin
                                state <= S_DRAIN;
                            end
                        end
                    end
                end

                // Consume the final synchronous memory result.
                S_DRAIN: begin
                    output_lane_count <= '0;
                    state <= S_OUTPUT;
                end

                // K_H*K_W*IN_CH products are complete for all LANES.
                // K_H*K_W*IN_CH 筆 mac 計算完畢，依次輸出
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
                                input_window_base
                                    <= input_window_base
                                     + INPUT_NEXT_W_STEP;
                                state <= S_CLEAR;
                            end else begin
                                // width 跑完，進行 group 輸出
                                out_w_count <= '0;
                                input_window_base <= '0;
                                if (out_group_count < OUT_GROUPS-1) begin
                                    out_group_count
                                        <= out_group_count + 1'b1;
                                    weight_group_base
                                        <= weight_group_base + KERNEL_WORDS;
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
