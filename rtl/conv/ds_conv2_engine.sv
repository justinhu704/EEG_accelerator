// Depthwise-separable Conv2 engine
// 輸出順序：width -> height -> channel group -> lane
module ds_conv2_engine #(
    parameter integer DATA_WIDTH        = 16,
    parameter integer WEIGHT_WIDTH      = 16,
    parameter integer BIAS_WIDTH        = 16,
    parameter integer INPUT_ADDR_WIDTH  = 16,
    parameter integer OUTPUT_ADDR_WIDTH = 16,

    parameter integer IN_H   = 20,
    parameter integer IN_W   = 156,
    parameter integer IN_CH  = 21,
    parameter integer K_H    = 2,
    parameter integer K_W    = 5,
    parameter integer OUT_CH = 20,
    parameter integer LANES  = 5,

    // product 與 bias 對齊後，再轉回 activation 格式
    parameter integer DW_BIAS_SHIFT = 10,
    parameter integer DW_OUT_SHIFT  = 15,
    parameter integer PW_BIAS_SHIFT = 10,
    parameter integer PW_OUT_SHIFT  = 15,
    parameter integer ACC_WIDTH     = 48
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    output logic busy,
    output logic done,

    // Conv1 banked RAM 接收邏輯位址，奇偶 bank 轉換由 RAM 內部完成。
    output logic [INPUT_ADDR_WIDTH-1:0] input_addr_kh0,
    output logic [INPUT_ADDR_WIDTH-1:0] input_addr_kh1,
    input  logic signed [DATA_WIDTH-1:0] input_data_kh0,
    input  logic signed [DATA_WIDTH-1:0] input_data_kh1,

    // Depthwise ROM：一個 word 同時提供 kh=0、kh=1 的權重
    output logic [$clog2(IN_CH*K_W)-1:0] dw_weight_addr,
    input  logic signed [(K_H*WEIGHT_WIDTH)-1:0] dw_weight_data,
    output logic [$clog2(IN_CH)-1:0] dw_bias_addr,
    input  logic signed [BIAS_WIDTH-1:0] dw_bias_data,

    // Pointwise ROM：一個 word 提供 LANES 個輸出 channel 的權重
    output logic [$clog2(IN_CH*(OUT_CH/LANES))-1:0] pw_weight_addr,
    input  logic signed [(LANES*WEIGHT_WIDTH)-1:0] pw_weight_data,
    output logic [$clog2(OUT_CH/LANES)-1:0] pw_bias_addr,
    input  logic signed [(LANES*BIAS_WIDTH)-1:0] pw_bias_data,

    // 串流輸出：同一個空間位置會連續輸出全部 OUT_CH
    output logic output_valid,
    output logic output_last,
    output logic signed [DATA_WIDTH-1:0] output_data,
    output logic [OUTPUT_ADDR_WIDTH-1:0] output_addr,
    output logic [$clog2(IN_H-K_H+1)-1:0] output_h,
    output logic [$clog2(IN_W-K_W+1)-1:0] output_w,
    output logic [$clog2(OUT_CH)-1:0] output_channel
);

    // ------------------------------------------------------------------
    // 尺寸與狀態
    // ------------------------------------------------------------------
    localparam integer OUT_H      = IN_H - K_H + 1;
    localparam integer OUT_W      = IN_W - K_W + 1;
    localparam integer OUT_GROUPS = OUT_CH / LANES;
    localparam integer PROD_WIDTH = DATA_WIDTH + WEIGHT_WIDTH;
    localparam integer PAIR_WIDTH = PROD_WIDTH + 1;

    localparam integer H_W      = (OUT_H > 1) ? $clog2(OUT_H) : 1;
    localparam integer W_W      = (OUT_W > 1) ? $clog2(OUT_W) : 1;
    localparam integer CH_W     = (IN_CH > 1) ? $clog2(IN_CH) : 1;
    localparam integer KW_W     = (K_W > 1) ? $clog2(K_W) : 1;
    localparam integer GROUP_W  = (OUT_GROUPS > 1) ? $clog2(OUT_GROUPS) : 1;
    localparam integer LANE_W   = (LANES > 1) ? $clog2(LANES) : 1;

    typedef enum logic [2:0] {
        S_IDLE,
        S_DW_PREP,
        S_DW_STREAM,
        S_DRAIN,
        S_OUTPUT
    } state_t;

    state_t state;

    logic [H_W-1:0]     out_h_count;
    logic [W_W-1:0]     out_w_count;
    logic [CH_W-1:0]    dw_issue_channel;
    logic [KW_W-1:0]    dw_issue_kw;
    logic [GROUP_W-1:0] output_group;
    logic [LANE_W-1:0]  output_lane;

    // Fused Pointwise 排程器會直接參與 ROM 位址產生。
    logic pw_issue_active;
    logic [GROUP_W-1:0] pw_issue_group;
    logic [CH_W-1:0] pw_pending_channel;
    logic signed [DATA_WIDTH-1:0] pw_pending_activation;

    logic [31:0] dw_weight_addr_full;
    logic [31:0] pw_weight_addr_full;
    logic [31:0] output_channel_full;

    assign busy = (state != S_IDLE);

    // ------------------------------------------------------------------
    // RAM / ROM 位址
    // ------------------------------------------------------------------
    always_comb begin
        input_addr_kh0 = out_h_count
                       + IN_H * ((out_w_count + dw_issue_kw)
                       + IN_W * dw_issue_channel);
        input_addr_kh1 = (out_h_count + 1'b1)
                       + IN_H * ((out_w_count + dw_issue_kw)
                       + IN_W * dw_issue_channel);

        dw_weight_addr_full = dw_issue_kw + K_W * dw_issue_channel;
        dw_weight_addr = dw_weight_addr_full[$clog2(IN_CH*K_W)-1:0];
        dw_bias_addr = dw_issue_channel;

        // Pointwise 位址由剛完成的 DW channel 與目前 group 共同決定。
        pw_weight_addr_full = pw_pending_channel + IN_CH * pw_issue_group;
        pw_weight_addr = pw_weight_addr_full[
            $clog2(IN_CH*(OUT_CH/LANES))-1:0];
        pw_bias_addr = pw_issue_group;
        output_channel_full = output_group * LANES + output_lane;
    end

    wire logic signed [WEIGHT_WIDTH-1:0] dw_w_kh0 =
        dw_weight_data[0 +: WEIGHT_WIDTH];
    wire logic signed [WEIGHT_WIDTH-1:0] dw_w_kh1 =
        dw_weight_data[WEIGHT_WIDTH +: WEIGHT_WIDTH];

    // ------------------------------------------------------------------
    // 飽和與定點轉換
    // ------------------------------------------------------------------
    function automatic logic signed [DATA_WIDTH-1:0] quantize_dw(
        input logic signed [ACC_WIDTH-1:0] acc,
        input logic signed [BIAS_WIDTH-1:0] bias
    );
        logic signed [ACC_WIDTH-1:0] bias_ext;
        logic signed [ACC_WIDTH-1:0] biased;
        logic signed [ACC_WIDTH-1:0] shifted;
        begin
            bias_ext = {{(ACC_WIDTH-BIAS_WIDTH){bias[BIAS_WIDTH-1]}}, bias};
            if (DW_BIAS_SHIFT >= 0)
                bias_ext = bias_ext <<< DW_BIAS_SHIFT;
            else
                bias_ext = bias_ext >>> (-DW_BIAS_SHIFT);

            biased = acc + bias_ext;
            if (DW_OUT_SHIFT >= 0)
                shifted = biased >>> DW_OUT_SHIFT;
            else
                shifted = biased <<< (-DW_OUT_SHIFT);

            if (shifted > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}}))
                quantize_dw = {1'b0, {(DATA_WIDTH-1){1'b1}}};
            else if (shifted < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}}))
                quantize_dw = {1'b1, {(DATA_WIDTH-1){1'b0}}};
            else
                quantize_dw = shifted[DATA_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] quantize_pw(
        input logic signed [ACC_WIDTH-1:0] acc,
        input logic signed [BIAS_WIDTH-1:0] bias
    );
        logic signed [ACC_WIDTH-1:0] bias_ext;
        logic signed [ACC_WIDTH-1:0] biased;
        logic signed [ACC_WIDTH-1:0] shifted;
        begin
            bias_ext = {{(ACC_WIDTH-BIAS_WIDTH){bias[BIAS_WIDTH-1]}}, bias};
            if (PW_BIAS_SHIFT >= 0)
                bias_ext = bias_ext <<< PW_BIAS_SHIFT;
            else
                bias_ext = bias_ext >>> (-PW_BIAS_SHIFT);

            biased = acc + bias_ext;
            if (PW_OUT_SHIFT >= 0)
                shifted = biased >>> PW_OUT_SHIFT;
            else
                shifted = biased <<< (-PW_OUT_SHIFT);

            if (shifted > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}}))
                quantize_pw = {1'b0, {(DATA_WIDTH-1){1'b1}}};
            else if (shifted < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}}))
                quantize_pw = {1'b1, {(DATA_WIDTH-1){1'b0}}};
            else
                quantize_pw = shifted[DATA_WIDTH-1:0];
        end
    endfunction

    // ------------------------------------------------------------------
    // Depthwise pipeline：read -> multiply -> pair add -> accumulate
    // channel tag 跟著資料前進，因此 channel 之間不必排空 pipeline。
    // ------------------------------------------------------------------
    logic dw_read_valid, dw_read_first, dw_read_last;
    logic dw_prod_valid, dw_prod_first, dw_prod_last;
    logic dw_pair_valid, dw_pair_first, dw_pair_last;
    logic [CH_W-1:0] dw_read_channel, dw_prod_channel, dw_pair_channel;
    logic signed [BIAS_WIDTH-1:0] dw_prod_bias, dw_pair_bias;

    logic signed [PROD_WIDTH-1:0] dw_product_kh0;
    logic signed [PROD_WIDTH-1:0] dw_product_kh1;
    logic signed [PAIR_WIDTH-1:0] dw_pair_sum;
    logic signed [ACC_WIDTH-1:0]  dw_accumulator;

    // ------------------------------------------------------------------
    // Fused Pointwise pipeline
    // 每完成一個 DW channel，就在下一個 DW channel 計算期間依序更新
    // OUT_GROUPS 組 Pointwise accumulator，不再儲存並重讀 dw_buffer。
    // ------------------------------------------------------------------
    logic pw_read_valid;
    logic [GROUP_W-1:0] pw_read_group;
    logic [CH_W-1:0] pw_read_channel;
    logic signed [DATA_WIDTH-1:0] pw_read_activation;

    logic pw_prod_valid;
    logic [GROUP_W-1:0] pw_prod_group;
    logic [CH_W-1:0] pw_prod_channel;
    logic signed [(LANES*BIAS_WIDTH)-1:0] pw_prod_bias;
    logic signed [PROD_WIDTH-1:0] pw_products [0:LANES-1];
    logic signed [ACC_WIDTH-1:0]
        pw_accumulators [0:OUT_GROUPS-1][0:LANES-1];
    logic signed [DATA_WIDTH-1:0]
        pw_results [0:OUT_GROUPS-1][0:LANES-1];

    integer lane;
    logic signed [ACC_WIDTH-1:0] dw_final_sum;
    logic signed [DATA_WIDTH-1:0] dw_quantized_value;
    logic signed [ACC_WIDTH-1:0] pw_final_sum;

    // ------------------------------------------------------------------
    // Pipeline registers 與控制流程
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            done             <= 1'b0;
            output_valid     <= 1'b0;
            output_last      <= 1'b0;
            output_data      <= '0;
            output_addr      <= '0;
            output_h         <= '0;
            output_w         <= '0;
            output_channel   <= '0;

            out_h_count      <= '0;
            out_w_count      <= '0;
            dw_issue_channel <= '0;
            dw_issue_kw      <= '0;
            output_group     <= '0;
            output_lane      <= '0;

            dw_read_valid    <= 1'b0;
            dw_prod_valid    <= 1'b0;
            dw_pair_valid    <= 1'b0;
            dw_read_first    <= 1'b0;
            dw_read_last     <= 1'b0;
            dw_prod_first    <= 1'b0;
            dw_prod_last     <= 1'b0;
            dw_pair_first    <= 1'b0;
            dw_pair_last     <= 1'b0;
            dw_read_channel  <= '0;
            dw_prod_channel  <= '0;
            dw_pair_channel  <= '0;
            dw_prod_bias     <= '0;
            dw_pair_bias     <= '0;
            dw_product_kh0   <= '0;
            dw_product_kh1   <= '0;
            dw_pair_sum      <= '0;
            dw_accumulator   <= '0;

            pw_issue_active       <= 1'b0;
            pw_issue_group        <= '0;
            pw_pending_channel    <= '0;
            pw_pending_activation <= '0;
            pw_read_valid         <= 1'b0;
            pw_read_group         <= '0;
            pw_read_channel       <= '0;
            pw_read_activation    <= '0;
            pw_prod_valid         <= 1'b0;
            pw_prod_group         <= '0;
            pw_prod_channel       <= '0;
            pw_prod_bias          <= '0;
        end else begin
            done         <= 1'b0;
            output_valid <= 1'b0;
            output_last  <= 1'b0;

            // 每拍推進 DW valid、first/last 與 channel tag。
            dw_prod_valid   <= dw_read_valid;
            dw_prod_first   <= dw_read_first;
            dw_prod_last    <= dw_read_last;
            dw_prod_channel <= dw_read_channel;
            dw_pair_valid   <= dw_prod_valid;
            dw_pair_first   <= dw_prod_first;
            dw_pair_last    <= dw_prod_last;
            dw_pair_channel <= dw_prod_channel;
            pw_prod_valid   <= pw_read_valid;
            pw_prod_group   <= pw_read_group;
            pw_prod_channel <= pw_read_channel;

            dw_read_valid <= 1'b0;
            pw_read_valid <= 1'b0;

            if (dw_read_valid) begin
                dw_product_kh0 <= $signed(input_data_kh0)
                                * $signed(dw_w_kh0);
                dw_product_kh1 <= $signed(input_data_kh1)
                                * $signed(dw_w_kh1);
                dw_prod_bias <= dw_bias_data;
            end

            if (dw_prod_valid) begin
                dw_pair_sum  <= $signed(dw_product_kh0)
                              + $signed(dw_product_kh1);
                dw_pair_bias <= dw_prod_bias;
            end

            if (dw_pair_valid) begin
                if (dw_pair_first)
                    dw_accumulator <= {{(ACC_WIDTH-PAIR_WIDTH){
                        dw_pair_sum[PAIR_WIDTH-1]}}, dw_pair_sum};
                else
                    dw_accumulator <= dw_accumulator
                                    + {{(ACC_WIDTH-PAIR_WIDTH){
                                        dw_pair_sum[PAIR_WIDTH-1]}},
                                       dw_pair_sum};

                if (dw_pair_last) begin
                    if (dw_pair_first)
                        dw_final_sum = {{(ACC_WIDTH-PAIR_WIDTH){
                            dw_pair_sum[PAIR_WIDTH-1]}}, dw_pair_sum};
                    else
                        dw_final_sum = dw_accumulator
                                     + {{(ACC_WIDTH-PAIR_WIDTH){
                                         dw_pair_sum[PAIR_WIDTH-1]}},
                                        dw_pair_sum};

                    dw_quantized_value = quantize_dw(
                        dw_final_sum, dw_pair_bias);

                    // K_W=5，兩個完成值間有五拍，足以排入四個 PW group。
                    pw_pending_channel    <= dw_pair_channel;
                    pw_pending_activation <= dw_quantized_value;
                    pw_issue_group        <= '0;
                    pw_issue_active       <= 1'b1;
                end
            end

            // 每個 DW activation 依序送入四組 Pointwise weights。
            if (pw_issue_active) begin
                pw_read_valid      <= 1'b1;
                pw_read_group      <= pw_issue_group;
                pw_read_channel    <= pw_pending_channel;
                pw_read_activation <= pw_pending_activation;

                if (pw_issue_group == OUT_GROUPS-1) begin
                    pw_issue_group  <= '0;
                    pw_issue_active <= 1'b0;
                end else begin
                    pw_issue_group <= pw_issue_group + 1'b1;
                end
            end

            if (pw_read_valid) begin
                for (lane = 0; lane < LANES; lane = lane + 1)
                    pw_products[lane] <= $signed(pw_read_activation)
                                       * $signed(pw_weight_data[
                                           lane*WEIGHT_WIDTH
                                           +: WEIGHT_WIDTH]);
                pw_prod_bias <= pw_bias_data;
            end

            if (pw_prod_valid) begin
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    if (pw_prod_channel == 0)
                        pw_final_sum = {{(ACC_WIDTH-PROD_WIDTH){
                            pw_products[lane][PROD_WIDTH-1]}},
                            pw_products[lane]};
                    else
                        pw_final_sum =
                            pw_accumulators[pw_prod_group][lane]
                            + {{(ACC_WIDTH-PROD_WIDTH){
                                pw_products[lane][PROD_WIDTH-1]}},
                               pw_products[lane]};

                    pw_accumulators[pw_prod_group][lane] <= pw_final_sum;

                    if (pw_prod_channel == IN_CH-1)
                        pw_results[pw_prod_group][lane] <= quantize_pw(
                            pw_final_sum,
                            pw_prod_bias[lane*BIAS_WIDTH +: BIAS_WIDTH]);
                end

                // 最後一組 Pointwise 結果完成後，才開始依序輸出 20 channels。
                if ((pw_prod_channel == IN_CH-1)
                 && (pw_prod_group == OUT_GROUPS-1)) begin
                    output_group <= '0;
                    output_lane  <= '0;
                    state        <= S_OUTPUT;
                end
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        out_h_count      <= '0;
                        out_w_count      <= '0;
                        dw_issue_channel <= '0;
                        dw_issue_kw      <= '0;
                        pw_issue_active  <= 1'b0;
                        state            <= S_DW_PREP;
                    end
                end

                // 預先送出第一個 RAM/ROM 位址，配合同步記憶體一拍 latency。
                S_DW_PREP: begin
                    dw_issue_channel <= '0;
                    dw_issue_kw      <= '0;
                    state            <= S_DW_STREAM;
                end

                // 所有 channel 與 kw 完全連續，不再逐 channel 進入 drain。
                S_DW_STREAM: begin
                    dw_read_valid   <= 1'b1;
                    dw_read_first   <= (dw_issue_kw == 0);
                    dw_read_last    <= (dw_issue_kw == K_W-1);
                    dw_read_channel <= dw_issue_channel;

                    if (dw_issue_kw == K_W-1) begin
                        dw_issue_kw <= '0;
                        if (dw_issue_channel == IN_CH-1) begin
                            state <= S_DRAIN;
                        end else begin
                            dw_issue_channel <= dw_issue_channel + 1'b1;
                        end
                    end else begin
                        dw_issue_kw <= dw_issue_kw + 1'b1;
                    end
                end

                // DW 與 fused PW 的尾端 valid 仍會自行前進。
                S_DRAIN: begin
                end

                S_OUTPUT: begin
                    output_valid   <= 1'b1;
                    output_data    <= pw_results[output_group][output_lane];
                    output_h       <= out_h_count;
                    output_w       <= out_w_count;
                    output_channel <= output_channel_full[
                        $clog2(OUT_CH)-1:0];
                    output_addr    <= out_h_count
                                    + OUT_H * (out_w_count
                                    + OUT_W * output_channel_full);
                    output_last    <= (out_w_count == OUT_W-1)
                                   && (out_h_count == OUT_H-1)
                                   && (output_group == OUT_GROUPS-1)
                                   && (output_lane == LANES-1);

                    if (output_lane != LANES-1) begin
                        output_lane <= output_lane + 1'b1;
                    end else begin
                        output_lane <= '0;
                        if (output_group != OUT_GROUPS-1) begin
                            output_group <= output_group + 1'b1;
                        end else if ((out_h_count == OUT_H-1)
                                  && (out_w_count == OUT_W-1)) begin
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            output_group <= '0;
                            if (out_h_count == OUT_H-1) begin
                                out_h_count <= '0;
                                out_w_count <= out_w_count + 1'b1;
                            end else begin
                                out_h_count <= out_h_count + 1'b1;
                            end
                            state <= S_DW_PREP;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (K_H != 2)
            $error("ds_conv2_engine currently requires K_H=2");
        if ((OUT_CH % LANES) != 0)
            $error("OUT_CH must be divisible by LANES");
        if (OUT_GROUPS >= K_W)
            $error("Fused PW scheduler requires OUT_GROUPS < K_W");
    end
`endif

endmodule
