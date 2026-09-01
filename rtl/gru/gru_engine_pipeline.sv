// Pipelined GRU inference engine.
//
// Input tensor : 15 features x 18 time steps, stored as Pool2 [18,1,15]
//                input_addr = time + 18 * feature
// Hidden state : 9 units, Q15, initialized to zero for every start pulse
// Output tensor: 9 hidden units x 18 time steps
//                output_addr = hidden + 9 * time
// Reset mode   : before-multiplication, Uh * (r .* h_previous)
//
// Compared with gru_engine.sv, this version keeps the verified gate math but
// pipelines both the gate and candidate paths:
//   - Split Wr/Wz/Ur/Uz address, multiply, and accumulation into stages.
//   1. Calculate r .* h_previous once per time step.
//   2. Stream Wh and Uh in their existing MATLAB column-major order.
//   3. Interleave the stream across nine independent candidate accumulators.
//   4. Register ROM outputs, multiplier outputs, and accumulator inputs.
//   5. Pipeline the final hidden-state mix into multiply/add/output stages.
//
// The original gru_engine.sv is intentionally left unchanged.
module gru_engine_pipeline #(
    parameter int INPUT_SIZE  = 15,
    parameter int HIDDEN_SIZE = 9,
    parameter int TIME_STEPS  = 18,
    parameter WR_FILE = "mem/weights/gru_Wr.mem",
    parameter WZ_FILE = "mem/weights/gru_Wz.mem",
    parameter WH_FILE = "mem/weights/gru_Wh.mem",
    parameter UR_FILE = "mem/weights/gru_Ur.mem",
    parameter UZ_FILE = "mem/weights/gru_Uz.mem",
    parameter UH_FILE = "mem/weights/gru_Uh.mem",
    parameter BR_FILE = "mem/weights/gru_br.mem",
    parameter BZ_FILE = "mem/weights/gru_bz.mem",
    parameter BH_FILE = "mem/weights/gru_bh.mem",
    parameter SIGMOID_FILE = "mem/lut/sigmoid_half_lut_q15.mem",
    parameter TANH_FILE    = "mem/lut/tanh_half_lut_q15.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // External synchronous activation RAM interface.
    output logic [31:0]        input_addr,
    input  logic signed [15:0] input_data,

    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int INPUT_WEIGHT_DEPTH = HIDDEN_SIZE * INPUT_SIZE;
    localparam int RECURRENT_WEIGHT_DEPTH = HIDDEN_SIZE * HIDDEN_SIZE;

    localparam int INPUT_INDEX_W = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE);
    localparam int INPUT_COUNT_W = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE + 1);
    localparam int HIDDEN_INDEX_W = (HIDDEN_SIZE <= 1) ? 1 : $clog2(HIDDEN_SIZE);
    localparam int TIME_INDEX_W = (TIME_STEPS <= 1) ? 1 : $clog2(TIME_STEPS);
    localparam int INPUT_ADDR_W = (INPUT_WEIGHT_DEPTH <= 1)
                                ? 1 : $clog2(INPUT_WEIGHT_DEPTH);
    localparam int RECURRENT_ADDR_W = (RECURRENT_WEIGHT_DEPTH <= 1)
                                    ? 1 : $clog2(RECURRENT_WEIGHT_DEPTH);

    // Q formats from the verified MATLAB export:
    // x=F13, h/r/z/candidate=F15, Wr=F15, Wz=F14, Wh=F15,
    // Ur=F15, Uz=F15, Uh=F14, biases=F15, LUT input=F9.
    // These values come from EEG_CNN_GRU_quantized.mat; the older
    // fixed_point_pkg table does not describe the current exported files.
    localparam int GATE_ACC_F = 30;
    localparam int LUT_INPUT_F = 9;
    localparam int CAND_ACC_F = 44;

    logic signed [15:0] wr_mem [0:INPUT_WEIGHT_DEPTH-1];
    logic signed [15:0] wz_mem [0:INPUT_WEIGHT_DEPTH-1];
    logic signed [15:0] wh_mem [0:INPUT_WEIGHT_DEPTH-1];
    logic signed [15:0] ur_mem [0:RECURRENT_WEIGHT_DEPTH-1];
    logic signed [15:0] uz_mem [0:RECURRENT_WEIGHT_DEPTH-1];
    logic signed [15:0] uh_mem [0:RECURRENT_WEIGHT_DEPTH-1];
    logic signed [15:0] br_mem [0:HIDDEN_SIZE-1];
    logic signed [15:0] bz_mem [0:HIDDEN_SIZE-1];
    logic signed [15:0] bh_mem [0:HIDDEN_SIZE-1];

    logic signed [15:0] x_buffer [0:INPUT_SIZE-1];
    logic signed [15:0] hidden_state [0:HIDDEN_SIZE-1];
    logic signed [15:0] hidden_next [0:HIDDEN_SIZE-1];
    logic signed [15:0] reset_gate [0:HIDDEN_SIZE-1];
    logic signed [15:0] update_gate [0:HIDDEN_SIZE-1];
    logic signed [31:0] gated_hidden [0:HIDDEN_SIZE-1];
    logic signed [63:0] candidate_accumulator [0:HIDDEN_SIZE-1];

    initial begin
        $readmemh(WR_FILE, wr_mem, 0, INPUT_WEIGHT_DEPTH-1);
        $readmemh(WZ_FILE, wz_mem, 0, INPUT_WEIGHT_DEPTH-1);
        $readmemh(WH_FILE, wh_mem, 0, INPUT_WEIGHT_DEPTH-1);
        $readmemh(UR_FILE, ur_mem, 0, RECURRENT_WEIGHT_DEPTH-1);
        $readmemh(UZ_FILE, uz_mem, 0, RECURRENT_WEIGHT_DEPTH-1);
        $readmemh(UH_FILE, uh_mem, 0, RECURRENT_WEIGHT_DEPTH-1);
        $readmemh(BR_FILE, br_mem, 0, HIDDEN_SIZE-1);
        $readmemh(BZ_FILE, bz_mem, 0, HIDDEN_SIZE-1);
        $readmemh(BH_FILE, bh_mem, 0, HIDDEN_SIZE-1);
    end

    typedef enum logic [4:0] {
        S_IDLE,
        S_PREP_X,
        S_LOAD_X,
        S_LOAD_DRAIN,
        S_GATE_CLEAR,
        S_GATE_BIAS,
        S_GATE_INPUT,
        S_GATE_INPUT_DRAIN,
        S_GATE_RECURRENT,
        S_GATE_RECURRENT_DRAIN,
        S_GATE_LUT,
        S_GATE_WAIT,
        S_GATED_H_PREP,
        S_GATED_H_CALC,
        S_CAND_INIT,
        S_WH_ISSUE,
        S_WH_DRAIN,
        S_UH_ISSUE,
        S_UH_DRAIN,
        S_ACT_ISSUE,
        S_ACT_DRAIN,
        S_COMMIT,
        S_DONE
    } state_t;
    state_t state;

    logic [TIME_INDEX_W-1:0] time_index;
    logic [INPUT_INDEX_W-1:0] issue_index;
    logic [INPUT_COUNT_W-1:0] receive_index;
    logic [INPUT_INDEX_W-1:0] feature_index;
    logic [HIDDEN_INDEX_W-1:0] recurrent_index;
    logic [HIDDEN_INDEX_W-1:0] gate_neuron;
    logic [HIDDEN_INDEX_W-1:0] gated_hidden_index;
    logic [HIDDEN_INDEX_W-1:0] hidden_copy_index;
    logic load_data_valid;

    logic signed [63:0] reset_accumulator;
    logic signed [63:0] update_accumulator;

    // Gate bias/address preparation. Address counters replace the long
    // gate_neuron + HIDDEN_SIZE * index combinational path.
    logic [INPUT_ADDR_W-1:0] gate_input_addr;
    logic [RECURRENT_ADDR_W-1:0] gate_recurrent_addr;
    logic signed [15:0] reset_bias_s1;
    logic signed [15:0] update_bias_s1;

    // Wr/Wz pipeline: weight/data register -> multiply -> accumulator.
    logic signed [15:0] wr_weight_s1;
    logic signed [15:0] wz_weight_s1;
    logic signed [15:0] gate_input_data_s1;
    logic gate_input_valid_s1;
    logic gate_input_last_s1;
    logic signed [31:0] wr_product_s2;
    logic signed [31:0] wz_product_s2;
    logic gate_input_valid_s2;
    logic gate_input_last_s2;

    // Ur/Uz pipeline: weight/data register -> multiply -> accumulator.
    logic signed [15:0] ur_weight_s1;
    logic signed [15:0] uz_weight_s1;
    logic signed [15:0] gate_recurrent_data_s1;
    logic gate_recurrent_valid_s1;
    logic gate_recurrent_last_s1;
    logic signed [31:0] ur_product_s2;
    logic signed [31:0] uz_product_s2;
    logic gate_recurrent_valid_s2;
    logic gate_recurrent_last_s2;

    // Wh pipeline: synchronous ROM read -> 16x16 multiply -> banked add.
    logic [INPUT_ADDR_W-1:0] wh_addr;
    logic [INPUT_INDEX_W-1:0] wh_feature;
    logic [HIDDEN_INDEX_W-1:0] wh_neuron;
    logic signed [15:0] wh_weight_s1;
    logic signed [15:0] wh_data_s1;
    logic [HIDDEN_INDEX_W-1:0] wh_neuron_s1;
    logic wh_valid_s1;
    logic wh_last_s1;
    logic signed [31:0] wh_product_s2;
    logic [HIDDEN_INDEX_W-1:0] wh_neuron_s2;
    logic wh_valid_s2;
    logic wh_last_s2;

    // Uh pipeline: synchronous ROM read -> 16x32 multiply -> banked add.
    logic [RECURRENT_ADDR_W-1:0] uh_addr;
    logic [HIDDEN_INDEX_W-1:0] uh_recurrent;
    logic [HIDDEN_INDEX_W-1:0] uh_neuron;
    logic signed [15:0] uh_weight_s1;
    logic signed [31:0] uh_data_s1;
    logic [HIDDEN_INDEX_W-1:0] uh_neuron_s1;
    logic uh_valid_s1;
    logic uh_last_s1;
    logic signed [47:0] uh_product_s2;
    logic [HIDDEN_INDEX_W-1:0] uh_neuron_s2;
    logic uh_valid_s2;
    logic uh_last_s2;

    logic sigmoid_in_valid;
    logic signed [15:0] reset_lut_input;
    logic signed [15:0] update_lut_input;
    logic reset_lut_valid;
    logic update_lut_valid;
    logic signed [15:0] reset_lut_output;
    logic signed [15:0] update_lut_output;

    logic [HIDDEN_INDEX_W-1:0] activation_neuron;
    logic activation_lut_valid;
    logic signed [15:0] activation_lut_input;
    logic [HIDDEN_INDEX_W-1:0] activation_lut_neuron;
    logic tanh_in_valid;
    logic signed [15:0] candidate_lut_input;
    logic tanh_out_valid;
    logic signed [15:0] tanh_output;
    logic [HIDDEN_INDEX_W-1:0] tanh_neuron_d1;

    // Hidden mix pipeline.
    // h_new = (1-z)*candidate + z*h_previous, all operands in Q15.
    logic hm_valid_s0;
    logic signed [15:0] hm_candidate_s0;
    logic signed [16:0] hm_update_s0;
    logic signed [16:0] hm_one_minus_s0;
    logic signed [15:0] hm_hidden_s0;
    logic [HIDDEN_INDEX_W-1:0] hm_neuron_s0;

    logic hm_valid_s1;
    logic signed [32:0] hm_candidate_product_s1;
    logic signed [32:0] hm_previous_product_s1;
    logic [HIDDEN_INDEX_W-1:0] hm_neuron_s1;

    logic hm_valid_s2;
    logic signed [63:0] hm_sum_s2;
    logic [HIDDEN_INDEX_W-1:0] hm_neuron_s2;

    integer loop_index;

    function automatic logic signed [15:0] saturate16(
        input logic signed [63:0] value
    );
        begin
            if (value > 64'sd32767)
                saturate16 = 16'sh7fff;
            else if (value < -64'sd32768)
                saturate16 = 16'sh8000;
            else
                saturate16 = value[15:0];
        end
    endfunction

    // 格式截斷與轉換邏輯：將 64-bit 累加器縮小為 F9 給 LUT 讀取
    always_comb begin
        reset_lut_input = saturate16(
            reset_accumulator >>> (GATE_ACC_F - LUT_INPUT_F)
        );
        update_lut_input = saturate16(
            update_accumulator >>> (GATE_ACC_F - LUT_INPUT_F)
        );
        candidate_lut_input = activation_lut_input;
        sigmoid_in_valid = (state == S_GATE_LUT);
        tanh_in_valid = activation_lut_valid;
    end

    // 對應 sigmoid 數值
    sigmoid_lut #(.MEM_FILE(SIGMOID_FILE)) u_reset_sigmoid (
        .clk(clk), .rst_n(rst_n),
        .in_valid(sigmoid_in_valid), .in_data(reset_lut_input),
        .out_valid(reset_lut_valid), .out_data(reset_lut_output)
    );

    sigmoid_lut #(.MEM_FILE(SIGMOID_FILE)) u_update_sigmoid (
        .clk(clk), .rst_n(rst_n),
        .in_valid(sigmoid_in_valid), .in_data(update_lut_input),
        .out_valid(update_lut_valid), .out_data(update_lut_output)
    );

    tanh_lut #(.MEM_FILE(TANH_FILE)) u_candidate_tanh (
        .clk(clk), .rst_n(rst_n),
        .in_valid(tanh_in_valid), .in_data(candidate_lut_input),
        .out_valid(tanh_out_valid), .out_data(tanh_output)
    );

    always_comb begin
        input_addr = $unsigned(time_index)
                   + TIME_STEPS * $unsigned(issue_index);
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            time_index <= '0;
            issue_index <= '0;
            receive_index <= '0;
            feature_index <= '0;
            recurrent_index <= '0;
            gate_neuron <= '0;
            gated_hidden_index <= '0;
            hidden_copy_index <= '0;
            load_data_valid <= 1'b0;
            reset_accumulator <= '0;
            update_accumulator <= '0;
            gate_input_addr <= '0;
            gate_recurrent_addr <= '0;
            reset_bias_s1 <= '0;
            update_bias_s1 <= '0;

            wr_weight_s1 <= '0;
            wz_weight_s1 <= '0;
            gate_input_data_s1 <= '0;
            gate_input_valid_s1 <= 1'b0;
            gate_input_last_s1 <= 1'b0;
            wr_product_s2 <= '0;
            wz_product_s2 <= '0;
            gate_input_valid_s2 <= 1'b0;
            gate_input_last_s2 <= 1'b0;

            ur_weight_s1 <= '0;
            uz_weight_s1 <= '0;
            gate_recurrent_data_s1 <= '0;
            gate_recurrent_valid_s1 <= 1'b0;
            gate_recurrent_last_s1 <= 1'b0;
            ur_product_s2 <= '0;
            uz_product_s2 <= '0;
            gate_recurrent_valid_s2 <= 1'b0;
            gate_recurrent_last_s2 <= 1'b0;

            wh_addr <= '0;
            wh_feature <= '0;
            wh_neuron <= '0;
            wh_weight_s1 <= '0;
            wh_data_s1 <= '0;
            wh_neuron_s1 <= '0;
            wh_valid_s1 <= 1'b0;
            wh_last_s1 <= 1'b0;
            wh_product_s2 <= '0;
            wh_neuron_s2 <= '0;
            wh_valid_s2 <= 1'b0;
            wh_last_s2 <= 1'b0;

            uh_addr <= '0;
            uh_recurrent <= '0;
            uh_neuron <= '0;
            uh_weight_s1 <= '0;
            uh_data_s1 <= '0;
            uh_neuron_s1 <= '0;
            uh_valid_s1 <= 1'b0;
            uh_last_s1 <= 1'b0;
            uh_product_s2 <= '0;
            uh_neuron_s2 <= '0;
            uh_valid_s2 <= 1'b0;
            uh_last_s2 <= 1'b0;

            activation_neuron <= '0;
            activation_lut_valid <= 1'b0;
            activation_lut_input <= '0;
            activation_lut_neuron <= '0;
            tanh_neuron_d1 <= '0;
            hm_valid_s0 <= 1'b0;
            hm_candidate_s0 <= '0;
            hm_update_s0 <= '0;
            hm_one_minus_s0 <= '0;
            hm_hidden_s0 <= '0;
            hm_neuron_s0 <= '0;
            hm_valid_s1 <= 1'b0;
            hm_candidate_product_s1 <= '0;
            hm_previous_product_s1 <= '0;
            hm_neuron_s1 <= '0;
            hm_valid_s2 <= 1'b0;
            hm_sum_s2 <= '0;
            hm_neuron_s2 <= '0;

            output_valid <= 1'b0;
            activation_lut_valid <= 1'b0;

            output_addr <= '0;
            output_data <= '0;

            for (loop_index = 0; loop_index < HIDDEN_SIZE;
                 loop_index = loop_index + 1) begin
                hidden_state[loop_index] <= '0;
                hidden_next[loop_index] <= '0;
                reset_gate[loop_index] <= '0;
                update_gate[loop_index] <= '0;
                gated_hidden[loop_index] <= '0;
                candidate_accumulator[loop_index] <= '0;
            end
        // IDLE 等待 UART 時保持所有 GRU pipeline 暫存器。
        end else if ((state != S_IDLE) || start) begin
            output_valid <= 1'b0;
            activation_lut_valid <= 1'b0;

            // -------------------------------------------------------------
            // Reset/Update Gate Pipeline
            // Stage 1 權重與資料暫存 -> Stage 2 乘法 -> Stage 3 累加
            // -------------------------------------------------------------
            gate_input_valid_s1 <= 1'b0;
            gate_input_valid_s2 <= gate_input_valid_s1;
            gate_input_last_s2 <= gate_input_last_s1;
            if (gate_input_valid_s1) begin
                wr_product_s2 <= $signed(wr_weight_s1)
                               * $signed(gate_input_data_s1);
                wz_product_s2 <= $signed(wz_weight_s1)
                               * $signed(gate_input_data_s1);
            end

            gate_recurrent_valid_s1 <= 1'b0;
            gate_recurrent_valid_s2 <= gate_recurrent_valid_s1;
            gate_recurrent_last_s2 <= gate_recurrent_last_s1;
            if (gate_recurrent_valid_s1) begin
                ur_product_s2 <= $signed(ur_weight_s1)
                               * $signed(gate_recurrent_data_s1);
                uz_product_s2 <= $signed(uz_weight_s1)
                               * $signed(gate_recurrent_data_s1);
            end

            // 乘法結果已經過暫存，這一級只保留64-bit累加器加法。
            if (gate_input_valid_s2) begin
                reset_accumulator <= reset_accumulator
                                   + ($signed(wr_product_s2) <<< 2);
                update_accumulator <= update_accumulator
                                    + ($signed(wz_product_s2) <<< 3);
            end else if (gate_recurrent_valid_s2) begin
                reset_accumulator <= reset_accumulator
                                   + $signed(ur_product_s2);
                update_accumulator <= update_accumulator
                                    + $signed(uz_product_s2);
            end

            // -------------------------------------------------------------
            // Wh, Uh Pipeline: Stage 1 -> Stage 2 -> 寫入累加器
            // -------------------------------------------------------------
            wh_valid_s1 <= 1'b0;
            wh_valid_s2 <= wh_valid_s1;
            wh_last_s2 <= wh_last_s1;
            // Wh 乘積 (Stage 1)
            if (wh_valid_s1) begin
                wh_product_s2 <= $signed(wh_weight_s1)
                               * $signed(wh_data_s1);
                wh_neuron_s2 <= wh_neuron_s1;
            end

            uh_valid_s1 <= 1'b0;
            uh_valid_s2 <= uh_valid_s1;
            uh_last_s2 <= uh_last_s1;
            // Uh 乘積 (Stage 1)
            if (uh_valid_s1) begin
                uh_product_s2 <= $signed(uh_weight_s1)
                               * $signed(uh_data_s1);
                uh_neuron_s2 <= uh_neuron_s1;
            end

            // 當 Stage 2 有效時，自動歸類加進專屬的神經元累加器中
            if (wh_valid_s2) begin
                candidate_accumulator[wh_neuron_s2]
                    <= candidate_accumulator[wh_neuron_s2]
                     + ($signed({{32{wh_product_s2[31]}}, wh_product_s2})
                        <<< 16);
            end else if (uh_valid_s2) begin
                candidate_accumulator[uh_neuron_s2]
                    <= candidate_accumulator[uh_neuron_s2]
                     + $signed({{16{uh_product_s2[47]}}, uh_product_s2});
            end

            // 記錄 tanh 計算的目標神經元
            if (tanh_in_valid)
                tanh_neuron_d1 <= activation_lut_neuron;

            // -------------------------------------------------------------
            // Hidden Mix 三級管線： Stage 0 -> Stage 1 -> Stage 2 -> 輸出
            // -------------------------------------------------------------
            // Stage 0: 鎖存 Tanh 結果與運算元
            hm_valid_s0 <= tanh_out_valid;
            if (tanh_out_valid) begin
                hm_candidate_s0 <= tanh_output;
                hm_update_s0 <= $signed({1'b0,
                                         update_gate[tanh_neuron_d1]});
                hm_one_minus_s0 <= 17'sd32768
                                 - $signed({1'b0,
                                            update_gate[tanh_neuron_d1]});
                hm_hidden_s0 <= hidden_state[tanh_neuron_d1];
                hm_neuron_s0 <= tanh_neuron_d1;
            end

            // Stage 1: 兩顆 DSP 執行平行乘法
            hm_valid_s1 <= hm_valid_s0;
            if (hm_valid_s0) begin
                hm_candidate_product_s1 <= $signed(hm_one_minus_s0)
                                         * $signed(hm_candidate_s0);
                hm_previous_product_s1 <= $signed(hm_update_s0)
                                        * $signed(hm_hidden_s0);
                hm_neuron_s1 <= hm_neuron_s0;
            end

            // Stage 2: 相加
            hm_valid_s2 <= hm_valid_s1;
            if (hm_valid_s1) begin
                hm_sum_s2
                    <= $signed({{31{hm_candidate_product_s1[32]}},
                                 hm_candidate_product_s1})
                     + $signed({{31{hm_previous_product_s1[32]}},
                                 hm_previous_product_s1});
                hm_neuron_s2 <= hm_neuron_s1;
            end

            // 最終 Stage: 右移、飽和截斷，並立刻拉高 output_valid
            if (hm_valid_s2) begin
                hidden_next[hm_neuron_s2]
                    <= saturate16(hm_sum_s2 >>> 15);
                output_valid <= 1'b1;
                output_addr <= $unsigned(hm_neuron_s2)
                             + HIDDEN_SIZE * $unsigned(time_index);
                output_data <= saturate16(hm_sum_s2 >>> 15);
            end

            case (state)
                S_IDLE: begin
                    load_data_valid <= 1'b0;
                    if (start) begin
                        time_index <= '0;
                        gate_input_valid_s1 <= 1'b0;
                        gate_input_valid_s2 <= 1'b0;
                        gate_recurrent_valid_s1 <= 1'b0;
                        gate_recurrent_valid_s2 <= 1'b0;
                        wh_valid_s1 <= 1'b0;
                        wh_valid_s2 <= 1'b0;
                        uh_valid_s1 <= 1'b0;
                        uh_valid_s2 <= 1'b0;
                        hm_valid_s0 <= 1'b0;
                        hm_valid_s1 <= 1'b0;
                        hm_valid_s2 <= 1'b0;
                        activation_lut_valid <= 1'b0;
                        for (loop_index = 0; loop_index < HIDDEN_SIZE;
                             loop_index = loop_index + 1) begin
                            hidden_state[loop_index] <= '0;
                            hidden_next[loop_index] <= '0;
                        end
                        state <= S_PREP_X;
                    end
                end

                S_PREP_X: begin
                    issue_index <= '0;
                    receive_index <= '0;
                    load_data_valid <= 1'b0;
                    state <= S_LOAD_X;
                end

                // 將15個 feature load 到 register，一筆 mac 需要15個 feature
                S_LOAD_X: begin
                    load_data_valid <= 1'b1;
                    if (load_data_valid) begin
                        x_buffer[receive_index] <= input_data;
                        receive_index <= receive_index + 1'b1;
                    end

                    // 讀完所有 input data(15 個)
                    if (issue_index == INPUT_SIZE-1)
                        state <= S_LOAD_DRAIN;
                    else
                        issue_index <= issue_index + 1'b1;
                end

                S_LOAD_DRAIN: begin
                    if (load_data_valid)
                        x_buffer[receive_index] <= input_data;
                    load_data_valid <= 1'b0;
                    gate_neuron <= '0;
                    state <= S_GATE_CLEAR;
                end

                // =============================================================
                // accumulators 填入 bias
                S_GATE_CLEAR: begin
                    // 先暫存 bias 與兩組起始位址，切斷 gate_neuron 長路徑。
                    reset_bias_s1 <= br_mem[gate_neuron];
                    update_bias_s1 <= bz_mem[gate_neuron];
                    gate_input_addr <= gate_neuron;
                    gate_recurrent_addr <= gate_neuron;
                    gate_input_valid_s1 <= 1'b0;
                    gate_input_valid_s2 <= 1'b0;
                    gate_recurrent_valid_s1 <= 1'b0;
                    gate_recurrent_valid_s2 <= 1'b0;
                    state <= S_GATE_BIAS;
                end

                // 將已暫存的 bias 對齊成 accumulator 格式。
                S_GATE_BIAS: begin
                    reset_accumulator <= $signed(reset_bias_s1) <<< 15;
                    update_accumulator <= $signed(update_bias_s1) <<< 15;
                    feature_index <= '0;
                    recurrent_index <= '0;
                    state <= S_GATE_INPUT;
                end

                // 與 Wr Wz 做 mac
                // 每個 clock 發出一筆，乘法與累加由上方 pipeline 接手。
                S_GATE_INPUT: begin
                    wr_weight_s1 <= wr_mem[gate_input_addr];
                    wz_weight_s1 <= wz_mem[gate_input_addr];
                    gate_input_data_s1 <= x_buffer[feature_index];
                    gate_input_valid_s1 <= 1'b1;
                    gate_input_last_s1 <= (feature_index == INPUT_SIZE-1);

                    if (feature_index == INPUT_SIZE-1) begin
                        state <= S_GATE_INPUT_DRAIN;
                    end else begin
                        feature_index <= feature_index + 1'b1;
                        gate_input_addr <= gate_input_addr + HIDDEN_SIZE;
                    end
                end

                // 等待 Wr/Wz 最後一筆乘積寫入 accumulator。
                S_GATE_INPUT_DRAIN: begin
                    if (gate_input_valid_s2 && gate_input_last_s2)
                        state <= S_GATE_RECURRENT;
                end

                // 與遞迴權重 Ur Uz 做 mac
                S_GATE_RECURRENT: begin
                    ur_weight_s1 <= ur_mem[gate_recurrent_addr];
                    uz_weight_s1 <= uz_mem[gate_recurrent_addr];
                    gate_recurrent_data_s1
                        <= hidden_state[recurrent_index];
                    gate_recurrent_valid_s1 <= 1'b1;
                    gate_recurrent_last_s1
                        <= (recurrent_index == HIDDEN_SIZE-1);

                    if (recurrent_index == HIDDEN_SIZE-1) begin
                        state <= S_GATE_RECURRENT_DRAIN;
                    end else begin
                        recurrent_index <= recurrent_index + 1'b1;
                        gate_recurrent_addr
                            <= gate_recurrent_addr + HIDDEN_SIZE;
                    end
                end

                // 等待 Ur/Uz 最後一筆乘積寫入 accumulator。
                S_GATE_RECURRENT_DRAIN: begin
                    if (gate_recurrent_valid_s2 && gate_recurrent_last_s2)
                        state <= S_GATE_LUT;
                end

                S_GATE_LUT:
                    state <= S_GATE_WAIT;

                // 儲存sigmoid 計算結果，並迴圈九次
                S_GATE_WAIT: begin
                    if (reset_lut_valid && update_lut_valid) begin
                        reset_gate[gate_neuron] <= reset_lut_output;
                        update_gate[gate_neuron] <= update_lut_output;
                        if (gate_neuron == HIDDEN_SIZE-1) begin
                            gated_hidden_index <= '0;
                            state <= S_GATED_H_PREP;
                        end else begin
                            gate_neuron <= gate_neuron + 1'b1;
                            state <= S_GATE_CLEAR;
                        end
                    end
                end

                // =============================================================
                // Compute r .* h once. All candidate neurons reuse this vector.
                S_GATED_H_PREP: begin
                    gated_hidden_index <= '0;
                    state <= S_GATED_H_CALC;
                end

                // reset_gate 乘上 hidden_state(ht-1)，將計算結果暫存
                S_GATED_H_CALC: begin
                    gated_hidden[gated_hidden_index]
                        <= $signed(reset_gate[gated_hidden_index])
                         * $signed(hidden_state[gated_hidden_index]);
                    // 9 個 hidden unit 做完
                    if (gated_hidden_index == HIDDEN_SIZE-1)
                        state <= S_CAND_INIT;
                    else
                        gated_hidden_index <= gated_hidden_index + 1'b1;
                end

                // 為9個candidate accumulators 填入bias
                S_CAND_INIT: begin
                    // 1 cycle 填入9個bias
                    for (loop_index = 0; loop_index < HIDDEN_SIZE;
                         loop_index = loop_index + 1) begin
                        candidate_accumulator[loop_index]
                            <= $signed({{48{bh_mem[loop_index][15]}},
                                        bh_mem[loop_index]}) <<< 29;
                    end
                    wh_addr <= '0;
                    wh_feature <= '0;
                    wh_neuron <= '0;
                    wh_valid_s1 <= 1'b0;
                    wh_valid_s2 <= 1'b0;
                    state <= S_WH_ISSUE;
                end

                // Wh * x：Weight 為 column-major，所以一條 address 間隔9個 memory (135 cycles)
                S_WH_ISSUE: begin
                    wh_weight_s1 <= wh_mem[wh_addr];
                    wh_data_s1 <= x_buffer[wh_feature];
                    wh_neuron_s1 <= wh_neuron;
                    wh_last_s1 <= (wh_addr == INPUT_WEIGHT_DEPTH-1);
                    wh_valid_s1 <= 1'b1;

                    // 135 個Wh做完 mac
                    if (wh_addr == INPUT_WEIGHT_DEPTH-1) begin
                        state <= S_WH_DRAIN;
                    end else begin
                        wh_addr <= wh_addr + 1'b1;
                        if (wh_neuron == HIDDEN_SIZE-1) begin
                            wh_neuron <= '0;
                            wh_feature <= wh_feature + 1'b1;
                        end else begin
                            wh_neuron <= wh_neuron + 1'b1;
                        end
                    end
                end

                // 等待 Wh pipeline 清空 (2 cycle)
                S_WH_DRAIN: begin
                    if (wh_valid_s2 && wh_last_s2) begin
                        uh_addr <= '0;
                        uh_recurrent <= '0;
                        uh_neuron <= '0;
                        uh_valid_s1 <= 1'b0;
                        uh_valid_s2 <= 1'b0;
                        state <= S_UH_ISSUE;
                    end
                end

                // Uh * h: Weight 為 column-major，所以一條 address 間隔9個 memory (81 cycles)
                S_UH_ISSUE: begin
                    uh_weight_s1 <= uh_mem[uh_addr];
                    uh_data_s1 <= gated_hidden[uh_recurrent];
                    uh_neuron_s1 <= uh_neuron;
                    uh_last_s1 <= (uh_addr == RECURRENT_WEIGHT_DEPTH-1);
                    uh_valid_s1 <= 1'b1;

                    if (uh_addr == RECURRENT_WEIGHT_DEPTH-1) begin
                        state <= S_UH_DRAIN;
                    end else begin
                        uh_addr <= uh_addr + 1'b1;
                        if (uh_neuron == HIDDEN_SIZE-1) begin
                            uh_neuron <= '0;
                            uh_recurrent <= uh_recurrent + 1'b1;
                        end else begin
                            uh_neuron <= uh_neuron + 1'b1;
                        end
                    end
                end

                // 等待 Uh pipeline 清空 (2 cycle)
                S_UH_DRAIN: begin
                    if (uh_valid_s2 && uh_last_s2) begin
                        activation_neuron <= '0;
                        state <= S_ACT_ISSUE;
                    end
                end

                // 送入 Tanh 與排空輸出 (9 cycle)
                S_ACT_ISSUE: begin
                    activation_lut_input
                        <= saturate16(
                            candidate_accumulator[activation_neuron]
                            >>> (CAND_ACC_F - LUT_INPUT_F)
                        );
                    activation_lut_neuron <= activation_neuron;
                    activation_lut_valid <= 1'b1;
                    if (activation_neuron == HIDDEN_SIZE-1)
                        state <= S_ACT_DRAIN;
                    else
                        activation_neuron <= activation_neuron + 1'b1;
                end

                // 等待 Hidden_Mix 完成 (4 cycle)
                S_ACT_DRAIN: begin
                    if (hm_valid_s2 &&
                        (hm_neuron_s2 == HIDDEN_SIZE-1)) begin
                        hidden_copy_index <= '0;
                        state <= S_COMMIT;
                    end
                end

                // 更新 hidden state (9 cycle)
                S_COMMIT: begin
                    hidden_state[hidden_copy_index]
                        <= hidden_next[hidden_copy_index];
                    if (hidden_copy_index == HIDDEN_SIZE-1) begin
                        if (time_index == TIME_STEPS-1)
                            state <= S_DONE;
                        else begin
                            time_index <= time_index + 1'b1;
                            state <= S_PREP_X;
                        end
                    end else begin
                        hidden_copy_index <= hidden_copy_index + 1'b1;
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
