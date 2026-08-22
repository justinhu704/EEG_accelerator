// Sequential GRU inference engine.
//
// Input tensor : 15 features x 18 time steps, stored as Pool2 [18,1,15]
//                input_addr = time + 18 * feature
// Hidden state : 9 units, Q15, initialized to zero for every start pulse
// Output tensor: 9 hidden units x 18 time steps
//                output_addr = hidden + 9 * time
// Reset mode   : before-multiplication, Uh * (r .* h_previous)
module gru_engine #(
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
        S_GATE_INPUT,
        S_GATE_RECURRENT,
        S_GATE_LUT,
        S_GATE_WAIT,
        S_CAND_CLEAR,
        S_CAND_INPUT,
        S_CAND_GATE_H,
        S_CAND_RECURRENT,
        S_CAND_LUT,
        S_CAND_WAIT,
        S_COMMIT,
        S_DONE
    } state_t;
    state_t state;

    integer time_index;
    integer issue_index;
    integer receive_index;
    integer feature_index;
    integer recurrent_index;
    integer gate_neuron;
    integer candidate_neuron;
    integer hidden_copy_index;
    logic load_data_valid;

    logic signed [63:0] reset_accumulator;
    logic signed [63:0] update_accumulator;
    logic signed [63:0] candidate_accumulator;

    logic signed [31:0] wr_product;
    logic signed [31:0] wz_product;
    logic signed [31:0] ur_product;
    logic signed [31:0] uz_product;
    logic signed [31:0] wh_product;
    logic signed [31:0] gated_hidden_product;
    logic signed [31:0] gated_hidden_register;
    logic signed [47:0] uh_product;

    logic sigmoid_in_valid;
    logic signed [15:0] reset_lut_input;
    logic signed [15:0] update_lut_input;
    logic reset_lut_valid;
    logic update_lut_valid;
    logic signed [15:0] reset_lut_output;
    logic signed [15:0] update_lut_output;

    logic tanh_in_valid;
    logic signed [15:0] candidate_lut_input;
    logic tanh_out_valid;
    logic signed [15:0] tanh_output;

    logic signed [63:0] hidden_mix;
    logic signed [63:0] hidden_scaled;
    logic signed [16:0] one_minus_update;
    logic signed [15:0] hidden_result;

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

    // Weight files use MATLAB column-major order: neuron + HIDDEN_SIZE*input.
    always_comb begin
        wr_product = $signed(wr_mem[gate_neuron
                                  + HIDDEN_SIZE * feature_index])
                   * $signed(x_buffer[feature_index]);
        wz_product = $signed(wz_mem[gate_neuron
                                  + HIDDEN_SIZE * feature_index])
                   * $signed(x_buffer[feature_index]);
        ur_product = $signed(ur_mem[gate_neuron
                                  + HIDDEN_SIZE * recurrent_index])
                   * $signed(hidden_state[recurrent_index]);
        uz_product = $signed(uz_mem[gate_neuron
                                  + HIDDEN_SIZE * recurrent_index])
                   * $signed(hidden_state[recurrent_index]);

        wh_product = $signed(wh_mem[candidate_neuron
                                  + HIDDEN_SIZE * feature_index])
                   * $signed(x_buffer[feature_index]);
        gated_hidden_product = $signed(reset_gate[recurrent_index])
                             * $signed(hidden_state[recurrent_index]);
        uh_product = $signed(uh_mem[candidate_neuron
                                  + HIDDEN_SIZE * recurrent_index])
                   * $signed(gated_hidden_register);
    end

    // Convert wide pre-activations to the signed Q9 expected by the LUTs.
    always_comb begin
        reset_lut_input = saturate16(
            reset_accumulator >>> (GATE_ACC_F - LUT_INPUT_F)
        );
        update_lut_input = saturate16(
            update_accumulator >>> (GATE_ACC_F - LUT_INPUT_F)
        );
        candidate_lut_input = saturate16(
            candidate_accumulator >>> (CAND_ACC_F - LUT_INPUT_F)
        );
        sigmoid_in_valid = (state == S_GATE_LUT);
        tanh_in_valid = (state == S_CAND_LUT);
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

    // h_new = (1-z)*candidate + z*h_previous, all operands in Q15.
    always_comb begin
        one_minus_update = 17'sd32768 - $signed({1'b0, update_gate[candidate_neuron]});
        hidden_mix = one_minus_update * $signed(tanh_output)
            + $signed({1'b0, update_gate[candidate_neuron]}) * $signed(hidden_state[candidate_neuron]);
        hidden_scaled = hidden_mix >>> 15;
        hidden_result = saturate16(hidden_scaled);
    end

    always_comb begin
        input_addr = time_index + TIME_STEPS * issue_index;
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            time_index <= 0;
            issue_index <= 0;
            receive_index <= 0;
            feature_index <= 0;
            recurrent_index <= 0;
            gate_neuron <= 0;
            candidate_neuron <= 0;
            hidden_copy_index <= 0;
            load_data_valid <= 1'b0;
            reset_accumulator <= '0;
            update_accumulator <= '0;
            candidate_accumulator <= '0;
            gated_hidden_register <= '0;
            output_valid <= 1'b0;
            output_addr <= '0;
            output_data <= '0;
            for (loop_index = 0; loop_index < HIDDEN_SIZE;
                 loop_index = loop_index + 1) begin
                hidden_state[loop_index] <= '0;
                hidden_next[loop_index] <= '0;
                reset_gate[loop_index] <= '0;
                update_gate[loop_index] <= '0;
            end
        end else begin
            output_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    load_data_valid <= 1'b0;
                    if (start) begin
                        time_index <= 0;
                        for (loop_index = 0; loop_index < HIDDEN_SIZE;
                             loop_index = loop_index + 1) begin
                            hidden_state[loop_index] <= '0;
                            hidden_next[loop_index] <= '0;
                        end
                        state <= S_PREP_X;
                    end
                end

                S_PREP_X: begin
                    issue_index <= 0;
                    receive_index <= 0;
                    load_data_valid <= 1'b0;
                    state <= S_LOAD_X;
                end

                // 將15個 feature load 到 register，一筆 mac 需要15個 feature
                S_LOAD_X: begin
                    load_data_valid <= 1'b1;
                    if (load_data_valid) begin
                        x_buffer[receive_index] <= input_data;
                        receive_index <= receive_index + 1;
                    end

                    // 讀完所有 input data(15 個)
                    if (issue_index == INPUT_SIZE-1)
                        state <= S_LOAD_DRAIN;
                    else
                        issue_index <= issue_index + 1;
                end

                S_LOAD_DRAIN: begin
                    if (load_data_valid)
                        x_buffer[receive_index] <= input_data;
                    load_data_valid <= 1'b0;
                    gate_neuron <= 0;
                    state <= S_GATE_CLEAR;
                end

                // =============================================================
                // accumulators 填入 bias
                S_GATE_CLEAR: begin
                    reset_accumulator <= $signed(br_mem[gate_neuron]) <<< 15;
                    update_accumulator <= $signed(bz_mem[gate_neuron]) <<< 15;
                    feature_index <= 0;
                    state <= S_GATE_INPUT;
                end

                // 與 Wr Wz 做 mac
                S_GATE_INPUT: begin
                    reset_accumulator <= reset_accumulator + ($signed(wr_product) <<< 2);
                    update_accumulator <= update_accumulator + ($signed(wz_product) <<< 3);
                    if (feature_index == INPUT_SIZE-1) begin
                        recurrent_index <= 0;
                        state <= S_GATE_RECURRENT;
                    end else begin
                        feature_index <= feature_index + 1;
                    end
                end

                // 與遞迴權重 Ur Uz 做 mac
                S_GATE_RECURRENT: begin
                    reset_accumulator <= reset_accumulator + $signed(ur_product);
                    update_accumulator <= update_accumulator + $signed(uz_product);
                    if (recurrent_index == HIDDEN_SIZE-1)
                        state <= S_GATE_LUT;
                    else
                        recurrent_index <= recurrent_index + 1;
                end

                S_GATE_LUT:
                    state <= S_GATE_WAIT;

                // 儲存sigmoid 計算結果，並迴圈九次
                S_GATE_WAIT: begin
                    if (reset_lut_valid && update_lut_valid) begin
                        reset_gate[gate_neuron] <= reset_lut_output;
                        update_gate[gate_neuron] <= update_lut_output;
                        if (gate_neuron == HIDDEN_SIZE-1) begin
                            candidate_neuron <= 0;
                            state <= S_CAND_CLEAR;
                        end else begin
                            gate_neuron <= gate_neuron + 1;
                            state <= S_GATE_CLEAR;
                        end
                    end
                end

                // =============================================================
                // accumulators 填入 bias
                S_CAND_CLEAR: begin
                    candidate_accumulator <= $signed(bh_mem[candidate_neuron]) <<< 29;
                    feature_index <= 0;
                    state <= S_CAND_INPUT;
                end

                S_CAND_INPUT: begin
                    // x(F13)*Wh(F15)=F28, align to F44.
                    candidate_accumulator <= candidate_accumulator + ($signed(wh_product) <<< 16);
                    if (feature_index == INPUT_SIZE-1) begin
                        recurrent_index <= 0;
                        state <= S_CAND_GATE_H;
                    end else begin
                        feature_index <= feature_index + 1;
                    end
                end

                // Register r*h before multiplying by Uh.
                // 做 pipeline
                S_CAND_GATE_H: begin
                    gated_hidden_register <= gated_hidden_product;
                    state <= S_CAND_RECURRENT;
                end

                S_CAND_RECURRENT: begin
                    // Uh(F14) * r(F15) * h(F15) is already F44.
                    candidate_accumulator <= candidate_accumulator + $signed(uh_product);
                    if (recurrent_index == HIDDEN_SIZE-1)
                        state <= S_CAND_LUT;
                    else begin
                        recurrent_index <= recurrent_index + 1;
                        state <= S_CAND_GATE_H;
                    end
                end

                // send to tanh LUT
                S_CAND_LUT:
                    state <= S_CAND_WAIT;

                S_CAND_WAIT: begin
                    if (tanh_out_valid) begin
                        hidden_next[candidate_neuron] <= hidden_result;
                        output_valid <= 1'b1;
                        output_addr <= candidate_neuron + HIDDEN_SIZE * time_index;
                        output_data <= hidden_result;

                        if (candidate_neuron == HIDDEN_SIZE-1) begin
                            hidden_copy_index <= 0;
                            state <= S_COMMIT;
                        end else begin
                            candidate_neuron <= candidate_neuron + 1;
                            state <= S_CAND_CLEAR;
                        end
                    end
                end

                S_COMMIT: begin
                    hidden_state[hidden_copy_index]
                        <= hidden_next[hidden_copy_index];
                    if (hidden_copy_index == HIDDEN_SIZE-1) begin
                        if (time_index == TIME_STEPS-1)
                            state <= S_DONE;
                        else begin
                            time_index <= time_index + 1;
                            state <= S_PREP_X;
                        end
                    end else begin
                        hidden_copy_index <= hidden_copy_index + 1;
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
