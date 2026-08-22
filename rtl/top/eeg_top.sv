// Complete inference path:
// CNN -> Pool2 -> GRU/flatten -> FC1 -> ReLU -> BN -> FC_out -> Argmax.
module eeg_top #(
    parameter INPUT_FILE   = "mem/golden/q_in_act.mem",
    parameter CONV1_W_FILE = "mem/weights/conv1_W.mem",
    parameter CONV1_B_FILE = "mem/weights/conv1_b.mem",
    parameter CONV1_PACKED_W_FILE = "mem/weights/conv1_W_x3.mem",
    parameter CONV1_PACKED_B_FILE = "mem/weights/conv1_b_x3.mem",
    parameter BN1_A_FILE   = "mem/weights/bn1_A.mem",
    parameter BN1_B_FILE   = "mem/weights/bn1_B.mem",
    parameter CONV2_W_FILE = "mem/weights/conv2_W.mem",
    parameter CONV2_B_FILE = "mem/weights/conv2_b.mem",
    parameter CONV2_PACKED_W_FILE = "mem/weights/conv2_W_x5.mem",
    parameter CONV2_KH2_PACKED_W_FILE = "mem/weights/conv2_W_x5_kh2.mem",
    parameter CONV2_PACKED_B_FILE = "mem/weights/conv2_b_x5.mem",
    parameter BN2_A_FILE   = "mem/weights/bn2_A.mem",
    parameter BN2_B_FILE   = "mem/weights/bn2_B.mem",
    parameter CONV3_W_FILE = "mem/weights/conv3_W.mem",
    parameter CONV3_B_FILE = "mem/weights/conv3_b.mem",
    parameter CONV3_PACKED_W_FILE = "mem/weights/conv3_W_x3.mem",
    parameter CONV3_PACKED_B_FILE = "mem/weights/conv3_b_x3.mem",
    parameter BN3_A_FILE   = "mem/weights/bn3_A.mem",
    parameter BN3_B_FILE   = "mem/weights/bn3_B.mem",
    parameter GRU_WR_FILE  = "mem/weights/gru_Wr.mem",
    parameter GRU_WZ_FILE  = "mem/weights/gru_Wz.mem",
    parameter GRU_WH_FILE  = "mem/weights/gru_Wh.mem",
    parameter GRU_UR_FILE  = "mem/weights/gru_Ur.mem",
    parameter GRU_UZ_FILE  = "mem/weights/gru_Uz.mem",
    parameter GRU_UH_FILE  = "mem/weights/gru_Uh.mem",
    parameter GRU_BR_FILE  = "mem/weights/gru_br.mem",
    parameter GRU_BZ_FILE  = "mem/weights/gru_bz.mem",
    parameter GRU_BH_FILE  = "mem/weights/gru_bh.mem",
    parameter SIGMOID_FILE = "mem/lut/sigmoid_half_lut_q15.mem",
    parameter TANH_FILE    = "mem/lut/tanh_half_lut_q15.mem",
    parameter FC1_W_FILE   = "mem/weights/fc_1_W.mem",
    parameter FC1_B_FILE   = "mem/weights/fc_1_b.mem",
    parameter FC_BN_A_FILE = "mem/weights/bn_2_A.mem",
    parameter FC_BN_B_FILE = "mem/weights/bn_2_B.mem",
    parameter FC_OUT_W_FILE = "mem/weights/fc_out_W.mem",
    parameter FC_OUT_B_FILE = "mem/weights/fc_out_b.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // Load one quantized EEG sample into input RAM before asserting start.
    // Valid addresses are 0..3359 (21x160, signed Q12 values).
    input  logic               input_write_en,
    input  logic [11:0]        input_write_addr,
    input  logic signed [15:0] input_write_data,
    output logic               input_ready,

    output logic [6:0]         class_index,
    output logic signed [15:0] winning_logit,

    // Verification/debug stream from the final 105-output FC layer.
    output logic               logit_valid,
    output logic [6:0]         logit_index,
    output logic signed [15:0] logit_data,

    // Synchronous debug readback of the stored 105 logits.
    input  logic [6:0]         logit_read_addr,
    output logic signed [15:0] logit_read_data
);
    logic cnn_start, cnn_done;
    logic cnn_input_ready;
    logic fc1_start, fc1_done;
    logic fc_out_start, fc_out_done;
    logic argmax_start, argmax_done;

    logic [31:0] cnn_ram_addr;
    logic signed [15:0] cnn_ram_data;

    logic [31:0] fc1_input_addr, fc1_output_addr;
    logic signed [15:0] fc1_output_data;
    logic fc1_output_valid;
    logic fc_relu_valid;
    logic signed [15:0] fc_relu_data;
    logic fc_bn_valid;
    logic signed [15:0] fc_bn_data;
    logic [31:0] fc_addr_d1, fc_addr_d2;
    logic fc_addr_valid_d1;
    logic fc1_bn_done;

    logic [6:0] fc_out_output_addr;
    logic signed [15:0] fc_out_output_data;
    logic fc_out_output_valid;

    eeg_controller u_controller (
        .clk(clk), .rst_n(rst_n), .start(start),
        .cnn_gru_done(cnn_done), .fc1_bn_done(fc1_bn_done),
        .fc_out_done(fc_out_done), .argmax_done(argmax_done),
        .cnn_gru_start(cnn_start), .fc1_start(fc1_start),
        .fc_out_start(fc_out_start), .argmax_start(argmax_start),
        .busy(busy), .done(done)
    );

    // This contains both activation RAMs and the complete CNN/GRU path.
    // Result RAM A addresses 0..161 hold the flattened GRU result.
    cnn_gru_top #(
        .INPUT_FILE(INPUT_FILE),
        .CONV1_W_FILE(CONV1_W_FILE), .CONV1_B_FILE(CONV1_B_FILE),
        .CONV1_PACKED_W_FILE(CONV1_PACKED_W_FILE),
        .CONV1_PACKED_B_FILE(CONV1_PACKED_B_FILE),
        .BN1_A_FILE(BN1_A_FILE), .BN1_B_FILE(BN1_B_FILE),
        .CONV2_W_FILE(CONV2_W_FILE), .CONV2_B_FILE(CONV2_B_FILE),
        .CONV2_PACKED_W_FILE(CONV2_PACKED_W_FILE),
        .CONV2_KH2_PACKED_W_FILE(CONV2_KH2_PACKED_W_FILE),
        .CONV2_PACKED_B_FILE(CONV2_PACKED_B_FILE),
        .BN2_A_FILE(BN2_A_FILE), .BN2_B_FILE(BN2_B_FILE),
        .CONV3_W_FILE(CONV3_W_FILE), .CONV3_B_FILE(CONV3_B_FILE),
        .CONV3_PACKED_W_FILE(CONV3_PACKED_W_FILE),
        .CONV3_PACKED_B_FILE(CONV3_PACKED_B_FILE),
        .BN3_A_FILE(BN3_A_FILE), .BN3_B_FILE(BN3_B_FILE),
        .GRU_WR_FILE(GRU_WR_FILE), .GRU_WZ_FILE(GRU_WZ_FILE),
        .GRU_WH_FILE(GRU_WH_FILE), .GRU_UR_FILE(GRU_UR_FILE),
        .GRU_UZ_FILE(GRU_UZ_FILE), .GRU_UH_FILE(GRU_UH_FILE),
        .GRU_BR_FILE(GRU_BR_FILE), .GRU_BZ_FILE(GRU_BZ_FILE),
        .GRU_BH_FILE(GRU_BH_FILE),
        .SIGMOID_FILE(SIGMOID_FILE), .TANH_FILE(TANH_FILE)
    ) u_cnn_gru (
        .clk(clk), .rst_n(rst_n), .start(cnn_start),
        .busy(), .done(cnn_done),
        .input_write_en(input_write_en && input_ready),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .input_ready(cnn_input_ready),
        .output_valid(), .output_addr(), .output_data(),
        .result_read_addr(cnn_ram_addr[15:0]),
        .result_read_data(cnn_ram_data)
    );

    assign cnn_ram_addr = fc1_input_addr;
    // cnn_gru_top becomes locally idle after GRU, but FC1 still needs the
    // final-result RAM B contents. Therefore global busy/done also gate input.
    assign input_ready = cnn_input_ready && !busy && !done;

    fc_engine #(
        .INPUT_SIZE(162), .OUTPUT_SIZE(40),
        .BIAS_SHIFT(16), .OUTPUT_SHIFT(17),
        .WEIGHT_FILE(FC1_W_FILE), .BIAS_FILE(FC1_B_FILE)
    ) u_fc1 (
        .clk(clk), .rst_n(rst_n), .start(fc1_start),
        .busy(), .done(fc1_done),
        .input_addr(fc1_input_addr), .input_data(cnn_ram_data),
        .output_valid(fc1_output_valid),
        .output_addr(fc1_output_addr), .output_data(fc1_output_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(0)) u_fc_relu (
        .in_valid(fc1_output_valid), .in_data(fc1_output_data),
        .out_valid(fc_relu_valid), .out_data(fc_relu_data)
    );

    // FC1/ReLU is Q13. A is Q13 and B is Q12, producing Q12.
    bn_affine #(
        .CHANNELS(40), .BIAS_SHIFT(14), .OUTPUT_SHIFT(14),
        .A_FILE(FC_BN_A_FILE), .B_FILE(FC_BN_B_FILE)
    ) u_fc_bn (
        .clk(clk), .rst_n(rst_n),
        .in_valid(fc_relu_valid), .in_data(fc_relu_data),
        .in_ch_idx(fc1_output_addr),
        .out_valid(fc_bn_valid), .out_data(fc_bn_data)
    );

    // Keep the FC address aligned with the two registered BN stages.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc_addr_valid_d1 <= 1'b0;
            fc_addr_d1 <= '0;
            fc_addr_d2 <= '0;
        end else begin
            fc_addr_valid_d1 <= fc1_output_valid;
            if (fc1_output_valid)
                fc_addr_d1 <= fc1_output_addr;
            if (fc_addr_valid_d1)
                fc_addr_d2 <= fc_addr_d1;
        end
    end

    assign fc1_bn_done = fc_bn_valid && (fc_addr_d2 == 32'd39);

    // Stream each FC1/BN result directly into FC_out. FC1 produces one value
    // about every 162 MAC clocks, allowing this engine to update all 105
    // class accumulators before the next BN value arrives.
    fc_out_streaming #(
        .INPUT_SIZE(40), .OUTPUT_SIZE(105),
        .BIAS_SHIFT(12), .OUTPUT_SHIFT(17),
        .WEIGHT_FILE(FC_OUT_W_FILE), .BIAS_FILE(FC_OUT_B_FILE)
    ) u_fc_out (
        .clk(clk), .rst_n(rst_n), .start(fc1_start),
        .busy(), .done(fc_out_done),
        .input_valid(fc_bn_valid), .input_index(fc_addr_d2[5:0]),
        .input_data(fc_bn_data),
        .output_valid(fc_out_output_valid),
        .output_addr(fc_out_output_addr), .output_data(fc_out_output_data)
    );

    // Store logits for debug/readback while simultaneously scanning them.
    /*activation_ram #(
        .DATA_W(16), .DEPTH(105), .ADDR_W(7), .MEM_FILE("")
    ) u_logit_ram (
        .clk(clk),
        .write_en(fc_out_output_valid),
        .write_addr(fc_out_output_addr[6:0]),
        .write_data(fc_out_output_data),
        .read_addr(logit_read_addr), .read_data(logit_read_data)
    );*/

    argmax_105 u_argmax (
        .clk(clk), .rst_n(rst_n), .start(argmax_start),
        .in_valid(fc_out_output_valid),
        .in_index(fc_out_output_addr), .in_data(fc_out_output_data),
        .busy(), .done(argmax_done),
        .class_index(class_index), .max_value(winning_logit)
    );

    assign logit_valid = fc_out_output_valid;
    assign logit_index = fc_out_output_addr;
    assign logit_data  = fc_out_output_data;
endmodule
