// Complete inference path:
// CNN -> Pool2 -> GRU/flatten -> FC1 -> ReLU -> BN -> FC_out -> Argmax.

module eeg_top #(
    parameter INPUT_FILE   = "mem/dsconv1_dsconv2/board/ram_a_sample0_q12.mem",
    parameter INPUT_EVEN_FILE = "mem/dsconv1_dsconv2/board/sample0_q12_even.mem",
    parameter INPUT_ODD_FILE = "mem/dsconv1_dsconv2/board/sample0_q12_odd.mem",
    parameter CONV1_DW_W_FILE = "mem/dsconv1_dsconv2/weights/conv1_depthwise_W_kh2.mem",
    parameter CONV1_DW_B_FILE = "mem/dsconv1_dsconv2/weights/conv1_depthwise_b.mem",
    parameter CONV1_PW_W_FILE = "mem/dsconv1_dsconv2/weights/conv1_pointwise_W_x3.mem",
    parameter CONV1_PW_B_FILE = "mem/dsconv1_dsconv2/weights/conv1_pointwise_b_x3.mem",
    parameter BN1_A_FILE   = "mem/dsconv1_dsconv2/weights/bn1_A.mem",
    parameter BN1_B_FILE   = "mem/dsconv1_dsconv2/weights/bn1_B.mem",
    parameter CONV2_DW_W_FILE = "mem/dsconv1_dsconv2/weights/conv2_depthwise_W_kh2.mem",
    parameter CONV2_DW_B_FILE = "mem/dsconv1_dsconv2/weights/conv2_depthwise_b.mem",
    parameter CONV2_PW_W_FILE = "mem/dsconv1_dsconv2/weights/conv2_pointwise_W_x5.mem",
    parameter CONV2_PW_B_FILE = "mem/dsconv1_dsconv2/weights/conv2_pointwise_b_x5.mem",
    parameter BN2_A_FILE   = "mem/dsconv1_dsconv2/weights/bn2_A.mem",
    parameter BN2_B_FILE   = "mem/dsconv1_dsconv2/weights/bn2_B.mem",
    parameter CONV3_W_FILE = "mem/dsconv1_dsconv2/weights/conv3_W.mem",
    parameter CONV3_B_FILE = "mem/dsconv1_dsconv2/weights/conv3_b.mem",
    parameter CONV3_PACKED_W_FILE = "mem/dsconv1_dsconv2/weights/conv3_W_x3.mem",
    parameter CONV3_PACKED_B_FILE = "mem/dsconv1_dsconv2/weights/conv3_b_x3.mem",
    parameter BN3_A_FILE   = "mem/dsconv1_dsconv2/weights/bn3_A.mem",
    parameter BN3_B_FILE   = "mem/dsconv1_dsconv2/weights/bn3_B.mem",
    parameter GRU_WR_FILE  = "mem/dsconv1_dsconv2/weights/gru_Wr.mem",
    parameter GRU_WZ_FILE  = "mem/dsconv1_dsconv2/weights/gru_Wz.mem",
    parameter GRU_WH_FILE  = "mem/dsconv1_dsconv2/weights/gru_Wh.mem",
    parameter GRU_UR_FILE  = "mem/dsconv1_dsconv2/weights/gru_Ur.mem",
    parameter GRU_UZ_FILE  = "mem/dsconv1_dsconv2/weights/gru_Uz.mem",
    parameter GRU_UH_FILE  = "mem/dsconv1_dsconv2/weights/gru_Uh.mem",
    parameter GRU_BR_FILE  = "mem/dsconv1_dsconv2/weights/gru_br.mem",
    parameter GRU_BZ_FILE  = "mem/dsconv1_dsconv2/weights/gru_bz.mem",
    parameter GRU_BH_FILE  = "mem/dsconv1_dsconv2/weights/gru_bh.mem",
    parameter ACTIVATION_LUT_FILE = "mem/lut/gru_activation_lut_q15.mem",
    parameter FC1_W_FILE   = "mem/dsconv1_dsconv2/weights/fc_1_W.mem",
    parameter FC1_B_FILE   = "mem/dsconv1_dsconv2/weights/fc_1_b.mem",
    parameter FC_BN_A_FILE = "mem/dsconv1_dsconv2/weights/bn_2_A.mem",
    parameter FC_BN_B_FILE = "mem/dsconv1_dsconv2/weights/bn_2_B.mem",
    parameter FC_OUT_W_FILE = "mem/dsconv1_dsconv2/weights/fc_out_W.mem",
    parameter FC_OUT_B_FILE = "mem/dsconv1_dsconv2/weights/fc_out_b.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // 在啟動 (start) 前，先將單一筆量化後的 EEG 樣本載入輸入 RAM。
    // 有效位址範圍為 0..3359（21x160，帶正負號的 Q12 格式數值）。
    input  logic               input_write_en,
    input  logic [11:0]        input_write_addr,
    input  logic signed [15:0] input_write_data,
    output logic               input_ready,

    output logic [6:0]         class_index,
    output logic signed [15:0] winning_logit
);
    logic cnn_start, cnn_done;
    logic cnn_input_ready;
    logic fc1_start, fc1_busy;
    logic argmax_start, argmax_done;

    logic signed [15:0] cnn_ram_data;

    logic [7:0] fc1_input_addr;
    logic [5:0] fc1_output_addr;
    logic signed [15:0] fc1_output_data;
    logic fc1_output_valid;
    logic fc_relu_valid;
    logic signed [15:0] fc_relu_data;
    logic fc_bn_valid;
    logic signed [15:0] fc_bn_data;
    logic [5:0] fc_addr_d1, fc_addr_d2;
    logic fc_addr_valid_d1;
    logic fc1_bn_done;

    logic [6:0] fc_out_output_addr;  // Max 104
    logic signed [15:0] fc_out_output_data;
    logic fc_out_output_valid;

    // 控制 CNN/GRU → FC1 → ReLU/BN → FC_out 的完整推理流程。
    eeg_controller u_controller (
        .clk(clk), .rst_n(rst_n), .start(start),
        .cnn_gru_done(cnn_done), .fc1_bn_done(fc1_bn_done),
        .argmax_done(argmax_done),
        .cnn_gru_start(cnn_start), .fc1_start(fc1_start),
        .argmax_start(argmax_start),
        .busy(busy), .done(done)
    );

    cnn_gru_top #(
        .INPUT_FILE(INPUT_FILE),
        .INPUT_EVEN_FILE(INPUT_EVEN_FILE), .INPUT_ODD_FILE(INPUT_ODD_FILE),
        .CONV1_DW_W_FILE(CONV1_DW_W_FILE), .CONV1_DW_B_FILE(CONV1_DW_B_FILE),
        .CONV1_PW_W_FILE(CONV1_PW_W_FILE),
        .CONV1_PW_B_FILE(CONV1_PW_B_FILE),
        .BN1_A_FILE(BN1_A_FILE), .BN1_B_FILE(BN1_B_FILE),
        .CONV2_DW_W_FILE(CONV2_DW_W_FILE),
        .CONV2_DW_B_FILE(CONV2_DW_B_FILE),
        .CONV2_PW_W_FILE(CONV2_PW_W_FILE),
        .CONV2_PW_B_FILE(CONV2_PW_B_FILE),
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
        .ACTIVATION_LUT_FILE(ACTIVATION_LUT_FILE)
    ) u_cnn_gru (
        .clk(clk), .rst_n(rst_n), .start(cnn_start),
        .busy(), .done(cnn_done),
        .input_write_en(input_write_en && input_ready),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .input_ready(cnn_input_ready),
        .output_valid(), .output_addr(), .output_data(),
        .result_read_en(fc1_start || fc1_busy),
        .result_read_addr({5'd0, fc1_input_addr}),
        .result_read_data(cnn_ram_data)
    );

    // 全域 busy/done 控制輸入
    assign input_ready = cnn_input_ready && !busy && !done;

    fc_engine #(
        .INPUT_SIZE(162), .OUTPUT_SIZE(40),
        // flatten(Q15) * weight(Q15) 為 Q30，右移 17 位得到 FC1 的 Q13。
        .BIAS_SHIFT(16), .OUTPUT_SHIFT(17),
        .WEIGHT_FILE(FC1_W_FILE), .BIAS_FILE(FC1_B_FILE)
    ) u_fc1 (
        .clk(clk), .rst_n(rst_n), .start(fc1_start),
        .busy(fc1_busy), .done(),
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

    assign fc1_bn_done = fc_bn_valid && (fc_addr_d2 == 6'd39);

    // 以 streaming 計算 FC_out
    // FC1 大約 162 個 cycle 產生一個 result，因此可在下一 BN 數值到來前更新 105 類別累加器
    fc_out_streaming #(
        .INPUT_SIZE(40), .OUTPUT_SIZE(105),
        .BIAS_SHIFT(12), .OUTPUT_SHIFT(17),
        .WEIGHT_FILE(FC_OUT_W_FILE), .BIAS_FILE(FC_OUT_B_FILE)
    ) u_fc_out (
        .clk(clk), .rst_n(rst_n), .start(fc1_start),
        .busy(), .done(),
        .input_valid(fc_bn_valid), .input_index(fc_addr_d2),
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

endmodule
