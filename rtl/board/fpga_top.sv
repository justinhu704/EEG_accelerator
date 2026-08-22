// DE1-SoC board-level top for the fixed-sample EEG classifier demo.
// KEY0: active-low reset. KEY1: reload Sample 0 and start inference.
// HEX3..HEX0: predicted subject ID as S001..S109. The mapping skips
// subjects S088, S092, S100 and S104, which are absent from the model.
module fpga_top #(
    parameter SAMPLE_FILE   = "mem/board/sample105_q12.mem",
    parameter CONV1_W_FILE  = "mem/weights/conv1_W.mem",
    parameter CONV1_B_FILE  = "mem/weights/conv1_b.mem",
    parameter CONV1_PACKED_W_FILE = "mem/weights/conv1_W_x3.mem",
    parameter CONV1_PACKED_B_FILE = "mem/weights/conv1_b_x3.mem",
    parameter BN1_A_FILE    = "mem/weights/bn1_A.mem",
    parameter BN1_B_FILE    = "mem/weights/bn1_B.mem",
    parameter CONV2_W_FILE  = "mem/weights/conv2_W.mem",
    parameter CONV2_B_FILE  = "mem/weights/conv2_b.mem",
    parameter CONV2_PACKED_W_FILE = "mem/weights/conv2_W_x5.mem",
    parameter CONV2_KH2_PACKED_W_FILE = "mem/weights/conv2_W_x5_kh2.mem",
    parameter CONV2_PACKED_B_FILE = "mem/weights/conv2_b_x5.mem",
    parameter BN2_A_FILE    = "mem/weights/bn2_A.mem",
    parameter BN2_B_FILE    = "mem/weights/bn2_B.mem",
    parameter CONV3_W_FILE  = "mem/weights/conv3_W.mem",
    parameter CONV3_B_FILE  = "mem/weights/conv3_b.mem",
    parameter CONV3_PACKED_W_FILE = "mem/weights/conv3_W_x3.mem",
    parameter CONV3_PACKED_B_FILE = "mem/weights/conv3_b_x3.mem",
    parameter BN3_A_FILE    = "mem/weights/bn3_A.mem",
    parameter BN3_B_FILE    = "mem/weights/bn3_B.mem",
    parameter GRU_WR_FILE   = "mem/weights/gru_Wr.mem",
    parameter GRU_WZ_FILE   = "mem/weights/gru_Wz.mem",
    parameter GRU_WH_FILE   = "mem/weights/gru_Wh.mem",
    parameter GRU_UR_FILE   = "mem/weights/gru_Ur.mem",
    parameter GRU_UZ_FILE   = "mem/weights/gru_Uz.mem",
    parameter GRU_UH_FILE   = "mem/weights/gru_Uh.mem",
    parameter GRU_BR_FILE   = "mem/weights/gru_br.mem",
    parameter GRU_BZ_FILE   = "mem/weights/gru_bz.mem",
    parameter GRU_BH_FILE   = "mem/weights/gru_bh.mem",
    parameter SIGMOID_FILE  = "mem/lut/sigmoid_half_lut_q15.mem",
    parameter TANH_FILE     = "mem/lut/tanh_half_lut_q15.mem",
    parameter FC1_W_FILE    = "mem/weights/fc_1_W.mem",
    parameter FC1_B_FILE    = "mem/weights/fc_1_b.mem",
    parameter FC_BN_A_FILE  = "mem/weights/bn_2_A.mem",
    parameter FC_BN_B_FILE  = "mem/weights/bn_2_B.mem",
    parameter FC_OUT_W_FILE = "mem/weights/fc_out_W.mem",
    parameter FC_OUT_B_FILE = "mem/weights/fc_out_b.mem"
) (
    input  logic       CLOCK_50,
    input  logic [3:0] KEY,
    output logic [6:0] HEX0,
    output logic [6:0] HEX1,
    output logic [6:0] HEX2,
    output logic [6:0] HEX3,
    output logic [6:0] HEX4,
    output logic [6:0] HEX5,
    output logic [9:0] LEDR
);
    logic [1:0] reset_sync;
    logic rst_n;
    logic key1_meta, key1_sync, key1_previous;
    logic start_request;

    logic loader_busy;
    logic input_write_en;
    logic [11:0] input_write_addr;
    logic signed [15:0] input_write_data;
    logic inference_start;

    logic core_busy, core_done, input_ready;
    logic [6:0] core_class_index;
    logic signed [15:0] core_winning_logit;
    logic result_valid;
    logic [6:0] class_latched;
    logic [6:0] subject_id;

    logic logit_valid_unused;
    logic [6:0] logit_index_unused;
    logic signed [15:0] logit_data_unused;
    logic signed [15:0] logit_read_data_unused;

    logic [3:0] bcd_hundreds, bcd_tens, bcd_ones;

    // Asynchronous assertion from KEY0, synchronous reset release.
    always_ff @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0])
            reset_sync <= 2'b00;
        else
            reset_sync <= {reset_sync[0], 1'b1};
    end
    assign rst_n = reset_sync[1];

    // Synchronize active-low KEY1 and convert one press into one clock pulse.
    // Button bounce is ignored while the loader/inference controller is busy.
    always_ff @(posedge CLOCK_50 or negedge rst_n) begin
        if (!rst_n) begin
            key1_meta     <= 1'b1;
            key1_sync     <= 1'b1;
            key1_previous <= 1'b1;
        end else begin
            key1_meta     <= KEY[1];
            key1_sync     <= key1_meta;
            key1_previous <= key1_sync;
        end
    end
    assign start_request = key1_previous && !key1_sync && !loader_busy;

    fixed_sample_loader #(
        .SAMPLE_SIZE(3360), .ADDR_W(12), .SAMPLE_FILE(SAMPLE_FILE)
    ) u_loader (
        .clk(CLOCK_50), .rst_n(rst_n),
        .request(start_request), .input_ready(input_ready),
        .inference_done(core_done),
        .input_write_en(input_write_en),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .inference_start(inference_start), .busy(loader_busy)
    );

    eeg_top #(
        // The loader supplies every input sample; do not initialize RAM A.
        .INPUT_FILE(""),
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
        .SIGMOID_FILE(SIGMOID_FILE), .TANH_FILE(TANH_FILE),
        .FC1_W_FILE(FC1_W_FILE), .FC1_B_FILE(FC1_B_FILE),
        .FC_BN_A_FILE(FC_BN_A_FILE), .FC_BN_B_FILE(FC_BN_B_FILE),
        .FC_OUT_W_FILE(FC_OUT_W_FILE), .FC_OUT_B_FILE(FC_OUT_B_FILE)
    ) u_eeg_top (
        .clk(CLOCK_50), .rst_n(rst_n), .start(inference_start),
        .busy(core_busy), .done(core_done),
        .input_write_en(input_write_en),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .input_ready(input_ready),
        .class_index(core_class_index),
        .winning_logit(core_winning_logit),
        .logit_valid(logit_valid_unused),
        .logit_index(logit_index_unused),
        .logit_data(logit_data_unused),
        .logit_read_addr(7'd0),
        .logit_read_data(logit_read_data_unused)
    );

    // Preserve the one-cycle result after core_done returns low.
    always_ff @(posedge CLOCK_50 or negedge rst_n) begin
        if (!rst_n) begin
            result_valid <= 1'b0;
            class_latched <= 7'd0;
        end else begin
            if (start_request)
                result_valid <= 1'b0;
            if (core_done) begin
                result_valid <= 1'b1;
                class_latched <= core_class_index;
            end
        end
    end

    class_to_subject_id u_class_to_subject_id (
        .class_index(class_latched),
        .subject_id(subject_id)
    );

    binary_to_bcd_7bit u_binary_to_bcd (
        .binary(subject_id),
        .hundreds(bcd_hundreds), .tens(bcd_tens), .ones(bcd_ones)
    );

    seven_seg_decoder u_hex0 (
        .digit(bcd_ones), .blank(!result_valid), .segments(HEX0)
    );
    seven_seg_decoder u_hex1 (
        .digit(bcd_tens), .blank(!result_valid), .segments(HEX1)
    );
    seven_seg_decoder u_hex2 (
        .digit(bcd_hundreds), .blank(!result_valid), .segments(HEX2)
    );

    // Active-low seven-segment pattern for the letter S. It intentionally
    // uses the same segments as digit 5, which reads as S on this display.
    assign HEX3 = result_valid ? 7'b0010010 : 7'b1111111;

    // Unused displays are off. LEDR0=busy, LEDR1=result valid,
    // LEDR2=input loader ready.
    assign HEX4 = 7'b1111111;
    assign HEX5 = 7'b1111111;
    assign LEDR = {7'b0, input_ready, result_valid,
                   (loader_busy || core_busy)};
endmodule
