// DE1-SoC UART streaming top. This is intentionally separate from fpga_top.sv.
// UART receives one complete sample, writes existing RAM A, runs inference,
// and returns the class. No third activation/input RAM is instantiated.
module fpga_uart_top #(
    // 921600 baud rate
    parameter integer UART_CLKS_PER_BIT = 54,
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
    input  logic       UART_RXD,
    output logic       UART_TXD,
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

    logic [7:0] rx_data;
    logic rx_valid, rx_framing_error, rx_busy;
    logic tx_start, tx_busy, tx_done;
    logic [7:0] tx_data;

    logic input_write_en;
    logic [11:0] input_write_addr;
    logic signed [15:0] input_write_data;
    logic inference_start;
    logic input_ready;
    logic loader_ready, loader_busy, packet_error, packet_loaded;
    logic [31:0] sample_id;

    logic core_busy, core_done;
    logic [6:0] core_class_index;
    logic signed [15:0] core_winning_logit;
    logic logit_valid_unused;
    logic [6:0] logit_index_unused;
    logic signed [15:0] logit_data_unused;
    logic signed [15:0] logit_read_data_unused;

    logic response_start;
    logic [7:0] response_status;
    logic response_busy, response_done;
    logic result_valid, packet_error_latched;
    logic [6:0] class_latched;
    logic [6:0] subject_id;
    logic [3:0] bcd_hundreds, bcd_tens, bcd_ones;

    // KEY0 provides asynchronous assertion and synchronous release.
    always_ff @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0])
            reset_sync <= 2'b00;
        else
            reset_sync <= {reset_sync[0], 1'b1};
    end
    assign rst_n = reset_sync[1];

    uart_rx #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) u_uart_rx (
        .clk(CLOCK_50), .rst_n(rst_n), .serial_rx(UART_RXD),
        .data(rx_data), .valid(rx_valid),
        .framing_error(rx_framing_error), .busy(rx_busy)
    );

    uart_sample_loader #(.SAMPLE_WORDS(3360), .ADDR_W(12)) u_loader (
        .clk(CLOCK_50), .rst_n(rst_n),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .rx_framing_error(rx_framing_error),
        .input_ready(input_ready),
        .input_write_en(input_write_en),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .inference_start(inference_start),
        .inference_done(core_done),
        .sample_id(sample_id),
        .loader_ready(loader_ready), .loader_busy(loader_busy),
        .packet_error(packet_error), .packet_loaded(packet_loaded)
    );

    eeg_top #(
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

    assign response_start  = core_done || packet_error;
    assign response_status = packet_error ? 8'd1 : 8'd0;

    uart_result_sender u_result_sender (
        .clk(CLOCK_50), .rst_n(rst_n), .start(response_start),
        .sample_id(sample_id), .status(response_status),
        .class_index(packet_error ? 7'd0 : core_class_index),
        .winning_logit(packet_error ? 16'sd0 : core_winning_logit),
        .tx_start(tx_start), .tx_data(tx_data),
        .tx_busy(tx_busy), .tx_done(tx_done),
        .busy(response_busy), .done(response_done)
    );

    uart_tx #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) u_uart_tx (
        .clk(CLOCK_50), .rst_n(rst_n),
        .start(tx_start), .data(tx_data),
        .serial_tx(UART_TXD), .busy(tx_busy), .done(tx_done)
    );

    always_ff @(posedge CLOCK_50 or negedge rst_n) begin
        if (!rst_n) begin
            result_valid         <= 1'b0;
            packet_error_latched <= 1'b0;
            class_latched        <= 7'd0;
        end else begin
            if (packet_loaded) begin
                result_valid         <= 1'b0;
                packet_error_latched <= 1'b0;
            end
            if (packet_error)
                packet_error_latched <= 1'b1;
            if (core_done) begin
                result_valid  <= 1'b1;
                class_latched <= core_class_index;
            end
        end
    end

    // Convert the model's compact class index (0..104) back to the original
    // PhysioNet subject number (S001..S109). The dataset omits S088, S092,
    // S100 and S104, so those identifiers are skipped by this mapping.
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

    // Active-low seven-segment pattern for the letter S (same as digit 5).
    assign HEX3 = result_valid ? 7'b0010010 : 7'b1111111;
    assign HEX4 = 7'b1111111;
    assign HEX5 = 7'b1111111;

    // LEDR0=core busy, LEDR1=result valid, LEDR2=loader ready,
    // LEDR3=receiving packet, LEDR4=transmitting response,
    // LEDR5=last packet had an error.
    assign LEDR = {4'b0, packet_error_latched, response_busy,
                   loader_busy, loader_ready, result_valid, core_busy};
endmodule
