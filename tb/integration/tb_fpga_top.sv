`timescale 1ns/1ps

// Board-level test: press KEY1, wait for complete inference, and verify that
// Sample 0 predicts class 0, which maps to subject S001 on HEX3..HEX0.
module tb_fpga_top;
    logic CLOCK_50;
    logic [3:0] KEY;
    logic [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5;
    logic [9:0] LEDR;

    fpga_top #(
        .SAMPLE_FILE("../mem/board/sample0_q12.mem"),
        .CONV1_W_FILE("../mem/weights/conv1_W.mem"),
        .CONV1_B_FILE("../mem/weights/conv1_b.mem"),
        .BN1_A_FILE("../mem/weights/bn1_A.mem"),
        .BN1_B_FILE("../mem/weights/bn1_B.mem"),
        .CONV2_W_FILE("../mem/weights/conv2_W.mem"),
        .CONV2_B_FILE("../mem/weights/conv2_b.mem"),
        .CONV2_PACKED_W_FILE("../mem/weights/conv2_W_x4.mem"),
        .CONV2_PACKED_B_FILE("../mem/weights/conv2_b_x4.mem"),
        .BN2_A_FILE("../mem/weights/bn2_A.mem"),
        .BN2_B_FILE("../mem/weights/bn2_B.mem"),
        .CONV3_W_FILE("../mem/weights/conv3_W.mem"),
        .CONV3_B_FILE("../mem/weights/conv3_b.mem"),
        .BN3_A_FILE("../mem/weights/bn3_A.mem"),
        .BN3_B_FILE("../mem/weights/bn3_B.mem"),
        .GRU_WR_FILE("../mem/weights/gru_Wr.mem"),
        .GRU_WZ_FILE("../mem/weights/gru_Wz.mem"),
        .GRU_WH_FILE("../mem/weights/gru_Wh.mem"),
        .GRU_UR_FILE("../mem/weights/gru_Ur.mem"),
        .GRU_UZ_FILE("../mem/weights/gru_Uz.mem"),
        .GRU_UH_FILE("../mem/weights/gru_Uh.mem"),
        .GRU_BR_FILE("../mem/weights/gru_br.mem"),
        .GRU_BZ_FILE("../mem/weights/gru_bz.mem"),
        .GRU_BH_FILE("../mem/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem"),
        .FC1_W_FILE("../mem/weights/fc_1_W.mem"),
        .FC1_B_FILE("../mem/weights/fc_1_b.mem"),
        .FC_BN_A_FILE("../mem/weights/bn_2_A.mem"),
        .FC_BN_B_FILE("../mem/weights/bn_2_B.mem"),
        .FC_OUT_W_FILE("../mem/weights/fc_out_W.mem"),
        .FC_OUT_B_FILE("../mem/weights/fc_out_b.mem")
    ) dut (
        .CLOCK_50(CLOCK_50), .KEY(KEY),
        .HEX0(HEX0), .HEX1(HEX1), .HEX2(HEX2),
        .HEX3(HEX3), .HEX4(HEX4), .HEX5(HEX5),
        .LEDR(LEDR)
    );

    always #10 CLOCK_50 = ~CLOCK_50;

    initial begin
        CLOCK_50 = 1'b0;
        KEY = 4'b1111;

        // Reset with KEY0, then press KEY1 once.
        #25 KEY[0] = 1'b0;
        #80 KEY[0] = 1'b1;
        repeat (5) @(posedge CLOCK_50);
        @(negedge CLOCK_50); KEY[1] = 1'b0;
        repeat (5) @(posedge CLOCK_50);
        @(negedge CLOCK_50); KEY[1] = 1'b1;

        wait (LEDR[1] === 1'b1);
        repeat (2) @(posedge CLOCK_50);

        if ((dut.class_latched == 7'd0) &&
            (dut.subject_id == 7'd1) &&
            (HEX3 == 7'b0010010) &&
            (HEX2 == 7'b1000000) &&
            (HEX1 == 7'b1000000) &&
            (HEX0 == 7'b1111001)) begin
            $display("PASS: fpga_top displays predicted subject S001 on HEX3..HEX0.");
        end else begin
            $display("FAIL: class=%0d subject=S%03d HEX3=%07b HEX2=%07b HEX1=%07b HEX0=%07b",
                     dut.class_latched, dut.subject_id,
                     HEX3, HEX2, HEX1, HEX0);
            $fatal(1);
        end
        $finish;
    end

    initial begin
        #400_000_000;
        $display("FAIL: tb_fpga_top timeout");
        $fatal(1);
    end
endmodule
