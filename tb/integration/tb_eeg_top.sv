`timescale 1ns/1ps

// End-to-end verification of one EEG sample:
// CNN -> BN/ReLU -> Pool -> GRU -> FC1 -> ReLU -> BN -> FC_out -> Argmax.
module tb_eeg_top;
    localparam int OUTPUT_SIZE = 105;
    localparam int MAX_CYCLES = 20_100_000;
    // The GRU uses finite LUTs, while MATLAB's golden uses continuous
    // sigmoid/tanh. This end-to-end bound is checked again after simulation.
    localparam int LOGIT_TOLERANCE = 512;

    logic clk, rst_n, start;
    logic busy, done;
    logic [6:0] class_index;
    logic signed [15:0] winning_logit;
    logic logit_valid;
    logic [6:0] logit_index;
    logic signed [15:0] logit_data;
    logic [6:0] logit_read_addr;
    logic signed [15:0] logit_read_data;
    logic signed [15:0] golden [0:4*OUTPUT_SIZE-1];

    integer stream_checked, stream_errors, stream_max_diff;
    integer ram_checked, ram_errors, ram_max_diff;
    integer expected_index, difference, cycle_count, i;
    logic signed [15:0] expected_max;

    eeg_top #(
        .INPUT_FILE("../mem/golden/q_in_act.mem"),
        .CONV1_W_FILE("../mem/weights/conv1_W.mem"),
        .CONV1_B_FILE("../mem/weights/conv1_b.mem"),
        .BN1_A_FILE("../mem/weights/bn1_A.mem"),
        .BN1_B_FILE("../mem/weights/bn1_B.mem"),
        .CONV2_W_FILE("../mem/weights/conv2_W.mem"),
        .CONV2_B_FILE("../mem/weights/conv2_b.mem"),
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
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .class_index(class_index), .winning_logit(winning_logit),
        .logit_valid(logit_valid), .logit_index(logit_index),
        .logit_data(logit_data),
        .logit_read_addr(logit_read_addr),
        .logit_read_data(logit_read_data)
    );

    always #5 clk = ~clk;

    function automatic integer abs_diff(
        input logic signed [15:0] actual,
        input logic signed [15:0] expected
    );
        integer value;
        begin
            value = $signed(actual) - $signed(expected);
            abs_diff = (value < 0) ? -value : value;
        end
    endfunction

    initial begin
        $readmemh("../mem/golden/q_fc_out_act.mem", golden);
        expected_index = 0;
        expected_max = golden[0];
        for (i = 1; i < OUTPUT_SIZE; i = i + 1) begin
            if ($signed(golden[i]) > $signed(expected_max)) begin
                expected_max = golden[i];
                expected_index = i;
            end
        end

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        logit_read_addr = '0;
        stream_checked = 0;
        stream_errors = 0;
        stream_max_diff = 0;
        ram_checked = 0;
        ram_errors = 0;
        ram_max_diff = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done);
        #1;

        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            @(negedge clk);
            logit_read_addr = i[6:0];
            @(posedge clk);
            #1;
            difference = abs_diff(logit_read_data, golden[i]);
            if (difference > ram_max_diff)
                ram_max_diff = difference;
            if (difference > LOGIT_TOLERANCE) begin
                if (ram_errors < 10)
                    $display("RAM mismatch index=%0d got=%0d expected=%0d diff=%0d",
                             i, logit_read_data, golden[i], difference);
                ram_errors = ram_errors + 1;
            end
            ram_checked = ram_checked + 1;
        end

        if ((stream_checked == OUTPUT_SIZE) &&
            (ram_checked == OUTPUT_SIZE) &&
            (stream_errors == 0) && (ram_errors == 0) &&
            (class_index == expected_index[6:0])) begin
            $display("PASS: complete EEG inference, predicted class=%0d.",
                     class_index);
            $display("Logit stream checked=%0d max_diff=%0d",
                     stream_checked, stream_max_diff);
            $display("Logit RAM checked=%0d max_diff=%0d",
                     ram_checked, ram_max_diff);
            $display("Winning RTL logit=%0d MATLAB class=%0d max=%0d",
                     winning_logit, expected_index, expected_max);
        end else begin
            $display("FAIL: complete EEG inference.");
            $display("Stream checked=%0d errors=%0d max_diff=%0d",
                     stream_checked, stream_errors, stream_max_diff);
            $display("RAM checked=%0d errors=%0d max_diff=%0d",
                     ram_checked, ram_errors, ram_max_diff);
            $display("RTL class=%0d expected class=%0d", class_index,
                     expected_index);
            $fatal(1);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (logit_valid) begin
            if ($isunknown(logit_index) || $isunknown(logit_data) ||
                (logit_index >= OUTPUT_SIZE)) begin
                stream_errors = stream_errors + 1;
            end else begin
                difference = abs_diff(logit_data, golden[logit_index]);
                if (difference > stream_max_diff)
                    stream_max_diff = difference;
                if (difference > LOGIT_TOLERANCE) begin
                    if (stream_errors < 10)
                        $display("Stream mismatch index=%0d got=%0d expected=%0d diff=%0d",
                                 logit_index, logit_data,
                                 golden[logit_index], difference);
                    stream_errors = stream_errors + 1;
                end
            end
            stream_checked = stream_checked + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_eeg_top timeout at cycle %0d", cycle_count);
            $fatal(1);
        end
    end
endmodule
