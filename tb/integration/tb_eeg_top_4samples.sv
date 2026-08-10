`timescale 1ns/1ps

// Repeated-inference test. It loads four different 21x160 EEG samples through
// the same external interface that a future FPGA board wrapper will use.
module tb_eeg_top_4samples;
    localparam int SAMPLES = 4;
    localparam int INPUT_SIZE = 21 * 160;
    localparam int OUTPUT_SIZE = 105;
    localparam int LOGIT_TOLERANCE = 512;

    logic clk, rst_n, start;
    logic busy, done;
    logic input_write_en, input_ready;
    logic [11:0] input_write_addr;
    logic signed [15:0] input_write_data;
    logic [6:0] class_index;
    logic signed [15:0] winning_logit;
    logic logit_valid;
    logic [6:0] logit_index;
    logic signed [15:0] logit_data;
    logic [6:0] logit_read_addr;
    logic signed [15:0] logit_read_data;

    logic signed [15:0] input_mem [0:SAMPLES*INPUT_SIZE-1];
    logic signed [15:0] golden_mem [0:SAMPLES*OUTPUT_SIZE-1];

    integer sample, i, difference;
    integer current_sample;
    integer stream_checked, stream_errors, stream_max_diff;
    integer ram_checked, ram_errors, ram_max_diff;
    integer prediction_matches;
    integer expected_index;
    logic signed [15:0] expected_max;

    eeg_top #(
        // Every sample is loaded explicitly, so no input RAM init file is used.
        .INPUT_FILE(""),
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
        .input_write_en(input_write_en),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .input_ready(input_ready),
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

    task automatic find_expected_class(input integer sample_number);
        integer k;
        integer base;
        begin
            base = sample_number * OUTPUT_SIZE;
            expected_index = 0;
            expected_max = golden_mem[base];
            for (k = 1; k < OUTPUT_SIZE; k = k + 1) begin
                if ($signed(golden_mem[base+k]) > $signed(expected_max)) begin
                    expected_max = golden_mem[base+k];
                    expected_index = k;
                end
            end
        end
    endtask

    task automatic load_one_sample(input integer sample_number);
        integer k;
        integer base;
        begin
            base = sample_number * INPUT_SIZE;
            wait (input_ready === 1'b1);
            for (k = 0; k < INPUT_SIZE; k = k + 1) begin
                @(negedge clk);
                input_write_en = 1'b1;
                input_write_addr = k[11:0];
                input_write_data = input_mem[base+k];
            end
            @(negedge clk);
            input_write_en = 1'b0;
            input_write_addr = '0;
            input_write_data = '0;
        end
    endtask

    task automatic pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    initial begin
        $readmemh("../mem/golden/q_in_act.mem", input_mem);
        $readmemh("../mem/golden/q_fc_out_act.mem", golden_mem);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        input_write_en = 1'b0;
        input_write_addr = '0;
        input_write_data = '0;
        logit_read_addr = '0;
        current_sample = 0;
        prediction_matches = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        for (sample = 0; sample < SAMPLES; sample = sample + 1) begin
            current_sample = sample;
            stream_checked = 0;
            stream_errors = 0;
            stream_max_diff = 0;
            ram_checked = 0;
            ram_errors = 0;
            ram_max_diff = 0;
            find_expected_class(sample);

            $display("Loading sample %0d...", sample);
            load_one_sample(sample);
            pulse_start();
            wait (done === 1'b1);
            #1;

            for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                @(negedge clk);
                logit_read_addr = i[6:0];
                @(posedge clk);
                #1;
                difference = abs_diff(
                    logit_read_data,
                    golden_mem[sample*OUTPUT_SIZE+i]
                );
                if (difference > ram_max_diff)
                    ram_max_diff = difference;
                if (difference > LOGIT_TOLERANCE)
                    ram_errors = ram_errors + 1;
                ram_checked = ram_checked + 1;
            end

            if (class_index == expected_index[6:0])
                prediction_matches = prediction_matches + 1;

            if ((stream_checked == OUTPUT_SIZE) &&
                (ram_checked == OUTPUT_SIZE) &&
                (stream_errors == 0) && (ram_errors == 0) &&
                (class_index == expected_index[6:0])) begin
                $display("Sample %0d PASS: RTL class=%0d MATLAB class=%0d stream_max_diff=%0d RAM_max_diff=%0d",
                         sample, class_index, expected_index,
                         stream_max_diff, ram_max_diff);
            end else begin
                $display("Sample %0d FAIL: RTL class=%0d MATLAB class=%0d stream=%0d/%0d errors=%0d RAM errors=%0d max_diff=%0d",
                         sample, class_index, expected_index,
                         stream_checked, OUTPUT_SIZE, stream_errors,
                         ram_errors, stream_max_diff);
                $fatal(1);
            end

            // Let done fall and the global controller return to IDLE before
            // loading the next sample.
            wait (input_ready === 1'b1);
        end

        $display("PASS: repeated inference completed.");
        $display("RTL/MATLAB prediction matches = %0d/%0d",
                 prediction_matches, SAMPLES);
        $finish;
    end

    always @(posedge clk) begin
        if (logit_valid) begin
            difference = abs_diff(
                logit_data,
                golden_mem[current_sample*OUTPUT_SIZE+logit_index]
            );
            if (difference > stream_max_diff)
                stream_max_diff = difference;
            if ($isunknown(logit_index) || $isunknown(logit_data) ||
                (logit_index >= OUTPUT_SIZE) ||
                (difference > LOGIT_TOLERANCE))
                stream_errors = stream_errors + 1;
            stream_checked = stream_checked + 1;
        end
    end

    // 100 million clocks is comfortably above four expected inferences.
    initial begin
        #1_000_000_000;
        $display("FAIL: tb_eeg_top_4samples timeout");
        $fatal(1);
    end
endmodule
