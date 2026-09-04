`timescale 1ns/1ps

// Cycle profiler for one complete EEG inference:
// CNN -> Pool2 -> GRU -> FC1/ReLU/BN -> FC_out -> Argmax.
//
// Cycle counts are elapsed clock intervals between the clock edge that accepts
// start and the clock edge where done is observed.
module tb_eeg_cycle_count;
    localparam int MAX_CYCLES = 20_100_000;
    localparam int EXPECTED_LOGITS = 105;
    localparam int EXPECTED_CLASS = 0;
    // 優化前 DS-Conv2 + BN2/ReLU2/Pool1 的實測基準。
    localparam int DS_CONV2_BASELINE_CYCLES = 823_086;
    localparam int CONV1_VALUES = 20 * 156 * 21;
    localparam int CONV2_VALUES = 19 * 152 * 20;
    localparam int POOL1_VALUES = 19 * 18 * 20;
    parameter real CLOCK_PERIOD_NS = 20.0;
    localparam real CLOCK_FREQ_MHZ = 1000.0 / CLOCK_PERIOD_NS;

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

    longint unsigned cycle_number;
    longint unsigned start_cycle;
    longint unsigned conv1_cycle;
    longint unsigned conv2_cycle;
    longint unsigned conv3_cycle;
    longint unsigned gru_cycle;
    longint unsigned fc1_cycle;
    longint unsigned fc_out_cycle;
    longint unsigned total_cycles;
    integer final_logit_count;
    integer logit_count;
    logic measuring;
    logic signed [15:0] conv1_golden [0:CONV1_VALUES-1];
    logic signed [15:0] conv2_golden [0:CONV2_VALUES-1];
    logic signed [15:0] pool1_golden [0:POOL1_VALUES-1];
    integer conv1_max_diff, conv2_max_diff, pool1_max_diff;
    integer stage_diff;

    eeg_top #(
        .INPUT_FILE("../mem/dsconv2/board/ram_a_sample0_q12.mem"),
        .CONV1_W_FILE("../mem/dsconv2/weights/conv1_W.mem"),
        .CONV1_B_FILE("../mem/dsconv2/weights/conv1_b.mem"),
        .CONV1_PACKED_W_FILE("../mem/dsconv2/weights/conv1_W_x3.mem"),
        .CONV1_PACKED_B_FILE("../mem/dsconv2/weights/conv1_b_x3.mem"),
        .BN1_A_FILE("../mem/dsconv2/weights/bn1_A.mem"),
        .BN1_B_FILE("../mem/dsconv2/weights/bn1_B.mem"),
        .CONV2_DW_W_FILE("../mem/dsconv2/weights/conv2_depthwise_W_kh2.mem"),
        .CONV2_DW_B_FILE("../mem/dsconv2/weights/conv2_depthwise_b.mem"),
        .CONV2_PW_W_FILE("../mem/dsconv2/weights/conv2_pointwise_W_x5.mem"),
        .CONV2_PW_B_FILE("../mem/dsconv2/weights/conv2_pointwise_b_x5.mem"),
        .BN2_A_FILE("../mem/dsconv2/weights/bn2_A.mem"),
        .BN2_B_FILE("../mem/dsconv2/weights/bn2_B.mem"),
        .CONV3_W_FILE("../mem/dsconv2/weights/conv3_W.mem"),
        .CONV3_B_FILE("../mem/dsconv2/weights/conv3_b.mem"),
        .CONV3_PACKED_W_FILE("../mem/dsconv2/weights/conv3_W_x3.mem"),
        .CONV3_PACKED_B_FILE("../mem/dsconv2/weights/conv3_b_x3.mem"),
        .BN3_A_FILE("../mem/dsconv2/weights/bn3_A.mem"),
        .BN3_B_FILE("../mem/dsconv2/weights/bn3_B.mem"),
        .GRU_WR_FILE("../mem/dsconv2/weights/gru_Wr.mem"),
        .GRU_WZ_FILE("../mem/dsconv2/weights/gru_Wz.mem"),
        .GRU_WH_FILE("../mem/dsconv2/weights/gru_Wh.mem"),
        .GRU_UR_FILE("../mem/dsconv2/weights/gru_Ur.mem"),
        .GRU_UZ_FILE("../mem/dsconv2/weights/gru_Uz.mem"),
        .GRU_UH_FILE("../mem/dsconv2/weights/gru_Uh.mem"),
        .GRU_BR_FILE("../mem/dsconv2/weights/gru_br.mem"),
        .GRU_BZ_FILE("../mem/dsconv2/weights/gru_bz.mem"),
        .GRU_BH_FILE("../mem/dsconv2/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem"),
        .FC1_W_FILE("../mem/dsconv2/weights/fc_1_W.mem"),
        .FC1_B_FILE("../mem/dsconv2/weights/fc_1_b.mem"),
        .FC_BN_A_FILE("../mem/dsconv2/weights/bn_2_A.mem"),
        .FC_BN_B_FILE("../mem/dsconv2/weights/bn_2_B.mem"),
        .FC_OUT_W_FILE("../mem/dsconv2/weights/fc_out_W.mem"),
        .FC_OUT_B_FILE("../mem/dsconv2/weights/fc_out_b.mem")
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

    always #(CLOCK_PERIOD_NS/2.0) clk = ~clk;

    initial begin
        $readmemh("../mem/dsconv2/golden/q_relu1_act_sample0.mem",
                  conv1_golden);
        $readmemh("../mem/dsconv2/golden/q_relu2_act_sample0.mem",
                  conv2_golden);
        $readmemh("../mem/dsconv2/golden/q_pool1_act_sample0.mem",
                  pool1_golden);
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        input_write_en = 1'b0;
        input_write_addr = '0;
        input_write_data = '0;
        logit_read_addr = '0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done);
        // Let the clocked profiler observe the S_DONE cycle before stopping.
        @(posedge clk);
        @(negedge clk);
        $finish;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_number <= 0;
            start_cycle <= 0;
            conv1_cycle <= 0;
            conv2_cycle <= 0;
            conv3_cycle <= 0;
            gru_cycle <= 0;
            fc1_cycle <= 0;
            fc_out_cycle <= 0;
            total_cycles <= 0;
            logit_count <= 0;
            measuring <= 1'b0;
            conv1_max_diff <= 0;
            conv2_max_diff <= 0;
            pool1_max_diff <= 0;
        end else begin
            cycle_number <= cycle_number + 1'b1;

            if (start && !measuring) begin
                start_cycle <= cycle_number;
                measuring <= 1'b1;
            end

            // Hierarchical references are intentionally used only by this
            // profiler. They do not add hardware to the synthesized design.
            if (dut.u_cnn_gru.conv1_start)
                conv1_cycle <= cycle_number;
            if (dut.u_cnn_gru.conv2_start)
                conv2_cycle <= cycle_number;
            if (dut.u_cnn_gru.conv3_start)
                conv3_cycle <= cycle_number;
            if (dut.u_cnn_gru.gru_start)
                gru_cycle <= cycle_number;
            if (dut.fc1_start)
                fc1_cycle <= cycle_number;
            if (dut.fc_out_start)
                fc_out_cycle <= cycle_number;

            if (logit_valid)
                logit_count <= logit_count + 1;

            // Keep cycle profiling and stage-order verification in one run.
            if (dut.u_cnn_gru.conv1_valid) begin
                stage_diff = $signed(dut.u_cnn_gru.conv1_data)
                           - $signed(conv1_golden[dut.u_cnn_gru.conv1_addr]);
                if (stage_diff < 0) stage_diff = -stage_diff;
                if (stage_diff > conv1_max_diff)
                    conv1_max_diff <= stage_diff;
            end
            if (dut.u_cnn_gru.conv2_valid) begin
                stage_diff = $signed(dut.u_cnn_gru.conv2_data)
                           - $signed(conv2_golden[dut.u_cnn_gru.conv2_addr]);
                if (stage_diff < 0) stage_diff = -stage_diff;
                if (stage_diff > conv2_max_diff)
                    conv2_max_diff <= stage_diff;
            end
            if (dut.u_cnn_gru.pool1_valid) begin
                stage_diff = $signed(dut.u_cnn_gru.pool1_data)
                           - $signed(pool1_golden[dut.u_cnn_gru.pool1_addr]);
                if (stage_diff < 0) stage_diff = -stage_diff;
                if (stage_diff > pool1_max_diff)
                    pool1_max_diff <= stage_diff;
            end

            if (done && measuring) begin
                total_cycles = cycle_number - start_cycle;
                // If the last logit and done arrive in the same cycle, the
                // nonblocking update above has not changed logit_count yet.
                final_logit_count = logit_count + (logit_valid ? 1 : 0);
                measuring <= 1'b0;

                $display("");
                $display("==================================================");
                $display("Complete EEG inference cycle report (%0.3f MHz)",
                         CLOCK_FREQ_MHZ);
                $display("==================================================");
                $display("Controller/start overhead : %0d cycles",
                         conv1_cycle - start_cycle);
                $display("Conv1 + BN1 + ReLU1       : %0d cycles",
                         conv2_cycle - conv1_cycle);
                $display("DS-Conv2 + BN2/ReLU2/Pool1: %0d cycles",
                         conv3_cycle - conv2_cycle);
                $display("DS-Conv2 cycles saved       : %0d cycles",
                         DS_CONV2_BASELINE_CYCLES
                         - (conv3_cycle - conv2_cycle));
                // Pool2 and Conv3 start together and operate as one streaming
                // interval, so separate start timestamps cannot divide them.
                $display("Conv3 + BN3 + ReLU3/Pool2 : %0d cycles",
                         gru_cycle - conv3_cycle);
                $display("GRU                       : %0d cycles",
                         fc1_cycle - gru_cycle);
                $display("FC1 + ReLU + BN           : %0d cycles",
                         fc_out_cycle - fc1_cycle);
                $display("FC_out + Argmax           : %0d cycles",
                         cycle_number - fc_out_cycle);
                $display("--------------------------------------------------");
                $display("TOTAL                     : %0d cycles",
                         total_cycles);
                $display("Hardware time at %0.3f MHz: %0.3f ms",
                         CLOCK_FREQ_MHZ,
                         total_cycles * CLOCK_PERIOD_NS / 1.0e6);
                $display("Produced logits           : %0d",
                         final_logit_count);
                $display("Predicted class            : %0d", class_index);
                $display("Winning logit              : %0d", winning_logit);
                $display("Conv1 max |RTL-golden|     : %0d", conv1_max_diff);
                $display("DS-Conv2 max |RTL-golden|  : %0d", conv2_max_diff);
                $display("Pool1 max |RTL-golden|     : %0d", pool1_max_diff);
                $display("==================================================");

                if (final_logit_count != EXPECTED_LOGITS) begin
                    $display("FAIL: expected %0d logits, received %0d.",
                             EXPECTED_LOGITS, final_logit_count);
                    $fatal(1);
                end else if ((conv3_cycle - conv2_cycle)
                          >= DS_CONV2_BASELINE_CYCLES) begin
                    $display("FAIL: DS-Conv2 cycle count did not improve.");
                    $fatal(1);
                end else if (class_index != EXPECTED_CLASS) begin
                    $display("FAIL: expected class %0d, received %0d.",
                             EXPECTED_CLASS, class_index);
                    $fatal(1);
                end else begin
                    $display("PASS: DS inference, class, and cycle count are valid.");
                end
            end

            if (measuring && ((cycle_number - start_cycle) > MAX_CYCLES)) begin
                $display("FAIL: cycle profiler timeout after %0d cycles.",
                         cycle_number - start_cycle);
                $fatal(1);
            end
        end
    end
endmodule
