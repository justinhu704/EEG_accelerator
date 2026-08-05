`timescale 1ns/1ps

// Verifies BN1, BN2, and BN3 against the first MATLAB sample.
module tb_bn;
    localparam int BN1_SIZE = 20 * 156 * 21;
    localparam int BN2_SIZE = 19 * 152 * 20;
    localparam int BN3_SIZE = 18 * 14  * 15;
    localparam int SAMPLES  = 4;
    localparam int BN1_BLOCK = 20 * 156;
    localparam int BN2_BLOCK = 19 * 152;
    localparam int BN3_BLOCK = 18 * 14;

    logic clk;
    logic rst_n;

    logic bn1_in_valid, bn2_in_valid, bn3_in_valid;
    logic signed [15:0] bn1_in_data, bn2_in_data, bn3_in_data;
    logic [31:0] bn1_ch, bn2_ch, bn3_ch;
    logic bn1_out_valid, bn2_out_valid, bn3_out_valid;
    logic signed [15:0] bn1_out_data, bn2_out_data, bn3_out_data;

    logic signed [15:0] bn1_input  [0:SAMPLES*BN1_SIZE-1];
    logic signed [15:0] bn1_golden [0:SAMPLES*BN1_SIZE-1];
    logic signed [15:0] bn2_input  [0:SAMPLES*BN2_SIZE-1];
    logic signed [15:0] bn2_golden [0:SAMPLES*BN2_SIZE-1];
    logic signed [15:0] bn3_input  [0:SAMPLES*BN3_SIZE-1];
    logic signed [15:0] bn3_golden [0:SAMPLES*BN3_SIZE-1];

    integer bn1_feed_count, bn2_feed_count, bn3_feed_count;
    integer bn1_check_count, bn2_check_count, bn3_check_count;
    integer bn1_one_lsb, bn2_one_lsb, bn3_one_lsb;
    integer error_count;
    integer difference;

    bn_affine #(
        .CHANNELS(21), .BIAS_SHIFT(11), .OUTPUT_SHIFT(13),
        .A_FILE("../mem/weights/bn1_A.mem"),
        .B_FILE("../mem/weights/bn1_B.mem")
    ) dut_bn1 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(bn1_in_valid), .in_data(bn1_in_data),
        .in_ch_idx(bn1_ch),
        .out_valid(bn1_out_valid), .out_data(bn1_out_data)
    );

    bn_affine #(
        .CHANNELS(20), .BIAS_SHIFT(11), .OUTPUT_SHIFT(14),
        .A_FILE("../mem/weights/bn2_A.mem"),
        .B_FILE("../mem/weights/bn2_B.mem")
    ) dut_bn2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(bn2_in_valid), .in_data(bn2_in_data),
        .in_ch_idx(bn2_ch),
        .out_valid(bn2_out_valid), .out_data(bn2_out_data)
    );

    bn_affine #(
        .CHANNELS(15), .BIAS_SHIFT(11), .OUTPUT_SHIFT(13),
        .A_FILE("../mem/weights/bn3_A.mem"),
        .B_FILE("../mem/weights/bn3_B.mem")
    ) dut_bn3 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(bn3_in_valid), .in_data(bn3_in_data),
        .in_ch_idx(bn3_ch),
        .out_valid(bn3_out_valid), .out_data(bn3_out_data)
    );

    always #5 clk = ~clk;

    task automatic require_file(input string file_name);
        integer file_handle;
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0) begin
                $display("FATAL: cannot open required file: %s", file_name);
                $display("Set the simulator working directory to C:/EEG_Project before starting simulation.");
                $fatal(1);
            end
            $fclose(file_handle);
        end
    endtask

    initial begin
        require_file("../mem/golden/q_conv1_act.mem");
        require_file("../mem/golden/q_bn1_act.mem");
        require_file("../mem/golden/q_conv2_act.mem");
        require_file("../mem/golden/q_bn2_act.mem");
        require_file("../mem/golden/q_conv3_act.mem");
        require_file("../mem/golden/q_bn3_act.mem");
        require_file("../mem/weights/bn1_A.mem");
        require_file("../mem/weights/bn1_B.mem");
        require_file("../mem/weights/bn2_A.mem");
        require_file("../mem/weights/bn2_B.mem");
        require_file("../mem/weights/bn3_A.mem");
        require_file("../mem/weights/bn3_B.mem");

        $readmemh("../mem/golden/q_conv1_act.mem", bn1_input);
        $readmemh("../mem/golden/q_bn1_act.mem",   bn1_golden);
        $readmemh("../mem/golden/q_conv2_act.mem", bn2_input);
        $readmemh("../mem/golden/q_bn2_act.mem",   bn2_golden);
        $readmemh("../mem/golden/q_conv3_act.mem", bn3_input);
        $readmemh("../mem/golden/q_bn3_act.mem",   bn3_golden);

        clk = 1'b0;
        rst_n = 1'b0;
        bn1_in_valid = 1'b0;
        bn2_in_valid = 1'b0;
        bn3_in_valid = 1'b0;
        bn1_in_data = '0;
        bn2_in_data = '0;
        bn3_in_data = '0;
        bn1_ch = '0;
        bn2_ch = '0;
        bn3_ch = '0;
        bn1_feed_count = 0;
        bn2_feed_count = 0;
        bn3_feed_count = 0;
        bn1_check_count = 0;
        bn2_check_count = 0;
        bn3_check_count = 0;
        bn1_one_lsb = 0;
        bn2_one_lsb = 0;
        bn3_one_lsb = 0;
        error_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        wait ((bn1_check_count == BN1_SIZE) &&
              (bn2_check_count == BN2_SIZE) &&
              (bn3_check_count == BN3_SIZE));
        repeat (2) @(posedge clk);

        if (error_count == 0) begin
            $display("PASS: BN1/BN2/BN3 match MATLAB within 1 LSB.");
            $display("BN1 checked=%0d exact=%0d one-LSB=%0d",
                     bn1_check_count, bn1_check_count-bn1_one_lsb,
                     bn1_one_lsb);
            $display("BN2 checked=%0d exact=%0d one-LSB=%0d",
                     bn2_check_count, bn2_check_count-bn2_one_lsb,
                     bn2_one_lsb);
            $display("BN3 checked=%0d exact=%0d one-LSB=%0d",
                     bn3_check_count, bn3_check_count-bn3_one_lsb,
                     bn3_one_lsb);
        end else begin
            $display("FAIL: BN verification errors=%0d", error_count);
        end
        $finish;
    end

    // Feed the first sample of all three layers in parallel.
    always @(negedge clk) begin
        if (!rst_n) begin
            bn1_in_valid = 1'b0;
            bn2_in_valid = 1'b0;
            bn3_in_valid = 1'b0;
        end else begin
            if (bn1_feed_count < BN1_SIZE) begin
                bn1_in_valid = 1'b1;
                bn1_in_data  = bn1_input[bn1_feed_count];
                bn1_ch       = bn1_feed_count / BN1_BLOCK;
                bn1_feed_count = bn1_feed_count + 1;
            end else begin
                bn1_in_valid = 1'b0;
            end

            if (bn2_feed_count < BN2_SIZE) begin
                bn2_in_valid = 1'b1;
                bn2_in_data  = bn2_input[bn2_feed_count];
                bn2_ch       = bn2_feed_count / BN2_BLOCK;
                bn2_feed_count = bn2_feed_count + 1;
            end else begin
                bn2_in_valid = 1'b0;
            end

            if (bn3_feed_count < BN3_SIZE) begin
                bn3_in_valid = 1'b1;
                bn3_in_data  = bn3_input[bn3_feed_count];
                bn3_ch       = bn3_feed_count / BN3_BLOCK;
                bn3_feed_count = bn3_feed_count + 1;
            end else begin
                bn3_in_valid = 1'b0;
            end
        end
    end

    // Compare outputs in stream order.
    always @(posedge clk) begin
        if (bn1_out_valid) begin
            if ($isunknown(bn1_out_data)) begin
                error_count = error_count + 1;
            end else begin
                difference = $signed(bn1_out_data)
                           - $signed(bn1_golden[bn1_check_count]);
                if (difference < 0) difference = -difference;
                if (difference == 1)
                    bn1_one_lsb = bn1_one_lsb + 1;
                else if (difference > 1) begin
                    if (error_count < 20)
                        $display("BN1 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 bn1_check_count, bn1_out_data,
                                 bn1_golden[bn1_check_count], difference);
                    error_count = error_count + 1;
                end
            end
            bn1_check_count = bn1_check_count + 1;
        end

        if (bn2_out_valid) begin
            if ($isunknown(bn2_out_data)) begin
                error_count = error_count + 1;
            end else begin
                difference = $signed(bn2_out_data)
                           - $signed(bn2_golden[bn2_check_count]);
                if (difference < 0) difference = -difference;
                if (difference == 1)
                    bn2_one_lsb = bn2_one_lsb + 1;
                else if (difference > 1) begin
                    if (error_count < 20)
                        $display("BN2 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 bn2_check_count, bn2_out_data,
                                 bn2_golden[bn2_check_count], difference);
                    error_count = error_count + 1;
                end
            end
            bn2_check_count = bn2_check_count + 1;
        end

        if (bn3_out_valid) begin
            if ($isunknown(bn3_out_data)) begin
                error_count = error_count + 1;
            end else begin
                difference = $signed(bn3_out_data)
                           - $signed(bn3_golden[bn3_check_count]);
                if (difference < 0) difference = -difference;
                if (difference == 1)
                    bn3_one_lsb = bn3_one_lsb + 1;
                else if (difference > 1) begin
                    if (error_count < 20)
                        $display("BN3 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 bn3_check_count, bn3_out_data,
                                 bn3_golden[bn3_check_count], difference);
                    error_count = error_count + 1;
                end
            end
            bn3_check_count = bn3_check_count + 1;
        end
    end

    initial begin
        #2000000;
        $display("FAIL: tb_bn timeout");
        $finish;
    end
endmodule
