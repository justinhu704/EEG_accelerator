`timescale 1ns/1ps

module tb_conv2;
    localparam int OUTPUT_SIZE = 19 * 152 * 20;
    localparam int MAX_CYCLES  = 30_000_000;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic output_valid;
    logic [31:0] output_addr;
    logic [31:0] output_ch_idx;
    logic signed [15:0] output_data;

    logic signed [15:0] expected_mem [0:OUTPUT_SIZE-1];

    integer checked_count;
    integer error_count;
    integer one_lsb_count;
    integer cycle_count;
    integer difference;
    integer file_handle;

    task automatic require_file(input string file_name);
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0) begin
                $display("FATAL: cannot open required file: %s", file_name);
                $display("Run the simulation with the working directory set to C:/EEG_Project/questa.");
                $fatal(1);
            end
            $fclose(file_handle);
        end
    endtask

    conv_layer_mem #(
        .IN_H         (20),
        .IN_W         (156),
        .IN_CH        (21),
        .K_H          (2),
        .K_W          (5),
        .OUT_CH       (20),
        .BIAS_SHIFT   (10),
        .OUTPUT_SHIFT (15),
        .INPUT_FILE   ("../mem/golden/q_relu1_act.mem"),
        .WEIGHT_FILE  ("../mem/weights/conv2_W.mem"),
        .BIAS_FILE    ("../mem/weights/conv2_b.mem")
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .output_valid (output_valid),
        .output_addr  (output_addr),
        .output_ch_idx(output_ch_idx),
        .output_data  (output_data)
    );

    always #5 clk = ~clk;

    initial begin
        require_file("../mem/golden/q_relu1_act.mem");
        require_file("../mem/weights/conv2_W.mem");
        require_file("../mem/weights/conv2_b.mem");
        require_file("../mem/golden/q_conv2_act.mem");

        $readmemh("../mem/golden/q_conv2_act.mem",
                  expected_mem, 0, OUTPUT_SIZE-1);

        clk           = 0;
        rst_n         = 0;
        start         = 0;
        checked_count = 0;
        error_count   = 0;
        one_lsb_count = 0;
        cycle_count   = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1;

        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        wait (done);
        @(posedge clk);

        if ((error_count == 0) && (checked_count == OUTPUT_SIZE)) begin
            $display("PASS: Conv2 matches MATLAB within 1 LSB.");
            $display("Checked outputs = %0d", checked_count);
            $display("Exact matches   = %0d", checked_count-one_lsb_count);
            $display("One-LSB cases   = %0d", one_lsb_count);
        end else begin
            $display("FAIL: Conv2 verification failed.");
            $display("Checked=%0d expected_count=%0d errors=%0d one_lsb=%0d",
                     checked_count, OUTPUT_SIZE, error_count, one_lsb_count);
        end

        $finish;
    end

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data)) begin
                if (error_count < 20)
                    $display("MISMATCH: DUT produced X/Z at output %0d",
                             checked_count);
                error_count = error_count + 1;
            end else if (output_addr >= OUTPUT_SIZE) begin
                $display("ERROR: output_addr out of range: %0d", output_addr);
                error_count = error_count + 1;
            end else if ($isunknown(expected_mem[output_addr])) begin
                if (error_count < 20)
                    $display("MISMATCH: golden value is X/Z at addr=%0d",
                             output_addr);
                error_count = error_count + 1;
            end else if (output_data !== expected_mem[output_addr]) begin
                difference = $signed(output_data)
                           - $signed(expected_mem[output_addr]);

                if (difference < 0)
                    difference = -difference;

                if (difference == 1) begin
                    one_lsb_count = one_lsb_count + 1;
                end else begin
                    if (error_count < 20)
                        $display("MISMATCH addr=%0d got=%0d (0x%h) expected=%0d (0x%h) diff=%0d",
                                 output_addr, output_data, output_data,
                                 expected_mem[output_addr],
                                 expected_mem[output_addr], difference);
                    error_count = error_count + 1;
                end
            end

            checked_count = checked_count + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: Conv2 simulation timeout at cycle %0d",
                     cycle_count);
            $finish;
        end
    end
endmodule
