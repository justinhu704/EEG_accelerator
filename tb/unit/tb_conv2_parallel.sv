`timescale 1ns/1ps

module tb_conv2_parallel;
    localparam int OUTPUT_SIZE = 19 * 152 * 20;
    localparam int MAX_CYCLES  = 5_000_000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic busy;
    logic done;
    logic output_valid;
    logic [31:0] output_addr;
    logic [31:0] output_ch_idx;
    logic signed [15:0] output_data;

    logic signed [15:0] expected_mem [0:OUTPUT_SIZE-1];
    logic seen [0:OUTPUT_SIZE-1];
    integer checked_count;
    integer exact_count;
    integer one_lsb_count;
    integer error_count;
    integer cycle_count;
    integer difference;
    integer i;

    always #5 clk = ~clk;

    conv_layer_parallel_mem #(
        .IN_H(20), .IN_W(156), .IN_CH(21),
        .K_H(2), .K_W(5), .OUT_CH(20), .LANES(4),
        .BIAS_SHIFT(10), .OUTPUT_SHIFT(15),
        .INPUT_FILE("../mem/golden/q_relu1_act.mem"),
        .PACKED_WEIGHT_FILE("../mem/weights/conv2_W_x4.mem"),
        .PACKED_BIAS_FILE("../mem/weights/conv2_b_x4.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .output_valid(output_valid), .output_addr(output_addr),
        .output_ch_idx(output_ch_idx), .output_data(output_data)
    );

    initial begin
        $readmemh("../mem/golden/q_conv2_act.mem", expected_mem);
        for (i = 0; i < OUTPUT_SIZE; i = i + 1)
            seen[i] = 1'b0;

        checked_count = 0;
        exact_count = 0;
        one_lsb_count = 0;
        error_count = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done);
        @(posedge clk);

        if ((error_count == 0) && (checked_count == OUTPUT_SIZE)) begin
            $display("PASS: 4-lane Conv2 matches MATLAB within 1 LSB.");
            $display("Checked outputs = %0d", checked_count);
            $display("Exact matches   = %0d", exact_count);
            $display("One-LSB cases   = %0d", one_lsb_count);
            $display("Parallel cycles = %0d", cycle_count);
        end else begin
            $fatal(1, "Parallel Conv2 failed: checked=%0d errors=%0d",
                   checked_count, error_count);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data) ||
                (output_addr >= OUTPUT_SIZE)) begin
                if (error_count < 20)
                    $display("INVALID output addr=%0d data=%0d", output_addr, output_data);
                error_count = error_count + 1;
            end else if (seen[output_addr]) begin
                if (error_count < 20)
                    $display("DUPLICATE output addr=%0d", output_addr);
                error_count = error_count + 1;
            end else begin
                seen[output_addr] = 1'b1;
                difference = $signed(output_data) - $signed(expected_mem[output_addr]);
                if (difference < 0)
                    difference = -difference;

                if (difference == 0)
                    exact_count = exact_count + 1;
                else if (difference == 1)
                    one_lsb_count = one_lsb_count + 1;
                else begin
                    if (error_count < 20)
                        $display("MISMATCH addr=%0d ch=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_ch_idx, output_data,
                                 expected_mem[output_addr], difference);
                    error_count = error_count + 1;
                end
                checked_count = checked_count + 1;
            end
        end

        if (cycle_count > MAX_CYCLES)
            $fatal(1, "Parallel Conv2 timeout at cycle %0d", cycle_count);
    end
endmodule
