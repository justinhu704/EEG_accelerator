`timescale 1ns/1ps

module tb_streaming_maxpool1;
    localparam int IN_H        = 19;
    localparam int IN_W        = 152;
    localparam int IN_CH       = 20;
    localparam int LANES       = 4;
    localparam int POOL_W      = 10;
    localparam int STRIDE_W    = 8;
    localparam int OUT_H       = IN_H;
    localparam int OUT_W       = ((IN_W - POOL_W) / STRIDE_W) + 1;
    localparam int INPUT_SIZE  = IN_H * IN_W * IN_CH;
    localparam int OUTPUT_SIZE = OUT_H * OUT_W * IN_CH;
    localparam int OUT_GROUPS  = IN_CH / LANES;
    localparam int MAX_CYCLES  = 100_000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic busy;
    logic done;

    logic input_valid = 1'b0;
    logic signed [15:0] input_data = '0;

    logic output_valid;
    logic [12:0] output_addr;
    logic signed [15:0] output_data;

    logic signed [15:0] input_mem    [0:INPUT_SIZE-1];
    logic signed [15:0] expected_mem [0:OUTPUT_SIZE-1];
    logic seen [0:OUTPUT_SIZE-1];

    integer group_index;
    integer w_index;
    integer h_index;
    integer lane_index;
    integer channel_index;
    integer input_addr;
    integer checked_count;
    integer exact_count;
    integer error_count;
    integer max_difference;
    integer difference;
    integer cycle_count;
    integer input_file;
    integer expected_file;
    integer scan_status;
    integer i;

    always #5 clk = ~clk;

    streaming_maxpool1 #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .POOL_W(POOL_W), .STRIDE_W(STRIDE_W),
        .LANES(LANES), .OUT_H(OUT_H), .OUT_W(OUT_W)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_valid(input_valid), .input_data(input_data),
        .output_valid(output_valid), .output_addr(output_addr),
        .output_data(output_data)
    );

    initial begin
        // The exported golden files currently contain multiple samples.
        // Read exactly the first sample so Questa does not report the benign
        // "too many data words" warning produced by $readmemh.
        input_file = $fopen("mem/golden/q_relu2_act.mem", "r");
        if (input_file == 0)
            $fatal(1, "Cannot open mem/golden/q_relu2_act.mem");
        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
            scan_status = $fscanf(input_file, "%h", input_mem[i]);
            if (scan_status != 1)
                $fatal(1, "ReLU2 input file ended at word %0d", i);
        end
        $fclose(input_file);

        expected_file = $fopen("mem/golden/q_pool1_act.mem", "r");
        if (expected_file == 0)
            $fatal(1, "Cannot open mem/golden/q_pool1_act.mem");
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            scan_status = $fscanf(expected_file, "%h", expected_mem[i]);
            if (scan_status != 1)
                $fatal(1, "Pool1 golden file ended at word %0d", i);
        end
        $fclose(expected_file);

        for (i = 0; i < OUTPUT_SIZE; i = i + 1)
            seen[i] = 1'b0;

        checked_count = 0;
        exact_count = 0;
        error_count = 0;
        max_difference = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // Reproduce the real serialized 4-lane Conv2/ReLU2 stream:
        // channel group -> width -> height -> lane.
        for (group_index = 0; group_index < OUT_GROUPS;
             group_index = group_index + 1) begin
            for (w_index = 0; w_index < IN_W; w_index = w_index + 1) begin
                for (h_index = 0; h_index < IN_H; h_index = h_index + 1) begin
                    for (lane_index = 0; lane_index < LANES;
                         lane_index = lane_index + 1) begin
                        channel_index = group_index * LANES + lane_index;
                        input_addr = h_index
                                   + IN_H * (w_index + IN_W * channel_index);
                        @(negedge clk);
                        input_valid = 1'b1;
                        input_data = input_mem[input_addr];
                    end
                end
            end
        end

        @(negedge clk);
        input_valid = 1'b0;
        input_data = '0;

        wait (done);
        @(posedge clk);

        if ((error_count == 0) &&
            (checked_count == OUTPUT_SIZE) &&
            (exact_count == OUTPUT_SIZE)) begin
            $display("PASS: Streaming Pool1 exactly matches MATLAB.");
            $display("Input values   = %0d", INPUT_SIZE);
            $display("Checked outputs= %0d", checked_count);
            $display("Exact matches  = %0d", exact_count);
            $display("Maximum diff   = %0d", max_difference);
            $display("Simulation cycles = %0d", cycle_count);
        end else begin
            $fatal(1,
                   "Streaming Pool1 failed: checked=%0d exact=%0d errors=%0d max_diff=%0d",
                   checked_count, exact_count, error_count, max_difference);
        end

        $finish;
    end

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data) ||
                (output_addr >= OUTPUT_SIZE)) begin
                if (error_count < 20)
                    $display("INVALID output addr=%0d data=%0d",
                             output_addr, output_data);
                error_count = error_count + 1;
            end else if (seen[output_addr]) begin
                if (error_count < 20)
                    $display("DUPLICATE output addr=%0d", output_addr);
                error_count = error_count + 1;
            end else begin
                seen[output_addr] = 1'b1;
                difference = $signed(output_data)
                           - $signed(expected_mem[output_addr]);
                if (difference < 0)
                    difference = -difference;

                if (difference > max_difference)
                    max_difference = difference;

                if (difference == 0) begin
                    exact_count = exact_count + 1;
                end else begin
                    if (error_count < 20)
                        $display("MISMATCH addr=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_data,
                                 expected_mem[output_addr], difference);
                    error_count = error_count + 1;
                end

                checked_count = checked_count + 1;
            end
        end

        if (cycle_count > MAX_CYCLES)
            $fatal(1, "Streaming Pool1 timeout at cycle %0d", cycle_count);
    end
endmodule
