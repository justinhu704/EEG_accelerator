`timescale 1ns/1ps

module tb_streaming_maxpool2;
    localparam int IN_H        = 18;
    localparam int IN_W        = 14;
    localparam int IN_CH       = 15;
    localparam int LANES       = 3;
    localparam int OUT_H       = 18;
    localparam int OUT_W       = 1;
    localparam int INPUT_SIZE  = IN_H * IN_W * IN_CH;
    localparam int OUTPUT_SIZE = OUT_H * OUT_W * IN_CH;
    localparam int OUT_GROUPS  = IN_CH / LANES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic busy, done;
    logic input_valid = 1'b0;
    logic signed [15:0] input_data = '0;
    logic output_valid;
    logic [12:0] output_addr;
    logic signed [15:0] output_data;

    logic signed [15:0] input_mem [0:INPUT_SIZE-1];
    logic signed [15:0] expected_mem [0:OUTPUT_SIZE-1];
    logic seen [0:OUTPUT_SIZE-1];
    integer group_index, w_index, h_index, lane_index, channel_index;
    integer input_addr, checked_count, error_count, difference;
    integer input_file, expected_file, scan_status, i;

    always #5 clk = ~clk;

    streaming_maxpool #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .POOL_W(10), .STRIDE_W(8), .LANES(LANES),
        .INPUT_F(12), .OUTPUT_F(13)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_valid(input_valid), .input_data(input_data),
        .output_valid(output_valid), .output_addr(output_addr),
        .output_data(output_data)
    );

    initial begin
        input_file = $fopen("mem/golden/q_relu3_act.mem", "r");
        if (input_file == 0) $fatal(1, "Cannot open q_relu3_act.mem");
        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
            scan_status = $fscanf(input_file, "%h", input_mem[i]);
            if (scan_status != 1) $fatal(1, "ReLU3 ended at %0d", i);
        end
        $fclose(input_file);

        expected_file = $fopen("mem/golden/q_pool2_act.mem", "r");
        if (expected_file == 0) $fatal(1, "Cannot open q_pool2_act.mem");
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            scan_status = $fscanf(expected_file, "%h", expected_mem[i]);
            if (scan_status != 1) $fatal(1, "Pool2 ended at %0d", i);
            seen[i] = 1'b0;
        end
        $fclose(expected_file);
        checked_count = 0;
        error_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        for (group_index = 0; group_index < OUT_GROUPS;
             group_index = group_index + 1)
            for (w_index = 0; w_index < IN_W; w_index = w_index + 1)
                for (h_index = 0; h_index < IN_H; h_index = h_index + 1)
                    for (lane_index = 0; lane_index < LANES;
                         lane_index = lane_index + 1) begin
                        channel_index = group_index * LANES + lane_index;
                        input_addr = h_index + IN_H *
                            (w_index + IN_W * channel_index);
                        @(negedge clk);
                        input_valid = 1'b1;
                        input_data = input_mem[input_addr];
                    end

        @(negedge clk); input_valid = 1'b0;
        wait (done);
        @(posedge clk);
        if ((error_count == 0) && (checked_count == OUTPUT_SIZE))
            $display("PASS: Streaming Pool2 exactly matches MATLAB (%0d values).",
                     checked_count);
        else
            $fatal(1, "Pool2 failed: checked=%0d errors=%0d",
                   checked_count, error_count);
        $finish;
    end

    always @(posedge clk) begin
        if (output_valid) begin
            if ((output_addr >= OUTPUT_SIZE) || seen[output_addr]) begin
                error_count = error_count + 1;
            end else begin
                seen[output_addr] = 1'b1;
                difference = $signed(output_data)
                           - $signed(expected_mem[output_addr]);
                if (difference != 0) begin
                    if (error_count < 20)
                        $display("MISMATCH addr=%0d got=%0d expected=%0d",
                                 output_addr, output_data,
                                 expected_mem[output_addr]);
                    error_count = error_count + 1;
                end
                checked_count = checked_count + 1;
            end
        end
    end
endmodule
