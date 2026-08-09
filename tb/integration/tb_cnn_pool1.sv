`timescale 1ns/1ps

// ==================================================
// Verifies:
// RAM A -> Conv1 -> BN1 -> ReLU1 -> RAM B
// RAM B -> Conv2 -> BN2 -> ReLU2 -> RAM A
// RAM A -> MaxPool1 -> RAM B
// ==================================================

module tb_cnn_pool1;
    localparam int SAMPLES = 4;
    localparam int POOL1_SIZE = 19 * 18 * 20;
    localparam int TOLERANCE = 2;
    localparam int MAX_CYCLES = 20_000_000;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic pool1_output_valid;
    logic [31:0] pool1_output_addr;
    logic signed [15:0] pool1_output_data;
    logic [15:0] ram_a_read_addr;
    logic signed [15:0] ram_a_read_data;
    logic [15:0] ram_b_read_addr;
    logic signed [15:0] ram_b_read_data;
    logic signed [15:0] expected_mem [0:SAMPLES*POOL1_SIZE-1];

    integer stream_checked;
    integer stream_errors;
    integer stream_max_difference;
    integer ram_checked;
    integer ram_errors;
    integer ram_max_difference;
    integer difference;
    integer cycle_count;
    integer i;

    cnn_stage2_top #(
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
        .BN3_B_FILE("../mem/weights/bn3_B.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .output_valid(), .output_addr(), .output_data(),
        .stage1_output_valid(), .stage1_output_addr(),
        .stage1_output_data(),
        .pool1_output_valid(pool1_output_valid),
        .pool1_output_addr(pool1_output_addr),
        .pool1_output_data(pool1_output_data),
        .ram_a_read_addr(ram_a_read_addr),
        .ram_a_read_data(ram_a_read_data),
        .ram_b_read_addr(ram_b_read_addr),
        .ram_b_read_data(ram_b_read_data)
    );

    always #5 clk = ~clk;

    function automatic integer absolute_difference(
        input logic signed [15:0] actual,
        input logic signed [15:0] expected
    );
        integer value;
        begin
            value = $signed(actual) - $signed(expected);
            if (value < 0)
                value = -value;
            absolute_difference = value;
        end
    endfunction

    task automatic require_file(input string file_name);
        integer file_handle;
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0) begin
                $display("FATAL: cannot open required file: %s", file_name);
                $fatal(1);
            end
            $fclose(file_handle);
        end
    endtask

    initial begin
        require_file("../mem/golden/q_in_act.mem");
        require_file("../mem/golden/q_pool1_act.mem");
        require_file("../mem/weights/conv1_W.mem");
        require_file("../mem/weights/conv1_b.mem");
        require_file("../mem/weights/bn1_A.mem");
        require_file("../mem/weights/bn1_B.mem");
        require_file("../mem/weights/conv2_W.mem");
        require_file("../mem/weights/conv2_b.mem");
        require_file("../mem/weights/bn2_A.mem");
        require_file("../mem/weights/bn2_B.mem");
        $readmemh("../mem/golden/q_pool1_act.mem", expected_mem);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        ram_a_read_addr = '0;
        ram_b_read_addr = '0;
        stream_checked = 0;
        stream_errors = 0;
        stream_max_difference = 0;
        ram_checked = 0;
        ram_errors = 0;
        ram_max_difference = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // done is asserted only after the last Pool1 result is written.
        wait (done);
        repeat (2) @(posedge clk);

        // Verify that every Pool1 result was actually stored in RAM B.
        for (i = 0; i < POOL1_SIZE; i = i + 1) begin
            @(negedge clk);
            ram_b_read_addr = i[15:0];
            @(posedge clk);
            #1;

            if ($isunknown(ram_b_read_data)) begin
                if (ram_errors < 20)
                    $display("RAM B X/Z at addr=%0d", i);
                ram_errors = ram_errors + 1;
            end else begin
                difference = absolute_difference(ram_b_read_data,
                                                 expected_mem[i]);
                if (difference > ram_max_difference)
                    ram_max_difference = difference;
                if (difference > TOLERANCE) begin
                    if (ram_errors < 20)
                        $display("RAM B mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 i, ram_b_read_data,
                                 expected_mem[i], difference);
                    ram_errors = ram_errors + 1;
                end
            end
            ram_checked = ram_checked + 1;
        end

        if ((stream_checked == POOL1_SIZE) &&
            (ram_checked == POOL1_SIZE) &&
            (stream_errors == 0) && (ram_errors == 0)) begin
            $display("PASS: CNN through Pool1 stored correctly in RAM B.");
            $display("Pool1 stream checked=%0d max_diff=%0d",
                     stream_checked, stream_max_difference);
            $display("RAM B checked=%0d max_diff=%0d",
                     ram_checked, ram_max_difference);
        end else begin
            $display("FAIL: CNN-to-Pool1 integration.");
            $display("Stream checked=%0d errors=%0d max_diff=%0d",
                     stream_checked, stream_errors,
                     stream_max_difference);
            $display("RAM B checked=%0d errors=%0d max_diff=%0d",
                     ram_checked, ram_errors, ram_max_difference);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (pool1_output_valid) begin
            if ($isunknown(pool1_output_addr) ||
                $isunknown(pool1_output_data) ||
                (pool1_output_addr >= POOL1_SIZE)) begin
                stream_errors = stream_errors + 1;
            end else begin
                difference = absolute_difference(
                    pool1_output_data, expected_mem[pool1_output_addr]
                );
                if (difference > stream_max_difference)
                    stream_max_difference = difference;
                if (difference > TOLERANCE) begin
                    if (stream_errors < 20)
                        $display("Pool1 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 pool1_output_addr, pool1_output_data,
                                 expected_mem[pool1_output_addr], difference);
                    stream_errors = stream_errors + 1;
                end
            end
            stream_checked = stream_checked + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_cnn_pool1 timeout at cycle %0d", cycle_count);
            $finish;
        end
    end
endmodule
