`timescale 1ns/1ps

// ==================================================
// Verifies RAM A -> Conv1 -> BN1 -> ReLU1 -> RAM B.
// ==================================================

module tb_cnn;
    localparam int OUTPUT_SIZE = 20 * 156 * 21;
    localparam int SAMPLES = 4;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic output_valid;
    logic [31:0] output_addr;
    logic signed [15:0] output_data;
    logic [15:0] ram_b_read_addr;
    logic signed [15:0] ram_b_read_data;

    logic signed [15:0] expected_mem [0:SAMPLES*OUTPUT_SIZE-1];

    integer stream_checked;
    integer stream_exact;
    integer stream_one_lsb;
    integer stream_two_lsb;
    integer stream_three_lsb;
    integer stream_errors;

    integer ram_checked;
    integer ram_exact;
    integer ram_one_lsb;
    integer ram_two_lsb;
    integer ram_three_lsb;
    integer ram_errors;

    integer difference;
    integer cycle_count;
    integer i;

    cnn_stage1_top #(
        .CONV_INPUT_FILE("../mem/golden/q_in_act.mem"),
        .CONV_WEIGHT_FILE("../mem/weights/conv1_W.mem"),
        .CONV_BIAS_FILE("../mem/weights/conv1_b.mem"),
        .BN_A_FILE("../mem/weights/bn1_A.mem"),
        .BN_B_FILE("../mem/weights/bn1_B.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .output_valid(output_valid),
        .output_addr(output_addr),
        .output_data(output_data),
        .ram_b_read_addr(ram_b_read_addr),
        .ram_b_read_data(ram_b_read_data)
    );

    always #5 clk = ~clk;

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
        require_file("../mem/weights/conv1_W.mem");
        require_file("../mem/weights/conv1_b.mem");
        require_file("../mem/weights/bn1_A.mem");
        require_file("../mem/weights/bn1_B.mem");
        require_file("../mem/golden/q_relu1_act.mem");
        $readmemh("../mem/golden/q_relu1_act.mem", expected_mem);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        ram_b_read_addr = '0;
        stream_checked = 0;
        stream_exact = 0;
        stream_one_lsb = 0;
        stream_two_lsb = 0;
        stream_three_lsb = 0;
        stream_errors = 0;
        ram_checked = 0;
        ram_exact = 0;
        ram_one_lsb = 0;
        ram_two_lsb = 0;
        ram_three_lsb = 0;
        ram_errors = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // done is generated only after the last ReLU1 value has been written.
        wait (done);
        repeat (2) @(posedge clk);

        // Read every RAM B address. The RAM is synchronous, so drive the
        // address before a rising edge and sample read_data after that edge.
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            @(negedge clk);
            ram_b_read_addr = i[15:0];
            @(posedge clk);
            #1;

            if ($isunknown(ram_b_read_data)) begin
                if (ram_errors < 20)
                    $display("RAM B X/Z at addr=%0d", i);
                ram_errors = ram_errors + 1;
            end else begin
                difference = $signed(ram_b_read_data)
                           - $signed(expected_mem[i]);
                if (difference < 0)
                    difference = -difference;

                if (difference == 0)
                    ram_exact = ram_exact + 1;
                else if (difference == 1)
                    ram_one_lsb = ram_one_lsb + 1;
                else if (difference == 2)
                    ram_two_lsb = ram_two_lsb + 1;
                else if (difference == 3)
                    ram_three_lsb = ram_three_lsb + 1;
                else begin
                    if (ram_errors < 20)
                        $display("RAM B mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 i, ram_b_read_data, expected_mem[i], difference);
                    ram_errors = ram_errors + 1;
                end
            end
            ram_checked = ram_checked + 1;
        end

        if ((stream_errors == 0) && (ram_errors == 0) &&
            (stream_checked == OUTPUT_SIZE) &&
            (ram_checked == OUTPUT_SIZE)) begin
            $display("PASS: RAM A -> Conv1 -> BN1 -> ReLU1 -> RAM B.");
            $display("Stream checked=%0d exact=%0d 1LSB=%0d 2LSB=%0d 3LSB=%0d",
                     stream_checked, stream_exact, stream_one_lsb,
                     stream_two_lsb, stream_three_lsb);
            $display("RAM B  checked=%0d exact=%0d 1LSB=%0d 2LSB=%0d 3LSB=%0d",
                     ram_checked, ram_exact, ram_one_lsb,
                     ram_two_lsb, ram_three_lsb);
        end else begin
            $display("FAIL: Stage-1 ping-pong verification failed.");
            $display("Stream checked=%0d errors=%0d", stream_checked, stream_errors);
            $display("RAM B checked=%0d errors=%0d", ram_checked, ram_errors);
        end
        $finish;
    end

    // Check the live ReLU1 stream while it is being written to RAM B.
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data) ||
                (output_addr >= OUTPUT_SIZE)) begin
                if (stream_errors < 20)
                    $display("Invalid stream output at item=%0d", stream_checked);
                stream_errors = stream_errors + 1;
            end else begin
                difference = $signed(output_data)
                           - $signed(expected_mem[output_addr]);
                if (difference < 0)
                    difference = -difference;

                if (difference == 0)
                    stream_exact = stream_exact + 1;
                else if (difference == 1)
                    stream_one_lsb = stream_one_lsb + 1;
                else if (difference == 2)
                    stream_two_lsb = stream_two_lsb + 1;
                else if (difference == 3)
                    stream_three_lsb = stream_three_lsb + 1;
                else begin
                    if (stream_errors < 20)
                        $display("Stream mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_data,
                                 expected_mem[output_addr], difference);
                    stream_errors = stream_errors + 1;
                end
            end
            stream_checked = stream_checked + 1;
        end

        if (cycle_count > 2000000) begin
            $display("FAIL: tb_cnn timeout");
            $finish;
        end
    end
endmodule
