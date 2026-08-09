`timescale 1ns/1ps

// ==================================================
// Verifies the complete two-stage ping-pong path:
// RAM A -> Conv1 -> BN1 -> ReLU1 -> RAM B
// RAM B -> Conv2 -> BN2 -> ReLU2 -> RAM A
// ==================================================

module tb_cnn_stage2;
    localparam int STAGE1_SIZE = 20 * 156 * 21;
    localparam int STAGE2_SIZE = 19 * 152 * 20;
    localparam int SAMPLES = 4;
    localparam int MAX_CYCLES = 20_000_000;
    localparam int STAGE1_TOLERANCE = 3;
    localparam int STAGE2_TOLERANCE = 2;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic output_valid;
    logic [31:0] output_addr;
    logic signed [15:0] output_data;
    logic stage1_output_valid;
    logic [31:0] stage1_output_addr;
    logic signed [15:0] stage1_output_data;
    logic [15:0] ram_a_read_addr;
    logic signed [15:0] ram_a_read_data;
    logic [15:0] ram_b_read_addr;
    logic signed [15:0] ram_b_read_data;

    logic signed [15:0] stage1_expected [0:SAMPLES*STAGE1_SIZE-1];
    logic signed [15:0] stage2_expected [0:SAMPLES*STAGE2_SIZE-1];

    integer stage1_checked;
    integer stage1_errors;
    integer stage1_max_difference;
    integer stage2_checked;
    integer stage2_errors;
    integer stage2_max_difference;
    integer ram_a_checked;
    integer ram_a_errors;
    integer ram_a_max_difference;
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
        .output_valid(output_valid), .output_addr(output_addr),
        .output_data(output_data),
        .stage1_output_valid(stage1_output_valid),
        .stage1_output_addr(stage1_output_addr),
        .stage1_output_data(stage1_output_data),
        .ram_a_read_addr(ram_a_read_addr),
        .ram_a_read_data(ram_a_read_data),
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

    initial begin
        require_file("../mem/golden/q_in_act.mem");
        require_file("../mem/golden/q_relu1_act.mem");
        require_file("../mem/golden/q_relu2_act.mem");
        require_file("../mem/weights/conv1_W.mem");
        require_file("../mem/weights/conv1_b.mem");
        require_file("../mem/weights/bn1_A.mem");
        require_file("../mem/weights/bn1_B.mem");
        require_file("../mem/weights/conv2_W.mem");
        require_file("../mem/weights/conv2_b.mem");
        require_file("../mem/weights/bn2_A.mem");
        require_file("../mem/weights/bn2_B.mem");

        $readmemh("../mem/golden/q_relu1_act.mem", stage1_expected);
        $readmemh("../mem/golden/q_relu2_act.mem", stage2_expected);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        ram_a_read_addr = '0;
        ram_b_read_addr = '0;
        stage1_checked = 0;
        stage1_errors = 0;
        stage1_max_difference = 0;
        stage2_checked = 0;
        stage2_errors = 0;
        stage2_max_difference = 0;
        ram_a_checked = 0;
        ram_a_errors = 0;
        ram_a_max_difference = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // done means the final ReLU2 value has been committed to RAM A.
        wait (done);
        repeat (2) @(posedge clk);

        // Read every final result back from RAM A. Its read is synchronous.
        for (i = 0; i < STAGE2_SIZE; i = i + 1) begin
            @(negedge clk);
            ram_a_read_addr = i[15:0];
            @(posedge clk);
            #1;

            if ($isunknown(ram_a_read_data)) begin
                if (ram_a_errors < 20)
                    $display("RAM A X/Z at addr=%0d", i);
                ram_a_errors = ram_a_errors + 1;
            end else begin
                difference = absolute_difference(ram_a_read_data,
                                                 stage2_expected[i]);
                if (difference > ram_a_max_difference)
                    ram_a_max_difference = difference;
                if (difference > STAGE2_TOLERANCE) begin
                    if (ram_a_errors < 20)
                        $display("RAM A mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 i, ram_a_read_data,
                                 stage2_expected[i], difference);
                    ram_a_errors = ram_a_errors + 1;
                end
            end
            ram_a_checked = ram_a_checked + 1;
        end

        if ((stage1_checked == STAGE1_SIZE) &&
            (stage2_checked == STAGE2_SIZE) &&
            (ram_a_checked == STAGE2_SIZE) &&
            (stage1_errors == 0) && (stage2_errors == 0) &&
            (ram_a_errors == 0)) begin
            $display("PASS: two-stage ping-pong CNN integration.");
            $display("Stage1 stream checked=%0d max_diff=%0d",
                     stage1_checked, stage1_max_difference);
            $display("Stage2 stream checked=%0d max_diff=%0d",
                     stage2_checked, stage2_max_difference);
            $display("RAM A final checked=%0d max_diff=%0d",
                     ram_a_checked, ram_a_max_difference);
        end else begin
            $display("FAIL: two-stage ping-pong CNN integration.");
            $display("Stage1 checked=%0d errors=%0d max_diff=%0d",
                     stage1_checked, stage1_errors,
                     stage1_max_difference);
            $display("Stage2 checked=%0d errors=%0d max_diff=%0d",
                     stage2_checked, stage2_errors,
                     stage2_max_difference);
            $display("RAM A checked=%0d errors=%0d max_diff=%0d",
                     ram_a_checked, ram_a_errors,
                     ram_a_max_difference);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (stage1_output_valid) begin
            if ($isunknown(stage1_output_addr) ||
                $isunknown(stage1_output_data) ||
                (stage1_output_addr >= STAGE1_SIZE)) begin
                stage1_errors = stage1_errors + 1;
            end else begin
                difference = absolute_difference(
                    stage1_output_data,
                    stage1_expected[stage1_output_addr]
                );
                if (difference > stage1_max_difference)
                    stage1_max_difference = difference;
                if (difference > STAGE1_TOLERANCE) begin
                    if (stage1_errors < 20)
                        $display("Stage1 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 stage1_output_addr, stage1_output_data,
                                 stage1_expected[stage1_output_addr],
                                 difference);
                    stage1_errors = stage1_errors + 1;
                end
            end
            stage1_checked = stage1_checked + 1;
        end

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data) ||
                (output_addr >= STAGE2_SIZE)) begin
                stage2_errors = stage2_errors + 1;
            end else begin
                difference = absolute_difference(
                    output_data, stage2_expected[output_addr]
                );
                if (difference > stage2_max_difference)
                    stage2_max_difference = difference;
                if (difference > STAGE2_TOLERANCE) begin
                    if (stage2_errors < 20)
                        $display("Stage2 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_data,
                                 stage2_expected[output_addr], difference);
                    stage2_errors = stage2_errors + 1;
                end
            end
            stage2_checked = stage2_checked + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_cnn_stage2 timeout at cycle %0d", cycle_count);
            $finish;
        end
    end
endmodule
