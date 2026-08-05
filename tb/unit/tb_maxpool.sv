`timescale 1ns/1ps

// Verifies Pool1 and Pool2 against the first MATLAB sample.
module tb_maxpool;
    localparam int SAMPLES = 4;

    localparam int POOL1_INPUT_SIZE  = 19 * 152 * 20;
    localparam int POOL1_OUTPUT_SIZE = 19 * 18  * 20;
    localparam int POOL2_INPUT_SIZE  = 18 * 14  * 15;
    localparam int POOL2_OUTPUT_SIZE = 18 * 1   * 15;

    logic clk;
    logic rst_n;
    logic pool1_start;
    logic pool2_start;

    logic pool1_busy;
    logic pool1_done;
    logic pool1_output_valid;
    logic [31:0] pool1_output_addr;
    logic signed [15:0] pool1_output_data;

    logic pool2_busy;
    logic pool2_done;
    logic pool2_output_valid;
    logic [31:0] pool2_output_addr;
    logic signed [15:0] pool2_output_data;

    logic signed [15:0] pool1_golden
        [0:SAMPLES*POOL1_OUTPUT_SIZE-1];
    logic signed [15:0] pool2_golden
        [0:SAMPLES*POOL2_OUTPUT_SIZE-1];

    integer pool1_checked;
    integer pool2_checked;
    integer error_count;
    logic pool1_done_seen;
    logic pool2_done_seen;

    // Pool1: ReLU2 F12 -> Pool1 F12, so no format shift.
    maxpool_engine #(
        .IN_H(19),
        .IN_W(152),
        .IN_CH(20),
        .POOL_H(1),
        .POOL_W(10),
        .STRIDE_H(1),
        .STRIDE_W(8),
        .INPUT_F(fixed_point_pkg::RELU2_OUT_F),
        .OUTPUT_F(fixed_point_pkg::POOL1_OUT_F),
        .INPUT_DEPTH(SAMPLES*POOL1_INPUT_SIZE),
        .INPUT_FILE("../mem/golden/q_relu2_act.mem")
    ) dut_pool1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(pool1_start),
        .busy(pool1_busy),
        .done(pool1_done),
        .output_valid(pool1_output_valid),
        .output_addr(pool1_output_addr),
        .output_data(pool1_output_data)
    );

    // Pool2: ReLU3 F13 -> Pool2 F14, so the maximum is shifted left once.
    maxpool_engine #(
        .IN_H(18),
        .IN_W(14),
        .IN_CH(15),
        .POOL_H(1),
        .POOL_W(10),
        .STRIDE_H(1),
        .STRIDE_W(8),
        .INPUT_F(fixed_point_pkg::RELU3_OUT_F),
        .OUTPUT_F(fixed_point_pkg::POOL2_OUT_F),
        .INPUT_DEPTH(SAMPLES*POOL2_INPUT_SIZE),
        .INPUT_FILE("../mem/golden/q_relu3_act.mem")
    ) dut_pool2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(pool2_start),
        .busy(pool2_busy),
        .done(pool2_done),
        .output_valid(pool2_output_valid),
        .output_addr(pool2_output_addr),
        .output_data(pool2_output_data)
    );

    always #5 clk = ~clk;

    task automatic require_file(input string file_name);
        integer file_handle;
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0) begin
                $display("FATAL: cannot open required file: %s", file_name);
                $display("Set the simulator working directory to C:/EEG_Project/questa before starting simulation.");
                $fatal(1);
            end
            $fclose(file_handle);
        end
    endtask

    initial begin
        require_file("../mem/golden/q_relu2_act.mem");
        require_file("../mem/golden/q_pool1_act.mem");
        require_file("../mem/golden/q_relu3_act.mem");
        require_file("../mem/golden/q_pool2_act.mem");

        $readmemh("../mem/golden/q_pool1_act.mem", pool1_golden);
        $readmemh("../mem/golden/q_pool2_act.mem", pool2_golden);

        clk = 1'b0;
        rst_n = 1'b0;
        pool1_start = 1'b0;
        pool2_start = 1'b0;
        pool1_checked = 0;
        pool2_checked = 0;
        error_count = 0;
        pool1_done_seen = 1'b0;
        pool2_done_seen = 1'b0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // A one-clock start pulse launches both engines.
        @(negedge clk);
        pool1_start = 1'b1;
        pool2_start = 1'b1;
        @(negedge clk);
        pool1_start = 1'b0;
        pool2_start = 1'b0;

        wait ((pool1_checked == POOL1_OUTPUT_SIZE) &&
              (pool2_checked == POOL2_OUTPUT_SIZE) &&
              pool1_done_seen && pool2_done_seen);
        repeat (2) @(posedge clk);

        if (error_count == 0) begin
            $display("PASS: Pool1 and Pool2 exactly match MATLAB.");
            $display("Pool1 checked=%0d", pool1_checked);
            $display("Pool2 checked=%0d", pool2_checked);
        end else begin
            $display("FAIL: MaxPool verification errors=%0d", error_count);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (pool1_done)
            pool1_done_seen = 1'b1;
        if (pool2_done)
            pool2_done_seen = 1'b1;

        if (pool1_output_valid) begin
            if ($isunknown(pool1_output_addr) ||
                $isunknown(pool1_output_data)) begin
                if (error_count < 20)
                    $display("Pool1 produced X/Z at output %0d",
                             pool1_checked);
                error_count = error_count + 1;
            end else if (pool1_output_addr >= POOL1_OUTPUT_SIZE) begin
                if (error_count < 20)
                    $display("Pool1 address out of range: %0d",
                             pool1_output_addr);
                error_count = error_count + 1;
            end else begin
                if (pool1_output_addr != pool1_checked) begin
                    if (error_count < 20)
                        $display("Pool1 address order error: got=%0d expected=%0d",
                                 pool1_output_addr, pool1_checked);
                    error_count = error_count + 1;
                end

                if (pool1_output_data !==
                    pool1_golden[pool1_output_addr]) begin
                    if (error_count < 20)
                        $display("Pool1 mismatch addr=%0d got=%0d expected=%0d",
                                 pool1_output_addr,
                                 pool1_output_data,
                                 pool1_golden[pool1_output_addr]);
                    error_count = error_count + 1;
                end
            end
            pool1_checked = pool1_checked + 1;
        end

        if (pool2_output_valid) begin
            if ($isunknown(pool2_output_addr) ||
                $isunknown(pool2_output_data)) begin
                if (error_count < 20)
                    $display("Pool2 produced X/Z at output %0d",
                             pool2_checked);
                error_count = error_count + 1;
            end else if (pool2_output_addr >= POOL2_OUTPUT_SIZE) begin
                if (error_count < 20)
                    $display("Pool2 address out of range: %0d",
                             pool2_output_addr);
                error_count = error_count + 1;
            end else begin
                if (pool2_output_addr != pool2_checked) begin
                    if (error_count < 20)
                        $display("Pool2 address order error: got=%0d expected=%0d",
                                 pool2_output_addr, pool2_checked);
                    error_count = error_count + 1;
                end

                if (pool2_output_data !==
                    pool2_golden[pool2_output_addr]) begin
                    if (error_count < 20)
                        $display("Pool2 mismatch addr=%0d got=%0d expected=%0d",
                                 pool2_output_addr,
                                 pool2_output_data,
                                 pool2_golden[pool2_output_addr]);
                    error_count = error_count + 1;
                end
            end
            pool2_checked = pool2_checked + 1;
        end
    end

    initial begin
        #2000000;
        $display("FAIL: tb_maxpool timeout");
        $finish;
    end
endmodule
