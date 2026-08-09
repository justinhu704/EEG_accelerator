`timescale 1ns/1ps

// FC1 unit test: 162 flattened GRU values -> 40 outputs.
module tb_fc;
    localparam int SAMPLES = 4;
    localparam int INPUT_SIZE = 162;
    localparam int OUTPUT_SIZE = 40;
    localparam int INPUT_DEPTH = SAMPLES * INPUT_SIZE;
    localparam int GOLDEN_DEPTH = SAMPLES * OUTPUT_SIZE;
    localparam int TOLERANCE = 1;
    localparam int MAX_CYCLES = 100_000;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic [31:0] input_addr;
    logic signed [15:0] input_data;
    logic output_valid;
    logic [31:0] output_addr;
    logic signed [15:0] output_data;
    logic signed [15:0] expected_mem [0:GOLDEN_DEPTH-1];

    integer checked_count;
    integer error_count;
    integer exact_count;
    integer one_lsb_count;
    integer max_difference;
    integer difference;
    integer cycle_count;

    activation_ram #(
        .DATA_W(16),
        .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/golden/q_flatten_act.mem")
    ) u_input_ram (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0), .write_data('0),
        .read_addr(input_addr[$clog2(INPUT_DEPTH)-1:0]),
        .read_data(input_data)
    );

    fc_engine #(
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .BIAS_SHIFT(16),
        .OUTPUT_SHIFT(17),
        .WEIGHT_FILE("../mem/weights/fc_1_W.mem"),
        .BIAS_FILE("../mem/weights/fc_1_b.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_addr(input_addr), .input_data(input_data),
        .output_valid(output_valid),
        .output_addr(output_addr), .output_data(output_data)
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
        require_file("../mem/golden/q_flatten_act.mem");
        require_file("../mem/golden/q_fc_1_act.mem");
        require_file("../mem/weights/fc_1_W.mem");
        require_file("../mem/weights/fc_1_b.mem");
        $readmemh("../mem/golden/q_fc_1_act.mem", expected_mem);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        checked_count = 0;
        error_count = 0;
        exact_count = 0;
        one_lsb_count = 0;
        max_difference = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done);
        repeat (2) @(posedge clk);

        if ((checked_count == OUTPUT_SIZE) && (error_count == 0)) begin
            $display("PASS: FC1 162 -> 40 matches MATLAB.");
            $display("Checked=%0d exact=%0d one_lsb=%0d max_diff=%0d",
                     checked_count, exact_count, one_lsb_count,
                     max_difference);
        end else begin
            $display("FAIL: FC1 verification.");
            $display("Checked=%0d expected=%0d errors=%0d max_diff=%0d",
                     checked_count, OUTPUT_SIZE, error_count,
                     max_difference);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data) ||
                (output_addr >= OUTPUT_SIZE)) begin
                error_count = error_count + 1;
            end else begin
                difference = absolute_difference(
                    output_data, expected_mem[output_addr]
                );
                if (difference > max_difference)
                    max_difference = difference;
                if (difference == 0)
                    exact_count = exact_count + 1;
                else if (difference == 1)
                    one_lsb_count = one_lsb_count + 1;

                if (difference > TOLERANCE) begin
                    if (error_count < 20)
                        $display("FC1 mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_data,
                                 expected_mem[output_addr], difference);
                    error_count = error_count + 1;
                end
            end
            checked_count = checked_count + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_fc timeout at cycle %0d", cycle_count);
            $finish;
        end
    end
endmodule
