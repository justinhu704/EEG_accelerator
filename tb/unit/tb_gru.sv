`timescale 1ns/1ps

// Verifies all 9 hidden units across all 18 GRU time steps.
module tb_gru;
    localparam int INPUT_DEPTH = 4 * 18 * 15;
    localparam int OUTPUT_SIZE = 9 * 18;
    localparam int GOLDEN_DEPTH = 4 * OUTPUT_SIZE;
    // q_flatten_act comes from MATLAB's continuous sigmoid/tanh, while the
    // RTL intentionally uses 256-entry LUT approximations. 512 Q15 codes
    // equal 0.015625 and bound the measured end-to-end LUT approximation.
    localparam int TOLERANCE = 512;
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
    integer max_difference;
    integer difference;
    integer cycle_count;

    activation_ram #(
        .DATA_W(16),
        .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/golden/q_pool2_act.mem")
    ) u_input_ram (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0),
        .write_data('0),
        .read_addr(input_addr[$clog2(INPUT_DEPTH)-1:0]),
        .read_data(input_data)
    );

    gru_engine #(
        .WR_FILE("../mem/weights/gru_Wr.mem"),
        .WZ_FILE("../mem/weights/gru_Wz.mem"),
        .WH_FILE("../mem/weights/gru_Wh.mem"),
        .UR_FILE("../mem/weights/gru_Ur.mem"),
        .UZ_FILE("../mem/weights/gru_Uz.mem"),
        .UH_FILE("../mem/weights/gru_Uh.mem"),
        .BR_FILE("../mem/weights/gru_br.mem"),
        .BZ_FILE("../mem/weights/gru_bz.mem"),
        .BH_FILE("../mem/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem")
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
        require_file("../mem/golden/q_pool2_act.mem");
        require_file("../mem/golden/q_flatten_act.mem");
        require_file("../mem/weights/gru_Wr.mem");
        require_file("../mem/weights/gru_Wz.mem");
        require_file("../mem/weights/gru_Wh.mem");
        require_file("../mem/weights/gru_Ur.mem");
        require_file("../mem/weights/gru_Uz.mem");
        require_file("../mem/weights/gru_Uh.mem");
        require_file("../mem/weights/gru_br.mem");
        require_file("../mem/weights/gru_bz.mem");
        require_file("../mem/weights/gru_bh.mem");
        $readmemh("../mem/golden/q_flatten_act.mem", expected_mem);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        checked_count = 0;
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

        wait (done);
        repeat (2) @(posedge clk);

        if ((checked_count == OUTPUT_SIZE) && (error_count == 0)) begin
            $display("PASS: GRU 15x18 -> 9x18 matches golden.");
            $display("Checked=%0d max_diff=%0d", checked_count,
                     max_difference);
        end else begin
            $display("FAIL: GRU verification.");
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
                if (difference > TOLERANCE) begin
                    if (error_count < 20)
                        $display("GRU mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_data,
                                 expected_mem[output_addr], difference);
                    error_count = error_count + 1;
                end
            end
            checked_count = checked_count + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_gru timeout at cycle %0d", cycle_count);
            $finish;
        end
    end
endmodule
