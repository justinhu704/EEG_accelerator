`timescale 1ns/1ps

// Isolates the final FC layer using MATLAB's post-BN 40-value input.
module tb_fc_out;
    localparam int INPUT_SIZE = 40;
    localparam int OUTPUT_SIZE = 105;
    localparam int SAMPLES = 4;
    localparam int MAX_CYCLES = 100_000;

    logic clk, rst_n, start, busy, done;
    logic [31:0] input_addr, output_addr;
    logic signed [15:0] input_data, output_data;
    logic output_valid;
    logic signed [15:0] expected [0:SAMPLES*OUTPUT_SIZE-1];
    integer checked, errors, max_diff, difference, cycles;

    activation_ram #(
        .DATA_W(16), .DEPTH(SAMPLES*INPUT_SIZE),
        .ADDR_W($clog2(SAMPLES*INPUT_SIZE)),
        .MEM_FILE("../mem/golden/q_bn_2_act.mem")
    ) u_input_ram (
        .clk(clk), .write_en(1'b0), .write_addr('0), .write_data('0),
        .read_addr(input_addr[$clog2(SAMPLES*INPUT_SIZE)-1:0]),
        .read_data(input_data)
    );

    fc_engine #(
        .INPUT_SIZE(INPUT_SIZE), .OUTPUT_SIZE(OUTPUT_SIZE),
        .BIAS_SHIFT(12), .OUTPUT_SHIFT(17),
        .WEIGHT_FILE("../mem/weights/fc_out_W.mem"),
        .BIAS_FILE("../mem/weights/fc_out_b.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_addr(input_addr), .input_data(input_data),
        .output_valid(output_valid), .output_addr(output_addr),
        .output_data(output_data)
    );

    always #5 clk = ~clk;

    function automatic integer abs_diff(
        input logic signed [15:0] actual,
        input logic signed [15:0] golden
    );
        integer value;
        begin
            value = $signed(actual) - $signed(golden);
            abs_diff = (value < 0) ? -value : value;
        end
    endfunction

    initial begin
        $readmemh("../mem/golden/q_fc_out_act.mem", expected);
        clk = 0; rst_n = 0; start = 0;
        checked = 0; errors = 0; max_diff = 0; cycles = 0;
        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1; start = 1;
        @(negedge clk); start = 0;
        wait (done);
        repeat (2) @(posedge clk);
        if ((checked == OUTPUT_SIZE) && (errors == 0))
            $display("PASS: FC_out 40 -> 105 exact, checked=%0d max_diff=%0d",
                     checked, max_diff);
        else begin
            $display("FAIL: FC_out checked=%0d errors=%0d max_diff=%0d",
                     checked, errors, max_diff);
            $fatal(1);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycles = cycles + 1;
        if (output_valid) begin
            difference = abs_diff(output_data, expected[output_addr]);
            if (difference > max_diff) max_diff = difference;
            if (difference > 1) begin
                if (errors < 10)
                    $display("Mismatch index=%0d got=%0d expected=%0d diff=%0d",
                             output_addr, output_data,
                             expected[output_addr], difference);
                errors = errors + 1;
            end
            checked = checked + 1;
        end
        if (cycles > MAX_CYCLES) begin
            $display("FAIL: tb_fc_out timeout");
            $fatal(1);
        end
    end
endmodule
