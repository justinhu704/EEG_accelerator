`timescale 1ns/1ps

module tb_argmax;
    localparam int OUTPUT_SIZE = 105;

    logic clk, rst_n, start, in_valid;
    logic [6:0] in_index;
    logic signed [15:0] in_data;
    logic busy, done;
    logic [6:0] class_index;
    logic signed [15:0] max_value;
    logic signed [15:0] golden [0:4*OUTPUT_SIZE-1];

    integer i;
    integer expected_index;
    logic signed [15:0] expected_max;

    argmax_105 dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .in_valid(in_valid), .in_index(in_index), .in_data(in_data),
        .busy(busy), .done(done),
        .class_index(class_index), .max_value(max_value)
    );

    always #5 clk = ~clk;

    initial begin
        $readmemh("../mem/golden/q_fc_out_act.mem", golden);

        expected_index = 0;
        expected_max = golden[0];
        for (i = 1; i < OUTPUT_SIZE; i = i + 1) begin
            if ($signed(golden[i]) > $signed(expected_max)) begin
                expected_max = golden[i];
                expected_index = i;
            end
        end

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        in_valid = 1'b0;
        in_index = '0;
        in_data = '0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // Include bubbles to prove that Argmax counts valid values, not clocks.
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            if ((i % 11) == 5) begin
                in_valid = 1'b0;
                @(negedge clk);
            end
            in_valid = 1'b1;
            in_index = i[6:0];
            in_data = golden[i];
            @(negedge clk);
        end
        in_valid = 1'b0;

        wait (done);
        #1;
        if ((class_index == expected_index[6:0]) &&
            ($signed(max_value) == $signed(expected_max))) begin
            $display("PASS: Argmax105 class=%0d max=%0d",
                     class_index, max_value);
        end else begin
            $display("FAIL: Argmax105 got class=%0d max=%0d expected class=%0d max=%0d",
                     class_index, max_value, expected_index, expected_max);
            $fatal(1);
        end
        $finish;
    end
endmodule
