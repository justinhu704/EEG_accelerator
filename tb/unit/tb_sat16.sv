// 飽和測試

`timescale 1ns/1ps

module tb_sat16;
    logic signed [47:0] value_in;
    logic signed [15:0] value_out;
    integer errors;

    sat16 dut (.*);

    task automatic check(
        input logic signed [47:0] test_value,
        input logic signed [15:0] expected
    );
        begin
            value_in = test_value;
            #1;
            if (value_out !== expected) begin
                $display("ERROR sat16: in=%0d got=%0d expected=%0d",
                         test_value, value_out, expected);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        check(48'sd1234,    16'sd1234);
        check(-48'sd1234,  -16'sd1234);
        check(48'sd32767,   16'sd32767);
        check(48'sd40000,   16'sd32767);
        check(-48'sd32768, -16'sd32768);
        check(-48'sd40000, -16'sd32768);

        if (errors == 0)
            $display("PASS: tb_sat16");
        else
            $display("FAIL: tb_sat16, errors=%0d", errors);
        $finish;
    end
endmodule
