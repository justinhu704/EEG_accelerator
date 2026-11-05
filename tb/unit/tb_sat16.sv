// 飽和測試

`timescale 1ns/1ps

module tb_sat16;
    logic signed [47:0] value_in_48;
    logic signed [15:0] value_out_48;
    logic signed [39:0] value_in_40;
    logic signed [15:0] value_out_40;
    integer errors;

    sat16 dut_48 (
        .value_in(value_in_48),
        .value_out(value_out_48)
    );

    // fc_out_streaming uses a 40-bit accumulator.  Test the exact interface
    // width so negative logits cannot be accidentally zero-extended.
    sat16 #(.INPUT_W(40)) dut_40 (
        .value_in(value_in_40),
        .value_out(value_out_40)
    );

    task automatic check(
        input logic signed [47:0] test_value,
        input logic signed [15:0] expected
    );
        begin
            value_in_48 = test_value;
            #1;
            if (value_out_48 !== expected) begin
                $display("ERROR sat16(48): in=%0d got=%0d expected=%0d",
                         test_value, value_out_48, expected);
                errors = errors + 1;
            end
        end
    endtask

    task automatic check_40(
        input logic signed [39:0] test_value,
        input logic signed [15:0] expected
    );
        begin
            value_in_40 = test_value;
            #1;
            if (value_out_40 !== expected) begin
                $display("ERROR sat16(40): in=%0d got=%0d expected=%0d",
                         test_value, value_out_40, expected);
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

        check_40(40'sd1234,    16'sd1234);
        check_40(-40'sd1234,  -16'sd1234);
        check_40(40'sd40000,   16'sd32767);
        check_40(-40'sd40000, -16'sd32768);

        if (errors == 0)
            $display("PASS: tb_sat16");
        else
            $display("FAIL: tb_sat16, errors=%0d", errors);
        $finish;
    end
endmodule
