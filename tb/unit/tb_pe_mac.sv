// 乘加法器測試

`timescale 1ns/1ps

module tb_pe_mac;
    logic clk;
    logic rst_n;
    logic clear_acc;
    logic mac_en;
    logic signed [15:0] data_in;
    logic signed [15:0] weight_in;
    logic signed [47:0] accumulator;

    pe_mac dut (.*);

    always #5 clk = ~clk;

    initial begin
        clk       = 0;
        rst_n     = 0;
        clear_acc = 0;
        mac_en    = 0;
        data_in   = 0;
        weight_in = 0;

        repeat (2) @(posedge clk);
        rst_n = 1;

        @(negedge clk);
        clear_acc = 1;
        @(negedge clk);
        clear_acc = 0;

        // 3*4 + (-2)*5 = 2
        data_in   = 16'sd3;
        weight_in = 16'sd4;
        mac_en    = 1;
        @(negedge clk);
        data_in   = -16'sd2;
        weight_in = 16'sd5;
        @(negedge clk);
        mac_en = 0;

        if (accumulator === 48'sd2)
            $display("PASS: tb_pe_mac, accumulator=%0d", accumulator);
        else
            $display("FAIL: tb_pe_mac, got=%0d expected=2", accumulator);
        $finish;
    end
endmodule
