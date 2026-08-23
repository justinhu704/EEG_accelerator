`timescale 1ns/1ps

module tb_conv1_banked_ram;
    localparam int ADDR_W = 16;

    logic clk, rst_n;
    logic write_en;
    logic [ADDR_W-1:0] write_addr;
    logic signed [15:0] write_data;
    logic [ADDR_W-1:0] read_addr_kh0, read_addr_kh1;
    logic signed [15:0] read_data_kh0, read_data_kh1;
    integer errors;

    always #5 clk = ~clk;

    conv1_banked_ram #(
        .INPUT_F(11), .STORED_F(5), .LOG_ADDR_W(ADDR_W),
        .BANK_DEPTH((20 * 156 * 21) / 2)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .write_en(write_en),
        .write_logical_addr(write_addr),
        .write_q11_data(write_data),
        .read_logical_addr_kh0(read_addr_kh0),
        .read_logical_addr_kh1(read_addr_kh1),
        .read_q11_data_kh0(read_data_kh0),
        .read_q11_data_kh1(read_data_kh1)
    );

    task automatic write_activation(
        input logic [ADDR_W-1:0] addr,
        input logic signed [15:0] data
    );
        begin
            @(negedge clk);
            write_en = 1'b1;
            write_addr = addr;
            write_data = data;
            @(negedge clk);
            write_en = 1'b0;
        end
    endtask

    task automatic check_pair(
        input logic [ADDR_W-1:0] addr0,
        input logic [ADDR_W-1:0] addr1,
        input integer expected0,
        input integer expected1
    );
        begin
            @(negedge clk);
            read_addr_kh0 = addr0;
            read_addr_kh1 = addr1;
            @(posedge clk);
            #1;
            if ($isunknown(read_data_kh0) ||
                ($signed(read_data_kh0) != expected0)) begin
                $display("KH0 mismatch addr=%0d got=%0d expected=%0d",
                         addr0, read_data_kh0, expected0);
                errors = errors + 1;
            end
            if ($isunknown(read_data_kh1) ||
                ($signed(read_data_kh1) != expected1)) begin
                $display("KH1 mismatch addr=%0d got=%0d expected=%0d",
                         addr1, read_data_kh1, expected1);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        write_en = 1'b0;
        write_addr = '0;
        write_data = '0;
        read_addr_kh0 = '0;
        read_addr_kh1 = 16'd1;
        errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Address bit zero is the height parity because height is the fastest
        // logical dimension. Adjacent kh values therefore enter opposite banks.
        write_activation(16'd0,    16'sd64);
        write_activation(16'd1,    16'sd128);
        write_activation(16'd2,    16'sd192);
        write_activation(16'd3120, 16'sd320);
        write_activation(16'd3121, 16'sd384);
        write_activation(16'd6240, 16'sd32767);
        write_activation(16'd6241, 16'sd640);

        check_pair(16'd0, 16'd1, 64, 128);
        // Also verify the odd-kh0 case swaps physical bank outputs correctly.
        check_pair(16'd1, 16'd2, 128, 192);
        check_pair(16'd3120, 16'd3121, 320, 384);
        check_pair(16'd6240, 16'd6241, 16320, 640);

        if (errors == 0)
            $display("PASS: Conv1 even/odd UQ5 banks restore Conv2 Q11 data.");
        else begin
            $display("FAIL: Conv1 banked RAM errors=%0d", errors);
            $fatal(1);
        end
        $finish;
    end
endmodule
