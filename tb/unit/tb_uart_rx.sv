`timescale 1ns/1ps
module tb_uart_rx;
    localparam integer CLKS_PER_BIT = 8;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic serial_rx = 1'b1;
    logic [7:0] data;
    logic valid, framing_error, busy;
    integer valid_count = 0;
    logic [7:0] received [0:2];

    always #5 clk = ~clk;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) dut (
        .clk(clk), .rst_n(rst_n), .serial_rx(serial_rx),
        .data(data), .valid(valid),
        .framing_error(framing_error), .busy(busy)
    );

    always @(posedge clk) begin
        if (valid) begin
            received[valid_count] <= data;
            valid_count <= valid_count + 1;
        end
        if (framing_error)
            $fatal(1, "Unexpected UART framing error");
    end

    task automatic send_byte(input logic [7:0] value);
        integer i;
        begin
            serial_rx = 1'b0;
            repeat (CLKS_PER_BIT) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                serial_rx = value[i];
                repeat (CLKS_PER_BIT) @(posedge clk);
            end
            serial_rx = 1'b1;
            repeat (CLKS_PER_BIT) @(posedge clk);
            repeat (2) @(posedge clk);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        send_byte(8'hA5);
        send_byte(8'h00);
        send_byte(8'hFF);
        repeat (5) @(posedge clk);

        if (valid_count != 3)
            $fatal(1, "Expected 3 bytes, received %0d", valid_count);
        if ((received[0] !== 8'hA5) || (received[1] !== 8'h00) ||
            (received[2] !== 8'hFF))
            $fatal(1, "UART RX data mismatch");

        $display("PASS: UART RX decoded A5 00 FF.");
        $finish;
    end
endmodule
