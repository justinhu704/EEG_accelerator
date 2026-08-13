`timescale 1ns/1ps
module tb_uart_tx;
    localparam integer CLKS_PER_BIT = 8;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic [7:0] data = '0;
    logic serial_tx, busy, done;

    always #5 clk = ~clk;

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .data(data),
        .serial_tx(serial_tx), .busy(busy), .done(done)
    );

    task automatic launch_byte(input logic [7:0] value);
        begin
            @(negedge clk);
            data = value;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task automatic receive_byte(output logic [7:0] value);
        integer i;
        begin
            @(negedge serial_tx);
            repeat (CLKS_PER_BIT/2) @(posedge clk);
            if (serial_tx !== 1'b0)
                $fatal(1, "UART TX start bit is not low");
            for (i = 0; i < 8; i = i + 1) begin
                repeat (CLKS_PER_BIT) @(posedge clk);
                value[i] = serial_tx;
            end
            repeat (CLKS_PER_BIT) @(posedge clk);
            if (serial_tx !== 1'b1)
                $fatal(1, "UART TX stop bit is not high");
        end
    endtask

    logic [7:0] decoded;
    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        fork
            launch_byte(8'h3C);
            receive_byte(decoded);
        join
        wait(done);
        if (decoded !== 8'h3C)
            $fatal(1, "UART TX decoded %02h, expected 3C", decoded);

        $display("PASS: UART TX serialized byte 3C.");
        $finish;
    end
endmodule
