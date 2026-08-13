`timescale 1ns/1ps
module tb_uart_result_sender;
    localparam integer CLKS_PER_BIT = 8;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic [31:0] sample_id = 32'h12345678;
    logic [7:0] status = 8'd0;
    logic [6:0] class_index = 7'd104;
    logic signed [15:0] winning_logit = -16'sd1234;
    logic tx_start;
    logic [7:0] tx_data;
    logic serial_line;
    logic tx_busy, tx_done;
    logic sender_busy, sender_done;
    logic [7:0] decoded_data;
    logic decoded_valid, framing_error, rx_busy;
    logic [7:0] response [0:11];
    integer response_count = 0;

    always #5 clk = ~clk;

    uart_result_sender dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .sample_id(sample_id), .status(status),
        .class_index(class_index), .winning_logit(winning_logit),
        .tx_start(tx_start), .tx_data(tx_data),
        .tx_busy(tx_busy), .tx_done(tx_done),
        .busy(sender_busy), .done(sender_done)
    );

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx (
        .clk(clk), .rst_n(rst_n), .start(tx_start), .data(tx_data),
        .serial_tx(serial_line), .busy(tx_busy), .done(tx_done)
    );

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx (
        .clk(clk), .rst_n(rst_n), .serial_rx(serial_line),
        .data(decoded_data), .valid(decoded_valid),
        .framing_error(framing_error), .busy(rx_busy)
    );

    always @(posedge clk) begin
        if (decoded_valid) begin
            response[response_count] <= decoded_data;
            response_count <= response_count + 1;
        end
        if (framing_error)
            $fatal(1, "Unexpected loopback framing error");
    end

    function automatic logic [15:0] crc_next(
        input logic [15:0] current_crc,
        input logic [7:0] value
    );
        logic [15:0] crc;
        integer i;
        begin
            crc = current_crc;
            for (i = 0; i < 8; i = i + 1) begin
                if (crc[15] ^ value[7-i])
                    crc = {crc[14:0],1'b0} ^ 16'h1021;
                else
                    crc = {crc[14:0],1'b0};
            end
            crc_next = crc;
        end
    endfunction

    logic [15:0] expected_crc;
    integer j;
    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait(sender_done);
        wait(response_count == 12);
        repeat (2) @(posedge clk);

        if ((response[0] !== 8'h5A) || (response[1] !== 8'hA5))
            $fatal(1, "Response magic mismatch");
        if ({response[5],response[4],response[3],response[2]} !== 32'h12345678)
            $fatal(1, "Response sample ID mismatch");
        if ((response[6] !== 8'd0) || (response[7] !== 8'd104))
            $fatal(1, "Response status/class mismatch");
        if ({response[9],response[8]} !== -16'sd1234)
            $fatal(1, "Response winning logit mismatch");

        expected_crc = 16'hFFFF;
        for (j = 2; j <= 9; j = j + 1)
            expected_crc = crc_next(expected_crc, response[j]);
        if ({response[11],response[10]} !== expected_crc)
            $fatal(1, "Response CRC mismatch");

        $display("PASS: UART result response class=104 CRC=%04h.", expected_crc);
        $finish;
    end
endmodule
