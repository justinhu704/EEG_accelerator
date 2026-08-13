`timescale 1ns/1ps
module tb_uart_sample_loader;
    localparam integer SAMPLE_WORDS = 4;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [7:0] rx_data = '0;
    logic rx_valid = 1'b0;
    logic rx_framing_error = 1'b0;
    logic input_ready = 1'b1;
    logic input_write_en;
    logic [11:0] input_write_addr;
    logic signed [15:0] input_write_data;
    logic inference_start;
    logic inference_done = 1'b0;
    logic [31:0] sample_id;
    logic loader_ready, loader_busy, packet_error, packet_loaded;
    logic signed [15:0] written [0:SAMPLE_WORDS-1];
    integer write_count = 0;
    integer start_count = 0;
    integer error_count = 0;

    always #5 clk = ~clk;

    uart_sample_loader #(
        .SAMPLE_WORDS(SAMPLE_WORDS), .ADDR_W(12)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .rx_framing_error(rx_framing_error),
        .input_ready(input_ready),
        .input_write_en(input_write_en),
        .input_write_addr(input_write_addr),
        .input_write_data(input_write_data),
        .inference_start(inference_start),
        .inference_done(inference_done),
        .sample_id(sample_id), .loader_ready(loader_ready),
        .loader_busy(loader_busy), .packet_error(packet_error),
        .packet_loaded(packet_loaded)
    );

    always @(posedge clk) begin
        if (input_write_en) begin
            if (input_write_addr !== write_count)
                $fatal(1, "Write address %0d, expected %0d",
                       input_write_addr, write_count);
            written[write_count] <= input_write_data;
            write_count <= write_count + 1;
        end
        if (inference_start)
            start_count <= start_count + 1;
        if (packet_error)
            error_count <= error_count + 1;
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

    task automatic send_byte(input logic [7:0] value);
        begin
            @(negedge clk);
            rx_data = value;
            rx_valid = 1'b1;
            @(negedge clk);
            rx_valid = 1'b0;
        end
    endtask

    task automatic send_body_byte(
        input logic [7:0] value,
        inout logic [15:0] crc
    );
        begin
            send_byte(value);
            crc = crc_next(crc, value);
        end
    endtask

    logic [15:0] crc;
    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);
        if (!loader_ready)
            $fatal(1, "Loader should be ready after reset");

        crc = 16'hFFFF;
        send_byte(8'hA5);
        send_byte(8'h5A);
        send_body_byte(8'h78, crc);
        send_body_byte(8'h56, crc);
        send_body_byte(8'h34, crc);
        send_body_byte(8'h12, crc);
        send_body_byte(8'h04, crc);
        send_body_byte(8'h00, crc);
        send_body_byte(8'h34, crc);
        send_body_byte(8'h12, crc);
        send_body_byte(8'hFE, crc);
        send_body_byte(8'hFF, crc);
        send_body_byte(8'h00, crc);
        send_body_byte(8'h80, crc);
        send_body_byte(8'hFF, crc);
        send_body_byte(8'h7F, crc);
        send_byte(crc[7:0]);
        send_byte(crc[15:8]);
        repeat (4) @(posedge clk);

        if (sample_id !== 32'h12345678)
            $fatal(1, "Sample ID mismatch: %08h", sample_id);
        if (write_count != SAMPLE_WORDS)
            $fatal(1, "Expected %0d writes, got %0d", SAMPLE_WORDS, write_count);
        if (start_count != 1 || error_count != 0)
            $fatal(1, "start_count=%0d error_count=%0d", start_count, error_count);
        if ((written[0] !== 16'sh1234) || (written[1] !== -16'sd2) ||
            (written[2] !== 16'sh8000) || (written[3] !== 16'sh7FFF))
            $fatal(1, "Payload write data mismatch");

        @(negedge clk);
        inference_done = 1'b1;
        @(negedge clk);
        inference_done = 1'b0;
        repeat (2) @(posedge clk);
        if (!loader_ready)
            $fatal(1, "Loader did not return to ready");

        $display("PASS: UART sample loader wrote RAM A and started inference.");
        $finish;
    end
endmodule
