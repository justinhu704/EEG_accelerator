`timescale 1ns/1ps
module tb_fpga_uart_top;
    localparam integer UART_CLKS_PER_BIT = 4;
    localparam integer SAMPLE_WORDS = 3360;
    logic CLOCK_50 = 1'b0;
    logic [3:0] KEY = 4'b1110;
    logic UART_RXD = 1'b1;
    logic UART_TXD;
    logic [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5;
    logic [9:0] LEDR;
    logic [15:0] sample_mem [0:SAMPLE_WORDS-1];

    logic [7:0] response_byte;
    logic response_valid, response_frame_error, response_rx_busy;
    logic [7:0] response [0:11];
    integer response_count = 0;
    integer i;
    logic [15:0] request_crc;
    logic [15:0] response_crc;

    always #10 CLOCK_50 = ~CLOCK_50;

    fpga_uart_top #(
        .UART_CLKS_PER_BIT(UART_CLKS_PER_BIT),
        .CONV1_W_FILE("../mem/weights/conv1_W.mem"),
        .CONV1_B_FILE("../mem/weights/conv1_b.mem"),
        .CONV1_PACKED_W_FILE("../mem/weights/conv1_W_x3.mem"),
        .CONV1_PACKED_B_FILE("../mem/weights/conv1_b_x3.mem"),
        .BN1_A_FILE("../mem/weights/bn1_A.mem"),
        .BN1_B_FILE("../mem/weights/bn1_B.mem"),
        .CONV2_W_FILE("../mem/weights/conv2_W.mem"),
        .CONV2_B_FILE("../mem/weights/conv2_b.mem"),
        .CONV2_PACKED_W_FILE("../mem/weights/conv2_W_x5.mem"),
        .CONV2_KH2_PACKED_W_FILE("../mem/weights/conv2_W_x5_kh2.mem"),
        .CONV2_PACKED_B_FILE("../mem/weights/conv2_b_x5.mem"),
        .BN2_A_FILE("../mem/weights/bn2_A.mem"),
        .BN2_B_FILE("../mem/weights/bn2_B.mem"),
        .CONV3_W_FILE("../mem/weights/conv3_W.mem"),
        .CONV3_B_FILE("../mem/weights/conv3_b.mem"),
        .CONV3_PACKED_W_FILE("../mem/weights/conv3_W_x3.mem"),
        .CONV3_PACKED_B_FILE("../mem/weights/conv3_b_x3.mem"),
        .BN3_A_FILE("../mem/weights/bn3_A.mem"),
        .BN3_B_FILE("../mem/weights/bn3_B.mem"),
        .GRU_WR_FILE("../mem/weights/gru_Wr.mem"),
        .GRU_WZ_FILE("../mem/weights/gru_Wz.mem"),
        .GRU_WH_FILE("../mem/weights/gru_Wh.mem"),
        .GRU_UR_FILE("../mem/weights/gru_Ur.mem"),
        .GRU_UZ_FILE("../mem/weights/gru_Uz.mem"),
        .GRU_UH_FILE("../mem/weights/gru_Uh.mem"),
        .GRU_BR_FILE("../mem/weights/gru_br.mem"),
        .GRU_BZ_FILE("../mem/weights/gru_bz.mem"),
        .GRU_BH_FILE("../mem/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem"),
        .FC1_W_FILE("../mem/weights/fc_1_W.mem"),
        .FC1_B_FILE("../mem/weights/fc_1_b.mem"),
        .FC_BN_A_FILE("../mem/weights/bn_2_A.mem"),
        .FC_BN_B_FILE("../mem/weights/bn_2_B.mem"),
        .FC_OUT_W_FILE("../mem/weights/fc_out_W.mem"),
        .FC_OUT_B_FILE("../mem/weights/fc_out_b.mem")
    ) dut (
        .CLOCK_50(CLOCK_50), .KEY(KEY),
        .UART_RXD(UART_RXD), .UART_TXD(UART_TXD),
        .HEX0(HEX0), .HEX1(HEX1), .HEX2(HEX2),
        .HEX3(HEX3), .HEX4(HEX4), .HEX5(HEX5), .LEDR(LEDR)
    );

    // Decode the FPGA response using the same physical UART timing.
    uart_rx #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) response_decoder (
        .clk(CLOCK_50), .rst_n(KEY[0]), .serial_rx(UART_TXD),
        .data(response_byte), .valid(response_valid),
        .framing_error(response_frame_error), .busy(response_rx_busy)
    );

    always @(posedge CLOCK_50) begin
        if (response_valid) begin
            response[response_count] <= response_byte;
            response_count <= response_count + 1;
        end
        if (response_frame_error)
            $fatal(1, "FPGA UART response framing error");
    end

    function automatic logic [15:0] crc_next(
        input logic [15:0] current_crc,
        input logic [7:0] value
    );
        logic [15:0] crc;
        integer bit_index;
        begin
            crc = current_crc;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                if (crc[15] ^ value[7-bit_index])
                    crc = {crc[14:0],1'b0} ^ 16'h1021;
                else
                    crc = {crc[14:0],1'b0};
            end
            crc_next = crc;
        end
    endfunction

    task automatic uart_send_byte(input logic [7:0] value);
        integer bit_index;
        begin
            UART_RXD = 1'b0;
            repeat (UART_CLKS_PER_BIT) @(posedge CLOCK_50);
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                UART_RXD = value[bit_index];
                repeat (UART_CLKS_PER_BIT) @(posedge CLOCK_50);
            end
            UART_RXD = 1'b1;
            repeat (UART_CLKS_PER_BIT) @(posedge CLOCK_50);
        end
    endtask

    task automatic send_body_byte(input logic [7:0] value);
        begin
            uart_send_byte(value);
            request_crc = crc_next(request_crc, value);
        end
    endtask

    initial begin
        $readmemh("../mem/board/sample0_q12.mem", sample_mem);
        repeat (10) @(posedge CLOCK_50);
        KEY[0] = 1'b1;
        repeat (10) @(posedge CLOCK_50);

        request_crc = 16'hFFFF;
        uart_send_byte(8'hA5);
        uart_send_byte(8'h5A);
        send_body_byte(8'h78);
        send_body_byte(8'h56);
        send_body_byte(8'h34);
        send_body_byte(8'h12);
        send_body_byte(8'h20); // 3360 = 0x0D20
        send_body_byte(8'h0D);
        for (i = 0; i < SAMPLE_WORDS; i = i + 1) begin
            send_body_byte(sample_mem[i][7:0]);
            send_body_byte(sample_mem[i][15:8]);
        end
        uart_send_byte(request_crc[7:0]);
        uart_send_byte(request_crc[15:8]);

        fork
            begin
                wait(response_count == 12);
            end
            begin
                #400000000;
                $fatal(1, "Timed out waiting for complete FPGA response");
            end
        join_any
        disable fork;
        repeat (5) @(posedge CLOCK_50);

        if ((response[0] !== 8'h5A) || (response[1] !== 8'hA5))
            $fatal(1, "Response magic mismatch");
        if ({response[5],response[4],response[3],response[2]} !== 32'h12345678)
            $fatal(1, "Response sample ID mismatch");
        if (response[6] !== 8'd0)
            $fatal(1, "FPGA returned status %0d", response[6]);
        if (response[7] !== 8'd0)
            $fatal(1, "Expected class 0, received %0d", response[7]);
        if ($signed({response[9],response[8]}) !== 16'sd22792)
            $fatal(1, "Expected winning logit 22792, received %0d",
                   $signed({response[9],response[8]}));

        response_crc = 16'hFFFF;
        for (i = 2; i <= 9; i = i + 1)
            response_crc = crc_next(response_crc, response[i]);
        if ({response[11],response[10]} !== response_crc)
            $fatal(1, "Response CRC mismatch");
        if ((dut.subject_id !== 7'd1) ||
            (HEX3 !== 7'b0010010) ||
            (HEX2 !== 7'b1000000) ||
            (HEX1 !== 7'b1000000) ||
            (HEX0 !== 7'b1111001))
            $fatal(1, "Seven-segment display does not show subject S001");

        $display("PASS: UART inference returned sample=%08h class=%0d subject=S%03d logit=%0d CRC=%04h.",
                 {response[5],response[4],response[3],response[2]},
                 response[7], dut.subject_id,
                 $signed({response[9],response[8]}), response_crc);
        $finish;
    end
endmodule
