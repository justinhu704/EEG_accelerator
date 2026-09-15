`timescale 1ns/1ps

// DS-Conv2 串流池化獨立測試：使用正式尺寸與真實輸出順序。
module tb_dsconv_streaming_pool;
    localparam integer IN_H = 19;
    localparam integer IN_W = 152;
    localparam integer IN_CH = 20;
    localparam integer POOL_W = 10;
    localparam integer STRIDE_W = 8;
    localparam integer OUT_H = IN_H;
    localparam integer OUT_W = ((IN_W - POOL_W) / STRIDE_W) + 1;
    localparam integer EXPECTED_OUTPUTS = OUT_H * OUT_W * IN_CH;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic busy;
    logic done;
    logic input_valid = 1'b0;
    logic input_last = 1'b0;
    logic signed [15:0] input_data = '0;
    logic [$clog2(IN_H)-1:0] input_h = '0;
    logic [$clog2(IN_W)-1:0] input_w = '0;
    logic [$clog2(IN_CH)-1:0] input_channel = '0;
    logic output_valid;
    logic [12:0] output_addr;
    logic signed [15:0] output_data;

    integer w, h, ch;
    integer output_count, error_count;
    integer decoded_ch, decoded_window, decoded_h;
    integer address_remainder, expected_data;

    always #10 clk = ~clk;

    dsconv_streaming_pool #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .POOL_W(POOL_W), .STRIDE_W(STRIDE_W),
        .INPUT_F(11), .OUTPUT_F(11)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .input_valid(input_valid), .input_last(input_last),
        .input_data(input_data), .input_h(input_h), .input_w(input_w),
        .input_channel(input_channel), .output_valid(output_valid),
        .output_addr(output_addr), .output_data(output_data)
    );

    // 輸入順序與 DS-Conv2 相同：width -> height -> channel。
    initial begin
        output_count = 0;
        error_count = 0;
        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait (busy);
        for (w = 0; w < IN_W; w = w + 1) begin
            for (h = 0; h < IN_H; h = h + 1) begin
                for (ch = 0; ch < IN_CH; ch = ch + 1) begin
                    @(negedge clk);
                    input_valid = 1'b1;
                    input_last = (w == IN_W-1) && (h == IN_H-1) &&
                                 (ch == IN_CH-1);
                    input_w = w;
                    input_h = h;
                    input_channel = ch;
                    input_data = w*100 + h*20 + ch;
                end
            end
        end

        @(negedge clk);
        input_valid = 1'b0;
        input_last = 1'b0;
        wait (done);
        @(negedge clk);
        if ((error_count == 0) && (output_count == EXPECTED_OUTPUTS))
            $display("PASS: streaming pool produced %0d correct outputs.", output_count);
        else
            $fatal(1, "Pool failed: outputs=%0d expected=%0d errors=%0d",
                   output_count, EXPECTED_OUTPUTS, error_count);
        $finish;
    end

    // 輸入值隨 width 增加，每個 window 的最大值位於最後一列。
    always @(negedge clk) begin
        if (rst_n && output_valid) begin
            decoded_ch = output_addr / (OUT_H * OUT_W);
            address_remainder = output_addr % (OUT_H * OUT_W);
            decoded_window = address_remainder / OUT_H;
            decoded_h = address_remainder % OUT_H;
            expected_data = (decoded_window*STRIDE_W + POOL_W-1)*100
                          + decoded_h*20 + decoded_ch;
            if ($signed(output_data) !== expected_data) begin
                $display("MISMATCH addr=%0d window=%0d h=%0d ch=%0d got=%0d expected=%0d",
                         output_addr, decoded_window, decoded_h, decoded_ch,
                         $signed(output_data), expected_data);
                error_count = error_count + 1;
            end
            output_count = output_count + 1;
        end
    end
endmodule
