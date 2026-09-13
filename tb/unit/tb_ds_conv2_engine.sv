`timescale 1ns/1ps

// DS-Conv2 獨立測試：保留正式設計的 21 input channels、20 output
// channels 與 5 lanes，只縮小成一個空間位置，方便觀察完整 pipeline。
module tb_ds_conv2_engine;
    localparam integer DATA_WIDTH  = 16;
    localparam integer WEIGHT_WIDTH = 16;
    localparam integer BIAS_WIDTH  = 16;
    localparam integer IN_H        = 2;
    localparam integer IN_W        = 5;
    localparam integer IN_CH       = 21;
    localparam integer K_H         = 2;
    localparam integer K_W         = 5;
    localparam integer OUT_CH      = 20;
    localparam integer LANES       = 5;
    localparam integer OUT_GROUPS  = OUT_CH / LANES;
    localparam integer INPUT_SIZE  = IN_H * IN_W * IN_CH;
    localparam integer DW_WORDS    = IN_CH * K_W;
    localparam integer PW_WORDS    = IN_CH * OUT_GROUPS;
    localparam integer MAX_CYCLES  = 500;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic busy;
    logic done;

    logic [15:0] input_addr_kh0;
    logic [15:0] input_addr_kh1;
    logic signed [DATA_WIDTH-1:0] input_data_kh0;
    logic signed [DATA_WIDTH-1:0] input_data_kh1;

    logic [$clog2(DW_WORDS)-1:0] dw_weight_addr;
    logic signed [(K_H*WEIGHT_WIDTH)-1:0] dw_weight_data;
    logic [$clog2(IN_CH)-1:0] dw_bias_addr;
    logic signed [BIAS_WIDTH-1:0] dw_bias_data;

    logic [$clog2(PW_WORDS)-1:0] pw_weight_addr;
    logic signed [(LANES*WEIGHT_WIDTH)-1:0] pw_weight_data;
    logic [$clog2(OUT_GROUPS)-1:0] pw_bias_addr;
    logic signed [(LANES*BIAS_WIDTH)-1:0] pw_bias_data;

    logic output_valid;
    logic output_last;
    logic signed [DATA_WIDTH-1:0] output_data;
    logic [15:0] output_addr;
    logic output_h;
    logic output_w;
    logic [$clog2(OUT_CH)-1:0] output_channel;

    // 一拍同步 RAM / ROM 模型，與 FPGA M10K 的讀取方式一致。
    logic signed [DATA_WIDTH-1:0] input_mem [0:INPUT_SIZE-1];
    logic signed [(K_H*WEIGHT_WIDTH)-1:0] dw_weight_mem [0:DW_WORDS-1];
    logic signed [BIAS_WIDTH-1:0] dw_bias_mem [0:IN_CH-1];
    logic signed [(LANES*WEIGHT_WIDTH)-1:0] pw_weight_mem [0:PW_WORDS-1];
    logic signed [(LANES*BIAS_WIDTH)-1:0] pw_bias_mem [0:OUT_GROUPS-1];

    integer ch;
    integer kw;
    integer kh;
    integer group_idx;
    integer lane_idx;
    integer cycle_count;
    integer output_count;
    integer expected_value;
    integer error_count;

    always #5 clk = ~clk;

    ds_conv2_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .INPUT_ADDR_WIDTH(16),
        .OUTPUT_ADDR_WIDTH(16),
        .IN_H(IN_H),
        .IN_W(IN_W),
        .IN_CH(IN_CH),
        .K_H(K_H),
        .K_W(K_W),
        .OUT_CH(OUT_CH),
        .LANES(LANES),
        // 測試資料使用整數，讓結果可以直接人工核對。
        .DW_BIAS_SHIFT(0),
        .DW_OUT_SHIFT(0),
        .PW_BIAS_SHIFT(0),
        .PW_OUT_SHIFT(0)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .input_addr_kh0(input_addr_kh0),
        .input_addr_kh1(input_addr_kh1),
        .input_data_kh0(input_data_kh0),
        .input_data_kh1(input_data_kh1),
        .dw_weight_addr(dw_weight_addr),
        .dw_weight_data(dw_weight_data),
        .dw_bias_addr(dw_bias_addr),
        .dw_bias_data(dw_bias_data),
        .pw_weight_addr(pw_weight_addr),
        .pw_weight_data(pw_weight_data),
        .pw_bias_addr(pw_bias_addr),
        .pw_bias_data(pw_bias_data),
        .output_valid(output_valid),
        .output_last(output_last),
        .output_data(output_data),
        .output_addr(output_addr),
        .output_h(output_h),
        .output_w(output_w),
        .output_channel(output_channel)
    );

    // ------------------------------------------------------------------
    // 測試資料
    // ------------------------------------------------------------------
    // input(c) = c+1，所有 DW/PW weights = 1。
    // 每個 DW 結果為 2*5*(c+1)，所有 channel 相加為 2310。
    // PW bias 設為 output channel 編號，因此預期輸出為 2310+channel。
    initial begin
        for (ch = 0; ch < IN_CH; ch = ch + 1) begin
            dw_bias_mem[ch] = '0;

            for (kw = 0; kw < K_W; kw = kw + 1) begin
                dw_weight_mem[kw + K_W*ch] = '0;
                for (kh = 0; kh < K_H; kh = kh + 1) begin
                    input_mem[kh + IN_H*(kw + IN_W*ch)] = ch + 1;
                    dw_weight_mem[kw + K_W*ch][kh*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sd1;
                end
            end
        end

        for (group_idx = 0; group_idx < OUT_GROUPS; group_idx = group_idx + 1) begin
            pw_bias_mem[group_idx] = '0;
            for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1)
                pw_bias_mem[group_idx][lane_idx*BIAS_WIDTH +: BIAS_WIDTH]
                    = group_idx*LANES + lane_idx;

            for (ch = 0; ch < IN_CH; ch = ch + 1) begin
                pw_weight_mem[ch + IN_CH*group_idx] = '0;
                for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1)
                    pw_weight_mem[ch + IN_CH*group_idx]
                        [lane_idx*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 16'sd1;
            end
        end
    end

    // ------------------------------------------------------------------
    // 同步記憶體讀取
    // ------------------------------------------------------------------
    always_ff @(posedge clk) begin
        input_data_kh0 <= input_mem[input_addr_kh0];
        input_data_kh1 <= input_mem[input_addr_kh1];
        dw_weight_data <= dw_weight_mem[dw_weight_addr];
        dw_bias_data   <= dw_bias_mem[dw_bias_addr];
        pw_weight_data <= pw_weight_mem[pw_weight_addr];
        pw_bias_data   <= pw_bias_mem[pw_bias_addr];
    end

    // ------------------------------------------------------------------
    // Reset、啟動與結束檢查
    // ------------------------------------------------------------------
    initial begin
        cycle_count = 0;
        output_count = 0;
        error_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done);
        @(posedge clk);

        if ((error_count == 0) && (output_count == OUT_CH)) begin
            $display("PASS: DS-Conv2 pipeline produced all %0d correct outputs.", OUT_CH);
            $display("Cycles from start = %0d", cycle_count);
        end else begin
            $fatal(1, "DS-Conv2 failed: outputs=%0d errors=%0d",
                   output_count, error_count);
        end
        $finish;
    end

    // ------------------------------------------------------------------
    // Output scoreboard
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (output_valid) begin
                expected_value = 2310 + output_channel;

                if ((output_addr !== output_channel) ||
                    ($signed(output_data) !== expected_value)) begin
                    $display("MISMATCH cycle=%0d ch=%0d addr=%0d got=%0d expected=%0d",
                             cycle_count, output_channel, output_addr,
                             $signed(output_data), expected_value);
                    error_count = error_count + 1;
                end

                if (output_last !== (output_channel == OUT_CH-1)) begin
                    $display("LAST ERROR cycle=%0d ch=%0d output_last=%0b",
                             cycle_count, output_channel, output_last);
                    error_count = error_count + 1;
                end

                output_count = output_count + 1;
            end

            if (cycle_count > MAX_CYCLES)
                $fatal(1, "DS-Conv2 timeout at cycle %0d", cycle_count);
        end
    end

endmodule
