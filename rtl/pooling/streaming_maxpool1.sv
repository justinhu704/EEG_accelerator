// input order: channel_group -> width -> height -> lane
// POOL_W = 10, STRIDE_W = 8

module streaming_maxpool1 #(
    parameter int IN_H     = 19,
    parameter int IN_W     = 152,
    parameter int IN_CH    = 20,
    parameter int POOL_W   = 10,
    parameter int STRIDE_W = 8,
    parameter int LANES    = 4,
    parameter int OUT_H    = IN_H,
    parameter int OUT_W    = ((IN_W - POOL_W) / STRIDE_W) + 1
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    input  logic               input_valid,
    input  logic signed [15:0] input_data,

    output logic               output_valid,
    output logic [12:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int OUT_GROUPS = IN_CH / LANES;
    localparam int OVERLAP    = POOL_W - STRIDE_W;

    // registers for two overlapping windows
    // max_even: 0, 2, 4, ... windows
    // max_odd:  1, 3, 5, ... windows
    logic signed [15:0] max_even [0:IN_H-1][0:LANES-1];
    logic signed [15:0] max_odd  [0:IN_H-1][0:LANES-1];

    logic active;
    integer group_count;
    integer w_count;
    integer h_count;
    integer lane_count;
    integer current_window;
    // 現在width中的第幾個stride(0~7)
    integer stride_phase;

    logic current_window_valid;
    logic previous_window_active;
    logic previous_window_ends;
    logic current_window_is_odd;
    integer output_channel;
    integer previous_window;

    always_comb begin
        busy = active;

        current_window_valid  = (current_window < OUT_W);
        // 判斷當前輸入是否屬於前一個pool的後半段
        previous_window_active = (current_window > 0) && (stride_phase < OVERLAP);
        // 判斷前一個pool的後半段是否結束
        previous_window_ends   = previous_window_active && (stride_phase == OVERLAP-1);
        // 判斷當前pool是否為奇數
        current_window_is_odd  = current_window[0];

        output_channel  = group_count * LANES + lane_count;
        previous_window = current_window - 1;
    end

    initial begin
        if (IN_CH % LANES != 0)
            $error("streaming_maxpool1 requires IN_CH to be divisible by LANES");
        if (POOL_W <= STRIDE_W)
            $error("streaming_maxpool1 requires overlapping pool windows");
        if (POOL_W > 2 * STRIDE_W)
            $error("streaming_maxpool1 supports at most two active windows");
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active         <= 1'b0;
            done           <= 1'b0;
            output_valid   <= 1'b0;
            output_addr    <= '0;
            output_data    <= '0;
            lane_count     <= 0;
            h_count        <= 0;
            w_count        <= 0;
            group_count    <= 0;
            current_window <= 0;
            stride_phase   <= 0;
        end else begin
            done         <= 1'b0;
            output_valid <= 1'b0;

            if (!active) begin
                if (start) begin
                    active         <= 1'b1;
                    lane_count     <= 0;
                    h_count        <= 0;
                    w_count        <= 0;
                    group_count    <= 0;
                    current_window <= 0;
                    stride_phase   <= 0;
                end
            end else if (input_valid) begin
                // ------------------------------
                // 更新最大值
                // ------------------------------
                // 當前 window 有效
                if (current_window_valid) begin
                    // 偶數 window
                    if (!current_window_is_odd) begin
                        // 第一次輸入或大於目前最大值
                        if ((stride_phase == 0) || ($signed(input_data) > $signed(max_even[h_count][lane_count])))
                            max_even[h_count][lane_count] <= input_data;
                    // 奇數 window
                    end else begin
                        if ((stride_phase == 0) || ($signed(input_data) > $signed(max_odd[h_count][lane_count])))
                            max_odd[h_count][lane_count] <= input_data;
                    end
                end

                // ------------------------------
                // 重疊資料，同時更新兩組最大值
                // ------------------------------
                if (previous_window_active) begin
                    if (!current_window_is_odd) begin
                        if ($signed(input_data) > $signed(max_odd[h_count][lane_count]))
                            max_odd[h_count][lane_count] <= input_data;
                    end else begin
                        if ($signed(input_data) > $signed(max_even[h_count][lane_count]))
                            max_even[h_count][lane_count] <= input_data;
                    end
                end

                // ------------------------------
                // 前一個window結束，輸出
                // ------------------------------
                if (previous_window_ends) begin
                    output_valid <= 1'b1;
                    output_addr  <= h_count + OUT_H * (previous_window + OUT_W * output_channel);

                    // nonblocking assignment
                    // max_even/max_odd 為舊值，需要再比較一次
                    if (!current_window_is_odd) begin
                        if ($signed(input_data) > $signed(max_odd[h_count][lane_count]))
                            output_data <= input_data;
                        else
                            output_data <= max_odd[h_count][lane_count];
                    end else begin
                        if ($signed(input_data) > $signed(max_even[h_count][lane_count]))
                            output_data <= input_data;
                        else
                            output_data <= max_even[h_count][lane_count];
                    end
                end

                // ------------------------------
                // counter 更新
                // ------------------------------
                if (lane_count < LANES-1) begin
                    lane_count <= lane_count + 1;
                end else begin
                    lane_count <= 0;
                    if (h_count < IN_H-1) begin
                        h_count <= h_count + 1;
                    end else begin
                        h_count <= 0;
                        if (w_count < IN_W-1) begin
                            w_count <= w_count + 1;
                            if (stride_phase < STRIDE_W-1) begin
                                stride_phase <= stride_phase + 1;
                            end else begin
                                stride_phase   <= 0;
                                current_window <= current_window + 1;
                            end
                        end else begin
                            w_count        <= 0;
                            stride_phase   <= 0;
                            current_window <= 0;
                            if (group_count < OUT_GROUPS-1) begin
                                group_count <= group_count + 1;
                            end else begin
                                group_count <= 0;
                                active      <= 1'b0;
                                done        <= 1'b1;
                            end
                        end
                    end
                end
            end
        end
    end
endmodule
