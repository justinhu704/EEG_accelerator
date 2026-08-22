// input order: channel_group -> width -> height -> lane
// POOL_W = 10, STRIDE_W = 8

module streaming_maxpool #(
    parameter int IN_H     = 19,
    parameter int IN_W     = 152,
    parameter int IN_CH    = 20,
    parameter int POOL_W   = 10,
    parameter int STRIDE_W = 8,
    parameter int LANES    = 4,
    parameter int INPUT_F  = 13,
    parameter int OUTPUT_F = 13,
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
    localparam int MAX_DEPTH  = IN_H * LANES;
    localparam int MAX_ADDR_W = (MAX_DEPTH <= 1) ? 1 : $clog2(MAX_DEPTH);
    localparam int GROUP_W    = (OUT_GROUPS <= 1) ? 1 : $clog2(OUT_GROUPS);
    localparam int W_COUNT_W  = (IN_W <= 1) ? 1 : $clog2(IN_W);
    localparam int H_COUNT_W  = (IN_H <= 1) ? 1 : $clog2(IN_H);
    localparam int LANE_W     = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam int WINDOW_W   = (OUT_W + 1 <= 1) ? 1 : $clog2(OUT_W + 1);
    localparam int STRIDE_W_W = (STRIDE_W <= 1) ? 1 : $clog2(STRIDE_W);
    localparam int CHANNEL_W  = (IN_CH <= 1) ? 1 : $clog2(IN_CH);

    // RAM for two overlapping windows
    // max_even: 0, 2, 4, ... windows
    // max_odd:  1, 3, 5, ... windows
    logic [MAX_ADDR_W-1:0] max_read_addr;
    logic [MAX_ADDR_W-1:0] max_even_write_addr;
    logic [MAX_ADDR_W-1:0] max_odd_write_addr;
    logic signed [15:0] max_even_read_data;
    logic signed [15:0] max_odd_read_data;
    logic signed [15:0] max_even_write_data;
    logic signed [15:0] max_odd_write_data;
    logic max_even_write_en;
    logic max_odd_write_en;

    logic active;
    logic draining;
    logic [GROUP_W-1:0] group_count;
    logic [W_COUNT_W-1:0] w_count;
    logic [H_COUNT_W-1:0] h_count;
    logic [LANE_W-1:0] lane_count;
    logic [WINDOW_W-1:0] current_window;
    // 現在width中的第幾個stride(0~7)
    logic [STRIDE_W_W-1:0] stride_phase;

    logic current_window_valid;
    logic previous_window_active;
    logic previous_window_ends;
    logic current_window_is_odd;
    logic [CHANNEL_W-1:0] output_channel;
    logic [WINDOW_W-1:0] previous_window;
    logic input_is_last;

    // Synchronous max RAM read and its metadata pipeline.
    logic stage1_valid;
    logic stage1_last;
    logic signed [15:0] stage1_data;
    logic [MAX_ADDR_W-1:0] stage1_max_addr;
    logic stage1_current_window_valid;
    logic stage1_previous_window_active;
    logic stage1_previous_window_ends;
    logic stage1_current_window_is_odd;
    logic stage1_first_value;
    logic [12:0] stage1_output_addr;
    logic signed [15:0] completed_max;
    logic signed [47:0] completed_extended;
    logic signed [47:0] completed_scaled;
    logic signed [15:0] completed_saturated;

    function automatic logic signed [15:0] max16(
        input logic signed [15:0] a,
        input logic signed [15:0] b
    );
        max16 = ($signed(a) > $signed(b)) ? a : b;
    endfunction

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
        input_is_last = (group_count == OUT_GROUPS-1) &&
                        (w_count == IN_W-1) &&
                        (h_count == IN_H-1) &&
                        (lane_count == LANES-1);
    end

    initial begin
        if (IN_CH % LANES != 0)
            $error("streaming_maxpool1 requires IN_CH to be divisible by LANES");
        if (POOL_W <= STRIDE_W)
            $error("streaming_maxpool1 requires overlapping pool windows");
        if (POOL_W > 2 * STRIDE_W)
            $error("streaming_maxpool1 supports at most two active windows");
    end

    activation_ram #(
        .DATA_W(16), .DEPTH(MAX_DEPTH), .ADDR_W(MAX_ADDR_W),
        .MEM_FILE("")
    ) u_max_even_ram (
        .clk(clk),
        .write_en(max_even_write_en),
        .write_addr(max_even_write_addr),
        .write_data(max_even_write_data),
        .read_addr(max_read_addr),
        .read_data(max_even_read_data)
    );

    // Pool1 has several overlapping output windows and needs both the even
    // and odd buffers. Pool2 has OUT_W=1, so Quartus removes this second RAM.
    generate
        if (OUT_W > 1) begin : gen_odd_max_ram
            activation_ram #(
                .DATA_W(16), .DEPTH(MAX_DEPTH), .ADDR_W(MAX_ADDR_W),
                .MEM_FILE("")
            ) u_max_odd_ram (
                .clk(clk),
                .write_en(max_odd_write_en),
                .write_addr(max_odd_write_addr),
                .write_data(max_odd_write_data),
                .read_addr(max_read_addr),
                .read_data(max_odd_read_data)
            );
        end else begin : gen_no_odd_max_ram
            always_comb max_odd_read_data = '0;
        end
    endgenerate

    // Convert the selected maximum to the fractional format expected by the
    // following layer. Pool1 uses Q13 -> Q13; Pool2 uses Q12 -> Q13.
    always_comb begin
        if (!stage1_current_window_is_odd)
            completed_max = max16(stage1_data, max_odd_read_data);
        else
            completed_max = max16(stage1_data, max_even_read_data);

        completed_extended = {{32{completed_max[15]}}, completed_max};
        if (OUTPUT_F >= INPUT_F)
            completed_scaled = completed_extended <<< (OUTPUT_F - INPUT_F);
        else
            completed_scaled = completed_extended >>> (INPUT_F - OUTPUT_F);
    end

    sat16 u_output_sat16 (
        .value_in(completed_scaled),
        .value_out(completed_saturated)
    );

    // ------------------------------
    // 更新最大值
    // ------------------------------
    // Pipeline 第 2 級：比較同步 RAM 輸出並寫回。
    always_comb begin
        max_even_write_en = 1'b0;
        max_odd_write_en = 1'b0;
        max_even_write_addr = stage1_max_addr;
        max_odd_write_addr = stage1_max_addr;
        max_even_write_data = stage1_data;
        max_odd_write_data = stage1_data;

        if (stage1_valid) begin
            // 當前 window 有效
            if (stage1_current_window_valid) begin
                // 偶數 window
                if (!stage1_current_window_is_odd) begin
                    max_even_write_en = 1'b1;
                    // 第一次輸入或大於目前最大值
                    if (!stage1_first_value)
                        max_even_write_data = max16(
                            stage1_data, max_even_read_data
                        );
                // 奇數 window
                end else begin
                    max_odd_write_en = 1'b1;
                    if (!stage1_first_value)
                        max_odd_write_data = max16(
                            stage1_data, max_odd_read_data
                        );
                end
            end

            // ------------------------------
            // 重疊資料，同時更新兩組最大值
            // ------------------------------
            if (stage1_previous_window_active) begin
                if (!stage1_current_window_is_odd) begin
                    max_odd_write_en = 1'b1;
                    max_odd_write_data = max16(
                        stage1_data, max_odd_read_data
                    );
                end else begin
                    max_even_write_en = 1'b1;
                    max_even_write_data = max16(
                        stage1_data, max_even_read_data
                    );
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active         <= 1'b0;
            draining       <= 1'b0;
            done           <= 1'b0;
            output_valid   <= 1'b0;
            output_addr    <= '0;
            output_data    <= '0;
            lane_count     <= '0;
            h_count        <= '0;
            w_count        <= '0;
            group_count    <= '0;
            current_window <= '0;
            stride_phase   <= '0;
            max_read_addr  <= '0;
            stage1_valid   <= 1'b0;
            stage1_last    <= 1'b0;
            stage1_data    <= '0;
            stage1_max_addr <= '0;
            stage1_current_window_valid <= 1'b0;
            stage1_previous_window_active <= 1'b0;
            stage1_previous_window_ends <= 1'b0;
            stage1_current_window_is_odd <= 1'b0;
            stage1_first_value <= 1'b0;
            stage1_output_addr <= '0;
        end else begin
            done         <= 1'b0;
            output_valid <= 1'b0;
            stage1_valid <= 1'b0;

            if (!active) begin
                if (start) begin
                    active         <= 1'b1;
                    draining       <= 1'b0;
                    lane_count     <= '0;
                    h_count        <= '0;
                    w_count        <= '0;
                    group_count    <= '0;
                    current_window <= '0;
                    stride_phase   <= '0;
                    max_read_addr  <= '0;
                end
            end else begin
                // ------------------------------
                // Pipeline 第 1 級：讀 RAM 並保存這一筆資料的 metadata。
                // ------------------------------
                if (!draining && input_valid) begin
                    stage1_valid <= 1'b1;
                    stage1_last <= input_is_last;
                    stage1_data <= input_data;
                    stage1_max_addr <= max_read_addr;
                    stage1_current_window_valid <= current_window_valid;
                    stage1_previous_window_active <= previous_window_active;
                    stage1_previous_window_ends <= previous_window_ends;
                    stage1_current_window_is_odd <= current_window_is_odd;
                    stage1_first_value <= (stride_phase == 0);
                    stage1_output_addr <= h_count + OUT_H *
                        (previous_window + OUT_W * output_channel);
                end

                // ------------------------------
                // 前一個window結束，輸出
                // ------------------------------
                if (stage1_valid && stage1_previous_window_ends) begin
                    output_valid <= 1'b1;
                    output_addr  <= stage1_output_addr;
                    // The RAM output is the value before this clock's
                    // writeback, so include stage1_data in the final compare.
                    output_data <= completed_saturated;
                end

                // ------------------------------
                // counter 更新
                // ------------------------------
                if (!draining && input_valid) begin
                    if (max_read_addr < MAX_DEPTH-1)
                        max_read_addr <= max_read_addr + 1'b1;
                    else
                        max_read_addr <= '0;

                    if (lane_count < LANES-1) begin
                        lane_count <= lane_count + 1'b1;
                    end else begin
                        lane_count <= '0;
                        if (h_count < IN_H-1) begin
                            h_count <= h_count + 1'b1;
                        end else begin
                            h_count <= '0;
                            if (w_count < IN_W-1) begin
                                w_count <= w_count + 1'b1;
                                if (stride_phase < STRIDE_W-1) begin
                                    stride_phase <= stride_phase + 1'b1;
                                end else begin
                                    stride_phase   <= '0;
                                    current_window <= current_window + 1'b1;
                                end
                            end else begin
                                w_count        <= '0;
                                stride_phase   <= '0;
                                current_window <= '0;
                                if (group_count < OUT_GROUPS-1)
                                    group_count <= group_count + 1'b1;
                                else
                                    group_count <= '0;
                            end
                        end
                    end

                    if (input_is_last)
                        draining <= 1'b1;
                end

                // Wait until the final synchronous RAM result has passed
                // through the compare/write pipeline before asserting done.
                if (stage1_valid && stage1_last) begin
                    active   <= 1'b0;
                    draining <= 1'b0;
                    done     <= 1'b1;
                end
            end
        end
    end
endmodule
