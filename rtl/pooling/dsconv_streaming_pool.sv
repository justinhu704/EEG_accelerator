// Streaming max-pool for DS-Conv position-major output.
// Input order: width -> height -> channel group -> lane.
module dsconv_streaming_pool #(
    parameter int IN_H     = 19,
    parameter int IN_W     = 152,
    parameter int IN_CH    = 20,
    parameter int POOL_W   = 10,
    parameter int STRIDE_W = 8,
    parameter int INPUT_F  = 11,
    parameter int OUTPUT_F = 11,
    parameter int OUT_H    = IN_H,
    parameter int OUT_W    = ((IN_W - POOL_W) / STRIDE_W) + 1
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    input  logic               input_valid,
    input  logic               input_last,
    input  logic signed [15:0] input_data,
    input  logic [$clog2(IN_H)-1:0]  input_h,
    input  logic [$clog2(IN_W)-1:0]  input_w,
    input  logic [$clog2(IN_CH)-1:0] input_channel,

    output logic               output_valid,
    output logic [12:0]        output_addr,
    output logic signed [15:0] output_data
);
    localparam int OVERLAP    = POOL_W - STRIDE_W;
    localparam int MAX_DEPTH  = IN_H * IN_CH;
    localparam int MAX_ADDR_W = (MAX_DEPTH <= 1) ? 1 : $clog2(MAX_DEPTH);
    localparam int WINDOW_W   = (OUT_W + 1 <= 1) ? 1 : $clog2(OUT_W + 1);
    localparam int STRIDE_W_W = (STRIDE_W <= 1) ? 1 : $clog2(STRIDE_W);

    logic active;
    logic draining;
    logic [WINDOW_W-1:0] current_window;
    logic [STRIDE_W_W-1:0] stride_phase;

    logic [MAX_ADDR_W-1:0] max_read_addr;
    logic [MAX_ADDR_W-1:0] max_even_write_addr;
    logic [MAX_ADDR_W-1:0] max_odd_write_addr;
    logic signed [15:0] max_even_read_data;
    logic signed [15:0] max_odd_read_data;
    logic signed [15:0] max_even_write_data;
    logic signed [15:0] max_odd_write_data;
    logic max_even_write_en;
    logic max_odd_write_en;

    // 同步 RAM 的 metadata pipeline。
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

    logic current_window_valid;
    logic previous_window_active;
    logic previous_window_ends;
    logic current_window_is_odd;
    logic [WINDOW_W-1:0] previous_window;
    logic end_of_width;
    logic [31:0] max_read_addr_full;
    logic [31:0] stage1_output_addr_full;

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
        max_read_addr_full = input_h * IN_CH + input_channel;
        max_read_addr = max_read_addr_full[MAX_ADDR_W-1:0];

        current_window_valid = (current_window < OUT_W);
        previous_window_active = (current_window > 0)
                               && (stride_phase < OVERLAP);
        previous_window_ends = previous_window_active
                             && (stride_phase == OVERLAP-1);
        current_window_is_odd = current_window[0];
        previous_window = current_window - 1'b1;
        end_of_width = (input_h == IN_H-1)
                     && (input_channel == IN_CH-1);
        stage1_output_addr_full = input_h + OUT_H
                                * (previous_window + OUT_W * input_channel);
    end

    initial begin
        if (POOL_W <= STRIDE_W)
            $error("dsconv_streaming_pool requires overlapping windows");
        if (POOL_W > 2 * STRIDE_W)
            $error("dsconv_streaming_pool supports at most two active windows");
    end

    activation_ram #(
        .DATA_W(16), .DEPTH(MAX_DEPTH), .ADDR_W(MAX_ADDR_W),
        .MEM_FILE(""), .USE_READ_ENABLE(1'b1)
    ) u_max_even_ram (
        .clk(clk),
        .write_en(max_even_write_en),
        .write_addr(max_even_write_addr),
        .write_data(max_even_write_data),
        .read_en(active && !draining && input_valid),
        .read_addr(max_read_addr),
        .read_data(max_even_read_data)
    );

    activation_ram #(
        .DATA_W(16), .DEPTH(MAX_DEPTH), .ADDR_W(MAX_ADDR_W),
        .MEM_FILE(""), .USE_READ_ENABLE(1'b1)
    ) u_max_odd_ram (
        .clk(clk),
        .write_en(max_odd_write_en),
        .write_addr(max_odd_write_addr),
        .write_data(max_odd_write_data),
        .read_en(active && !draining && input_valid),
        .read_addr(max_read_addr),
        .read_data(max_odd_read_data)
    );

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

    // 比較同步 RAM 輸出，並更新目前與重疊中的前一個 window。
    always_comb begin
        max_even_write_en = 1'b0;
        max_odd_write_en = 1'b0;
        max_even_write_addr = stage1_max_addr;
        max_odd_write_addr = stage1_max_addr;
        max_even_write_data = stage1_data;
        max_odd_write_data = stage1_data;

        if (stage1_valid) begin
            if (stage1_current_window_valid) begin
                if (!stage1_current_window_is_odd) begin
                    max_even_write_en = 1'b1;
                    if (!stage1_first_value)
                        max_even_write_data = max16(stage1_data,
                                                    max_even_read_data);
                end else begin
                    max_odd_write_en = 1'b1;
                    if (!stage1_first_value)
                        max_odd_write_data = max16(stage1_data,
                                                   max_odd_read_data);
                end
            end

            if (stage1_previous_window_active) begin
                if (!stage1_current_window_is_odd) begin
                    max_odd_write_en = 1'b1;
                    max_odd_write_data = max16(stage1_data,
                                               max_odd_read_data);
                end else begin
                    max_even_write_en = 1'b1;
                    max_even_write_data = max16(stage1_data,
                                                max_even_read_data);
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            draining <= 1'b0;
            done <= 1'b0;
            output_valid <= 1'b0;
            output_addr <= '0;
            output_data <= '0;
            current_window <= '0;
            stride_phase <= '0;
            stage1_valid <= 1'b0;
            stage1_last <= 1'b0;
            stage1_data <= '0;
            stage1_max_addr <= '0;
            stage1_current_window_valid <= 1'b0;
            stage1_previous_window_active <= 1'b0;
            stage1_previous_window_ends <= 1'b0;
            stage1_current_window_is_odd <= 1'b0;
            stage1_first_value <= 1'b0;
            stage1_output_addr <= '0;
        end else begin
            done <= 1'b0;
            output_valid <= 1'b0;
            stage1_valid <= 1'b0;

            if (!active) begin
                if (start) begin
                    active <= 1'b1;
                    draining <= 1'b0;
                    current_window <= '0;
                    stride_phase <= '0;
                end
            end else begin
                if (!draining && input_valid) begin
                    stage1_valid <= 1'b1;
                    stage1_last <= input_last;
                    stage1_data <= input_data;
                    stage1_max_addr <= max_read_addr;
                    stage1_current_window_valid <= current_window_valid;
                    stage1_previous_window_active <= previous_window_active;
                    stage1_previous_window_ends <= previous_window_ends;
                    stage1_current_window_is_odd <= current_window_is_odd;
                    stage1_first_value <= (stride_phase == 0);
                    stage1_output_addr <= stage1_output_addr_full[12:0];

                    // width 只在完整接收該欄的所有 height/channel 後推進。
                    if (end_of_width) begin
                        if (input_w == IN_W-1) begin
                            current_window <= '0;
                            stride_phase <= '0;
                        end else if (stride_phase == STRIDE_W-1) begin
                            current_window <= current_window + 1'b1;
                            stride_phase <= '0;
                        end else begin
                            stride_phase <= stride_phase + 1'b1;
                        end
                    end

                    if (input_last)
                        draining <= 1'b1;
                end

                if (stage1_valid && stage1_previous_window_ends) begin
                    output_valid <= 1'b1;
                    output_addr <= stage1_output_addr;
                    output_data <= completed_saturated;
                end

                if (stage1_valid && stage1_last) begin
                    active <= 1'b0;
                    draining <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
