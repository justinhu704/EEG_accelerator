// Batch-normalization inference affine transform:
// y = A[channel] * x + B[channel]
//
// BIAS_SHIFT aligns B with the fractional format of x*A.
// OUTPUT_SHIFT converts the aligned sum to the requested output format.
module bn_affine #(
    parameter int CHANNELS     = 21,
    parameter int BIAS_SHIFT   = 11,
    parameter int OUTPUT_SHIFT = 13,
    parameter     A_FILE       = "mem/weights/bn1_A.mem",
    parameter     B_FILE       = "mem/weights/bn1_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic signed [15:0] in_data,
    input  logic [31:0]        in_ch_idx,
    output logic               out_valid,
    output logic signed [15:0] out_data
);
    logic signed [15:0] A_mem [0:CHANNELS-1];
    logic signed [15:0] B_mem [0:CHANNELS-1];

    logic               stage1_valid;
    logic signed [15:0] stage1_data;
    logic signed [15:0] stage1_A;
    logic signed [15:0] stage1_B;

    logic               stage2_valid;
    logic signed [31:0] stage2_product;
    logic signed [47:0] stage2_B_aligned;

    logic signed [47:0] stage1_B_extended;
    logic signed [47:0] stage2_product_extended;
    logic signed [47:0] aligned_sum;
    logic signed [47:0] scaled_sum;
    logic signed [15:0] saturated_out;

    initial begin
        $readmemh(A_FILE, A_mem, 0, CHANNELS-1);
        $readmemh(B_FILE, B_mem, 0, CHANNELS-1);
    end

    // Pipeline stage 1: capture data and per-channel parameters.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 1'b0;
            stage1_data  <= '0;
            stage1_A     <= '0;
            stage1_B     <= '0;
        end else begin
            stage1_valid <= in_valid;
            if (in_valid) begin
                stage1_data <= in_data;
                stage1_A    <= A_mem[in_ch_idx];
                stage1_B    <= B_mem[in_ch_idx];
            end
        end
    end

    always_comb begin
        stage1_B_extended = {{32{stage1_B[15]}}, stage1_B};
    end

    // Pipeline stage 2: multiply and align B to the product format.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_valid     <= 1'b0;
            stage2_product   <= '0;
            stage2_B_aligned <= '0;
        end else begin
            stage2_valid <= stage1_valid;
            if (stage1_valid) begin
                stage2_product   <= stage1_data * stage1_A;
                stage2_B_aligned <= stage1_B_extended <<< BIAS_SHIFT;
            end
        end
    end

    // Scale the result and clamp it to signed 16-bit.
    always_comb begin
        stage2_product_extended =
            {{16{stage2_product[31]}}, stage2_product};
        aligned_sum = stage2_product_extended + stage2_B_aligned;
        scaled_sum  = aligned_sum >>> OUTPUT_SHIFT;
    end

    sat16 u_sat16 (
        .value_in  (scaled_sum),
        .value_out (saturated_out)
    );

    always_comb begin
        out_valid = stage2_valid;
        out_data  = stage2_valid ? saturated_out : 16'sd0;
    end
endmodule
