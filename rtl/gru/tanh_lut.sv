// 256-entry synchronous half-tanh lookup table used by the GRU candidate state.
//
// Input format : signed Q9 (real value = in_data / 2^9)
// Output format: signed Q15 in the range -32768 .. +32767
// Stored range : 0 .. +3.46484375 (0 .. 1774 in Q9)
// Negative inputs use tanh(-x) = -tanh(x).
// Latency      : one clock; one new input may be accepted every clock.
module tanh_lut #(
    parameter MEM_FILE = "mem/lut/tanh_half_lut_q15.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic signed [15:0] in_data,
    output logic               out_valid,
    output logic signed [15:0] out_data
);
    localparam int LUT_DEPTH = 256;
    localparam int RANGE_RAW = 1774;

    // index = floor(abs(x) * 255 / range), implemented without a divider.
    localparam int INDEX_GAIN  = 150725;
    localparam int INDEX_SHIFT = 20;

    logic signed [15:0] lut [0:LUT_DEPTH-1];
    logic [10:0] magnitude;
    logic [28:0] index_product;
    logic [7:0]  lut_addr;
    logic signed [15:0] rom_data;
    logic negative_d1;
    logic signed [16:0] negated_value;

    initial begin
        $readmemh(MEM_FILE, lut, 0, LUT_DEPTH-1);
    end

    always_comb begin
        magnitude = '0;
        index_product = '0;
        lut_addr = '0;

        if (($signed(in_data) <= -RANGE_RAW) ||
            ($signed(in_data) >=  RANGE_RAW)) begin
            magnitude = RANGE_RAW;
            lut_addr = 8'd255;
        end else begin
            if (in_data[15])
                magnitude = -$signed(in_data);
            else
                magnitude = in_data;
            index_product = magnitude * INDEX_GAIN;
            lut_addr = index_product >> INDEX_SHIFT;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            rom_data <= '0;
            negative_d1 <= 1'b0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                rom_data <= lut[lut_addr];
                negative_d1 <= in_data[15];
            end
        end
    end

    always_comb begin
        negated_value = -$signed(rom_data);
        if (!out_valid)
            out_data = '0;
        else if (negative_d1)
            out_data = negated_value[15:0];
        else
            out_data = rom_data;
    end
endmodule
