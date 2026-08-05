// ReLU with an optional fixed-point left shift:
// y = max(0, x) * 2^OUTPUT_LEFT_SHIFT
module relu #(
    parameter int OUTPUT_LEFT_SHIFT = 0
) (
    input  logic               in_valid,
    input  logic signed [15:0] in_data,
    output logic               out_valid,
    output logic signed [15:0] out_data
);
    logic signed [47:0] scaled_positive;
    logic signed [15:0] saturated_positive;

    always_comb begin
        scaled_positive = 48'sd0;
        if (in_valid && !in_data[15])
            scaled_positive = {{32{1'b0}}, in_data}
                            <<< OUTPUT_LEFT_SHIFT;
    end

    sat16 u_sat16 (
        .value_in  (scaled_positive),
        .value_out (saturated_positive)
    );

    always_comb begin
        out_valid = in_valid;
        if (!in_valid || in_data[15])
            out_data = 16'sd0;
        else
            out_data = saturated_positive;
    end
endmodule
