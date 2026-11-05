// saturates a signed value to 16-bit signed

module sat16 #(
    parameter int INPUT_W = 48
) (
    input  logic signed [INPUT_W-1:0] value_in,
    output logic signed [15:0] value_out
);
    localparam logic signed [INPUT_W-1:0] SAT_MAX =
        {{(INPUT_W-16){1'b0}}, 16'sh7fff};
    localparam logic signed [INPUT_W-1:0] SAT_MIN =
        {{(INPUT_W-16){1'b1}}, 16'sh8000};

    always_comb begin
        // avoid overflow
        if (value_in > SAT_MAX)
            value_out = 16'sh7fff;
        else if (value_in < SAT_MIN)
            value_out = 16'sh8000;
        else
            value_out = value_in[15:0];
    end
endmodule
