module sat16 (
    input  logic signed [47:0] value_in,
    output logic signed [15:0] value_out
);
    always_comb begin
        if (value_in > 48'sd32767)
            value_out = 16'sh7fff;
        else if (value_in < -48'sd32768)
            value_out = 16'sh8000;
        else
            value_out = value_in[15:0];
    end
endmodule
