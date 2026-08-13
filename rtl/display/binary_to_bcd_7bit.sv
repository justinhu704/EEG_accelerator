// Divider-free double-dabble conversion for unsigned values 0..127.
module binary_to_bcd_7bit (
    input  logic [6:0] binary,
    output logic [3:0] hundreds,
    output logic [3:0] tens,
    output logic [3:0] ones
);
    logic [18:0] shift_reg;
    integer i;

    always_comb begin
        shift_reg = '0;
        shift_reg[6:0] = binary;

        for (i = 0; i < 7; i = i + 1) begin
            if (shift_reg[10:7] >= 5)
                shift_reg[10:7] = shift_reg[10:7] + 4'd3;
            if (shift_reg[14:11] >= 5)
                shift_reg[14:11] = shift_reg[14:11] + 4'd3;
            if (shift_reg[18:15] >= 5)
                shift_reg[18:15] = shift_reg[18:15] + 4'd3;
            shift_reg = shift_reg << 1;
        end

        ones     = shift_reg[10:7];
        tens     = shift_reg[14:11];
        hundreds = shift_reg[18:15];
    end
endmodule
