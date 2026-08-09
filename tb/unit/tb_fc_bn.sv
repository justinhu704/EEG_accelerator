`timescale 1ns/1ps

// Verifies the classifier's FC1 -> ReLU -> BN transform independently.
module tb_fc_bn;
    localparam int SIZE = 40;
    localparam int SAMPLES = 4;

    logic clk, rst_n, in_valid;
    logic [31:0] in_index;
    logic signed [15:0] in_data;
    logic relu_valid, bn_valid;
    logic signed [15:0] relu_data, bn_data;
    logic signed [15:0] fc1_mem [0:SAMPLES*SIZE-1];
    logic signed [15:0] relu_golden [0:SAMPLES*SIZE-1];
    logic signed [15:0] bn_golden [0:SAMPLES*SIZE-1];
    logic [31:0] addr_d1, addr_d2;
    logic valid_d1;
    integer i, relu_checked, bn_checked, errors;

    relu u_relu (
        .in_valid(in_valid), .in_data(in_data),
        .out_valid(relu_valid), .out_data(relu_data)
    );

    bn_affine #(
        .CHANNELS(40), .BIAS_SHIFT(14), .OUTPUT_SHIFT(14),
        .A_FILE("../mem/weights/bn_2_A.mem"),
        .B_FILE("../mem/weights/bn_2_B.mem")
    ) u_bn (
        .clk(clk), .rst_n(rst_n),
        .in_valid(relu_valid), .in_data(relu_data),
        .in_ch_idx(in_index), .out_valid(bn_valid), .out_data(bn_data)
    );

    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 0; addr_d1 <= 0; addr_d2 <= 0;
        end else begin
            valid_d1 <= in_valid;
            if (in_valid) addr_d1 <= in_index;
            if (valid_d1) addr_d2 <= addr_d1;
        end
    end

    initial begin
        $readmemh("../mem/golden/q_fc_1_act.mem", fc1_mem);
        $readmemh("../mem/golden/q_fc_relu_act.mem", relu_golden);
        $readmemh("../mem/golden/q_bn_2_act.mem", bn_golden);
        clk = 0; rst_n = 0; in_valid = 0; in_index = 0; in_data = 0;
        relu_checked = 0; bn_checked = 0; errors = 0;
        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;
        for (i = 0; i < SIZE; i = i + 1) begin
            in_valid = 1;
            in_index = i;
            in_data = fc1_mem[i];
            @(negedge clk);
        end
        in_valid = 0;
        repeat (4) @(posedge clk);
        if ((relu_checked == SIZE) && (bn_checked == SIZE) && (errors == 0))
            $display("PASS: FC1 ReLU and BN exact, ReLU=%0d BN=%0d",
                     relu_checked, bn_checked);
        else begin
            $display("FAIL: FC ReLU/BN, ReLU=%0d BN=%0d errors=%0d",
                     relu_checked, bn_checked, errors);
            $fatal(1);
        end
        $finish;
    end

    always @(posedge clk) begin
        if (relu_valid) begin
            if ($signed(relu_data) !== $signed(relu_golden[in_index])) begin
                $display("ReLU mismatch index=%0d got=%0d expected=%0d",
                         in_index, relu_data, relu_golden[in_index]);
                errors = errors + 1;
            end
            relu_checked = relu_checked + 1;
        end
        if (bn_valid) begin
            if ($signed(bn_data) !== $signed(bn_golden[addr_d2])) begin
                $display("BN mismatch index=%0d got=%0d expected=%0d",
                         addr_d2, bn_data, bn_golden[addr_d2]);
                errors = errors + 1;
            end
            bn_checked = bn_checked + 1;
        end
    end
endmodule
