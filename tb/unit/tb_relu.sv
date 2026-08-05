`timescale 1ns/1ps

// Verifies ReLU1, ReLU2, and ReLU3 against the first MATLAB sample.
module tb_relu;
    localparam int RELU1_SIZE = 20 * 156 * 21;
    localparam int RELU2_SIZE = 19 * 152 * 20;
    localparam int RELU3_SIZE = 18 * 14  * 15;
    localparam int SAMPLES = 4;

    logic relu1_in_valid, relu2_in_valid, relu3_in_valid;
    logic signed [15:0] relu1_in_data, relu2_in_data, relu3_in_data;
    logic relu1_out_valid, relu2_out_valid, relu3_out_valid;
    logic signed [15:0] relu1_out_data, relu2_out_data, relu3_out_data;

    logic signed [15:0] relu1_input  [0:SAMPLES*RELU1_SIZE-1];
    logic signed [15:0] relu1_golden [0:SAMPLES*RELU1_SIZE-1];
    logic signed [15:0] relu2_input  [0:SAMPLES*RELU2_SIZE-1];
    logic signed [15:0] relu2_golden [0:SAMPLES*RELU2_SIZE-1];
    logic signed [15:0] relu3_input  [0:SAMPLES*RELU3_SIZE-1];
    logic signed [15:0] relu3_golden [0:SAMPLES*RELU3_SIZE-1];

    integer i;
    integer error_count;

    relu #(.OUTPUT_LEFT_SHIFT(0)) dut_relu1 (
        .in_valid(relu1_in_valid), .in_data(relu1_in_data),
        .out_valid(relu1_out_valid), .out_data(relu1_out_data)
    );

    // ReLU2 converts F11 input to F12 output.
    relu #(.OUTPUT_LEFT_SHIFT(1)) dut_relu2 (
        .in_valid(relu2_in_valid), .in_data(relu2_in_data),
        .out_valid(relu2_out_valid), .out_data(relu2_out_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(0)) dut_relu3 (
        .in_valid(relu3_in_valid), .in_data(relu3_in_data),
        .out_valid(relu3_out_valid), .out_data(relu3_out_data)
    );

    task automatic require_file(input string file_name);
        integer file_handle;
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0) begin
                $display("FATAL: cannot open required file: %s", file_name);
                $display("Set the simulator working directory to C:/EEG_Project before starting simulation.");
                $fatal(1);
            end
            $fclose(file_handle);
        end
    endtask

    initial begin
        require_file("../mem/golden/q_bn1_act.mem");
        require_file("../mem/golden/q_relu1_act.mem");
        require_file("../mem/golden/q_bn2_act.mem");
        require_file("../mem/golden/q_relu2_act.mem");
        require_file("../mem/golden/q_bn3_act.mem");
        require_file("../mem/golden/q_relu3_act.mem");

        $readmemh("../mem/golden/q_bn1_act.mem",   relu1_input);
        $readmemh("../mem/golden/q_relu1_act.mem", relu1_golden);
        $readmemh("../mem/golden/q_bn2_act.mem",   relu2_input);
        $readmemh("../mem/golden/q_relu2_act.mem", relu2_golden);
        $readmemh("../mem/golden/q_bn3_act.mem",   relu3_input);
        $readmemh("../mem/golden/q_relu3_act.mem", relu3_golden);

        error_count = 0;
        relu1_in_valid = 1'b0;
        relu2_in_valid = 1'b0;
        relu3_in_valid = 1'b0;
        relu1_in_data = '0;
        relu2_in_data = '0;
        relu3_in_data = '0;
        #1;

        for (i = 0; i < RELU1_SIZE; i = i + 1) begin
            relu1_in_valid = 1'b1;
            relu1_in_data = relu1_input[i];
            #1;
            if ((relu1_out_valid !== 1'b1) ||
                $isunknown(relu1_out_data) ||
                (relu1_out_data !== relu1_golden[i])) begin
                if (error_count < 20)
                    $display("ReLU1 mismatch addr=%0d got=%0d expected=%0d",
                             i, relu1_out_data, relu1_golden[i]);
                error_count = error_count + 1;
            end
        end
        relu1_in_valid = 1'b0;

        for (i = 0; i < RELU2_SIZE; i = i + 1) begin
            relu2_in_valid = 1'b1;
            relu2_in_data = relu2_input[i];
            #1;
            if ((relu2_out_valid !== 1'b1) ||
                $isunknown(relu2_out_data) ||
                (relu2_out_data !== relu2_golden[i])) begin
                if (error_count < 20)
                    $display("ReLU2 mismatch addr=%0d got=%0d expected=%0d",
                             i, relu2_out_data, relu2_golden[i]);
                error_count = error_count + 1;
            end
        end
        relu2_in_valid = 1'b0;

        for (i = 0; i < RELU3_SIZE; i = i + 1) begin
            relu3_in_valid = 1'b1;
            relu3_in_data = relu3_input[i];
            #1;
            if ((relu3_out_valid !== 1'b1) ||
                $isunknown(relu3_out_data) ||
                (relu3_out_data !== relu3_golden[i])) begin
                if (error_count < 20)
                    $display("ReLU3 mismatch addr=%0d got=%0d expected=%0d",
                             i, relu3_out_data, relu3_golden[i]);
                error_count = error_count + 1;
            end
        end
        relu3_in_valid = 1'b0;
        #1;

        if (error_count == 0) begin
            $display("PASS: ReLU1/ReLU2/ReLU3 exactly match MATLAB.");
            $display("ReLU1 checked=%0d", RELU1_SIZE);
            $display("ReLU2 checked=%0d", RELU2_SIZE);
            $display("ReLU3 checked=%0d", RELU3_SIZE);
        end else begin
            $display("FAIL: ReLU verification errors=%0d", error_count);
        end
        $finish;
    end
endmodule
