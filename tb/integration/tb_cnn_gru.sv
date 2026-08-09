`timescale 1ns/1ps

// Full standalone-top integration test: CNN -> Pool2 -> GRU -> RAM A.
module tb_cnn_gru;
    localparam int SAMPLES = 4;
    localparam int OUTPUT_SIZE = 9 * 18;
    // Bound for the 256-entry LUT approximation versus MATLAB's continuous
    // sigmoid/tanh result (512 Q15 codes = 0.015625).
    localparam int TOLERANCE = 512;
    localparam int MAX_CYCLES = 20_000_000;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic output_valid;
    logic [31:0] output_addr;
    logic signed [15:0] output_data;
    logic [15:0] ram_a_read_addr;
    logic signed [15:0] ram_a_read_data;
    logic signed [15:0] expected_mem [0:SAMPLES*OUTPUT_SIZE-1];

    integer stream_checked;
    integer stream_errors;
    integer stream_max_difference;
    integer ram_checked;
    integer ram_errors;
    integer ram_max_difference;
    integer difference;
    integer cycle_count;
    integer i;

    cnn_gru_top #(
        .INPUT_FILE("../mem/golden/q_in_act.mem"),
        .CONV1_W_FILE("../mem/weights/conv1_W.mem"),
        .CONV1_B_FILE("../mem/weights/conv1_b.mem"),
        .BN1_A_FILE("../mem/weights/bn1_A.mem"),
        .BN1_B_FILE("../mem/weights/bn1_B.mem"),
        .CONV2_W_FILE("../mem/weights/conv2_W.mem"),
        .CONV2_B_FILE("../mem/weights/conv2_b.mem"),
        .BN2_A_FILE("../mem/weights/bn2_A.mem"),
        .BN2_B_FILE("../mem/weights/bn2_B.mem"),
        .CONV3_W_FILE("../mem/weights/conv3_W.mem"),
        .CONV3_B_FILE("../mem/weights/conv3_b.mem"),
        .BN3_A_FILE("../mem/weights/bn3_A.mem"),
        .BN3_B_FILE("../mem/weights/bn3_B.mem"),
        .GRU_WR_FILE("../mem/weights/gru_Wr.mem"),
        .GRU_WZ_FILE("../mem/weights/gru_Wz.mem"),
        .GRU_WH_FILE("../mem/weights/gru_Wh.mem"),
        .GRU_UR_FILE("../mem/weights/gru_Ur.mem"),
        .GRU_UZ_FILE("../mem/weights/gru_Uz.mem"),
        .GRU_UH_FILE("../mem/weights/gru_Uh.mem"),
        .GRU_BR_FILE("../mem/weights/gru_br.mem"),
        .GRU_BZ_FILE("../mem/weights/gru_bz.mem"),
        .GRU_BH_FILE("../mem/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .output_valid(output_valid),
        .output_addr(output_addr), .output_data(output_data),
        .ram_a_read_addr(ram_a_read_addr),
        .ram_a_read_data(ram_a_read_data)
    );

    always #5 clk = ~clk;

    function automatic integer absolute_difference(
        input logic signed [15:0] actual,
        input logic signed [15:0] expected
    );
        integer value;
        begin
            value = $signed(actual) - $signed(expected);
            if (value < 0)
                value = -value;
            absolute_difference = value;
        end
    endfunction

    task automatic require_file(input string file_name);
        integer file_handle;
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0) begin
                $display("FATAL: cannot open required file: %s", file_name);
                $fatal(1);
            end
            $fclose(file_handle);
        end
    endtask

    initial begin
        require_file("../mem/golden/q_in_act.mem");
        require_file("../mem/golden/q_flatten_act.mem");
        require_file("../mem/weights/conv1_W.mem");
        require_file("../mem/weights/conv2_W.mem");
        require_file("../mem/weights/conv3_W.mem");
        require_file("../mem/weights/gru_Wr.mem");
        require_file("../mem/weights/gru_Uh.mem");
        require_file("../mem/lut/sigmoid_half_lut_q15.mem");
        require_file("../mem/lut/tanh_half_lut_q15.mem");
        $readmemh("../mem/golden/q_flatten_act.mem", expected_mem);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        ram_a_read_addr = '0;
        stream_checked = 0;
        stream_errors = 0;
        stream_max_difference = 0;
        ram_checked = 0;
        ram_errors = 0;
        ram_max_difference = 0;
        cycle_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // done occurs after GRU has finished and all 162 writes are complete.
        wait (done);
        repeat (2) @(posedge clk);

        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            @(negedge clk);
            ram_a_read_addr = i[15:0];
            @(posedge clk);
            #1;

            if ($isunknown(ram_a_read_data)) begin
                if (ram_errors < 20)
                    $display("RAM A X/Z at addr=%0d", i);
                ram_errors = ram_errors + 1;
            end else begin
                difference = absolute_difference(ram_a_read_data,
                                                 expected_mem[i]);
                if (difference > ram_max_difference)
                    ram_max_difference = difference;
                if (difference > TOLERANCE) begin
                    if (ram_errors < 20)
                        $display("RAM A mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 i, ram_a_read_data,
                                 expected_mem[i], difference);
                    ram_errors = ram_errors + 1;
                end
            end
            ram_checked = ram_checked + 1;
        end

        if ((stream_checked == OUTPUT_SIZE) &&
            (ram_checked == OUTPUT_SIZE) &&
            (stream_errors == 0) && (ram_errors == 0)) begin
            $display("PASS: standalone CNN + GRU stored in RAM A.");
            $display("GRU stream checked=%0d max_diff=%0d",
                     stream_checked, stream_max_difference);
            $display("RAM A checked=%0d max_diff=%0d",
                     ram_checked, ram_max_difference);
        end else begin
            $display("FAIL: CNN + GRU integration.");
            $display("Stream checked=%0d errors=%0d max_diff=%0d",
                     stream_checked, stream_errors,
                     stream_max_difference);
            $display("RAM A checked=%0d errors=%0d max_diff=%0d",
                     ram_checked, ram_errors, ram_max_difference);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (output_valid) begin
            if ($isunknown(output_addr) || $isunknown(output_data) ||
                (output_addr >= OUTPUT_SIZE)) begin
                stream_errors = stream_errors + 1;
            end else begin
                difference = absolute_difference(
                    output_data, expected_mem[output_addr]
                );
                if (difference > stream_max_difference)
                    stream_max_difference = difference;
                if (difference > TOLERANCE) begin
                    if (stream_errors < 20)
                        $display("GRU mismatch addr=%0d got=%0d expected=%0d diff=%0d",
                                 output_addr, output_data,
                                 expected_mem[output_addr], difference);
                    stream_errors = stream_errors + 1;
                end
            end
            stream_checked = stream_checked + 1;
        end

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_cnn_gru timeout at cycle %0d", cycle_count);
            $finish;
        end
    end
endmodule
