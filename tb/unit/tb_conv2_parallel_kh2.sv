`timescale 1ns/1ps

// Exact regression between the verified Lane5 counter Conv2 and the new
// Lane5 KH2 pipeline. All addressed raw Conv2 outputs must be bit-identical.
module tb_conv2_parallel_kh2;
    localparam int IN_H = 20;
    localparam int IN_W = 156;
    localparam int IN_CH = 21;
    localparam int K_H = 2;
    localparam int K_W = 5;
    localparam int OUT_CH = 20;
    localparam int LANES = 5;
    localparam int OUT_H = 19;
    localparam int OUT_W = 152;
    localparam int OUT_GROUPS = OUT_CH / LANES;
    localparam int INPUT_DEPTH = IN_H * IN_W * IN_CH;
    localparam int OLD_WEIGHT_DEPTH = K_H * K_W * IN_CH * OUT_GROUPS;
    localparam int KH2_WEIGHT_DEPTH = K_W * IN_CH * OUT_GROUPS;
    localparam int OUTPUT_SIZE = OUT_H * OUT_W * OUT_CH;
    localparam int MAX_CYCLES = 3_000_000;

    logic clk, rst_n, start;

    logic old_busy, old_done, old_valid;
    logic [31:0] old_input_addr, old_weight_addr, old_bias_addr;
    logic [31:0] old_output_addr, old_output_ch;
    logic signed [15:0] old_input_data, old_output_data;
    logic signed [(16*LANES)-1:0] old_weight_data, old_bias_data;

    logic new_busy, new_done, new_valid;
    logic [31:0] new_input_addr0, new_input_addr1;
    logic [31:0] new_weight_addr, new_bias_addr;
    logic [31:0] new_output_addr, new_output_ch;
    logic signed [15:0] new_input_data0, new_input_data1;
    logic signed [15:0] new_output_data;
    logic signed [(32*LANES)-1:0] new_weight_data;
    logic signed [(16*LANES)-1:0] new_bias_data;

    logic signed [15:0] old_results [0:OUTPUT_SIZE-1];
    logic signed [15:0] new_results [0:OUTPUT_SIZE-1];
    logic old_seen [0:OUTPUT_SIZE-1];
    logic new_seen [0:OUTPUT_SIZE-1];
    logic old_finished, new_finished;
    integer old_count, new_count, old_done_cycle, new_done_cycle;
    integer error_count, cycle_count, compare_index;

    always #5 clk = ~clk;

    activation_ram #(
        .DATA_W(16), .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/golden/q_relu1_act.mem")
    ) u_old_input_ram (
        .clk(clk), .write_en(1'b0), .write_addr('0), .write_data('0),
        .read_addr(old_input_addr[$clog2(INPUT_DEPTH)-1:0]),
        .read_data(old_input_data)
    );

    activation_ram_2r1w #(
        .DATA_W(16), .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/golden/q_relu1_act.mem")
    ) u_new_input_ram (
        .clk(clk), .port_a_write_en(1'b0), .port_a_addr(
            new_input_addr0[$clog2(INPUT_DEPTH)-1:0]),
        .port_a_write_data('0), .port_a_read_data(new_input_data0),
        .port_b_read_addr(new_input_addr1[$clog2(INPUT_DEPTH)-1:0]),
        .port_b_read_data(new_input_data1)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OLD_WEIGHT_DEPTH),
        .ADDR_W($clog2(OLD_WEIGHT_DEPTH)),
        .MEM_FILE("../mem/weights/conv2_W_x5.mem")
    ) u_old_weight_rom (
        .clk(clk), .addr(old_weight_addr[$clog2(OLD_WEIGHT_DEPTH)-1:0]),
        .data(old_weight_data)
    );

    weight_rom #(
        .DATA_W(32*LANES), .DEPTH(KH2_WEIGHT_DEPTH),
        .ADDR_W($clog2(KH2_WEIGHT_DEPTH)),
        .MEM_FILE("../mem/weights/conv2_W_x5_kh2.mem")
    ) u_new_weight_rom (
        .clk(clk), .addr(new_weight_addr[$clog2(KH2_WEIGHT_DEPTH)-1:0]),
        .data(new_weight_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OUT_GROUPS),
        .ADDR_W($clog2(OUT_GROUPS)),
        .MEM_FILE("../mem/weights/conv2_b_x5.mem")
    ) u_old_bias_rom (
        .clk(clk), .addr(old_bias_addr[$clog2(OUT_GROUPS)-1:0]),
        .data(old_bias_data)
    );

    weight_rom #(
        .DATA_W(16*LANES), .DEPTH(OUT_GROUPS),
        .ADDR_W($clog2(OUT_GROUPS)),
        .MEM_FILE("../mem/weights/conv2_b_x5.mem")
    ) u_new_bias_rom (
        .clk(clk), .addr(new_bias_addr[$clog2(OUT_GROUPS)-1:0]),
        .data(new_bias_data)
    );

    conv_engine_parallel_counter #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W), .LANES(LANES),
        .BIAS_SHIFT(10), .OUTPUT_SHIFT(15)
    ) u_old_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(old_busy), .done(old_done),
        .input_addr(old_input_addr), .input_data(old_input_data),
        .weight_addr(old_weight_addr), .weight_data(old_weight_data),
        .bias_addr(old_bias_addr), .bias_data(old_bias_data),
        .output_valid(old_valid), .output_addr(old_output_addr),
        .output_ch_idx(old_output_ch), .output_data(old_output_data)
    );

    conv_engine_parallel_kh2 #(
        .IN_H(IN_H), .IN_W(IN_W), .IN_CH(IN_CH),
        .K_H(K_H), .K_W(K_W), .OUT_CH(OUT_CH),
        .OUT_H(OUT_H), .OUT_W(OUT_W), .LANES(LANES),
        .BIAS_SHIFT(10), .OUTPUT_SHIFT(15)
    ) u_new_conv (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(new_busy), .done(new_done),
        .input_addr_kh0(new_input_addr0),
        .input_addr_kh1(new_input_addr1),
        .input_data_kh0(new_input_data0),
        .input_data_kh1(new_input_data1),
        .weight_addr(new_weight_addr), .weight_data(new_weight_data),
        .bias_addr(new_bias_addr), .bias_data(new_bias_data),
        .output_valid(new_valid), .output_addr(new_output_addr),
        .output_ch_idx(new_output_ch), .output_data(new_output_data)
    );

    task automatic require_file(input string file_name);
        integer file_handle;
        begin
            file_handle = $fopen(file_name, "r");
            if (file_handle == 0)
                $fatal(1, "Cannot open required file: %s", file_name);
            $fclose(file_handle);
        end
    endtask

    initial begin
        require_file("../mem/golden/q_relu1_act.mem");
        require_file("../mem/weights/conv2_W_x5.mem");
        require_file("../mem/weights/conv2_W_x5_kh2.mem");
        require_file("../mem/weights/conv2_b_x5.mem");
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        old_finished = 1'b0;
        new_finished = 1'b0;
        old_count = 0;
        new_count = 0;
        old_done_cycle = 0;
        new_done_cycle = 0;
        error_count = 0;
        cycle_count = 0;
        for (compare_index = 0; compare_index < OUTPUT_SIZE;
             compare_index = compare_index + 1) begin
            old_seen[compare_index] = 1'b0;
            new_seen[compare_index] = 1'b0;
        end

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait (old_finished && new_finished);
        repeat (2) @(posedge clk);

        if ((old_count != OUTPUT_SIZE) || (new_count != OUTPUT_SIZE)) begin
            $display("COUNT mismatch old=%0d new=%0d expected=%0d",
                     old_count, new_count, OUTPUT_SIZE);
            error_count = error_count + 1;
        end

        for (compare_index = 0; compare_index < OUTPUT_SIZE;
             compare_index = compare_index + 1) begin
            if (!old_seen[compare_index] || !new_seen[compare_index] ||
                (old_results[compare_index] !== new_results[compare_index])) begin
                if (error_count < 20)
                    $display("Mismatch addr=%0d old=%0d new=%0d",
                             compare_index, old_results[compare_index],
                             new_results[compare_index]);
                error_count = error_count + 1;
            end
        end

        if (error_count == 0) begin
            $display("PASS: KH2 Conv2 exactly matches verified Conv2.");
            $display("Checked=%0d old_cycles=%0d kh2_cycles=%0d saved=%0d",
                     OUTPUT_SIZE, old_done_cycle, new_done_cycle,
                     old_done_cycle - new_done_cycle);
        end else begin
            $display("FAIL: KH2 Conv2 errors=%0d", error_count);
            $fatal(1);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (old_valid) begin
            if ($isunknown(old_output_addr) ||
                (old_output_addr >= OUTPUT_SIZE) || old_seen[old_output_addr])
                error_count = error_count + 1;
            else begin
                old_seen[old_output_addr] = 1'b1;
                old_results[old_output_addr] = old_output_data;
                old_count = old_count + 1;
            end
        end
        if (new_valid) begin
            if ($isunknown(new_output_addr) ||
                (new_output_addr >= OUTPUT_SIZE) || new_seen[new_output_addr])
                error_count = error_count + 1;
            else begin
                new_seen[new_output_addr] = 1'b1;
                new_results[new_output_addr] = new_output_data;
                new_count = new_count + 1;
            end
        end
        if (old_done) begin
            old_finished = 1'b1;
            old_done_cycle = cycle_count;
        end
        if (new_done) begin
            new_finished = 1'b1;
            new_done_cycle = cycle_count;
        end
        if (cycle_count > MAX_CYCLES)
            $fatal(1, "KH2 Conv2 timeout at cycle %0d", cycle_count);
    end
endmodule
