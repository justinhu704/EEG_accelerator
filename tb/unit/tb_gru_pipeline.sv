`timescale 1ns/1ps

// Compares the original sequential GRU with the pipelined GRU. The engines
// have different latency, so results are stored and compared by output_addr.
module tb_gru_pipeline;
    localparam int INPUT_DEPTH = 4 * 18 * 15;
    localparam int OUTPUT_SIZE = 9 * 18;
    localparam int MAX_CYCLES = 100_000;

    logic clk;
    logic rst_n;
    logic start;

    logic old_busy;
    logic old_done;
    logic [31:0] old_input_addr;
    logic signed [15:0] old_input_data;
    logic old_output_valid;
    logic [31:0] old_output_addr;
    logic signed [15:0] old_output_data;

    logic new_busy;
    logic new_done;
    logic [31:0] new_input_addr;
    logic signed [15:0] new_input_data;
    logic new_output_valid;
    logic [31:0] new_output_addr;
    logic signed [15:0] new_output_data;

    logic signed [15:0] old_results [0:OUTPUT_SIZE-1];
    logic signed [15:0] new_results [0:OUTPUT_SIZE-1];
    logic old_finished;
    logic new_finished;

    integer old_count;
    integer new_count;
    integer error_count;
    integer cycle_count;
    integer old_done_cycle;
    integer new_done_cycle;
    integer compare_index;

    activation_ram #(
        .DATA_W(16),
        .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/golden/q_pool2_act.mem")
    ) u_old_input_ram (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0),
        .write_data('0),
        .read_addr(old_input_addr[$clog2(INPUT_DEPTH)-1:0]),
        .read_data(old_input_data)
    );

    activation_ram #(
        .DATA_W(16),
        .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/golden/q_pool2_act.mem")
    ) u_new_input_ram (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0),
        .write_data('0),
        .read_addr(new_input_addr[$clog2(INPUT_DEPTH)-1:0]),
        .read_data(new_input_data)
    );

    gru_engine #(
        .WR_FILE("../mem/weights/gru_Wr.mem"),
        .WZ_FILE("../mem/weights/gru_Wz.mem"),
        .WH_FILE("../mem/weights/gru_Wh.mem"),
        .UR_FILE("../mem/weights/gru_Ur.mem"),
        .UZ_FILE("../mem/weights/gru_Uz.mem"),
        .UH_FILE("../mem/weights/gru_Uh.mem"),
        .BR_FILE("../mem/weights/gru_br.mem"),
        .BZ_FILE("../mem/weights/gru_bz.mem"),
        .BH_FILE("../mem/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem")
    ) u_old_gru (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(old_busy), .done(old_done),
        .input_addr(old_input_addr), .input_data(old_input_data),
        .output_valid(old_output_valid),
        .output_addr(old_output_addr), .output_data(old_output_data)
    );

    gru_engine_pipeline #(
        .WR_FILE("../mem/weights/gru_Wr.mem"),
        .WZ_FILE("../mem/weights/gru_Wz.mem"),
        .WH_FILE("../mem/weights/gru_Wh.mem"),
        .UR_FILE("../mem/weights/gru_Ur.mem"),
        .UZ_FILE("../mem/weights/gru_Uz.mem"),
        .UH_FILE("../mem/weights/gru_Uh.mem"),
        .BR_FILE("../mem/weights/gru_br.mem"),
        .BZ_FILE("../mem/weights/gru_bz.mem"),
        .BH_FILE("../mem/weights/gru_bh.mem"),
        .SIGMOID_FILE("../mem/lut/sigmoid_half_lut_q15.mem"),
        .TANH_FILE("../mem/lut/tanh_half_lut_q15.mem")
    ) u_new_gru (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(new_busy), .done(new_done),
        .input_addr(new_input_addr), .input_data(new_input_data),
        .output_valid(new_output_valid),
        .output_addr(new_output_addr), .output_data(new_output_data)
    );

    always #5 clk = ~clk;

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
        require_file("../mem/golden/q_pool2_act.mem");
        require_file("../mem/weights/gru_Wr.mem");
        require_file("../mem/weights/gru_Wz.mem");
        require_file("../mem/weights/gru_Wh.mem");
        require_file("../mem/weights/gru_Ur.mem");
        require_file("../mem/weights/gru_Uz.mem");
        require_file("../mem/weights/gru_Uh.mem");
        require_file("../mem/weights/gru_br.mem");
        require_file("../mem/weights/gru_bz.mem");
        require_file("../mem/weights/gru_bh.mem");

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        old_finished = 1'b0;
        new_finished = 1'b0;
        old_count = 0;
        new_count = 0;
        error_count = 0;
        cycle_count = 0;
        old_done_cycle = 0;
        new_done_cycle = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (old_finished && new_finished);
        repeat (2) @(posedge clk);

        if ((old_count != OUTPUT_SIZE) || (new_count != OUTPUT_SIZE)) begin
            $display("COUNT mismatch old=%0d new=%0d expected=%0d",
                     old_count, new_count, OUTPUT_SIZE);
            error_count = error_count + 1;
        end

        for (compare_index = 0; compare_index < OUTPUT_SIZE;
             compare_index = compare_index + 1) begin
            if ($isunknown(old_results[compare_index]) ||
                $isunknown(new_results[compare_index]) ||
                (old_results[compare_index] !== new_results[compare_index])) begin
                if (error_count < 20)
                    $display("Mismatch addr=%0d old=%0d new=%0d",
                             compare_index, old_results[compare_index],
                             new_results[compare_index]);
                error_count = error_count + 1;
            end
        end

        if (error_count == 0) begin
            $display("PASS: pipelined GRU exactly matches original GRU.");
            $display("Checked outputs=%0d old_cycles=%0d new_cycles=%0d",
                     OUTPUT_SIZE, old_done_cycle, new_done_cycle);
        end else begin
            $display("FAIL: pipelined GRU comparison errors=%0d", error_count);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (old_output_valid) begin
            if (!$isunknown(old_output_addr) &&
                (old_output_addr < OUTPUT_SIZE)) begin
                old_results[old_output_addr] = old_output_data;
                old_count = old_count + 1;
            end else begin
                error_count = error_count + 1;
            end
        end

        if (new_output_valid) begin
            if (!$isunknown(new_output_addr) &&
                (new_output_addr < OUTPUT_SIZE)) begin
                new_results[new_output_addr] = new_output_data;
                new_count = new_count + 1;
            end else begin
                error_count = error_count + 1;
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

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: tb_gru_pipeline timeout at cycle %0d", cycle_count);
            $finish;
        end
    end
endmodule
