`timescale 1ns/1ps

// Exhaustive equivalence check between the original three lookup modules and
// the shared dual-read GRU activation ROM for every signed 16-bit input.
module tb_gru_activation_lut;
    logic clk;
    logic rst_n;
    logic gate_in_valid;
    logic signed [15:0] reset_in_data;
    logic signed [15:0] update_in_data;
    logic candidate_in_valid;
    logic signed [15:0] candidate_in_data;

    logic old_reset_valid;
    logic signed [15:0] old_reset_data;
    logic old_update_valid;
    logic signed [15:0] old_update_data;
    logic old_candidate_valid;
    logic signed [15:0] old_candidate_data;
    logic old_reset_valid_d1;
    logic old_reset_valid_d2;
    logic signed [15:0] old_reset_data_d1;
    logic signed [15:0] old_reset_data_d2;
    logic old_update_valid_d1;
    logic old_update_valid_d2;
    logic signed [15:0] old_update_data_d1;
    logic signed [15:0] old_update_data_d2;
    logic old_candidate_valid_d1;
    logic old_candidate_valid_d2;
    logic signed [15:0] old_candidate_data_d1;
    logic signed [15:0] old_candidate_data_d2;

    logic shared_gate_valid;
    logic signed [15:0] shared_reset_data;
    logic signed [15:0] shared_update_data;
    logic shared_candidate_valid;
    logic signed [15:0] shared_candidate_data;

    integer sample_index;
    integer error_count;
    integer gate_compare_count;
    integer candidate_compare_count;

    sigmoid_lut #(
        .MEM_FILE("../mem/lut/sigmoid_half_lut_q15.mem")
    ) u_old_reset (
        .clk(clk), .rst_n(rst_n), .in_valid(gate_in_valid),
        .in_data(reset_in_data), .out_valid(old_reset_valid),
        .out_data(old_reset_data)
    );

    sigmoid_lut #(
        .MEM_FILE("../mem/lut/sigmoid_half_lut_q15.mem")
    ) u_old_update (
        .clk(clk), .rst_n(rst_n), .in_valid(gate_in_valid),
        .in_data(update_in_data), .out_valid(old_update_valid),
        .out_data(old_update_data)
    );

    tanh_lut #(
        .MEM_FILE("../mem/lut/tanh_half_lut_q15.mem")
    ) u_old_candidate (
        .clk(clk), .rst_n(rst_n), .in_valid(candidate_in_valid),
        .in_data(candidate_in_data), .out_valid(old_candidate_valid),
        .out_data(old_candidate_data)
    );

    gru_activation_lut #(
        .MEM_FILE("../mem/lut/gru_activation_lut_q15.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .gate_in_valid(gate_in_valid),
        .reset_in_data(reset_in_data),
        .update_in_data(update_in_data),
        .gate_out_valid(shared_gate_valid),
        .reset_out_data(shared_reset_data),
        .update_out_data(shared_update_data),
        .candidate_in_valid(candidate_in_valid),
        .candidate_in_data(candidate_in_data),
        .candidate_out_valid(shared_candidate_valid),
        .candidate_out_data(shared_candidate_data)
    );

    always #5 clk = ~clk;

    // The timing-optimized shared ROM has two more pipeline stages than the
    // legacy lookup blocks. Delay the legacy reference by two clocks so the
    // exhaustive comparison remains cycle aligned.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            old_reset_valid_d1 <= 1'b0;
            old_reset_valid_d2 <= 1'b0;
            old_reset_data_d1 <= '0;
            old_reset_data_d2 <= '0;
            old_update_valid_d1 <= 1'b0;
            old_update_valid_d2 <= 1'b0;
            old_update_data_d1 <= '0;
            old_update_data_d2 <= '0;
            old_candidate_valid_d1 <= 1'b0;
            old_candidate_valid_d2 <= 1'b0;
            old_candidate_data_d1 <= '0;
            old_candidate_data_d2 <= '0;
        end else begin
            old_reset_valid_d1 <= old_reset_valid;
            old_reset_valid_d2 <= old_reset_valid_d1;
            old_reset_data_d1 <= old_reset_data;
            old_reset_data_d2 <= old_reset_data_d1;
            old_update_valid_d1 <= old_update_valid;
            old_update_valid_d2 <= old_update_valid_d1;
            old_update_data_d1 <= old_update_data;
            old_update_data_d2 <= old_update_data_d1;
            old_candidate_valid_d1 <= old_candidate_valid;
            old_candidate_valid_d2 <= old_candidate_valid_d1;
            old_candidate_data_d1 <= old_candidate_data;
            old_candidate_data_d2 <= old_candidate_data_d1;
        end
    end

    task automatic check_gate_outputs;
        begin
            if (!old_reset_valid_d2 || !old_update_valid_d2 ||
                (shared_reset_data !== old_reset_data_d2) ||
                (shared_update_data !== old_update_data_d2)) begin
                if (error_count < 20)
                    $display("Gate mismatch vector=%0d reset old=%h new=%h update old=%h new=%h",
                             gate_compare_count, old_reset_data_d2,
                             shared_reset_data, old_update_data_d2,
                             shared_update_data);
                error_count = error_count + 1;
            end
            gate_compare_count = gate_compare_count + 1;
        end
    endtask

    task automatic check_candidate_output;
        begin
            if (!old_candidate_valid_d2 ||
                (shared_candidate_data !== old_candidate_data_d2)) begin
                if (error_count < 20)
                    $display("Tanh mismatch vector=%0d old=%h new=%h",
                             candidate_compare_count, old_candidate_data_d2,
                             shared_candidate_data);
                error_count = error_count + 1;
            end
            candidate_compare_count = candidate_compare_count + 1;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        gate_in_valid = 1'b0;
        candidate_in_valid = 1'b0;
        reset_in_data = '0;
        update_in_data = '0;
        candidate_in_data = '0;
        error_count = 0;
        gate_compare_count = 0;
        candidate_compare_count = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Test both sigmoid ports. The complemented update sequence also
        // traverses all 65,536 signed input bit patterns.
        for (sample_index = 0; sample_index < 65536;
             sample_index = sample_index + 1) begin
            @(negedge clk);
            if (shared_gate_valid)
                check_gate_outputs();
            gate_in_valid = 1'b1;
            reset_in_data = sample_index[15:0];
            update_in_data = ~sample_index[15:0];
        end
        @(negedge clk);
        if (shared_gate_valid)
            check_gate_outputs();
        gate_in_valid = 1'b0;
        repeat (4) begin
            @(negedge clk);
            if (shared_gate_valid)
                check_gate_outputs();
        end

        // Test tanh over the complete signed 16-bit input space.
        for (sample_index = 0; sample_index < 65536;
             sample_index = sample_index + 1) begin
            @(negedge clk);
            if (shared_candidate_valid)
                check_candidate_output();
            candidate_in_valid = 1'b1;
            candidate_in_data = sample_index[15:0];
        end
        @(negedge clk);
        if (shared_candidate_valid)
            check_candidate_output();
        candidate_in_valid = 1'b0;
        repeat (4) begin
            @(negedge clk);
            if (shared_candidate_valid)
                check_candidate_output();
        end

        if ((gate_compare_count != 65536) ||
            (candidate_compare_count != 65536)) begin
            $fatal(1, "FAIL: compared gate=%0d tanh=%0d expected=65536",
                   gate_compare_count, candidate_compare_count);
        end else if (error_count == 0)
            $display("PASS: shared GRU LUT is bit-exact for all 16-bit inputs.");
        else
            $fatal(1, "FAIL: shared GRU LUT errors=%0d", error_count);
        $finish;
    end
endmodule
