`timescale 1ns/1ps

// Bit-exact regression for the current DSConv1/DSConv2 model's pipelined GRU.
// The legacy gru_engine uses older Q formats, so the authoritative reference
// is a captured sample-0 output vector from the pre-optimization pipeline.
module tb_gru_pipeline;
    localparam int INPUT_DEPTH = 18 * 15;
    localparam int OUTPUT_SIZE = 9 * 18;
    localparam int MAX_CYCLES = 20_000;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic [31:0] input_addr;
    logic signed [15:0] input_data;
    logic output_valid;
    logic [31:0] output_addr;
    logic signed [15:0] output_data;

    logic signed [15:0] golden [0:OUTPUT_SIZE-1];
    logic signed [15:0] results [0:OUTPUT_SIZE-1];
    integer output_count;
    integer error_count;
    integer cycle_count;
    integer done_cycle;
    integer compare_index;

    activation_ram #(
        .DATA_W(16),
        .DEPTH(INPUT_DEPTH),
        .ADDR_W($clog2(INPUT_DEPTH)),
        .MEM_FILE("../mem/dsconv1_dsconv2/golden/q_pool2_act_sample0.mem"),
        .USE_READ_ENABLE(1'b1)
    ) u_input_ram (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0),
        .write_data('0),
        .read_en(1'b1),
        .read_addr(input_addr[$clog2(INPUT_DEPTH)-1:0]),
        .read_data(input_data)
    );

    gru_engine_pipeline #(
        .WR_FILE("../mem/dsconv1_dsconv2/weights/gru_Wr.mem"),
        .WZ_FILE("../mem/dsconv1_dsconv2/weights/gru_Wz.mem"),
        .WH_FILE("../mem/dsconv1_dsconv2/weights/gru_Wh.mem"),
        .UR_FILE("../mem/dsconv1_dsconv2/weights/gru_Ur.mem"),
        .UZ_FILE("../mem/dsconv1_dsconv2/weights/gru_Uz.mem"),
        .UH_FILE("../mem/dsconv1_dsconv2/weights/gru_Uh.mem"),
        .BR_FILE("../mem/dsconv1_dsconv2/weights/gru_br.mem"),
        .BZ_FILE("../mem/dsconv1_dsconv2/weights/gru_bz.mem"),
        .BH_FILE("../mem/dsconv1_dsconv2/weights/gru_bh.mem"),
        .ACTIVATION_LUT_FILE("../mem/lut/gru_activation_lut_q15.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .input_addr(input_addr), .input_data(input_data),
        .output_valid(output_valid), .output_addr(output_addr),
        .output_data(output_data)
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
        require_file("../mem/dsconv1_dsconv2/golden/q_pool2_act_sample0.mem");
        require_file("../tb/data/gru_pipeline_rtl_sample0.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_Wr.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_Wz.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_Wh.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_Ur.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_Uz.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_Uh.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_br.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_bz.mem");
        require_file("../mem/dsconv1_dsconv2/weights/gru_bh.mem");
        $readmemh("../tb/data/gru_pipeline_rtl_sample0.mem", golden);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        output_count = 0;
        error_count = 0;
        cycle_count = 0;
        done_cycle = 0;

        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done);
        repeat (2) @(posedge clk);

        if (output_count != OUTPUT_SIZE) begin
            $display("COUNT mismatch got=%0d expected=%0d",
                     output_count, OUTPUT_SIZE);
            error_count = error_count + 1;
        end

        for (compare_index = 0; compare_index < OUTPUT_SIZE;
             compare_index = compare_index + 1) begin
            if ($isunknown(results[compare_index]) ||
                (results[compare_index] !== golden[compare_index])) begin
                if (error_count < 20)
                    $display("Mismatch addr=%0d got=%0d expected=%0d",
                             compare_index, results[compare_index],
                             golden[compare_index]);
                error_count = error_count + 1;
            end
        end

        if (error_count == 0) begin
            $display("PASS: optimized GRU matches pre-optimization RTL.");
            $display("Checked outputs=%0d cycles=%0d",
                     OUTPUT_SIZE, done_cycle);
        end else begin
            $display("FAIL: optimized GRU errors=%0d", error_count);
            $fatal(1);
        end
        $finish;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;

        if (output_valid) begin
            if (!$isunknown(output_addr) && (output_addr < OUTPUT_SIZE)) begin
                results[output_addr] = output_data;
                output_count = output_count + 1;
            end else begin
                error_count = error_count + 1;
            end
        end

        if (done)
            done_cycle = cycle_count;

        if (cycle_count > MAX_CYCLES) begin
            $display("FAIL: optimized GRU timeout at cycle %0d", cycle_count);
            $fatal(1);
        end
    end
endmodule
