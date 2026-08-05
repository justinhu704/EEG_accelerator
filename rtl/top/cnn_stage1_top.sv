// First ping-pong-buffer integration stage:
// RAM A -> Conv1 -> BN1 -> ReLU1 -> RAM B
module cnn_stage1_top #(
    parameter CONV_INPUT_FILE  = "mem/golden/q_in_act.mem",
    parameter CONV_WEIGHT_FILE = "mem/weights/conv1_W.mem",
    parameter CONV_BIAS_FILE   = "mem/weights/conv1_b.mem",
    parameter BN_A_FILE        = "mem/weights/bn1_A.mem",
    parameter BN_B_FILE        = "mem/weights/bn1_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // Stage-1 streaming output, retained for waveform/debug checking.
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data,

    // Temporary verification/read port for RAM B. Conv2 will use this port
    // as its activation input in the next integration stage.
    input  logic [15:0]        ram_b_read_addr,
    output logic signed [15:0] ram_b_read_data
);
    localparam int RAM_DEPTH = 65520;
    localparam int RAM_ADDR_W = 16;
    localparam int OUT_H = 20;
    localparam int OUT_W = 156;
    localparam int OUT_CH = 21;
    localparam int OUTPUT_SIZE = OUT_H * OUT_W * OUT_CH;
    localparam int CONV1_WEIGHT_DEPTH = 2 * 5 * 1 * 21;
    localparam int CONV1_BIAS_DEPTH = 21;

    logic conv_busy;
    logic conv_done;
    logic conv_valid;
    logic [31:0] conv_input_addr;
    logic [31:0] conv_weight_addr;
    logic [31:0] conv_bias_addr;
    logic [31:0] conv_addr;
    logic [31:0] conv_channel;
    logic signed [15:0] conv_input_data;
    logic signed [15:0] conv_weight_data;
    logic signed [15:0] conv_bias_data;
    logic signed [15:0] conv_data;

    logic bn_valid;
    logic signed [15:0] bn_data;
    logic relu_valid;
    logic signed [15:0] relu_data;

    logic conv_valid_d1;
    logic conv_valid_d2;
    logic [31:0] addr_d1;
    logic [31:0] addr_d2;
    logic running;

    // RAM A contains the EEG input for stage 1. Its write port is disabled
    // during inference. Later stages will reuse this bank as a destination.
    activation_ram #(
        .DATA_W(16),
        .DEPTH(RAM_DEPTH),
        .ADDR_W(RAM_ADDR_W),
        .MEM_FILE(CONV_INPUT_FILE)
    ) u_ram_a (
        .clk(clk),
        .write_en(1'b0),
        .write_addr('0),
        .write_data('0),
        .read_addr(conv_input_addr[RAM_ADDR_W-1:0]),
        .read_data(conv_input_data)
    );

    // Conv1 weights and biases are constants, so they remain in ROM.
    weight_rom #(
        .DATA_W(16),
        .DEPTH(CONV1_WEIGHT_DEPTH),
        .ADDR_W($clog2(CONV1_WEIGHT_DEPTH)),
        .MEM_FILE(CONV_WEIGHT_FILE)
    ) u_conv1_weight_rom (
        .clk(clk),
        .addr(conv_weight_addr[$clog2(CONV1_WEIGHT_DEPTH)-1:0]),
        .data(conv_weight_data)
    );

    weight_rom #(
        .DATA_W(16),
        .DEPTH(CONV1_BIAS_DEPTH),
        .ADDR_W($clog2(CONV1_BIAS_DEPTH)),
        .MEM_FILE(CONV_BIAS_FILE)
    ) u_conv1_bias_rom (
        .clk(clk),
        .addr(conv_bias_addr[$clog2(CONV1_BIAS_DEPTH)-1:0]),
        .data(conv_bias_data)
    );

    conv_engine #(
        .IN_H(21), .IN_W(160), .IN_CH(1),
        .K_H(2), .K_W(5), .OUT_CH(21),
        .BIAS_SHIFT(12), .OUTPUT_SHIFT(14)
    ) u_conv1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(conv_busy),
        .done(conv_done),
        .input_addr(conv_input_addr),
        .input_data(conv_input_data),
        .weight_addr(conv_weight_addr),
        .weight_data(conv_weight_data),
        .bias_addr(conv_bias_addr),
        .bias_data(conv_bias_data),
        .output_valid(conv_valid),
        .output_addr(conv_addr),
        .output_ch_idx(conv_channel),
        .output_data(conv_data)
    );

    bn_affine #(
        .CHANNELS(21),
        .BIAS_SHIFT(11),
        .OUTPUT_SHIFT(13),
        .A_FILE(BN_A_FILE),
        .B_FILE(BN_B_FILE)
    ) u_bn1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(conv_valid),
        .in_data(conv_data),
        .in_ch_idx(conv_channel),
        .out_valid(bn_valid),
        .out_data(bn_data)
    );

    relu #(.OUTPUT_LEFT_SHIFT(0)) u_relu1 (
        .in_valid(bn_valid),
        .in_data(bn_data),
        .out_valid(relu_valid),
        .out_data(relu_data)
    );

    // BN has two registered pipeline stages. Delay the Conv output address by
    // the same two clocks so the ReLU value is written to the correct address.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_valid_d1 <= 1'b0;
            conv_valid_d2 <= 1'b0;
            addr_d1 <= '0;
            addr_d2 <= '0;
        end else begin
            conv_valid_d1 <= conv_valid;
            conv_valid_d2 <= conv_valid_d1;
            if (conv_valid)
                addr_d1 <= conv_addr;
            if (conv_valid_d1)
                addr_d2 <= addr_d1;
        end
    end

    // RAM B receives every valid ReLU1 output. The independent read port is
    // used by the testbench now and will be connected to Conv2 later.
    activation_ram #(
        .DATA_W(16),
        .DEPTH(RAM_DEPTH),
        .ADDR_W(RAM_ADDR_W),
        .MEM_FILE("")
    ) u_ram_b (
        .clk(clk),
        .write_en(relu_valid),
        .write_addr(addr_d2[RAM_ADDR_W-1:0]),
        .write_data(relu_data),
        .read_addr(ram_b_read_addr),
        .read_data(ram_b_read_data)
    );

    // done is asserted on the same edge that commits the final RAM B write.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start)
                running <= 1'b1;
            if (relu_valid && (addr_d2 == OUTPUT_SIZE-1)) begin
                running <= 1'b0;
                done <= 1'b1;
            end
        end
    end

    always_comb begin
        busy = running || conv_busy || conv_valid_d1 || conv_valid_d2;
        output_valid = relu_valid;
        output_addr = addr_d2;
        output_data = relu_data;
    end

    // conv_done occurs before the BN/ReLU pipeline and is intentionally not
    // used as the integrated-stage completion signal.
endmodule
