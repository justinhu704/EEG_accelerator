// Complete convolutional feature extractor:
// Conv1 -> BN1 -> ReLU1 -> Conv2 -> BN2 -> ReLU2 -> Pool1
// -> Conv3 -> BN3 -> ReLU3 -> Pool2.
// Activations alternate between RAM A and RAM B.
module cnn_stage3_top #(
    parameter INPUT_FILE   = "mem/golden/q_in_act.mem",
    parameter CONV1_W_FILE = "mem/weights/conv1_W.mem",
    parameter CONV1_B_FILE = "mem/weights/conv1_b.mem",
    parameter BN1_A_FILE   = "mem/weights/bn1_A.mem",
    parameter BN1_B_FILE   = "mem/weights/bn1_B.mem",
    parameter CONV2_W_FILE = "mem/weights/conv2_W.mem",
    parameter CONV2_B_FILE = "mem/weights/conv2_b.mem",
    parameter BN2_A_FILE   = "mem/weights/bn2_A.mem",
    parameter BN2_B_FILE   = "mem/weights/bn2_B.mem",
    parameter CONV3_W_FILE = "mem/weights/conv3_W.mem",
    parameter CONV3_B_FILE = "mem/weights/conv3_b.mem",
    parameter BN3_A_FILE   = "mem/weights/bn3_A.mem",
    parameter BN3_B_FILE   = "mem/weights/bn3_B.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,
    output logic               output_valid,
    output logic [31:0]        output_addr,
    output logic signed [15:0] output_data,

    // Final Pool2 data is stored at RAM B addresses 0..269.
    input  logic [15:0]        ram_b_read_addr,
    output logic signed [15:0] ram_b_read_data
);
    logic [15:0] unused_ram_a_read_addr;
    logic signed [15:0] unused_ram_a_read_data;

    always_comb unused_ram_a_read_addr = '0;

    cnn_stage2_top #(
        .RUN_FULL_CNN(1'b1),
        .INPUT_FILE(INPUT_FILE),
        .CONV1_W_FILE(CONV1_W_FILE), .CONV1_B_FILE(CONV1_B_FILE),
        .BN1_A_FILE(BN1_A_FILE), .BN1_B_FILE(BN1_B_FILE),
        .CONV2_W_FILE(CONV2_W_FILE), .CONV2_B_FILE(CONV2_B_FILE),
        .BN2_A_FILE(BN2_A_FILE), .BN2_B_FILE(BN2_B_FILE),
        .CONV3_W_FILE(CONV3_W_FILE), .CONV3_B_FILE(CONV3_B_FILE),
        .BN3_A_FILE(BN3_A_FILE), .BN3_B_FILE(BN3_B_FILE)
    ) u_core (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .done(done),
        .output_valid(), .output_addr(), .output_data(),
        .stage1_output_valid(), .stage1_output_addr(),
        .stage1_output_data(),
        .pool1_output_valid(), .pool1_output_addr(),
        .pool1_output_data(),
        .pool2_output_valid(output_valid),
        .pool2_output_addr(output_addr),
        .pool2_output_data(output_data),
        .ram_a_read_addr(unused_ram_a_read_addr),
        .ram_a_read_data(unused_ram_a_read_data),
        .ram_b_read_addr(ram_b_read_addr),
        .ram_b_read_data(ram_b_read_data)
    );
endmodule
