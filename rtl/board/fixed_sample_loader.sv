// Copies one fixed sample from synchronous ROM into eeg_top's input RAM,
// then emits a one-clock inference start pulse. A new request is accepted
// only after the preceding inference has completed.

// 把一筆EEG資料寫進RAM A
module fixed_sample_loader #(
    parameter int SAMPLE_SIZE = 3360,
    parameter int ADDR_W = 12,
    parameter SAMPLE_FILE = "mem/board/sample0_q12.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               request,
    input  logic               input_ready,
    input  logic               inference_done,
    output logic               input_write_en,
    output logic [ADDR_W-1:0]  input_write_addr,
    output logic signed [15:0] input_write_data,
    output logic               inference_start,
    output logic               busy
);
    typedef enum logic [1:0] {
        S_IDLE,
        S_LOAD,
        S_DRAIN,
        S_WAIT_INFERENCE
    } state_t;
    state_t state;

    logic [ADDR_W-1:0] issue_addr;
    logic [ADDR_W-1:0] issued_addr_d;
    logic issue_valid;
    logic issue_valid_d;
    logic signed [15:0] rom_data;

    sample_rom #(
        .DATA_W(16), .DEPTH(SAMPLE_SIZE), .ADDR_W(ADDR_W),
        .MEM_FILE(SAMPLE_FILE)
    ) u_sample_rom (
        .clk(clk), .addr(issue_addr), .data(rom_data)
    );

    always_comb begin
        issue_valid = (state == S_LOAD);
        input_write_en   = issue_valid_d;
        input_write_addr = issued_addr_d;
        input_write_data = rom_data;
        busy = (state != S_IDLE);
    end

    // issue_valid_d and issued_addr_d align the ROM's one-clock read latency
    // with the input RAM write address and data.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            issue_addr      <= '0;
            issued_addr_d   <= '0;
            issue_valid_d   <= 1'b0;
            inference_start <= 1'b0;
        end else begin
            issue_valid_d <= issue_valid;
            if (issue_valid)
                issued_addr_d <= issue_addr;

            inference_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    issue_addr <= '0;
                    if (request && input_ready)
                        state <= S_LOAD;
                end

                S_LOAD: begin
                    if (issue_addr == SAMPLE_SIZE-1)
                        state <= S_DRAIN;
                    else
                        issue_addr <= issue_addr + 1'b1;
                end

                // The final ROM value is committed to input RAM on this edge.
                // The start pulse is observed by eeg_top on the next edge.
                S_DRAIN: begin
                    inference_start <= 1'b1;
                    state <= S_WAIT_INFERENCE;
                end

                S_WAIT_INFERENCE:
                    if (inference_done)
                        state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
