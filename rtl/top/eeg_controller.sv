// Global scheduler for one complete EEG inference.
// It only controls start/done handshakes; all arithmetic stays in the engines.
module eeg_controller (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic cnn_gru_done,
    input  logic fc1_bn_done,
    input  logic fc_out_done,
    input  logic argmax_done,
    output logic cnn_gru_start,
    output logic fc1_start,
    output logic fc_out_start,
    output logic argmax_start,
    output logic busy,
    output logic done
);
    typedef enum logic [3:0] {
        S_IDLE,
        S_START_CNN_GRU,
        S_RUN_CNN_GRU,
        S_START_FC1,
        S_RUN_FC1,
        S_START_FC_OUT,
        S_RUN_FC_OUT,
        S_DONE
    } state_t;
    state_t state;

    always_comb begin
        cnn_gru_start = (state == S_START_CNN_GRU);
        fc1_start     = (state == S_START_FC1);
        fc_out_start  = (state == S_START_FC_OUT);
        // Clear Argmax in the same cycle that FC_out starts. FC_out has many
        // cycles of latency before its first valid output.
        argmax_start  = (state == S_START_FC_OUT);
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE:
                    if (start)
                        state <= S_START_CNN_GRU;
                S_START_CNN_GRU: state <= S_RUN_CNN_GRU;
                S_RUN_CNN_GRU:
                    if (cnn_gru_done)
                        state <= S_START_FC1;
                S_START_FC1: state <= S_RUN_FC1;
                S_RUN_FC1:
                    // Wait for BN's final value, not FC1's earlier done pulse.
                    if (fc1_bn_done)
                        state <= S_START_FC_OUT;
                S_START_FC_OUT: state <= S_RUN_FC_OUT;
                S_RUN_FC_OUT:
                    if (fc_out_done && argmax_done)
                        state <= S_DONE;
                    else if (argmax_done)
                        state <= S_DONE;
                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
