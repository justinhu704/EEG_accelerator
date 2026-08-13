// Streaming argmax for the 105 signed logits produced by the final FC layer.
// No softmax is required when only the winning class is needed, because
// softmax preserves the ordering of the logits.
module argmax_105 (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               in_valid,
    input  logic [6:0]         in_index,
    input  logic signed [15:0] in_data,
    output logic               busy,
    output logic               done,
    output logic [6:0]         class_index,
    output logic signed [15:0] max_value
);
    logic [6:0] sample_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy        <= 1'b0;
            done        <= 1'b0;
            sample_count <= 7'd0;
            class_index <= 7'd0;
            max_value   <= 16'sh8000;
        end else begin
            done <= 1'b0;

            if (start) begin
                busy         <= 1'b1;
                sample_count <= 7'd0;
                class_index  <= 7'd0;
                max_value    <= 16'sh8000;
            end else if (busy && in_valid) begin
                // Strict '>' keeps the smaller/first index when values tie.
                if ($signed(in_data) > $signed(max_value)) begin
                    max_value   <= in_data;
                    class_index <= in_index;
                end

                if (sample_count == 7'd104) begin
                    // Include the 105th value in the registered final result.
                    if ($signed(in_data) > $signed(max_value)) begin
                        max_value   <= in_data;
                        class_index <= in_index;
                    end
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    sample_count <= sample_count + 7'd1;
                end
            end
        end
    end
endmodule
