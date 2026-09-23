// Shared GRU activation lookup memory.
//
// Each 32-bit word stores {tanh_q15, sigmoid_q15} for the same normalized
// table index. During the gate phase the two ROM ports read reset/update
// sigmoid values concurrently. During the later candidate phase port A reads
// tanh. The phases are mutually exclusive in gru_engine_pipeline.
//
// The lookup path is split into three registered stages:
//   input capture -> address generation -> synchronous ROM read.
// This keeps the accumulator saturation, constant address multiply and block
// memory access out of the same timing path while retaining one request/cycle.
module gru_activation_lut #(
    parameter MEM_FILE = "mem/lut/gru_activation_lut_q15.mem"
) (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               gate_in_valid,
    input  logic signed [15:0] reset_in_data,
    input  logic signed [15:0] update_in_data,
    output logic               gate_out_valid,
    output logic signed [15:0] reset_out_data,
    output logic signed [15:0] update_out_data,

    input  logic               candidate_in_valid,
    input  logic signed [15:0] candidate_in_data,
    output logic               candidate_out_valid,
    output logic signed [15:0] candidate_out_data
);
    localparam int LUT_DEPTH = 256;
    localparam int SIGMOID_RANGE_RAW = 3193;
    localparam int SIGMOID_INDEX_GAIN = 83741;
    localparam int TANH_RANGE_RAW = 1774;
    localparam int TANH_INDEX_GAIN = 150725;
    localparam int INDEX_SHIFT = 20;

    // One true dual-read ROM replaces two sigmoid ROMs and one tanh ROM.
    (* ramstyle = "M10K" *) logic [31:0] lut [0:LUT_DEPTH-1];

    logic gate_request_d1;
    logic candidate_request_d1;
    logic signed [15:0] reset_input_d1;
    logic signed [15:0] update_input_d1;
    logic signed [15:0] candidate_input_d1;

    logic [11:0] port_a_magnitude;
    logic [17:0] port_a_gain;
    (* multstyle = "logic" *) logic [28:0] port_a_index_product;
    logic [7:0] port_a_addr;

    logic [11:0] port_b_magnitude;
    (* multstyle = "logic" *) logic [28:0] port_b_index_product;
    logic [7:0] port_b_addr;

    logic gate_request_d2;
    logic candidate_request_d2;
    logic [7:0] port_a_addr_d2;
    logic [7:0] port_b_addr_d2;
    logic reset_negative_d2;
    logic update_negative_d2;
    logic candidate_negative_d2;

    logic [31:0] port_a_data;
    logic [31:0] port_b_data;
    logic reset_negative_d3;
    logic update_negative_d3;
    logic candidate_negative_d3;
    logic signed [16:0] reset_reflected;
    logic signed [16:0] update_reflected;
    logic signed [16:0] candidate_negated;

    initial begin
        $readmemh(MEM_FILE, lut, 0, LUT_DEPTH-1);
    end

    // Port A is shared by reset sigmoid and candidate tanh because those GRU
    // phases never overlap. Port B is used by update sigmoid in the gate phase.
    // Constant products are kept bit-exact but explicitly mapped to logic.
    always_comb begin
        port_a_magnitude = '0;
        port_a_gain = '0;
        port_a_index_product = '0;
        port_a_addr = '0;

        if (gate_request_d1) begin
            port_a_gain = SIGMOID_INDEX_GAIN;
            if (($signed(reset_input_d1) <= -SIGMOID_RANGE_RAW) ||
                ($signed(reset_input_d1) >=  SIGMOID_RANGE_RAW)) begin
                port_a_addr = 8'd255;
            end else begin
                if (reset_input_d1[15])
                    port_a_magnitude = -$signed(reset_input_d1);
                else
                    port_a_magnitude = reset_input_d1;
                port_a_index_product = port_a_magnitude * port_a_gain;
                port_a_addr = port_a_index_product >> INDEX_SHIFT;
            end
        end else if (candidate_request_d1) begin
            port_a_gain = TANH_INDEX_GAIN;
            if (($signed(candidate_input_d1) <= -TANH_RANGE_RAW) ||
                ($signed(candidate_input_d1) >=  TANH_RANGE_RAW)) begin
                port_a_addr = 8'd255;
            end else begin
                if (candidate_input_d1[15])
                    port_a_magnitude = -$signed(candidate_input_d1);
                else
                    port_a_magnitude = candidate_input_d1;
                port_a_index_product = port_a_magnitude * port_a_gain;
                port_a_addr = port_a_index_product >> INDEX_SHIFT;
            end
        end

        port_b_magnitude = '0;
        port_b_index_product = '0;
        port_b_addr = '0;
        if (gate_request_d1) begin
            if (($signed(update_input_d1) <= -SIGMOID_RANGE_RAW) ||
                ($signed(update_input_d1) >=  SIGMOID_RANGE_RAW)) begin
                port_b_addr = 8'd255;
            end else begin
                if (update_input_d1[15])
                    port_b_magnitude = -$signed(update_input_d1);
                else
                    port_b_magnitude = update_input_d1;
                port_b_index_product = port_b_magnitude
                                     * SIGMOID_INDEX_GAIN;
                port_b_addr = port_b_index_product >> INDEX_SHIFT;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gate_request_d1 <= 1'b0;
            candidate_request_d1 <= 1'b0;
            reset_input_d1 <= '0;
            update_input_d1 <= '0;
            candidate_input_d1 <= '0;
            gate_request_d2 <= 1'b0;
            candidate_request_d2 <= 1'b0;
            port_a_addr_d2 <= '0;
            port_b_addr_d2 <= '0;
            reset_negative_d2 <= 1'b0;
            update_negative_d2 <= 1'b0;
            candidate_negative_d2 <= 1'b0;
            gate_out_valid <= 1'b0;
            candidate_out_valid <= 1'b0;
            port_a_data <= '0;
            port_b_data <= '0;
            reset_negative_d3 <= 1'b0;
            update_negative_d3 <= 1'b0;
            candidate_negative_d3 <= 1'b0;
        end else begin
            // Stage 1: isolate the GRU accumulators from LUT address logic.
            gate_request_d1 <= gate_in_valid;
            candidate_request_d1 <= candidate_in_valid;
            if (gate_in_valid) begin
                reset_input_d1 <= reset_in_data;
                update_input_d1 <= update_in_data;
            end else if (candidate_in_valid) begin
                candidate_input_d1 <= candidate_in_data;
            end

            // Stage 2: register the mapped ROM addresses and signs.
            gate_request_d2 <= gate_request_d1;
            candidate_request_d2 <= candidate_request_d1;
            if (gate_request_d1) begin
                port_a_addr_d2 <= port_a_addr;
                port_b_addr_d2 <= port_b_addr;
                reset_negative_d2 <= reset_input_d1[15];
                update_negative_d2 <= update_input_d1[15];
            end else if (candidate_request_d1) begin
                port_a_addr_d2 <= port_a_addr;
                candidate_negative_d2 <= candidate_input_d1[15];
            end

            // Stage 3: synchronous dual-port ROM read.
            gate_out_valid <= gate_request_d2;
            candidate_out_valid <= candidate_request_d2;
            if (gate_request_d2) begin
                port_a_data <= lut[port_a_addr_d2];
                port_b_data <= lut[port_b_addr_d2];
                reset_negative_d3 <= reset_negative_d2;
                update_negative_d3 <= update_negative_d2;
            end else if (candidate_request_d2) begin
                port_a_data <= lut[port_a_addr_d2];
                candidate_negative_d3 <= candidate_negative_d2;
            end
        end
    end

    always_comb begin
        reset_reflected = 17'sd32768
                        - $signed({1'b0, port_a_data[15:0]});
        update_reflected = 17'sd32768
                         - $signed({1'b0, port_b_data[15:0]});
        candidate_negated = -$signed(port_a_data[31:16]);

        if (!gate_out_valid) begin
            reset_out_data = '0;
            update_out_data = '0;
        end else begin
            reset_out_data = reset_negative_d3
                           ? reset_reflected[15:0]
                           : port_a_data[15:0];
            update_out_data = update_negative_d3
                            ? update_reflected[15:0]
                            : port_b_data[15:0];
        end

        if (!candidate_out_valid)
            candidate_out_data = '0;
        else if (candidate_negative_d3)
            candidate_out_data = candidate_negated[15:0];
        else
            candidate_out_data = port_a_data[31:16];
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && gate_in_valid && candidate_in_valid)
            $error("GRU gate and candidate LUT requests must not overlap");
    end
`endif
endmodule
