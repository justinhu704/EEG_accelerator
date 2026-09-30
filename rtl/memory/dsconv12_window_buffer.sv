// Six-column rolling buffer between DS-Conv1 and DS-Conv2.
// ReLU1 remains signed 16-bit Q11. Even/odd heights are separate so
// DS-Conv2 can read kh=0 and kh=1 in the same clock.
module dsconv12_window_buffer #(
    parameter int IN_H = 20,
    parameter int IN_CH = 21,
    parameter int COLS = 6
) (
    input  logic clk,
    input  logic rst_n,
    input  logic write_en,
    input  logic [$clog2(COLS)-1:0] write_slot,
    input  logic [$clog2(IN_H)-1:0] write_h,
    input  logic [$clog2(IN_CH)-1:0] write_channel,
    input  logic signed [15:0] write_data,
    input  logic read_en,
    input  logic [$clog2(COLS)-1:0] window_base_slot,
    input  logic [$clog2(IN_H)-1:0] read_h,
    input  logic [$clog2(COLS)-1:0] read_kw,
    input  logic [$clog2(IN_CH)-1:0] read_channel,
    output logic signed [15:0] read_data_kh0,
    output logic signed [15:0] read_data_kh1
);
    localparam int HALF_H = IN_H / 2;
    localparam int BANK_DEPTH = HALF_H * COLS * IN_CH;
    localparam int BANK_ADDR_W = $clog2(BANK_DEPTH);

    (* ramstyle = "M10K" *) logic signed [15:0] even_mem [0:BANK_DEPTH-1];
    (* ramstyle = "M10K" *) logic signed [15:0] odd_mem  [0:BANK_DEPTH-1];

    logic [BANK_ADDR_W-1:0] write_addr;
    logic [BANK_ADDR_W-1:0] even_read_addr, odd_read_addr;
    logic signed [15:0] even_read_data, odd_read_data;
    logic kh0_odd_d1;

    logic [$clog2(COLS):0] read_slot_sum;
    logic [$clog2(COLS)-1:0] read_slot;
    logic [BANK_ADDR_W-1:0] read_bank_base;


    always_comb begin
        write_addr = (write_h >> 1)
                   + HALF_H * (write_slot + COLS * write_channel);

        // Engine 直接提供 h/kw/channel，不再用除法與餘數反解線性位址。
        read_slot_sum = window_base_slot + read_kw;
        if (read_slot_sum >= COLS)
            read_slot = read_slot_sum - COLS;
        else
            read_slot = read_slot_sum[$clog2(COLS)-1:0];

        read_bank_base = HALF_H * (read_slot + COLS * read_channel);
        if (!read_h[0]) begin
            even_read_addr = read_bank_base + (read_h >> 1);
            odd_read_addr  = read_bank_base + ((read_h + 1'b1) >> 1);
        end else begin
            odd_read_addr  = read_bank_base + (read_h >> 1);
            even_read_addr = read_bank_base + ((read_h + 1'b1) >> 1);
        end

        if (kh0_odd_d1) begin
            read_data_kh0 = odd_read_data;
            read_data_kh1 = even_read_data;
        end else begin
            read_data_kh0 = even_read_data;
            read_data_kh1 = odd_read_data;
        end
    end

    always_ff @(posedge clk) begin
        if (write_en) begin
            if (write_h[0])
                odd_mem[write_addr] <= write_data;
            else
                even_mem[write_addr] <= write_data;
        end
        if (read_en) begin
            even_read_data <= even_mem[even_read_addr];
            odd_read_data  <= odd_mem[odd_read_addr];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            kh0_odd_d1 <= 1'b0;
        else if (read_en)
            kh0_odd_d1 <= read_h[0];
    end

`ifndef SYNTHESIS
    initial begin
        if ((IN_H % 2) != 0) $error("IN_H must be even");
    end
`endif
endmodule
