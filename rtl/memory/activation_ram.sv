module activation_ram #(
    parameter int DATA_W = 16,
    parameter int DEPTH  = 65520,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter     MEM_FILE = ""
) (
    input  logic                        clk,

    input  logic                        write_en,
    input  logic [ADDR_W-1:0]           write_addr,
    input  logic signed [DATA_W-1:0]    write_data,

    input  logic [ADDR_W-1:0]           read_addr,
    output logic signed [DATA_W-1:0]    read_data
);
    // memory blocks. Other synthesis tools may safely ignore this attribute.
    (* ramstyle = "M10K" *)
    logic signed [DATA_W-1:0] mem [0:DEPTH-1];

    // Optional power-up contents. On FPGA this data is included in the
    // programming image when the selected device/tool supports RAM init.
    initial begin
        if (MEM_FILE != "")
            $readmemh(MEM_FILE, mem, 0, DEPTH-1);
    end

    // Independent synchronous write port.
    always_ff @(posedge clk) begin
        if (write_en)
            mem[write_addr] <= write_data;
    end

    // Independent synchronous read port (one-clock read latency).
    always_ff @(posedge clk) begin
        read_data <= mem[read_addr];
    end
endmodule
