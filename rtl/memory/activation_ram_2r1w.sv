// True-dual-port activation RAM used by the KH2 Conv2 engine.
//
// Port A is time-multiplexed: earlier stages write ReLU1 through it, then
// Conv2 uses it as the kh=0 read port.  Port B is the independent kh=1 read
// port.  The system never requests a Port-A write and read in the same cycle.
module activation_ram_2r1w #(
    parameter int DATA_W = 16,
    parameter int DEPTH  = 65520,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter     MEM_FILE = ""
) (
    input  logic                       clk,

    input  logic                       port_a_write_en,
    input  logic [ADDR_W-1:0]          port_a_addr,
    input  logic signed [DATA_W-1:0]   port_a_write_data,
    output logic signed [DATA_W-1:0]   port_a_read_data,

    input  logic [ADDR_W-1:0]          port_b_read_addr,
    output logic signed [DATA_W-1:0]   port_b_read_data
);
    (* ramstyle = "M10K" *)
    logic signed [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        if (MEM_FILE != "")
            $readmemh(MEM_FILE, mem, 0, DEPTH-1);
    end

    // Port A performs either one write or one synchronous read.
    always_ff @(posedge clk) begin
        if (port_a_write_en)
            mem[port_a_addr] <= port_a_write_data;
        else
            port_a_read_data <= mem[port_a_addr];
    end

    // Port B is an independent synchronous read port.
    always_ff @(posedge clk) begin
        port_b_read_data <= mem[port_b_read_addr];
    end
endmodule
