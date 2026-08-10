# DE1-SoC onboard CLOCK_50: 50 MHz = 20.000 ns period.
create_clock -name clk -period 20.000 [get_ports {clk}]
