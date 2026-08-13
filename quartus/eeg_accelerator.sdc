# DE1-SoC onboard CLOCK_50: 50 MHz = 20.000 ns period.
create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]

# KEY0/KEY1 are asynchronous board buttons. They enter explicit synchronizer
# logic in fpga_top, so external button-to-register paths are not timed as
# synchronous I/O paths.
set_false_path -from [get_ports {KEY[*]}]

# LEDs and seven-segment displays are human-visible indicators with no
# external synchronous receiver or board-level output timing requirement.
set_false_path -to [get_ports {HEX0[*] HEX1[*] HEX2[*] HEX3[*] HEX4[*] HEX5[*] LEDR[*]}]

# UART_RXD is asynchronous to CLOCK_50 and enters a two-flop synchronizer.
set_false_path -from [get_ports {UART_RXD}]

# UART_TXD has no external source-synchronous timing requirement.
set_false_path -to [get_ports {UART_TXD}]