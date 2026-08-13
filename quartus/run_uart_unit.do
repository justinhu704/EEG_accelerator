transcript on
if {![file isdirectory rtl]} {
    if {[file isdirectory ../rtl]} {cd ..} else {error "Run from EEG_Project or EEG_Project/quartus"}
}
if {![file exists work]} {vlib work}

vlog -sv rtl/uart/uart_crc16_ccitt.sv
vlog -sv rtl/uart/uart_rx.sv
vlog -sv rtl/uart/uart_tx.sv
vlog -sv rtl/uart/uart_sample_loader.sv
vlog -sv rtl/uart/uart_result_sender.sv
vlog -sv tb/unit/tb_uart_rx.sv
vlog -sv tb/unit/tb_uart_tx.sv
vlog -sv tb/unit/tb_uart_sample_loader.sv
vlog -sv tb/unit/tb_uart_result_sender.sv

vsim work.tb_uart_rx
onfinish stop
run -all
quit -sim

vsim work.tb_uart_tx
onfinish stop
run -all
quit -sim

vsim work.tb_uart_sample_loader
onfinish stop
run -all
quit -sim

vsim work.tb_uart_result_sender
onfinish stop
run -all
