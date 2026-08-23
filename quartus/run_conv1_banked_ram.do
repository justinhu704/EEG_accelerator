transcript on
if {![file exists work]} {vlib work}

vlog -sv ../rtl/memory/conv1_banked_ram.sv
vlog -sv ../tb/unit/tb_conv1_banked_ram.sv

vsim work.tb_conv1_banked_ram
run -all
