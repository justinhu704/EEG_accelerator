transcript on

if {![file isdirectory rtl]} {
    if {[file isdirectory ../rtl]} {cd ..} else {error "Run from EEG_Project or EEG_Project/quartus"}
}

if {![file exists work]} {vlib work}
vlog -sv rtl/memory/activation_ram.sv
vlog -sv rtl/common/sat16.sv
vlog -sv rtl/pooling/streaming_maxpool.sv
vlog -sv tb/unit/tb_streaming_maxpool2.sv
vsim work.tb_streaming_maxpool2
run -all
