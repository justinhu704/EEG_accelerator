transcript on

# Accept execution from either EEG_Project or EEG_Project/quartus.
if {![file isdirectory rtl]} {
    if {[file isdirectory ../rtl]} {cd ..} else {error "Run from EEG_Project or EEG_Project/quartus"}
}

if {![file exists work]} {vlib work}

vlog -sv rtl/pooling/streaming_maxpool1.sv
vlog -sv tb/unit/tb_streaming_maxpool1.sv

vsim work.tb_streaming_maxpool1
run -all
