transcript on

# Accept execution from either EEG_Project or EEG_Project/quartus.
if {![file isdirectory rtl]} {
    if {[file isdirectory ../rtl]} {cd ..} else {error "Run from EEG_Project or EEG_Project/quartus"}
}

# Generate the packed 4-lane ROM files from the original MATLAB export.
exec python host/pack_parallel_conv_weights.py

if {![file exists work]} {vlib work}

vlog -sv rtl/common/sat16.sv
vlog -sv rtl/common/pe_mac.sv
vlog -sv rtl/memory/weight_rom.sv
vlog -sv rtl/memory/activation_ram.sv
vlog -sv rtl/conv/conv_engine_parallel.sv
vlog -sv tb/unit/tb_conv2_parallel.sv

vsim work.tb_conv2_parallel
run -all
