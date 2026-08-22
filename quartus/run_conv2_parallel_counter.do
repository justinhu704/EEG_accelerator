transcript on

# Keep the simulation working directory at EEG_Project/quartus because the
# testbench memory parameters intentionally use ../mem paths.
if {[file isdirectory rtl] && [file isdirectory quartus]} {
    cd quartus
} elseif {![file isdirectory ../rtl]} {
    error "Run from EEG_Project or EEG_Project/quartus"
}

# Generate the packed 4-lane ROM files from the original MATLAB export.
exec python ../host/pack_parallel_conv_weights.py \
    --weights ../mem/weights/conv2_W.mem \
    --bias ../mem/weights/conv2_b.mem \
    --output-weights ../mem/weights/conv2_W_x4.mem \
    --output-bias ../mem/weights/conv2_b_x4.mem

if {[file exists work]} {vdel -lib work -all}
vlib work
vmap work work

vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/conv/conv_engine_parallel.sv
vlog -sv ../rtl/conv/conv_engine_parallel_counter.sv
vlog -sv ../tb/unit/tb_conv2_parallel_counter.sv

vsim -voptargs=+acc work.tb_conv2_parallel_counter
run -all
