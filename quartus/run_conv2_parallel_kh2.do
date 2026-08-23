transcript on

if {[file isdirectory rtl] && [file isdirectory quartus]} {
    cd quartus
} elseif {![file isdirectory ../rtl]} {
    error "Run from EEG_Project or EEG_Project/quartus"
}

# Preserve the existing x5 files and create a separate double-width KH2 ROM.
exec python ../host/pack_parallel_conv_weights.py \
    --weights ../mem/weights/conv2_W.mem \
    --bias ../mem/weights/conv2_b.mem \
    --output-weights ../mem/weights/conv2_W_x5_kh2.mem \
    --output-bias ../mem/weights/conv2_b_x5_kh2.mem \
    --lanes 5 --pair-kh

if {[file exists work]} {vdel -lib work -all}
vlib work
vmap work work

vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/memory/conv1_banked_ram.sv
vlog -sv ../rtl/conv/conv_engine_parallel_counter.sv
vlog -sv ../rtl/conv/conv_engine_parallel_kh2.sv
vlog -sv ../tb/unit/tb_conv2_parallel_kh2.sv

vsim -voptargs=+acc work.tb_conv2_parallel_kh2
run -all
