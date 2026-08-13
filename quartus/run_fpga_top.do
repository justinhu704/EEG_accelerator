transcript on
if {![file exists work]} {vlib work}

vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/bn_relu/relu.sv
vlog -sv ../rtl/bn_relu/bn_affine.sv
vlog -sv ../rtl/conv/conv_controller.sv
vlog -sv ../rtl/conv/conv_engine.sv
vlog -sv ../rtl/pooling/maxpool_engine.sv
vlog -sv ../rtl/gru/sigmoid_lut.sv
vlog -sv ../rtl/gru/tanh_lut.sv
vlog -sv ../rtl/gru/gru_engine.sv
vlog -sv ../rtl/fc/fc_engine.sv
vlog -sv ../rtl/fc/argmax_105.sv
vlog -sv ../rtl/top/cnn_gru_top.sv
vlog -sv ../rtl/top/eeg_controller.sv
vlog -sv ../rtl/top/eeg_top.sv
vlog -sv ../rtl/board/sample_rom.sv
vlog -sv ../rtl/board/fixed_sample_loader.sv
vlog -sv ../rtl/display/binary_to_bcd_7bit.sv
vlog -sv ../rtl/display/seven_seg_decoder.sv
vlog -sv ../rtl/display/class_to_subject_id.sv
vlog -sv ../rtl/board/fpga_top.sv
vlog -sv ../tb/integration/tb_fpga_top.sv

vsim work.tb_fpga_top
run -all
