transcript on
exec python ../host/pack_parallel_conv_weights.py --weights ../mem/weights/conv2_W.mem --bias ../mem/weights/conv2_b.mem --output-weights ../mem/weights/conv2_W_x4.mem --output-bias ../mem/weights/conv2_b_x4.mem
if {![file exists work]} {vlib work}

vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/bn_relu/relu.sv
vlog -sv ../rtl/bn_relu/bn_affine.sv
vlog -sv ../rtl/conv/conv_controller.sv
vlog -sv ../rtl/conv/conv_engine.sv
vlog -sv ../rtl/conv/conv_engine_parallel.sv
vlog -sv ../rtl/pooling/maxpool_engine.sv
vlog -sv ../rtl/pooling/streaming_maxpool1.sv
vlog -sv ../rtl/gru/sigmoid_lut.sv
vlog -sv ../rtl/gru/tanh_lut.sv
vlog -sv ../rtl/gru/gru_engine.sv
vlog -sv ../rtl/gru/gru_engine_pipeline.sv
vlog -sv ../rtl/fc/fc_engine.sv
vlog -sv ../rtl/fc/argmax_105.sv
vlog -sv ../rtl/top/cnn_gru_top.sv
vlog -sv ../rtl/top/eeg_controller.sv
vlog -sv ../rtl/top/eeg_top.sv
vlog -sv ../tb/integration/tb_eeg_top_4samples.sv

vsim work.tb_eeg_top_4samples
run -all
