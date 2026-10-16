transcript on
if {![file exists work]} {vlib work}

vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/memory/input_banked_ram.sv
vlog -sv ../rtl/memory/conv1_banked_ram.sv
vlog -sv ../rtl/memory/dsconv12_window_buffer.sv
vlog -sv ../rtl/memory/pool2_gru_ram.sv
vlog -sv ../rtl/bn_relu/relu.sv
vlog -sv ../rtl/bn_relu/bn_affine.sv
vlog -sv ../rtl/conv/conv_controller.sv
vlog -sv ../rtl/conv/conv_engine.sv
vlog -sv ../rtl/conv/conv_engine_parallel.sv
vlog -sv ../rtl/conv/conv_engine_parallel_counter.sv
vlog -sv ../rtl/conv/conv_engine_parallel_kh2.sv
vlog -sv ../rtl/conv/conv_bn_relu_parallel_kh2_block.sv
vlog -sv ../rtl/conv/ds_conv2_engine.sv
vlog -sv ../rtl/conv/ds_conv2_bn_relu_block.sv
vlog -sv ../rtl/conv/dsconv12_overlap_scheduler.sv
vlog -sv ../rtl/pooling/maxpool_engine.sv
vlog -sv ../rtl/pooling/streaming_maxpool.sv
vlog -sv ../rtl/pooling/dsconv_streaming_pool.sv
vlog -sv ../rtl/gru/sigmoid_lut.sv
vlog -sv ../rtl/gru/tanh_lut.sv
vlog -sv ../rtl/gru/gru_activation_lut.sv
vlog -sv ../rtl/gru/gru_engine.sv
vlog -sv ../rtl/gru/gru_engine_pipeline.sv
vlog -sv ../rtl/fc/fc_engine.sv
vlog -sv ../rtl/fc/fc_out_streaming.sv
vlog -sv ../rtl/fc/argmax_105.sv
vlog -sv ../rtl/top/cnn_gru_top.sv
vlog -sv ../rtl/top/eeg_controller.sv
vlog -sv ../rtl/top/eeg_top.sv
vlog -sv ../tb/integration/tb_eeg_cycle_count.sv

vsim -gCLOCK_PERIOD_NS=10.0 work.tb_eeg_cycle_count
vcd file ../output/conv12_timing.vcd
vcd add /tb_eeg_cycle_count/clk
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv1_start
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv1_valid
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv1_last
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv2_start
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv2_valid
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv2_last
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv1_column
vcd add /tb_eeg_cycle_count/dut/u_cnn_gru/conv2_column
run -all
vcd flush
quit -f
