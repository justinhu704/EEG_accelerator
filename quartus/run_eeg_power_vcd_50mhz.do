transcript on
exec python ../host/pack_parallel_conv_weights.py --weights ../mem/weights/conv1_W.mem --bias ../mem/weights/conv1_b.mem --output-weights ../mem/weights/conv1_W_x3.mem --output-bias ../mem/weights/conv1_b_x3.mem --kh 2 --kw 5 --in-ch 1 --out-ch 21 --lanes 3
exec python ../host/pack_parallel_conv_weights.py --weights ../mem/weights/conv2_W.mem --bias ../mem/weights/conv2_b.mem --output-weights ../mem/weights/conv2_W_x5.mem --output-bias ../mem/weights/conv2_b_x5.mem --lanes 5
exec python ../host/pack_parallel_conv_weights.py --weights ../mem/weights/conv2_W.mem --bias ../mem/weights/conv2_b.mem --output-weights ../mem/weights/conv2_W_x5_kh2.mem --output-bias ../mem/weights/conv2_b_x5_kh2.mem --lanes 5 --pair-kh
exec python ../host/pack_parallel_conv_weights.py --weights ../mem/weights/conv3_W.mem --bias ../mem/weights/conv3_b.mem --output-weights ../mem/weights/conv3_W_x3.mem --output-bias ../mem/weights/conv3_b_x3.mem --kh 2 --kw 5 --in-ch 20 --out-ch 15 --lanes 3

# 使用乾淨的 library，避免載入修改前的舊 RTL。
if {[file exists work]} {
    vdel -all -lib work
}
vlib work

vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/memory/conv1_banked_ram.sv
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
vlog -sv ../rtl/pooling/maxpool_engine.sv
vlog -sv ../rtl/pooling/streaming_maxpool.sv
vlog -sv ../rtl/pooling/dsconv_streaming_pool.sv
vlog -sv ../rtl/gru/sigmoid_lut.sv
vlog -sv ../rtl/gru/tanh_lut.sv
vlog -sv ../rtl/gru/gru_engine.sv
vlog -sv ../rtl/gru/gru_engine_pipeline.sv
vlog -sv ../rtl/fc/fc_engine.sv
vlog -sv ../rtl/fc/fc_out_streaming.sv
vlog -sv ../rtl/fc/argmax_105.sv
vlog -sv ../rtl/top/cnn_gru_top.sv
vlog -sv ../rtl/top/eeg_controller.sv
vlog -sv ../rtl/top/eeg_top.sv
vlog -sv ../tb/integration/tb_eeg_cycle_count.sv

# Override testbench clock period: 20.000 ns = 50 MHz.
vsim -gCLOCK_PERIOD_NS=20.000 work.tb_eeg_cycle_count

# 記錄一筆完整推論的實際切換活動，供 Power Analyzer 使用。
file mkdir ../power
vcd file ../power/eeg_active_50mhz.vcd
vcd add -r /tb_eeg_cycle_count/dut/*

run -all

vcd flush
vcd off
