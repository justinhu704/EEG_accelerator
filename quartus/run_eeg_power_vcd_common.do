transcript on
onerror {quit -code 1}

if {![info exists CLOCK_PERIOD_NS] || ![info exists CLOCK_FREQ_TAG]} {
    puts "ERROR: Set CLOCK_PERIOD_NS and CLOCK_FREQ_TAG before running this script."
    quit -code 1
}

# 產生目前 DSConv1 -> DSConv2 streaming 架構所需的權重。
exec python ../host/pack_dsconv1_dsconv2_weights.py
exec python ../host/pack_parallel_conv_weights.py --weights ../mem/dsconv1_dsconv2/weights/conv3_W.mem --bias ../mem/dsconv1_dsconv2/weights/conv3_b.mem --output-weights ../mem/dsconv1_dsconv2/weights/conv3_W_x3.mem --output-bias ../mem/dsconv1_dsconv2/weights/conv3_b_x3.mem --kh 2 --kw 5 --in-ch 20 --out-ch 15 --lanes 3

# 使用乾淨的 library，避免載入修改前的舊 RTL。
if {[file exists work]} {
    vdel -all -lib work
}
vlib work

# Common / memory
vlog -sv ../rtl/common/sat16.sv
vlog -sv ../rtl/common/pe_mac.sv
vlog -sv ../rtl/memory/weight_rom.sv
vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/memory/input_banked_ram.sv
vlog -sv ../rtl/memory/conv1_banked_ram.sv
vlog -sv ../rtl/memory/dsconv12_window_buffer.sv
vlog -sv ../rtl/memory/pool2_gru_ram.sv

# BN / convolution / pooling
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

# GRU / FC / top / testbench
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

set CLOCK_FREQ_MHZ [expr {1000.0 / $CLOCK_PERIOD_NS}]
puts "Simulation clock: $CLOCK_FREQ_MHZ MHz ($CLOCK_PERIOD_NS ns)"

# 保留內部節點，讓完整 DUT switching activity 能寫入 VCD。
vsim -voptargs=+acc -gCLOCK_PERIOD_NS=$CLOCK_PERIOD_NS work.tb_eeg_cycle_count

file mkdir ../power
set VCD_PATH "../power/eeg_active_${CLOCK_FREQ_TAG}.vcd"
vcd file $VCD_PATH
vcd add -r /tb_eeg_cycle_count/dut/*

run -all

vcd flush
vcd off
puts "VCD generated: $VCD_PATH"
