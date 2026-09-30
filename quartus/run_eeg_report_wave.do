# RTL report waveform for one complete EEG inference.
#
# Usage from the ModelSim / Questa transcript:
#   do C:/EEG_Project/quartus/run_eeg_report_wave.do
#
# From PowerShell, use the ModelSim 2020.1 Windows GUI launcher:
#   & "C:\ModelSim\modelsim_ase\win32aloem\modelsim.exe" -do \
#       "C:\EEG_Project\quartus\run_eeg_report_wave.do"

transcript on

# ModelSim Intel FPGA Starter 2020.1 does not report the active .do path from
# [info script]; it returns an internal Tcl path instead. Use the known project
# simulation directory so that every ../rtl, ../tb and ../mem path is stable.
cd C:/EEG_Project/quartus

# Regenerate the packed ROM images used by the current DS-Conv1/DS-Conv2 RTL.
exec python ../host/pack_dsconv1_dsconv2_weights.py
exec python ../host/pack_parallel_conv_weights.py \
    --weights ../mem/dsconv1_dsconv2/weights/conv3_W.mem \
    --bias ../mem/dsconv1_dsconv2/weights/conv3_b.mem \
    --output-weights ../mem/dsconv1_dsconv2/weights/conv3_W_x3.mem \
    --output-bias ../mem/dsconv1_dsconv2/weights/conv3_b_x3.mem \
    --kh 2 --kw 5 --in-ch 20 --out-ch 15 --lanes 3

if {![file exists work]} {
    vlib work
}

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

# Use the same direct core-clock constraint as Quartus and cycle-count tests.
set sdc_file "eeg_accelerator.sdc"
set sdc_fp [open $sdc_file r]
set sdc_text [read $sdc_fp]
close $sdc_fp

if {![regexp {set[ \t]+CORE_CLOCK_PERIOD_NS[ \t]+([0-9.]+)} \
      $sdc_text sdc_match clock_period_ns]} {
    puts "ERROR: Cannot find CORE_CLOCK_PERIOD_NS in $sdc_file"
    quit -code 1
}

set clock_freq_mhz [expr {1000.0 / $clock_period_ns}]
puts "SDC core period  : $clock_period_ns ns"
puts "Simulation clock : $clock_freq_mhz MHz"

vsim -gCLOCK_PERIOD_NS=$clock_period_ns work.tb_eeg_cycle_count

# ModelSim 2020.1 only accepts onfinish after a design has been elaborated.
# Keep the GUI open when the testbench reaches $finish.
onfinish stop

view wave
delete wave *

configure wave -namecolwidth 285
configure wave -valuecolwidth 110
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -timelineunits us

# -----------------------------------------------------------------------------
# Group 1: complete-inference control and the two levels of FSM.
# -----------------------------------------------------------------------------
add wave -divider {System control}
add wave sim:/tb_eeg_cycle_count/clk
add wave sim:/tb_eeg_cycle_count/rst_n
add wave sim:/tb_eeg_cycle_count/start
add wave sim:/tb_eeg_cycle_count/busy
add wave sim:/tb_eeg_cycle_count/done
add wave -radix symbolic sim:/tb_eeg_cycle_count/dut/u_controller/state
add wave -radix symbolic sim:/tb_eeg_cycle_count/dut/u_cnn_gru/state

# -----------------------------------------------------------------------------
# Group 2: busy/valid bars that make the actual overlap visible.
# Conv1 and Conv2 are scheduled by conv12_busy. Pool1 overlaps DS-Conv2;
# Conv3 starts after five Pool1 columns, and Pool2 consumes Conv3 immediately.
# -----------------------------------------------------------------------------
add wave -divider {CNN and pooling overlap}
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv12_busy
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv1_start
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv1_busy
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv2_start
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv2_busy
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/pool1_busy
add wave -radix unsigned sim:/tb_eeg_cycle_count/dut/u_cnn_gru/pool1_columns_ready
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv3_input_ready
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/conv3_busy
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/pool2_busy

# -----------------------------------------------------------------------------
# Group 3: sequential back end and FC1/FC-out overlap.
# FC-out begins with FC1 and consumes each valid BN feature as it arrives.
# -----------------------------------------------------------------------------
add wave -divider {GRU and classifier}
add wave sim:/tb_eeg_cycle_count/dut/u_cnn_gru/gru_busy
add wave sim:/tb_eeg_cycle_count/dut/fc1_busy
add wave sim:/tb_eeg_cycle_count/dut/fc_bn_valid
add wave -radix unsigned sim:/tb_eeg_cycle_count/dut/fc_addr_d2
add wave sim:/tb_eeg_cycle_count/dut/u_fc_out/busy
add wave sim:/tb_eeg_cycle_count/dut/fc_out_output_valid
add wave sim:/tb_eeg_cycle_count/dut/argmax_done

# -----------------------------------------------------------------------------
# Group 4: final 105-logit stream and classification result.
# -----------------------------------------------------------------------------
add wave -divider {Final logits and result}
add wave sim:/tb_eeg_cycle_count/logit_valid
add wave -radix unsigned sim:/tb_eeg_cycle_count/logit_index
add wave -radix decimal sim:/tb_eeg_cycle_count/logit_data
add wave -radix unsigned sim:/tb_eeg_cycle_count/class_index
add wave -radix decimal sim:/tb_eeg_cycle_count/winning_logit

puts "Running the complete RTL inference for the report waveform..."
run -all

wave zoom full
puts ""
puts "============================================================"
puts "Report waveform is ready in the Wave window."
puts "For the report, use the full-range busy overview and zoom in"
puts "on the final logit_valid/logit_index/done interval."
puts "============================================================"

