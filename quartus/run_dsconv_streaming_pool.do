transcript on

# 可從 EEG_Project 或 EEG_Project/quartus 執行。
if {![file isdirectory rtl]} {
    if {[file isdirectory ../rtl]} {cd ..} else {error "Run from EEG_Project or EEG_Project/quartus"}
}

if {![file exists work]} {vlib work}

vlog -sv rtl/common/sat16.sv
vlog -sv rtl/memory/activation_ram.sv
vlog -sv rtl/pooling/dsconv_streaming_pool.sv
vlog -sv tb/unit/tb_dsconv_streaming_pool.sv

# +acc 保留 RAM 與 pipeline 內部訊號。
vsim -voptargs=+acc work.tb_dsconv_streaming_pool

view wave
delete wave *
configure wave -namecolwidth 290
configure wave -valuecolwidth 120

add wave -divider {Clock / control}
add wave sim:/tb_dsconv_streaming_pool/clk
add wave sim:/tb_dsconv_streaming_pool/rst_n
add wave sim:/tb_dsconv_streaming_pool/start
add wave sim:/tb_dsconv_streaming_pool/busy
add wave sim:/tb_dsconv_streaming_pool/done

add wave -divider {DS-Conv2 input stream}
add wave sim:/tb_dsconv_streaming_pool/input_valid
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/input_w
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/input_h
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/input_channel
add wave -radix decimal sim:/tb_dsconv_streaming_pool/input_data

add wave -divider {Window controller}
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/dut/current_window
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/dut/stride_phase
add wave sim:/tb_dsconv_streaming_pool/dut/previous_window_active
add wave sim:/tb_dsconv_streaming_pool/dut/previous_window_ends
add wave sim:/tb_dsconv_streaming_pool/dut/current_window_is_odd
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/dut/max_read_addr

add wave -divider {Synchronous RAM pipeline}
add wave sim:/tb_dsconv_streaming_pool/dut/stage1_valid
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/stage1_data
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/dut/stage1_max_addr
add wave sim:/tb_dsconv_streaming_pool/dut/stage1_previous_window_ends
add wave sim:/tb_dsconv_streaming_pool/dut/stage1_current_window_is_odd
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/max_even_read_data
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/max_odd_read_data

add wave -divider {MAX result / RAM write}
add wave sim:/tb_dsconv_streaming_pool/dut/max_even_write_en
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/max_even_write_data
add wave sim:/tb_dsconv_streaming_pool/dut/max_odd_write_en
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/max_odd_write_data
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/completed_max
add wave -radix decimal sim:/tb_dsconv_streaming_pool/dut/completed_saturated

add wave -divider {Registered output}
add wave sim:/tb_dsconv_streaming_pool/output_valid
add wave -radix unsigned sim:/tb_dsconv_streaming_pool/output_addr
add wave -radix decimal sim:/tb_dsconv_streaming_pool/output_data

# W9/H0/CH0 約在 68.5 us，顯示 Window 0/1 的重疊更新。
run 70 us
wave zoom range 68400 ns 69200 ns
