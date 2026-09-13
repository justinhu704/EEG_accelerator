transcript on

# 可從 EEG_Project 或 EEG_Project/quartus 執行。
if {![file isdirectory rtl]} {
    if {[file isdirectory ../rtl]} {cd ..} else {error "Run from EEG_Project or EEG_Project/quartus"}
}

if {![file exists work]} {vlib work}

vlog -sv rtl/conv/ds_conv2_engine.sv
vlog -sv tb/unit/tb_ds_conv2_engine.sv

# +acc 保留內部 pipeline 訊號，方便在 Wave 視窗觀察。
vsim -voptargs=+acc work.tb_ds_conv2_engine

view wave
delete wave *
configure wave -namecolwidth 260
configure wave -valuecolwidth 120

add wave -divider {Clock / control}
add wave sim:/tb_ds_conv2_engine/clk
add wave sim:/tb_ds_conv2_engine/rst_n
add wave sim:/tb_ds_conv2_engine/start
add wave sim:/tb_ds_conv2_engine/busy
add wave -radix symbolic sim:/tb_ds_conv2_engine/dut/state

add wave -divider {DW address issue}
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/dw_issue_channel
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/dw_issue_kw
add wave -radix unsigned sim:/tb_ds_conv2_engine/input_addr_kh0
add wave -radix unsigned sim:/tb_ds_conv2_engine/input_addr_kh1
add wave -radix unsigned sim:/tb_ds_conv2_engine/dw_weight_addr

add wave -divider {Depthwise pipeline}
add wave sim:/tb_ds_conv2_engine/dut/dw_read_valid
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/dw_read_channel
add wave sim:/tb_ds_conv2_engine/dut/dw_read_first
add wave sim:/tb_ds_conv2_engine/dut/dw_read_last
add wave sim:/tb_ds_conv2_engine/dut/dw_prod_valid
add wave -radix decimal sim:/tb_ds_conv2_engine/dut/dw_product_kh0
add wave -radix decimal sim:/tb_ds_conv2_engine/dut/dw_product_kh1
add wave sim:/tb_ds_conv2_engine/dut/dw_pair_valid
add wave -radix decimal sim:/tb_ds_conv2_engine/dut/dw_pair_sum
add wave -radix decimal sim:/tb_ds_conv2_engine/dut/dw_accumulator
add wave -radix decimal sim:/tb_ds_conv2_engine/dut/dw_quantized_value

add wave -divider {Fused pointwise scheduler}
add wave sim:/tb_ds_conv2_engine/dut/pw_issue_active
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_pending_channel
add wave -radix decimal sim:/tb_ds_conv2_engine/dut/pw_pending_activation
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_issue_group
add wave sim:/tb_ds_conv2_engine/dut/pw_read_valid
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_read_channel
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_read_group
add wave -radix unsigned sim:/tb_ds_conv2_engine/pw_weight_addr

add wave -divider {Pointwise pipeline lane 0}
add wave sim:/tb_ds_conv2_engine/dut/pw_prod_valid
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_prod_channel
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_prod_group
add wave -radix decimal {sim:/tb_ds_conv2_engine/dut/pw_products[0]}
add wave -radix decimal {sim:/tb_ds_conv2_engine/dut/pw_accumulators[0][0]}
add wave sim:/tb_ds_conv2_engine/dut/pw_finalize_valid
add wave -radix unsigned sim:/tb_ds_conv2_engine/dut/pw_finalize_group
add wave -radix decimal {sim:/tb_ds_conv2_engine/dut/pw_finalize_sums[0]}
add wave -radix decimal {sim:/tb_ds_conv2_engine/dut/pw_results[0][0]}

add wave -divider {Serialized output}
add wave sim:/tb_ds_conv2_engine/output_valid
add wave -radix unsigned sim:/tb_ds_conv2_engine/output_channel
add wave -radix decimal sim:/tb_ds_conv2_engine/output_data
add wave -radix unsigned sim:/tb_ds_conv2_engine/output_addr
add wave sim:/tb_ds_conv2_engine/output_last
add wave sim:/tb_ds_conv2_engine/done

run -all
wave zoom full
