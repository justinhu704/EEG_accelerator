if {[file exists work]} {vdel -lib work -all}
vlib work
vmap work work

vlog -sv ../rtl/memory/activation_ram.sv
vlog -sv ../rtl/gru/sigmoid_lut.sv
vlog -sv ../rtl/gru/tanh_lut.sv
vlog -sv ../rtl/gru/gru_engine.sv
vlog -sv ../rtl/gru/gru_engine_pipeline.sv
vlog -sv ../tb/unit/tb_gru_pipeline.sv

vsim -voptargs=+acc work.tb_gru_pipeline
run -all
