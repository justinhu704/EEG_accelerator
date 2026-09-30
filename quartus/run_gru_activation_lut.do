if {[file exists work]} {vdel -lib work -all}
vlib work
vmap work work

vlog -sv ../rtl/gru/sigmoid_lut.sv
vlog -sv ../rtl/gru/tanh_lut.sv
vlog -sv ../rtl/gru/gru_activation_lut.sv
vlog -sv ../tb/unit/tb_gru_activation_lut.sv

vsim -voptargs=+acc work.tb_gru_activation_lut
run -all
