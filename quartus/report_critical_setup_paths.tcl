project_open eeg_accelerator -revision eeg_accelerator
create_timing_netlist
read_sdc quartus/eeg_accelerator.sdc
update_timing_netlist

report_timing -setup -npaths 20 -detail full_path \
    -file output_files/eeg_accelerator.critical_setup.rpt

delete_timing_netlist
project_close
