# FPGA EEG CNN-GRU Accelerator

This repository is an undergraduate project on running an EEG
classification model on an FPGA. The model contains three CNN stages, a GRU,
two fully connected layers, and a final argmax operation. The inference path is
written in SystemVerilog and uses fixed-point arithmetic so that it can run on a
DE1-SoC board.

The current design accepts one EEG sample from a PC through UART, performs the
complete CNN-GRU inference, and sends the predicted class and winning logit back
to the PC. MATLAB and Python scripts are included for exporting the test data,
packing weights, sending samples, and recording the FPGA results.

## System Overview

```text
MATLAB/Python host
       |
       | UART request: sample ID + 3360 Q12 input values + CRC
       v
DE1-SoC FPGA
       |
       +-- Conv1 (3 lanes) -> BN -> ReLU -> 8-bit even/odd feature RAM
       +-- DS-Conv2 (5 pointwise lanes) -> BN -> ReLU -> streaming MaxPool1
       +-- Conv3 (3 lanes) -> BN -> ReLU -> streaming MaxPool2
       +-- pipelined GRU -> pipelined FC1 -> ReLU -> BN
       +-- streaming output FC -> Argmax
       |
       v
UART response: predicted class + winning logit + CRC
```

The input shape is `21 x 160`, stored as 3360 signed 16-bit Q12 values. The
classifier produces 105 output classes. On the board, the class index is also
mapped back to its subject ID and shown on the seven-segment displays.

## Hardware and Tools

| Item | Setting used in this project |
|---|---|
| FPGA board | Terasic DE1-SoC |
| FPGA device | Cyclone V `5CSEMA5F31C6` |
| Board clock | 50 MHz `CLOCK_50` input |
| Current timing target | 100 MHz (10 ns constraint) |
| Quartus project | Quartus Prime Lite 25.1 |
| RTL language | SystemVerilog |
| Simulation | Questa Altera FPGA / ModelSim |
| Host environment | MATLAB and Python 3 |
| UART setting | 921600 baud, 8 data bits, no parity, 1 stop bit |

The Python UART scripts only require `pyserial`. Install it with:

```powershell
python -m pip install -r host\requirements.txt
```

## Repository Layout

```text
EEG_Project/
|-- rtl/        SystemVerilog modules for CNN, GRU, memory, UART, and board I/O
|-- tb/         Unit and full-pipeline testbenches
|-- host/       MATLAB/Python dataset export and UART tools
|-- mem/        Fixed-point weights, lookup tables, and board test samples
|-- quartus/    Quartus settings, timing constraints, and simulation scripts
|-- scripts/    Utilities for generating RTL support data
|-- docs/       UART packet format and implementation notes
|-- eeg_accelerator.qpf
|-- eeg_accelerator.qsf
`-- README.md
```

The Quartus top-level entity is `fpga_uart_top`, implemented in
`rtl/board/fpga_uart_top.sv`. Some older or alternative RTL modules are kept for
comparison and separate simulation.

## Running a Simulation

1. Open ModelSim or QuestaSim and change the working directory to `quartus/`.
2. Run one of the `.do` files from the Transcript window.

For example, the following script compiles and runs the UART top-level
testbench:

```tcl
do run_fpga_uart_top.do
```

Other useful scripts include:

- `run_eeg_top.do` for the complete inference path
- `run_eeg_cycle_count.do` for RTL cycle counting
- `run_gru_pipeline.do` for the GRU pipeline
- `run_uart_unit.do` for the UART modules

The scripts for the parallel convolution designs first regenerate their packed
weight files with `host/pack_parallel_conv_weights.py` and then compile the RTL
and testbench files.

## Compiling for the DE1-SoC

1. Open `eeg_accelerator.qpf` in Quartus Prime.
2. Check that `fpga_uart_top` is selected as the top-level entity.
3. Start compilation from **Processing > Start Compilation**.
4. Program the generated `.sof` file to the DE1-SoC.

The device selection and board pin assignments are stored in
`eeg_accelerator.qsf`. Model weight and lookup-table `.mem` files must be in the
paths referenced by the RTL before compilation.

## Sending a Test Sample through UART

The MATLAB export script creates:

- `host/data/test_inputs_q12.bin`
- `host/data/test_labels.csv`

Before connecting the board, the files and packet format can be checked without
opening a COM port:

```powershell
python host\send_eeg_uart.py --dry-run
```

To send samples to the FPGA, replace `COM10` with the port shown in Windows
Device Manager:

```powershell
python host\send_eeg_uart.py --port COM10 --baud 921600
```

The current hardware uses one shared 16-bit activation RAM and an 8-bit banked
feature RAM. The banked RAM separates even and odd height rows so Conv2 can read
two adjacent rows at the same time. It is reused for the Pool2 output after the
Conv1 feature map is no longer needed. The host sends one sample and waits for
its response before sending the next sample. The complete request and response
packet fields are documented in `docs/uart_protocol.md`.

## Current Project Status

The current inference path uses a depthwise separable Conv2, streaming pooling,
even/odd feature-memory banks, and pipelined GRU and fully connected stages.
Conv1 and Conv3 use three MAC lanes, while the pointwise stage of DS-Conv2 uses
five lanes. The DS-Conv2 stage reduced its measured RTL cycle count from 823,086
to 392,774 cycles.

The full RTL inference takes 1,018,709 cycles. This corresponds to about
20.374 ms with the board's 50 MHz input clock, or 10.187 ms at 100 MHz. The
current Quartus build meets the 100 MHz timing constraint with a reported Fmax
of 108.92 MHz in the slow 1100 mV, 85 C corner. The design still requires a PLL
or another 100 MHz clock source to run at 100 MHz on the board; changing the SDC
constraint alone does not change the physical input clock.

Current resource usage is 4,450 ALMs, 6,179 registers, 127 M10K blocks, and 34
DSP blocks. The latest full UART dataset run classified 115,211 of 119,075
samples correctly, giving 96.75% accuracy.

PowerPlay currently estimates 622.79 mW total thermal power and 176.16 mW core
dynamic power for the 100 MHz build. The report has low estimation confidence
because the available VCD does not cover enough internal switching activity, so
these power values should only be treated as preliminary estimates.

## License

This project currently has no license file.
