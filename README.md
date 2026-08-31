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
       +-- CNN1 -> BN -> ReLU
       +-- CNN2 -> BN -> ReLU -> MaxPool
       +-- CNN3 -> BN -> ReLU -> MaxPool
       +-- GRU -> FC -> ReLU -> BN -> FC -> Argmax
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

The current hardware reuses two activation RAMs during inference, so the host
sends one sample and waits for its response before sending the next sample. The
complete request and response packet fields are documented in
`docs/uart_protocol.md`.

## Current Project Status

The repository currently contains the complete RTL hierarchy, module-level
testbenches, full-pipeline testbenches, UART host tools, and a cycle-counting
testbench. Parallel convolution versions are also included to reduce the number
of inference cycles.

The next results to document are the final Quartus resource usage, timing
results, FPGA-only inference latency, and accuracy comparison with the software
model. These values are intentionally not listed here until they have been
measured on the final build.

## License

This project currently has no license file.
