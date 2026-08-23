# FPGA-Based EEG CNN-GRU Accelerator

This project implements an FPGA-based accelerator for EEG signal classification. The inference pipeline combines convolutional neural network (CNN) layers with a gated recurrent unit (GRU) to process EEG data efficiently in hardware.

The design is written primarily in SystemVerilog and is developed using Intel Quartus Prime. Python host utilities are included for weight preparation, UART communication, dataset inference, and performance benchmarking.

## Features

- Hardware-accelerated EEG inference
- SystemVerilog RTL implementation
- Convolution, batch normalization, and ReLU processing
- Streaming max-pooling
- Fully connected layers
- GRU-based temporal processing
- On-chip activation and weight memory
- UART communication between the FPGA and host computer
- Unit testbenches and ModelSim/Questa simulation scripts
- Python utilities for data transfer and benchmarking

## Project Structure

The Quartus top-level entity is `fpga_uart_top`.

```text
EEG_Project/
├── rtl/
│   ├── bn_relu/    # Batch-normalization and ReLU operators
│   ├── board/      # DE1-SoC top levels, sample loading, and board I/O
│   ├── common/     # Shared arithmetic and fixed-point helpers
│   ├── conv/       # Sequential and parallel convolution engines
│   ├── display/    # Class mapping and seven-segment display logic
│   ├── fc/         # Fully connected layers and final argmax
│   ├── gru/        # GRU engines and activation lookup tables
│   ├── memory/     # Activation RAM, weight ROM, and banked buffers
│   ├── pooling/    # Memory-based and streaming max-pooling
│   ├── top/        # CNN-GRU pipeline and inference control
│   └── uart/       # UART receive, transmit, packet loading, and results
│
├── tb/
│   ├── unit/       # Module-level SystemVerilog testbenches
│   └── integration/ # Full-pipeline, FPGA top, and UART tests
│
├── host/           # Python/MATLAB export, UART, packing, and benchmark tools
├── mem/
│   └── board/      # Version-controlled fixed-point board test samples
├── quartus/        # Timing constraints and ModelSim/Questa run scripts
├── scripts/        # RTL support-data generation utilities
├── docs/           # Data formats, memory map, UART, and verification notes
│
├── eeg_accelerator.qpf  # Quartus project descriptor
├── eeg_accelerator.qsf  # Device, pin, top-level, and RTL source assignments
├── eeg_accelerator_description.txt
├── .gitignore
└── README.md
```

The production hierarchy starts at `rtl/board/fpga_uart_top.sv`. Some RTL
files are retained as alternative implementations or for standalone
verification. Generated build, simulation, dataset, and result files are
excluded through `.gitignore`.

## Requirements

### FPGA Development

- Intel Quartus Prime
- ModelSim Intel FPGA Edition or QuestaSim
- A supported Intel FPGA development board
- USB-UART connection

### Host Computer

- Python 3
- Required Python packages used by the scripts in `host/`

The exact FPGA device, pin assignments, and top-level entity are defined in `eeg_accelerator.qsf`.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/justinhu704/EEG_Project.git
cd EEG_Project
```

Replace `justinhu704` with your GitHub username.

### 2. Open the Quartus Project

Open the following file in Intel Quartus Prime:

```text
eeg_accelerator.qpf
```

Review the FPGA device, pin assignments, clock settings, and top-level entity before compilation.

### 3. Run RTL Simulations

The `tb/unit/` directory contains unit testbenches for individual RTL modules.

Simulation command files are located in the `quartus/` directory. Run the appropriate `.do` script using ModelSim or QuestaSim.

For example:

```text
quartus/run_relu1_packed_buffer_adapter.do
```

Check the simulation transcript and waveforms to verify the expected behavior.

### 4. Compile the FPGA Design

In Quartus Prime:

1. Open `eeg_accelerator.qpf`.
2. Select **Processing > Start Compilation**.
3. Confirm that compilation completes without critical errors.
4. Review timing analysis and resource utilization.
5. Program the generated FPGA configuration file onto the board.

## Project Status

This project is under active development. RTL interfaces, model parameters, host communication protocols, and test procedures may continue to change.

## License

No license has been specified for this project.