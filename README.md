# FPGA EEG CNN-GRU Accelerator

This project implements a fixed-point EEG classifier on the Terasic DE1-SoC.
The complete path includes two depthwise-separable convolution stages, one
standard convolution stage, two streaming max-pooling stages, a GRU, two fully
connected layers, and a final argmax. A PC sends one EEG sample through UART,
and the FPGA returns the predicted class and winning logit.

The RTL is written in SystemVerilog. MATLAB is used to export the quantized
model and test data, while the Python tools prepare memory files and communicate
with the board.

## Current Architecture

```text
PC / Python
    |
    | UART request: sample_id + 3360 signed Q12 values + CRC-16
    v
fpga_uart_top
    |
    +-- UART receiver and packet loader
    |
    +-- DSConv1: DW 2x5 + PW 1->21 (3 PW lanes) + BN + ReLU
    |      output: 20 x 156 x 21
    |      |
    |      `-- six-column even/odd rolling buffer
    |
    +-- DSConv2: DW 2x5 + PW 21->20 (5 PW lanes) + BN + ReLU
    |      output: 19 x 152 x 20
    |      `-- double result bank overlaps computation and serialization
    |
    +-- Streaming MaxPool1: width 10, stride 8
    |      output: 19 x 18 x 20
    |
    +-- Conv3: 2x5, 20->15 (3 MAC lanes) + BN + ReLU
    |      output: 18 x 14 x 15
    |
    +-- Streaming MaxPool2: width 10, stride 8
    |      output: 18 x 1 x 15
    |
    +-- Pipelined GRU: 15 inputs, 9 hidden units, 18 time steps
    |
    +-- FC1: 162->40 + ReLU + BN
    |
    +-- Streaming FC output: 40->105
    |
    `-- Argmax
           |
           v
       UART response: sample_id + status + class + winning logit + CRC-16
```

The input tensor contains 21 EEG channels and 160 time samples. It is stored as
3360 signed 16-bit Q12 values. The classifier has 105 output classes. On the
board, the compact class index is mapped back to the original subject ID and
shown on the seven-segment displays.

## Pipeline and Memory Organization

DSConv1 and DSConv2 run with a producer-consumer schedule. DSConv1 writes each
completed column into a six-column rolling buffer while DSConv2 reads the
previous five-column window. The buffer is divided into even and odd height
banks, allowing both rows of the 2x5 depthwise kernel to be read in the same
cycle.

Inside each depthwise-separable engine, depthwise results are sent directly to
the pointwise stage. Two result banks are used at the output: one bank is
serialized while the other receives the next spatial result. This removes the
previous wait between computation and channel-by-channel output.

Pool1 starts as soon as DSConv2 begins producing valid data. Conv3 starts after
the first five Pool1 columns are ready and stalls only when the next required
column has not arrived. Pool2 consumes Conv3 output as a stream.

The main memories are:

- `input_banked_ram`: even/odd banks for the original EEG input.
- `dsconv12_window_buffer`: six-column even/odd rolling buffer between DSConv1
  and DSConv2.
- Pool1 internal even/odd max buffers in `dsconv_streaming_pool`.
- Shared activation RAM for UART input, Pool1 output, GRU output, and FC1 input
  at different phases of inference.
- `pool2_gru_ram`: small buffer between Pool2 and the GRU.
- M10K-based weight, bias, and activation memories where supported by Quartus.

## Current RTL Result

The latest complete ModelSim regression uses the 10 ns period from
`quartus/eeg_accelerator.sdc`.

| Measurement | Current result |
|---|---:|
| Clock used for cycle-to-time conversion | 100 MHz |
| Complete inference | 556,030 cycles |
| Calculated inference time | 5.5603 ms |
| Main model operations | 5,389,008 operations |
| Effective throughput | 0.969 GOPS |
| Produced logits | 105 |
| Regression prediction | Class 0 |
| Maximum checked layer difference | 2 LSB |

The operation count treats one multiply-accumulate as two operations. Pooling,
address generation, activation functions, and saturation are not included in
the GOPS count.

Reducing DSConv1 from seven pointwise lanes to three increased the complete
inference by only 20 cycles because DSConv1 remains hidden behind the slower
DSConv2 stage. The RTL regression still produces the same class and winning
logit.

The latest 105-subject UART demonstration produced 101 correct predictions out
of 105, or 96.19%. This number describes the one-sample-per-subject demo set; it
is not a replacement for complete test-set accuracy.

The current source changes have not yet been through a new Quartus Full
Compilation. Updated Fmax, ALM, register, M10K, DSP, and Power Analyzer results
must therefore be taken from the next build rather than from older reports.

## Clock and UART

The DE1-SoC board provides a physical 50 MHz `CLOCK_50` input. The current
`fpga_uart_top` uses this clock directly and sets `UART_CLKS_PER_BIT=54` for
921600 baud. The SDC currently contains a 10 ns timing target for 100 MHz timing
analysis and cycle reporting, but changing the SDC does not change the physical
board clock. A PLL or another real 100 MHz clock source is required before the
board itself can run at 100 MHz; the UART divider must also be updated for that
clock.

UART format: 921600 baud, 8 data bits, no parity, 1 stop bit. Multibyte fields
are little-endian. The complete packet definition is in
[`docs/uart_protocol.md`](docs/uart_protocol.md).

## Hardware and Tools

| Item | Setting |
|---|---|
| FPGA board | Terasic DE1-SoC |
| FPGA device | Cyclone V `5CSEMA5F31C6` |
| Quartus project | Quartus Prime Lite 25.1 |
| RTL | SystemVerilog |
| Simulation | Questa Altera FPGA / ModelSim |
| Host tools | MATLAB and Python 3 |
| UART | 921600 baud, 8N1 |

Install the Python dependency with:

```powershell
python -m pip install -r host\requirements.txt
```

## Repository Layout

```text
EEG_Project/
|-- rtl/        synthesizable RTL for the accelerator, memories, UART, and I/O
|-- tb/         unit and integration testbenches
|-- host/       MATLAB export, weight packing, UART, and benchmark tools
|-- mem/        generated fixed-point data, weights, LUTs, and golden results
|-- quartus/    SDC and ModelSim/PowerPlay scripts
|-- docs/       UART protocol and implementation notes
|-- scripts/    supporting generation utilities
|-- eeg_accelerator.qpf
|-- eeg_accelerator.qsf
`-- README.md
```

The Quartus top-level entity is `fpga_uart_top`. The QSF also contains older and
alternative modules used for comparison, but they are not all instantiated by
the current top-level design.

Generated model data under `mem/` is intentionally excluded from Git. The
required MATLAB exports and packed memory files must exist locally before
simulation or compilation.

## Preparing Packed Weights

The DSConv packing script generates the two-row depthwise words, three-lane
DSConv1 pointwise words, five-lane DSConv2 pointwise words, banked input sample,
and sample-0 golden files:

```powershell
python host\pack_dsconv1_dsconv2_weights.py
```

Conv3 uses a separate three-lane packing script. The complete cycle-count script
runs both packing steps automatically.

## Running Simulation

Start ModelSim or QuestaSim in the `quartus` directory. For the complete
functional and cycle regression:

```tcl
do run_eeg_cycle_count.do
```

The test checks the final class, all 105 logits, selected intermediate tensors,
and the total cycle count.

Other useful scripts:

- `run_ds_conv2_engine.do`: depthwise/pointwise pipeline and result-bank wave.
- `run_streaming_maxpool1.do`: first streaming pool.
- `run_streaming_maxpool2.do`: second streaming pool.
- `run_gru_pipeline.do`: pipelined GRU.
- `run_fpga_uart_top.do`: complete UART top-level test.
- `run_uart_unit.do`: UART receiver, transmitter, and packet logic.
- `run_eeg_power_vcd_25mhz.do`, `run_eeg_power_vcd_50mhz.do`, and
  `run_eeg_power_vcd_100mhz.do`: VCD generation for Power Analyzer.

From PowerShell, the complete regression can also be started with:

```powershell
cd quartus
vsim -c -do run_eeg_cycle_count.do
```

## Compiling for the DE1-SoC

1. Generate the required `.mem` files.
2. Open `eeg_accelerator.qpf` in Quartus Prime.
3. Confirm that `fpga_uart_top` is the top-level entity.
4. Run **Processing > Start Compilation**.
5. Review Fitter resource usage and the slow-corner setup timing report.
6. Program the generated `.sof` file onto the DE1-SoC.

Run a new Full Compilation whenever lane counts, packed ROM widths, memory
organization, or pipeline structure changes. ModelSim verifies functionality
and cycle count, but it cannot determine final DSP use or Fmax.

## UART Dataset Test

The MATLAB exporter creates:

- `host/data/test_inputs_q12.bin`
- `host/data/test_labels.csv`

Validate the files and packet construction without opening a COM port:

```powershell
python host\send_eeg_uart.py --dry-run
```

Send the test set to the board:

```powershell
python host\send_eeg_uart.py --port COM10 --baud 921600
```

Prepare and run the one-sample-per-subject demonstration:

```powershell
python host\demo_105_subjects_uart.py --prepare-only --rebuild-dataset
python host\demo_105_subjects_uart.py --port COM10 --baud 921600
```

Replace `COM10` with the port shown in Windows Device Manager. The host waits
for the FPGA response before sending the next sample because activation memory
is reused during inference.

## License

This repository currently has no license file.
