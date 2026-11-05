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
- Shared activation RAM for Pool1 output and the final GRU output. FC1 reads
  the GRU result from this RAM.
- `pool2_gru_ram`: small buffer between Pool2 and the GRU.
- M10K-based weight, bias, and activation memories where supported by Quartus.

## Current RTL Result

The reported complete ModelSim regression uses a 10 ns core-clock test period.

| Measurement | Current result |
|---|---:|
| Clock used for cycle-to-time conversion | 100 MHz |
| Complete inference | 556,984 cycles |
| Calculated inference time | 5.56984 ms |
| Main model MACs | 2,694,504 MACs |
| Main model operations | 5,389,008 operations |
| Effective throughput | 0.968 GOPS |
| Core inference rate, excluding UART | 179.5 inferences/s |
| Throughput per DSP block | 0.0346 GOPS/DSP |
| Produced logits | 105 |
| Regression prediction | Class 0 |
| Regression winning logit | 22,397 |

The operation count treats one multiply-accumulate as two operations. Pooling,
address generation, activation functions, saturation, and some elementwise
operations are not included in the GOPS count. These figures describe core
inference; UART transfer and host overhead are excluded.

Reducing DSConv1 from seven pointwise lanes to three increased the complete
inference by only 20 cycles because DSConv1 remains hidden behind the slower
DSConv2 stage. The RTL regression still produces the same class and winning
logit.

The final accuracy evaluation uses the complete 119,075-sample test set, rather
than the one-sample-per-subject UART demonstration:

| Evaluation | Correct predictions | Accuracy |
|---|---:|---:|
| MATLAB floating-point model | — | 96.54% |
| MATLAB fixed-point model | 114,965 / 119,075 | 96.55% |
| FPGA hardware | 114,965 / 119,075 | 96.55% |
| MATLAB fixed-point / FPGA class agreement | All test samples | 100% |

The following post-fit results for the Cyclone V `5CSEMA5F31C6` are the results
reported in Sections 5.3 and 5.4 of the
[project report](https://docs.google.com/document/d/1sEWBq4gSkROigj2ZEsMR7D_hydCRK2R4ZTHnyQXvEkg/edit).
They describe the 100 MHz PLL implementation, including the shared GRU
activation LUT. A separate local compilation may produce slightly different
placement and resource figures.

| Post-fit measurement | Reported result |
|---|---:|
| Fmax | 111.62 MHz |
| Logic utilization | 4,977 / 32,070 ALMs (approximately 16%) |
| Registers | 7,064 |
| M10K blocks | 75 / 397 (19%) |
| Block-memory bits | 448,040 / 4,065,280 (11%) |
| DSP blocks | 28 / 87 (32%) |
| GRU DSP blocks | 6 |
| PLLs | 1 / 6 |
| Total pins | 59 / 457 (13%) |

GRU multiplier sharing and the activation LUT optimization reduced GRU DSP
usage from 13 to 6 blocks and total DSP usage from 35 to 28 blocks (20%).

The reset/update sigmoid and candidate tanh tables share one dual-read
256 x 32-bit ROM, using one M10K. An exhaustive ModelSim comparison over all
signed 16-bit inputs, the 162-output GRU regression, and the complete EEG
inference regression are bit-exact.

## Power Estimates

The frequency sweep below follows Table 9 of the project report. Latency is
core inference time. GOPS/W uses total power, while energy per inference in
the last column uses core dynamic power only. Table entries are reproduced
as reported.

| Frequency | Inference latency | Core dynamic power | Total power | GOPS/W | Dynamic energy/inference |
|---|---:|---:|---:|---:|---:|
| 100 MHz | 5.570 ms | 162.26 mW | 607.12 mW | 1.589 | 0.903 mJ |
| 50 MHz | 11.140 ms | 74.34 mW | 506.58 mW | 0.955 | 0.828 mJ |
| 25 MHz | 22.279 ms | 38.17 mW | 465.27 mW | 0.520 | 0.850 mJ |
| 6 MHz | 92.831 ms | 8.90 mW | 432.20 mW | 0.134 | 0.826 mJ |

These are Quartus Power Analyzer estimates with Low confidence, not measured
whole-board power. The VCD is generated at `eeg_top`, so UART switching activity
is not fully covered. The report's abstract and Section 5.5 prose quote
164.16 mW for core dynamic power; this table uses the 162.26 mW value in Table 9,
which also matches the saved 100 MHz Power Analyzer report.
Recalculating the 100 MHz row from 5,389,008 operations, 556,984 cycles, and
the listed power gives approximately 1.594 GOPS/W and 0.904 mJ of dynamic
energy per inference; the report lists 1.589 GOPS/W and 0.903 mJ.

## Clock and UART

The DE1-SoC board provides a physical 50 MHz `CLOCK_50` input. The board top sets
`USE_PLL=1`, and `pll_50_to_100` generates the 100 MHz clock used by the UART
and complete inference path. Reset remains asserted until the PLL reports that
the generated clock is locked. The SDC constrains the external oscillator to
20 ns and derives the PLL-generated clock and clock uncertainty.

With the 100 MHz core clock, `UART_CLKS_PER_BIT=109` produces approximately
917.4 kbaud, which is compatible with the standard 921600-baud host setting.

At a 100 MHz core clock, the UART format is 921600 baud, 8 data bits, no
parity, and 1 stop bit. Multibyte fields are little-endian. The complete packet
definition is in
[`docs/uart_protocol.md`](docs/uart_protocol.md).

Each request is 6,730 bytes, including the 6,720-byte EEG payload, and each
response is 12 bytes. The report estimates about 78.7 ms per sample end to end,
or 12.7 samples/s, at 100 MHz. This includes serial transfer and core inference;
actual host-side elapsed time also includes software and USB/driver overhead.
The dataset's 62.5 ms window stride produces 16 windows/s, so processing every
continuous window requires a faster host-to-FPGA interface.

## Hardware and Tools

| Item | Setting |
|---|---|
| FPGA board | Terasic DE1-SoC |
| FPGA device | Cyclone V `5CSEMA5F31C6` |
| Quartus project | Quartus Prime Lite 25.1 |
| RTL | SystemVerilog |
| Simulation | Questa Altera FPGA / ModelSim |
| Host tools | MATLAB and Python 3 |
| UART | 921600 baud at 100 MHz core, 8N1 |

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
for the FPGA response before sending the next sample because the current UART
loader and accelerator process one complete sample at a time.

## License

This repository currently has no license file.
