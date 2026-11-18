# EEG UART host tools

| Folder | Contents |
|---|---|
| `packing/` | Pack DSConv and parallel convolution weights and the GRU LUT into RTL memory files. |
| `uart/` | Send test windows to the FPGA, run the 105-subject demo, and benchmark UART timing. |
| `matlab/` | Export the mixed-session MATLAB cache to the UART dataset format. |
| `data/` | Full and demo input binaries and label CSVs. |
| `results/` | Saved FPGA predictions and UART timing results. |

## 1. Export the MATLAB test set

Open `matlab/run_export_uart_dataset.m` in MATLAB and press Run if the cache is
at the original OneDrive location. For a cache stored elsewhere, use its actual
path. From the repository root in MATLAB:

```matlab
addpath(fullfile(pwd, 'host', 'matlab'));
export_uart_dataset("D:\path\to\EEG_105_stride10_mixedSess.mat");
```

The default destination is `host/data/`. Pass a second argument to export to
a separate directory and preserve the existing `test_inputs_q12.bin` and
`test_labels.csv`.

## 2. Validate files without hardware

From the repository root:

```powershell
python host\uart\send_eeg_uart.py --dry-run
```

## 3. Install UART support and run

```powershell
python -m pip install -r host\requirements.txt
python host\uart\send_eeg_uart.py --port COM10 --baud 921600
```

Replace `COM10` with the USB-to-UART COM port shown by Windows Device Manager.
The script sends one sample at a time and waits for its FPGA response.
