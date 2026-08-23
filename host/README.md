# EEG UART host tools

## 1. Export the MATLAB test set

The easiest method is to open `run_export_uart_dataset.m` in MATLAB and press
Run. Alternatively, call the function directly:

```matlab
export_uart_dataset( ...
    "C:\...\EEG_105_stride10_mixedSess.mat", ...
    "C:\EEG_Project\host\data");
```

This creates `test_inputs_q12.bin` and `test_labels.csv`.

## 2. Validate files without hardware

```powershell
python host\send_eeg_uart.py --dry-run
```

## 3. Install UART support and run

```powershell
python -m pip install -r host\requirements.txt
python host\send_eeg_uart.py --port COM10 --baud 921600
```

Replace `COM10` with the USB-to-UART COM port shown by Windows Device Manager.
The script sends only one sample at a time and waits for its FPGA response.
