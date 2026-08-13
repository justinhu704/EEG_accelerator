# EEG FPGA UART protocol

UART settings: 115200 baud, 8 data bits, no parity, 1 stop bit (8N1).
Multibyte integers are little-endian. The input payload contains signed Q12
16-bit values in MATLAB column-major order.

## Request: PC to FPGA

| Offset | Size | Field |
|---:|---:|---|
| 0 | 2 | Magic `A5 5A` |
| 2 | 4 | Unsigned sample ID |
| 6 | 2 | Word count; must be 3360 |
| 8 | 6720 | 3360 signed 16-bit input words |
| 6728 | 2 | CRC-16/CCITT-FALSE |

The request CRC covers sample ID, word count, and payload. It excludes magic.

## Response: FPGA to PC

| Offset | Size | Field |
|---:|---:|---|
| 0 | 2 | Magic `5A A5` |
| 2 | 4 | Unsigned sample ID |
| 6 | 1 | Status: 0=success, 1=bad request |
| 7 | 1 | Predicted class 0 through 104 |
| 8 | 2 | Signed 16-bit winning logit |
| 10 | 2 | CRC-16/CCITT-FALSE |

The response CRC covers sample ID through winning logit. It excludes magic.

CRC parameters: polynomial `0x1021`, initial value `0xFFFF`, no reflection,
no final XOR.

The PC must wait for the response before sending the next sample because the
current design deliberately reuses RAM A and RAM B during inference.
