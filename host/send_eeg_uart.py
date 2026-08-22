#!/usr/bin/env python3
"""Send quantized EEG test samples to the FPGA and calculate accuracy."""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO

SAMPLE_WORDS = 3360
BYTES_PER_SAMPLE = SAMPLE_WORDS * 2
REQUEST_MAGIC = b"\xA5\x5A"
RESPONSE_MAGIC = b"\x5A\xA5"
RESPONSE_SIZE_AFTER_MAGIC = 10


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: polynomial 0x1021, init 0xFFFF."""
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_request(sample_id: int, payload: bytes) -> bytes:
    if len(payload) != BYTES_PER_SAMPLE:
        raise ValueError(
            f"sample {sample_id} has {len(payload)} bytes; "
            f"expected {BYTES_PER_SAMPLE}"
        )
    body = struct.pack("<IH", sample_id, SAMPLE_WORDS) + payload
    return REQUEST_MAGIC + body + struct.pack("<H", crc16_ccitt(body))


def load_labels(path: Path) -> list[dict[str, int]]:
    labels: list[dict[str, int]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "fpga_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            sample_id = int(row["sample_id"])
            fpga_label = int(row["fpga_label"])
            if sample_id != len(labels):
                raise ValueError(
                    f"sample_id must be contiguous from 0; got {sample_id} "
                    f"at row {len(labels) + 2}"
                )
            if not 0 <= fpga_label <= 104:
                raise ValueError(f"invalid fpga_label {fpga_label}")
            labels.append(
                {
                    "sample_id": sample_id,
                    "fpga_label": fpga_label,
                    "subject_label": row.get(
                        "subject_label", row.get("matlab_label", str(fpga_label))
                    ),
                }
            )
    if not labels:
        raise ValueError(f"{path} contains no labels")
    return labels


def read_exact(stream: BinaryIO, count: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(count - len(result))
        if not chunk:
            raise TimeoutError(
                f"UART response stopped after {len(result)} of {count} bytes"
            )
        result.extend(chunk)
    return bytes(result)


def read_response(serial_port: BinaryIO) -> tuple[int, int, int, int]:
    matched = 0
    while matched < len(RESPONSE_MAGIC):
        byte = read_exact(serial_port, 1)[0]
        if byte == RESPONSE_MAGIC[matched]:
            matched += 1
        else:
            matched = 1 if byte == RESPONSE_MAGIC[0] else 0

    remainder = read_exact(serial_port, RESPONSE_SIZE_AFTER_MAGIC)
    body = remainder[:8]
    received_crc = struct.unpack("<H", remainder[8:10])[0]
    calculated_crc = crc16_ccitt(body)
    if received_crc != calculated_crc:
        raise ValueError(
            f"response CRC mismatch: received 0x{received_crc:04X}, "
            f"calculated 0x{calculated_crc:04X}"
        )
    sample_id, status, predicted_class, winning_logit = struct.unpack(
        "<IBBh", body
    )
    return sample_id, status, predicted_class, winning_logit


def validate_dataset(inputs_path: Path, labels: list[dict[str, int]]) -> None:
    actual_bytes = inputs_path.stat().st_size
    expected_bytes = len(labels) * BYTES_PER_SAMPLE
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{inputs_path} has {actual_bytes} bytes; expected {expected_bytes} "
            f"for {len(labels)} samples"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--inputs", type=Path, default=Path("host/data/test_inputs_q12.bin")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("host/data/test_labels.csv")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("host/results/fpga_predictions.csv")
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--limit", type=int, help="send only the first N samples")
    parser.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="print progress every N samples (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate files and packets without opening a COM port"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.report_every <= 0:
        raise ValueError("--report-every must be greater than zero")
    labels = load_labels(args.labels)
    validate_dataset(args.inputs, labels)
    sample_count = len(labels) if args.limit is None else min(args.limit, len(labels))

    if args.dry_run:
        with args.inputs.open("rb") as input_file:
            first_payload = input_file.read(BYTES_PER_SAMPLE)
        first_packet = build_request(0, first_payload)
        print(f"Dataset valid: {len(labels)} samples")
        print(f"Each payload: {BYTES_PER_SAMPLE} bytes")
        print(f"Each UART request: {len(first_packet)} bytes")
        print(f"First request CRC: 0x{struct.unpack('<H', first_packet[-2:])[0]:04X}")
        return 0

    if not args.port:
        raise ValueError("--port is required unless --dry-run is used")

    try:
        import serial  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "pyserial is not installed; run: pip install -r host/requirements.txt"
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct_count = 0
    start_time = time.monotonic()

    with serial.Serial(
        args.port,
        args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
        write_timeout=args.timeout,
    ) as uart, args.inputs.open("rb") as input_file, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "sample_id", "subject_label", "true_class",
                "predicted_class", "winning_logit", "correct"
            ],
        )
        writer.writeheader()
        uart.reset_input_buffer()
        uart.reset_output_buffer()

        for index in range(sample_count):
            label = labels[index]
            payload = input_file.read(BYTES_PER_SAMPLE)
            packet = build_request(index, payload)
            uart.write(packet)
            uart.flush()

            response_id, status, prediction, winning_logit = read_response(uart)
            if response_id != index:
                raise ValueError(
                    f"response sample_id {response_id} does not match {index}"
                )
            if status != 0:
                raise RuntimeError(
                    f"FPGA rejected sample {index}; status={status}"
                )
            if not 0 <= prediction <= 104:
                raise ValueError(f"invalid FPGA prediction {prediction}")

            is_correct = prediction == label["fpga_label"]
            correct_count += int(is_correct)
            writer.writerow(
                {
                    "sample_id": index,
                    "subject_label": label["subject_label"],
                    "true_class": label["fpga_label"],
                    "predicted_class": prediction,
                    "winning_logit": winning_logit,
                    "correct": int(is_correct),
                }
            )
            output_file.flush()

            completed = index + 1
            if completed % args.report_every == 0 or completed == sample_count:
                running_accuracy = 100.0 * correct_count / completed
                print(
                    f"[{completed}/{sample_count}] true={label['fpga_label']} "
                    f"pred={prediction} logit={winning_logit} "
                    f"accuracy={running_accuracy:.2f}%"
                )

    elapsed = time.monotonic() - start_time
    accuracy = 100.0 * correct_count / sample_count
    print(f"Finished: {correct_count}/{sample_count} correct ({accuracy:.2f}%)")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
