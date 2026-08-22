#!/usr/bin/env python3
"""Measure FPGA EEG inference latency over the existing UART protocol."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from send_eeg_uart import (
    BYTES_PER_SAMPLE,
    RESPONSE_SIZE_AFTER_MAGIC,
    build_request,
    load_labels,
    read_response,
    validate_dataset,
)

DEFAULT_COUNT = 10
UART_BITS_PER_BYTE_8N1 = 10
RESPONSE_BYTES = 2 + RESPONSE_SIZE_AFTER_MAGIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM10", help="UART COM port")
    # fpga_uart_top.sv currently uses 54 clocks/bit at 50 MHz
    # (925925.9 baud); 921600 is the closest common PC UART rate.
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="first dataset sample_id to send",
    )
    parser.add_argument(
        "--inputs", type=Path, default=Path("host/data/test_inputs_q12.bin")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("host/data/test_labels.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("host/results/fpga_uart_timing_10.csv"),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--settle",
        type=float,
        default=0.25,
        help="seconds to wait after opening the COM port (not measured)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and show settings without opening the COM port",
    )
    return parser.parse_args()


def print_statistics(name: str, values_ms: list[float]) -> None:
    print(f"{name} average : {statistics.mean(values_ms):.3f} ms")
    print(f"{name} median  : {statistics.median(values_ms):.3f} ms")
    print(f"{name} minimum : {min(values_ms):.3f} ms")
    print(f"{name} maximum : {max(values_ms):.3f} ms")
    print(f"{name} std dev : {statistics.pstdev(values_ms):.3f} ms")


def main() -> int:
    args = parse_args()
    if args.baud <= 0:
        raise ValueError("--baud must be greater than zero")
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.settle < 0:
        raise ValueError("--settle cannot be negative")

    labels = load_labels(args.labels)
    validate_dataset(args.inputs, labels)
    stop_index = args.start_index + args.count
    if stop_index > len(labels):
        raise ValueError(
            f"requested samples [{args.start_index}, {stop_index - 1}], "
            f"but dataset only contains {len(labels)} samples"
        )

    with args.inputs.open("rb") as input_file:
        input_file.seek(args.start_index * BYTES_PER_SAMPLE)
        first_payload = input_file.read(BYTES_PER_SAMPLE)
    request_bytes = len(build_request(args.start_index, first_payload))
    request_wire_ms = (
        request_bytes * UART_BITS_PER_BYTE_8N1 / args.baud * 1000.0
    )
    response_wire_ms = (
        RESPONSE_BYTES * UART_BITS_PER_BYTE_8N1 / args.baud * 1000.0
    )

    print(f"Port              : {args.port}")
    print(f"Baud rate         : {args.baud}")
    print(f"Samples           : {args.count} (ID {args.start_index}..{stop_index - 1})")
    print(f"Request size      : {request_bytes} bytes")
    print(f"Response size     : {RESPONSE_BYTES} bytes")
    print(f"Request wire time : {request_wire_ms:.3f} ms (theoretical 8N1)")
    print(f"Response wire time: {response_wire_ms:.3f} ms (theoretical 8N1)")

    if args.dry_run:
        print("Dry run complete; no data was sent to the FPGA.")
        return 0

    try:
        import serial  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "pyserial is not installed; run: pip install -r host/requirements.txt"
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    round_trip_values: list[float] = []
    estimated_inference_values: list[float] = []
    correct_count = 0

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
                "sample_id",
                "true_class",
                "predicted_class",
                "winning_logit",
                "correct",
                "host_transmit_ms",
                "response_wait_ms",
                "round_trip_ms",
                "response_wire_estimate_ms",
                "estimated_inference_ms",
            ],
        )
        writer.writeheader()

        uart.reset_input_buffer()
        uart.reset_output_buffer()
        if args.settle:
            time.sleep(args.settle)
        input_file.seek(args.start_index * BYTES_PER_SAMPLE)

        for sequence, sample_id in enumerate(
            range(args.start_index, stop_index), start=1
        ):
            payload = input_file.read(BYTES_PER_SAMPLE)
            packet = build_request(sample_id, payload)

            # Packet construction and CRC calculation are intentionally excluded.
            start_ns = time.perf_counter_ns()
            written = uart.write(packet)
            if written != len(packet):
                raise OSError(
                    f"UART wrote {written} of {len(packet)} request bytes"
                )
            uart.flush()
            transmit_done_ns = time.perf_counter_ns()

            response_id, status, prediction, winning_logit = read_response(uart)
            response_done_ns = time.perf_counter_ns()

            if response_id != sample_id:
                raise ValueError(
                    f"response sample_id {response_id} does not match {sample_id}"
                )
            if status != 0:
                raise RuntimeError(
                    f"FPGA rejected sample {sample_id}; status={status}"
                )
            if not 0 <= prediction <= 104:
                raise ValueError(f"invalid FPGA prediction {prediction}")

            transmit_ms = (transmit_done_ns - start_ns) / 1_000_000.0
            response_wait_ms = (
                response_done_ns - transmit_done_ns
            ) / 1_000_000.0
            round_trip_ms = (response_done_ns - start_ns) / 1_000_000.0
            # The FPGA starts after receiving a complete valid request. Subtracting
            # the 12-byte response's ideal wire time gives the closest estimate
            # available without a hardware cycle counter in the response packet.
            estimated_inference_ms = max(
                0.0, response_wait_ms - response_wire_ms
            )

            true_class = labels[sample_id]["fpga_label"]
            is_correct = prediction == true_class
            correct_count += int(is_correct)
            round_trip_values.append(round_trip_ms)
            estimated_inference_values.append(estimated_inference_ms)

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "true_class": true_class,
                    "predicted_class": prediction,
                    "winning_logit": winning_logit,
                    "correct": int(is_correct),
                    "host_transmit_ms": f"{transmit_ms:.6f}",
                    "response_wait_ms": f"{response_wait_ms:.6f}",
                    "round_trip_ms": f"{round_trip_ms:.6f}",
                    "response_wire_estimate_ms": f"{response_wire_ms:.6f}",
                    "estimated_inference_ms": f"{estimated_inference_ms:.6f}",
                }
            )
            output_file.flush()

            print(
                f"[{sequence:02d}/{args.count}] sample={sample_id} "
                f"true={true_class} pred={prediction} "
                f"correct={'Y' if is_correct else 'N'} "
                f"round-trip={round_trip_ms:.3f} ms "
                f"estimated-inference={estimated_inference_ms:.3f} ms"
            )

    print("\nTiming summary")
    print("--------------")
    print_statistics("Round-trip", round_trip_values)
    print()
    print_statistics("Estimated inference", estimated_inference_values)
    accuracy = 100.0 * correct_count / args.count
    print(f"\nAccuracy         : {correct_count}/{args.count} ({accuracy:.2f}%)")
    print(f"CSV results      : {args.output}")
    print(
        "Note: estimated inference still includes USB/UART driver latency; "
        "an RTL cycle counter is required for exact FPGA-only latency."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
