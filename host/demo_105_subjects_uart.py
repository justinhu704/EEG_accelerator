#!/usr/bin/env python3
"""Run a 105-subject FPGA UART demonstration using one test sample per subject."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from send_eeg_uart import (
    BYTES_PER_SAMPLE,
    build_request,
    read_response,
)

SUBJECT_COUNT = 105

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_INPUTS = SCRIPT_DIR / "data" / "test_inputs_q12.bin"
DEFAULT_SOURCE_LABELS = SCRIPT_DIR / "data" / "test_labels.csv"
DEFAULT_DEMO_INPUTS = SCRIPT_DIR / "data" / "demo_105_first_inputs_q12.bin"
DEFAULT_DEMO_LABELS = SCRIPT_DIR / "data" / "demo_105_first_labels.csv"
DEFAULT_RESULTS = SCRIPT_DIR / "results" / "fpga_demo_105_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM10", help="UART COM port")
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="UART baud rate (default: 921600 for 50 MHz / 54 clocks per bit)",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--settle",
        type=float,
        default=0.25,
        help="seconds to wait after opening the COM port (not measured)",
    )
    parser.add_argument("--source-inputs", type=Path, default=DEFAULT_SOURCE_INPUTS)
    parser.add_argument("--source-labels", type=Path, default=DEFAULT_SOURCE_LABELS)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_DEMO_INPUTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_DEMO_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="recreate the 105-subject dataset from the complete test set",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create/validate the 105-subject dataset without opening UART",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show settings and validate packets without opening UART",
    )
    return parser.parse_args()


def read_first_subject_rows(labels_path: Path) -> list[dict[str, int | str]]:
    """Select the first test-set occurrence of every subject."""
    selected: dict[str, dict[str, int | str]] = {}

    with labels_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "subject_label", "fpga_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_path} must contain {sorted(required)}")

        for row in reader:
            subject = row["subject_label"]
            if subject not in selected:
                selected[subject] = {
                    "source_sample_id": int(row["sample_id"]),
                    "subject_label": subject,
                    "fpga_label": int(row["fpga_label"]),
                }

    rows = sorted(selected.values(), key=lambda row: int(row["fpga_label"]))
    if len(rows) != SUBJECT_COUNT:
        raise ValueError(
            f"expected {SUBJECT_COUNT} subjects in {labels_path}, found {len(rows)}"
        )

    class_ids = [int(row["fpga_label"]) for row in rows]
    if class_ids != list(range(SUBJECT_COUNT)):
        raise ValueError("fpga_label values must contain every class from 0 to 104")

    return rows


def create_demo_dataset(
    source_inputs: Path,
    source_labels: Path,
    demo_inputs: Path,
    demo_labels: Path,
) -> None:
    rows = read_first_subject_rows(source_labels)
    source_size = source_inputs.stat().st_size
    if source_size % BYTES_PER_SAMPLE != 0:
        raise ValueError(
            f"{source_inputs} size {source_size} is not a multiple of "
            f"{BYTES_PER_SAMPLE} bytes"
        )
    source_sample_count = source_size // BYTES_PER_SAMPLE

    demo_inputs.parent.mkdir(parents=True, exist_ok=True)
    demo_labels.parent.mkdir(parents=True, exist_ok=True)
    inputs_tmp = demo_inputs.with_suffix(demo_inputs.suffix + ".tmp")
    labels_tmp = demo_labels.with_suffix(demo_labels.suffix + ".tmp")

    try:
        with source_inputs.open("rb") as source_file, inputs_tmp.open("wb") as output_file:
            for demo_id, row in enumerate(rows):
                source_id = int(row["source_sample_id"])
                if not 0 <= source_id < source_sample_count:
                    raise ValueError(f"invalid source sample_id {source_id}")
                source_file.seek(source_id * BYTES_PER_SAMPLE)
                payload = source_file.read(BYTES_PER_SAMPLE)
                if len(payload) != BYTES_PER_SAMPLE:
                    raise ValueError(f"could not read source sample {source_id}")
                output_file.write(payload)

        with labels_tmp.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "sample_id",
                    "source_sample_id",
                    "subject_label",
                    "fpga_label",
                ],
            )
            writer.writeheader()
            for demo_id, row in enumerate(rows):
                writer.writerow({"sample_id": demo_id, **row})

        inputs_tmp.replace(demo_inputs)
        labels_tmp.replace(demo_labels)
    finally:
        inputs_tmp.unlink(missing_ok=True)
        labels_tmp.unlink(missing_ok=True)


def load_demo_labels(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "source_sample_id",
            "subject_label",
            "fpga_label",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(required)}")

        for row in reader:
            sample_id = int(row["sample_id"])
            if sample_id != len(rows):
                raise ValueError(
                    f"demo sample_id must be contiguous from 0; got {sample_id}"
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": int(row["source_sample_id"]),
                    "subject_label": row["subject_label"],
                    "fpga_label": int(row["fpga_label"]),
                }
            )

    if len(rows) != SUBJECT_COUNT:
        raise ValueError(f"{path} contains {len(rows)} rows; expected 105")
    return rows


def validate_demo_dataset(inputs_path: Path, labels_path: Path) -> list[dict[str, int | str]]:
    rows = load_demo_labels(labels_path)
    expected_size = SUBJECT_COUNT * BYTES_PER_SAMPLE
    actual_size = inputs_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{inputs_path} has {actual_size} bytes; expected {expected_size}"
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.baud <= 0:
        raise ValueError("--baud must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.settle < 0:
        raise ValueError("--settle cannot be negative")
    if args.rebuild_dataset or not (args.inputs.exists() and args.labels.exists()):
        print("Preparing the first test sample from each of 105 subjects...")
        create_demo_dataset(
            args.source_inputs,
            args.source_labels,
            args.inputs,
            args.labels,
        )

    labels = validate_demo_dataset(args.inputs, args.labels)
    print(f"Dataset : {args.inputs}")
    print(f"Labels  : {args.labels}")
    print(f"Samples : {len(labels)} subjects x 1 test sample")

    print(f"Port    : {args.port}")
    print(f"Baud    : {args.baud}")

    if args.prepare_only:
        print("Dataset preparation/validation complete; UART was not opened.")
        return 0

    if args.dry_run:
        for sample_id in range(SUBJECT_COUNT):
            with args.inputs.open("rb") as input_file:
                input_file.seek(sample_id * BYTES_PER_SAMPLE)
                payload = input_file.read(BYTES_PER_SAMPLE)
            build_request(sample_id, payload)
        print("Dry run complete; all 105 UART packets are valid.")
        return 0

    try:
        import serial  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "pyserial is not installed; run: pip install -r host/requirements.txt"
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
                "source_sample_id",
                "subject_label",
                "true_class",
                "predicted_class",
                "winning_logit",
                "correct",
            ],
        )
        writer.writeheader()

        uart.reset_input_buffer()
        uart.reset_output_buffer()
        if args.settle:
            time.sleep(args.settle)

        demo_start_ns = time.perf_counter_ns()

        for sequence, label in enumerate(labels, start=1):
            sample_id = int(label["sample_id"])
            payload = input_file.read(BYTES_PER_SAMPLE)
            if len(payload) != BYTES_PER_SAMPLE:
                raise ValueError(f"could not read demo sample {sample_id}")
            packet = build_request(sample_id, payload)

            print(
                f"[{sequence:03d}/{SUBJECT_COUNT}] Sending "
                f"subject={label['subject_label']} "
                f"source_sample={label['source_sample_id']} "
                f"true={label['fpga_label']}...",
                flush=True,
            )

            written = uart.write(packet)
            if written != len(packet):
                raise OSError(f"UART wrote {written} of {len(packet)} bytes")
            response_id, status, prediction, winning_logit = read_response(uart)
            if response_id != sample_id:
                raise ValueError(
                    f"response sample_id {response_id} does not match {sample_id}"
                )
            if status != 0:
                raise RuntimeError(f"FPGA rejected sample {sample_id}; status={status}")
            if not 0 <= prediction < SUBJECT_COUNT:
                raise ValueError(f"invalid FPGA prediction {prediction}")

            true_class = int(label["fpga_label"])
            is_correct = prediction == true_class
            correct_count += int(is_correct)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_sample_id": label["source_sample_id"],
                    "subject_label": label["subject_label"],
                    "true_class": true_class,
                    "predicted_class": prediction,
                    "winning_logit": winning_logit,
                    "correct": int(is_correct),
                }
            )
            output_file.flush()

            running_accuracy = 100.0 * correct_count / sequence
            print(
                f"              pred={prediction:3d} "
                f"correct={'YES' if is_correct else 'NO ':3s} | "
                f"accuracy={running_accuracy:6.2f}%",
                flush=True,
            )

        demo_elapsed_ms = (time.perf_counter_ns() - demo_start_ns) / 1_000_000.0

    accuracy = 100.0 * correct_count / SUBJECT_COUNT
    print("\nDemo summary")
    print("------------")
    print(f"Subjects                     : {SUBJECT_COUNT}")
    print(f"Correct                      : {correct_count}/{SUBJECT_COUNT}")
    print(f"Accuracy                     : {accuracy:.2f}%")
    print(f"Total elapsed                : {demo_elapsed_ms / 1000.0:.3f} seconds")
    print(f"CSV results                  : {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
