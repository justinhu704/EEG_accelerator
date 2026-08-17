#!/usr/bin/env python3
"""Pack scalar convolution weights into output-channel-parallel ROM words."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_hex16(path: Path) -> list[int]:
    values: list[int] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        token = line.split("//", 1)[0].strip()
        if not token:
            continue
        try:
            values.append(int(token, 16) & 0xFFFF)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid hex value {token!r}") from error
    return values


def pack_lanes(values: list[int], lanes: int) -> str:
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFFFF) << (16 * lane)
    return f"{packed:0{lanes * 4}X}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("mem/weights/conv2_W.mem"))
    parser.add_argument("--bias", type=Path, default=Path("mem/weights/conv2_b.mem"))
    parser.add_argument("--output-weights", type=Path,
                        default=Path("mem/weights/conv2_W_x4.mem"))
    parser.add_argument("--output-bias", type=Path,
                        default=Path("mem/weights/conv2_b_x4.mem"))
    parser.add_argument("--kh", type=int, default=2)
    parser.add_argument("--kw", type=int, default=5)
    parser.add_argument("--in-ch", type=int, default=21)
    parser.add_argument("--out-ch", type=int, default=20)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()

    if args.lanes <= 0:
        raise ValueError("--lanes must be positive")

    weights = read_hex16(args.weights)
    biases = read_hex16(args.bias)
    kernel_size = args.kh * args.kw * args.in_ch
    expected_weights = kernel_size * args.out_ch
    if len(weights) != expected_weights:
        raise ValueError(
            f"{args.weights} contains {len(weights)} values; expected {expected_weights}"
        )
    if len(biases) != args.out_ch:
        raise ValueError(
            f"{args.bias} contains {len(biases)} values; expected {args.out_ch}"
        )

    groups = (args.out_ch + args.lanes - 1) // args.lanes
    packed_weights: list[str] = []
    packed_biases: list[str] = []

    for group in range(groups):
        base_channel = group * args.lanes
        for kernel_index in range(kernel_size):
            lane_values = []
            for lane in range(args.lanes):
                channel = base_channel + lane
                value = weights[kernel_index + kernel_size * channel] if channel < args.out_ch else 0
                lane_values.append(value)
            packed_weights.append(pack_lanes(lane_values, args.lanes))

        packed_biases.append(pack_lanes([
            biases[base_channel + lane] if base_channel + lane < args.out_ch else 0
            for lane in range(args.lanes)
        ], args.lanes))

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    args.output_bias.parent.mkdir(parents=True, exist_ok=True)
    args.output_weights.write_text("\n".join(packed_weights) + "\n")
    args.output_bias.write_text("\n".join(packed_biases) + "\n")

    print(
        f"Packed {len(weights)} weights into {len(packed_weights)} "
        f"x {args.lanes * 16}-bit words"
    )
    print(
        f"Packed {len(biases)} biases into {len(packed_biases)} "
        f"x {args.lanes * 16}-bit words"
    )


if __name__ == "__main__":
    main()
