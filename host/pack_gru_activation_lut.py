#!/usr/bin/env python3
"""Pack the existing 16-bit sigmoid/tanh tables into one 32-bit GRU ROM."""

from argparse import ArgumentParser
from pathlib import Path


def read_mem(path: Path) -> list[int]:
    values: list[int] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if line:
            values.append(int(line, 16) & 0xFFFF)
    if len(values) != 256:
        raise ValueError(f"{path} contains {len(values)} entries; expected 256")
    return values


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--sigmoid", type=Path,
                        default=Path("mem/lut/sigmoid_half_lut_q15.mem"))
    parser.add_argument("--tanh", type=Path,
                        default=Path("mem/lut/tanh_half_lut_q15.mem"))
    parser.add_argument("--output", type=Path,
                        default=Path("mem/lut/gru_activation_lut_q15.mem"))
    args = parser.parse_args()

    sigmoid = read_mem(args.sigmoid)
    tanh = read_mem(args.tanh)
    packed = [(tanh_value << 16) | sigmoid_value
              for sigmoid_value, tanh_value in zip(sigmoid, tanh)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{value:08X}\n" for value in packed), encoding="ascii"
    )
    print(f"Packed {len(packed)} GRU activation LUT words into {args.output}")


if __name__ == "__main__":
    main()
