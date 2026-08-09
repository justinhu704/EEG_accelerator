"""Generate the 512-entry Q15 activation tables used by the GRU RTL."""

from __future__ import annotations

import math
from pathlib import Path


LUT_DEPTH = 256
SIGMOID_RANGE = 6.236328125
TANH_RANGE = 3.46484375
Q15_SCALE = 1 << 15


def quantize_q15(value: float) -> int:
    """Round to signed Q15 and saturate to the 16-bit output range."""
    scaled = math.floor(value * Q15_SCALE + 0.5)
    return max(-32768, min(32767, scaled))


def write_mem(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{value & 0xFFFF:04X}\n" for value in values),
        encoding="ascii",
    )


def make_half_table(limit: float, function) -> list[int]:
    return [
        quantize_q15(function(limit * i / (LUT_DEPTH - 1)))
        for i in range(LUT_DEPTH)
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "mem" / "lut"

    sigmoid = make_half_table(
        SIGMOID_RANGE, lambda x: 1.0 / (1.0 + math.exp(-x))
    )
    tanh = make_half_table(TANH_RANGE, math.tanh)

    assert all(a <= b for a, b in zip(sigmoid, sigmoid[1:]))
    assert all(a <= b for a, b in zip(tanh, tanh[1:]))
    assert sigmoid[0] == Q15_SCALE // 2
    assert tanh[0] == 0

    write_mem(output_dir / "sigmoid_half_lut_q15.mem", sigmoid)
    write_mem(output_dir / "tanh_half_lut_q15.mem", tanh)

    print(f"Generated {len(sigmoid)} positive-half sigmoid entries")
    print(f"Generated {len(tanh)} positive-half tanh entries")
    print(f"Sigmoid stored endpoints: {sigmoid[0]}, {sigmoid[-1]}")
    print(f"Tanh stored endpoints: {tanh[0]}, {tanh[-1]}")


if __name__ == "__main__":
    main()
