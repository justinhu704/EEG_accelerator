#!/usr/bin/env python3
"""Pack MATLAB DS-Conv1/DS-Conv2 exports for the RTL ROM interfaces."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "mem" / "dsconv1_dsconv2"
WEIGHTS = BASE / "weights"
GOLDEN = BASE / "golden"


def read_hex16(path: Path) -> list[int]:
    values = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        token = line.split("//", 1)[0].strip()
        if token:
            try:
                values.append(int(token, 16) & 0xFFFF)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid {token!r}") from error
    return values


def pack(values: list[int]) -> str:
    word = sum((value & 0xFFFF) << (16 * lane)
               for lane, value in enumerate(values))
    return f"{word:0{4 * len(values)}X}"


def write_words(path: Path, words: list[str]) -> None:
    path.write_text("\n".join(words) + "\n")


def pack_depthwise(name: str, channels: int) -> None:
    values = read_hex16(WEIGHTS / f"{name}_depthwise_W.mem")
    if len(values) != 2 * 5 * channels:
        raise ValueError(f"unexpected {name} depthwise count: {len(values)}")
    write_words(
        WEIGHTS / f"{name}_depthwise_W_kh2.mem",
        [pack([values[2 * (kw + 5 * ch)],
               values[2 * (kw + 5 * ch) + 1]])
         for ch in range(channels) for kw in range(5)],
    )


def pack_pointwise(name: str, in_ch: int, out_ch: int, lanes: int) -> None:
    values = read_hex16(WEIGHTS / f"{name}_pointwise_W.mem")
    bias = read_hex16(WEIGHTS / f"{name}_pointwise_b.mem")
    if len(values) != in_ch * out_ch or len(bias) != out_ch:
        raise ValueError(f"unexpected {name} pointwise weight or bias count")
    groups = out_ch // lanes
    write_words(
        WEIGHTS / f"{name}_pointwise_W_x{lanes}.mem",
        [pack([values[ch + in_ch * (group * lanes + lane)]
               for lane in range(lanes)])
         for group in range(groups) for ch in range(in_ch)],
    )
    write_words(
        WEIGHTS / f"{name}_pointwise_b_x{lanes}.mem",
        [pack(bias[group * lanes:(group + 1) * lanes])
         for group in range(groups)],
    )


def main() -> None:
    pack_depthwise("conv1", 1)
    pack_pointwise("conv1", 1, 21, 3)
    pack_depthwise("conv2", 21)
    pack_pointwise("conv2", 21, 20, 5)

    golden_input = read_hex16(GOLDEN / "q_in_act.mem")
    if len(golden_input) < 21 * 160:
        raise ValueError("golden input does not contain one complete sample")
    board_dir = BASE / "board"
    board_dir.mkdir(parents=True, exist_ok=True)
    write_words(board_dir / "sample0_q12.mem",
                [f"{value:04X}" for value in golden_input[:21 * 160]])
    write_words(board_dir / "sample0_q12_even.mem",
                [f"{value:04X}" for value in golden_input[:21 * 160:2]])
    write_words(board_dir / "sample0_q12_odd.mem",
                [f"{value:04X}" for value in golden_input[1:21 * 160:2]])
    ram_values = golden_input[:21 * 160] + [0] * (19 * 18 * 20 - 21 * 160)
    write_words(board_dir / "ram_a_sample0_q12.mem",
                [f"{value:04X}" for value in ram_values])

    sample_sizes = {
        "q_relu1_act.mem": 20 * 156 * 21,
        "q_relu2_act.mem": 19 * 152 * 20,
        "q_pool1_act.mem": 19 * 18 * 20,
        "q_pool2_act.mem": 18 * 1 * 15,
        "q_fc_out_act.mem": 105,
    }
    for name, size in sample_sizes.items():
        values = read_hex16(GOLDEN / name)
        if len(values) < size:
            raise ValueError(f"{name} does not contain one complete sample")
        write_words(GOLDEN / f"{Path(name).stem}_sample0.mem",
                    [f"{value:04X}" for value in values[:size]])

    print("Created DS-Conv1/2 packed weights and sample-0 files.")


if __name__ == "__main__":
    main()
