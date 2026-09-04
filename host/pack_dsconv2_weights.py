#!/usr/bin/env python3
"""Pack MATLAB DS-Conv2 weights for the RTL ROM interfaces."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "mem" / "dsconv2" / "weights"


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
    word = 0
    for lane, value in enumerate(values):
        word |= (value & 0xFFFF) << (16 * lane)
    return f"{word:0{4 * len(values)}X}"


def write_words(path: Path, words: list[str]) -> None:
    path.write_text("\n".join(words) + "\n")


def main() -> None:
    # MATLAB depthwise shape is [kh, kw, 1, 1, input_channel].
    dw = read_hex16(WEIGHTS / "conv2_depthwise_W.mem")
    if len(dw) != 2 * 5 * 21:
        raise ValueError(f"expected 210 depthwise weights, got {len(dw)}")
    write_words(
        WEIGHTS / "conv2_depthwise_W_kh2.mem",
        [pack([dw[2 * (kw + 5 * ch)], dw[2 * (kw + 5 * ch) + 1]])
         for ch in range(21) for kw in range(5)],
    )

    # MATLAB pointwise shape is [1, 1, input_channel, output_channel].
    pw = read_hex16(WEIGHTS / "conv2_pointwise_W.mem")
    bias = read_hex16(WEIGHTS / "conv2_pointwise_b.mem")
    if len(pw) != 21 * 20 or len(bias) != 20:
        raise ValueError("unexpected pointwise weight or bias count")

    write_words(
        WEIGHTS / "conv2_pointwise_W_x5.mem",
        [pack([pw[in_ch + 21 * (group * 5 + lane)] for lane in range(5)])
         for group in range(4) for in_ch in range(21)],
    )
    write_words(
        WEIGHTS / "conv2_pointwise_b_x5.mem",
        [pack(bias[group * 5:(group + 1) * 5]) for group in range(4)],
    )

    # fixed_sample_loader needs exactly one 21x160 sample.
    golden_input = read_hex16(ROOT / "mem" / "dsconv2" / "golden"
                              / "q_in_act.mem")
    if len(golden_input) < 21 * 160:
        raise ValueError("DS golden input does not contain one complete sample")
    board_dir = ROOT / "mem" / "dsconv2" / "board"
    board_dir.mkdir(parents=True, exist_ok=True)
    write_words(board_dir / "sample0_q12.mem",
                [f"{value:04X}" for value in golden_input[:21 * 160]])
    # RAM A is 6840 words deep; pad unused addresses to keep $readmemh quiet.
    ram_a_values = golden_input[:21 * 160] + [0] * (19 * 18 * 20 - 21 * 160)
    write_words(board_dir / "ram_a_sample0_q12.mem",
                [f"{value:04X}" for value in ram_a_values])

    sample_sizes = {
        "q_relu1_act.mem": 20 * 156 * 21,
        "q_relu2_act.mem": 19 * 152 * 20,
        "q_pool1_act.mem": 19 * 18 * 20,
        "q_pool2_act.mem": 18 * 1 * 15,
        "q_fc_out_act.mem": 105,
    }
    golden_dir = ROOT / "mem" / "dsconv2" / "golden"
    for name, size in sample_sizes.items():
        values = read_hex16(golden_dir / name)
        if len(values) < size:
            raise ValueError(f"{name} does not contain one complete sample")
        stem = Path(name).stem
        write_words(golden_dir / f"{stem}_sample0.mem",
                    [f"{value:04X}" for value in values[:size]])

    print("Created DS packed weights and one 3360-word board sample.")


if __name__ == "__main__":
    main()
