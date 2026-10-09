"""Does a TinyPolicy-shaped network fit the KR260 layer engine? Prints memory, MACs and latency.

    python hls/engine/capacity.py                                   # v3r2 and TinyPolicy-L
    python hls/engine/capacity.py --conv 32 64 128 192 256 --fc 384 [--image-channels 6]

Shape the engine runs (plans/fpga-capacity.md): 5 convs (3x3, stride 2, pad 1) on a 96x96 image,
flatten (HWC) + 10 aux bytes, 2 hidden FCs of width F, FC to 48. Every output width is a multiple of
16 (one 128-bit weight word = 16 output channels). Model calibrated on the built engine: v3r2
estimates 8.1 ms vs 8.03 ms measured on the board (Python call, 16 lanes, 100 MHz).
"""

import argparse
import math

URAM_FOR_WEIGHTS = 62  # 64 on the K26; 2 go to other buffers in the built design
URAM_WORDS = 4096  # one URAM288 is 4096 x 72 bit; a 128-bit word needs 2 side by side
BRAM18 = 288  # BRAM18K tiles on the K26 (144 BRAM36)
OVERHEAD = 30  # cycles per (output pixel, 16-channel group): pipeline fill + requant + write
PY_MS = 0.5  # PYNQ register handshake + cache flush per call
AUX, OUT = 10, 48


def layers(conv, fc, ci0):
    out, ci, hw = [], ci0, 96
    for co in conv:
        out.append(("conv", ci, co, hw // 2, 9 * ci))
        ci, hw = co, hw // 2
    flat = hw * hw * ci + AUX
    for i, co in enumerate([fc, fc, OUT]):
        cin = flat if i == 0 else fc
        out.append(("fc", cin, co, 1, cin))
    return out, flat


def report(name, conv, fc, ci0=3, lanes=16, mhz=100.0):
    ls, flat = layers(conv, fc, ci0)
    errs = [f"{t} {ci}->{co}: width not a multiple of 16" for t, ci, co, _, _ in ls if co % 16]
    if len(ls) > 8:
        errs.append("more than 8 layers: header holds 8 shifts")
    words = sum(co // 16 * k for _, _, co, _, k in ls)
    params = words * 16 + sum(co for _, _, co, _, _ in ls)
    uram = 2 * math.ceil(words / URAM_WORDS)
    act = max([96 * 96 * ci0, flat] + [hw * hw * co for t, _, co, hw, _ in ls if t == "conv"])
    in_bytes = 96 * 96 * ci0 + AUX
    bram18 = 2 * math.ceil(act / 2048) + math.ceil(in_bytes / 16 / 512) * 4 + 19  # v3r2: 71 as built
    macs = sum(hw * hw * co * k for _, _, co, hw, k in ls)
    worst_acc = max(k for *_, k in ls) * 128 * 128
    if uram > URAM_FOR_WEIGHTS:
        errs.append(f"weights need {uram} URAM, {URAM_FOR_WEIGHTS} available")
    if bram18 > BRAM18 * 0.8:
        errs.append(f"activation buffers need ~{bram18} BRAM18 (> 80% of {BRAM18})")
    if worst_acc >= 2**31:
        errs.append("int32 accumulator can overflow")
    cycles = in_bytes + in_bytes / 16
    for _, _, co, hw, k in ls:
        cycles += hw * hw * math.ceil(co / lanes) * (k + OVERHEAD)
    ms = cycles / (mhz * 1e3) + PY_MS
    print(f"== {name}: conv {conv}, fc {fc}, image channels {ci0}")
    for t, ci, co, hw, k in ls:
        w, m = co // 16 * k, hw * hw * co * k
        print(f"   {t:4} {ci:5} -> {co:4}  out {hw}x{hw}  words {w:7,}  MACs {m:11,}")
    print(f"   parameters {params:,} | weights.bin {words * 16 + 4 * (params - words * 16):,} B | "
          f"weight words {words:,} -> URAM {uram + 2}/64 | "
          f"act buffer {act:,} B -> ~{bram18} BRAM18 of {BRAM18}")
    print(f"   MACs {macs:,} | est. FPGA call {ms:.1f} ms at {lanes} lanes, {mhz:.0f} MHz | "
          f"packet {in_bytes:,} B | {'FITS' if not errs else 'DOES NOT FIT: ' + '; '.join(errs)}")
    return not errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, nargs=5, help="output channels of conv1..conv5")
    ap.add_argument("--fc", type=int, help="hidden FC width F (fc6, fc7)")
    ap.add_argument("--image-channels", type=int, default=3, help="3 = front RGB, 6 = front + wrist")
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--mhz", type=float, default=100.0)
    a = ap.parse_args()
    if a.conv:
        raise SystemExit(0 if report("custom", a.conv, a.fc, a.image_channels, a.lanes, a.mhz) else 1)
    report("v3r2 (built, 8.03 ms measured)", [16, 32, 64, 96, 128], 256)
    report("TinyPolicy-L", [32, 64, 128, 192, 256], 384)
    report("TinyPolicy-L, front + wrist", [32, 64, 128, 192, 256], 384, ci0=6)
    report("TinyPolicy-L on engine v2", [32, 64, 128, 192, 256], 384, lanes=64, mhz=150)


if __name__ == "__main__":
    main()
