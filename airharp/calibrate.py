"""Loudness-match the instruments, and time the render loop.

    python -m airharp.calibrate            # report only
    python -m airharp.calibrate --wav out  # also write out.wav to listen to

Two instruments can have identical peak levels and still be a fistful apart in
perceived volume -- a plucked string is nearly all transient, a bell is nearly
all sustain. This measures one-second RMS instead and prints the trim each
instrument needs, which is where the `gain` values in `INSTRUMENTS` come from.
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np

from .audio import BLOCK, INSTRUMENTS, SR, Engine

TARGET_RMS = 0.085
OCTAVES = (130.8, 261.6, 523.3)


def _render(index, freq, seconds, amp=0.8, bright=0.6, damp=0.6, reverb=False,
            release_at=None):
    eng = Engine(silent=True, reverb=reverb)
    eng.master = 1.0
    eng.set_instrument(index)
    eng.set_chord("cal", [freq], amp, bright, damp)
    blocks = int(seconds * SR / BLOCK)
    cut = None if release_at is None else int(release_at * SR / BLOCK)
    out = []
    for b in range(blocks):
        if cut is not None and b == cut:
            eng.release("cal")
        out.append(eng.render_block())
    return np.concatenate(out)


def _t60(buf):
    env = np.abs(buf[:len(buf) // BLOCK * BLOCK].reshape(-1, BLOCK)).max(axis=1)
    peak = env.max()
    below = env < peak * 1e-3
    return (int(np.argmax(below)) * BLOCK / SR) if below.any() else float("inf")


def measure():
    rows = []
    for i, inst in enumerate(INSTRUMENTS):
        rms, peak, decay = [], [], []
        for f in OCTAVES:
            buf = _render(i, f, 6.0)
            rms.append(float(np.sqrt((buf[:SR] ** 2).mean())))
            peak.append(float(np.abs(buf).max()))
            decay.append(_t60(buf))
        mean_rms = float(np.mean(rms))
        held = float(np.sqrt((_render(i, 261.6, 3.0)[-SR // 2:] ** 2).mean()))
        rows.append({
            "name": inst.name,
            "gain": inst.gain,
            "sustains": inst.sustains,
            "held": held,
            "rms": mean_rms,
            "peak": max(peak),
            "t60": decay,
            "suggest": TARGET_RMS / max(mean_rms, 1e-9) * inst.gain,
        })
    return rows


def benchmark(voices=24, blocks=250):
    out = []
    for i, inst in enumerate(INSTRUMENTS):
        eng = Engine(silent=True)
        eng.set_instrument(i)
        for k in range(voices):
            eng.set_chord("v%d" % k, [160.0 + 33.0 * k], 0.5, 0.6, 0.6)
        eng.render_block()
        start = time.perf_counter()
        for _ in range(blocks):
            eng.render_block()
        per = (time.perf_counter() - start) / blocks
        out.append((inst.name, per * 1e3, per / (BLOCK / SR) * 100.0, len(eng.voices)))
    return out


def demo_wav(path, seconds_each=2.6):
    """A rising pentatonic run per instrument, so the balance can be heard."""
    scale = [261.6, 293.7, 329.6, 392.0, 440.0, 523.3]
    chunks = []
    for i in range(len(INSTRUMENTS)):
        eng = Engine(silent=True)
        eng.set_instrument(i)
        n = int(seconds_each * SR / BLOCK)
        gap = max(1, n // (len(scale) + 2))
        buf = []
        for b in range(n):
            if b % gap == 0 and b // gap < len(scale):
                eng.set_chord("demo", [scale[b // gap]], 0.75, 0.65, 0.6)
            if b == n - int(0.5 * SR / BLOCK):
                eng.release("demo")
            buf.append(eng.render_block())
        chunks.append(np.concatenate(buf))
    audio = np.concatenate(chunks)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path, len(audio) / SR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", help="write a demo wav to this path (no extension needed)")
    ap.add_argument("--voices", type=int, default=24)
    args = ap.parse_args(argv)

    print(f"{'instrument':10s} {'mode':>5s} {'gain':>5s} {'rms(1s)':>8s} "
          f"{'held':>7s} {'peak':>6s} {'t60 low/mid/high':>20s} {'suggested':>10s}")
    for r in measure():
        t60 = "/".join("inf" if np.isinf(x) else f"{x:.1f}" for x in r["t60"])
        mode = "hold" if r["sustains"] else "ring"
        print(f"{r['name']:10s} {mode:>5s} {r['gain']:5.2f} {r['rms']:8.4f} "
              f"{r['held']:7.4f} {r['peak']:6.3f} {t60:>20s} {r['suggest']:10.2f}")

    print(f"\n{'instrument':10s} {'ms/block':>9s} {'% realtime':>11s} {'voices':>7s}")
    for name, ms, pct, live in benchmark(voices=args.voices):
        print(f"{name:10s} {ms:9.3f} {pct:11.1f} {live:7d}")

    if args.wav:
        path = args.wav if args.wav.endswith(".wav") else args.wav + ".wav"
        path, secs = demo_wav(path)
        print(f"\nwrote {path} ({secs:.1f}s)")


if __name__ == "__main__":
    main()
