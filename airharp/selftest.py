"""Check the three things that can go wrong on a new machine.

    python -m airharp.selftest

Camera, hand model, audio device -- each reported separately, so a failure
says which one rather than just leaving a black window. `--shot out.png`
saves a rendered frame, which is also how the screenshots in the README are
made.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .app import MODEL
from .audio import INSTRUMENTS, SR, Engine
from .music import DEGREES, Key
from .poses import PoseReader
from .tracking import Tracker
from .ui import HUD

OK, BAD = "  ok  ", " FAIL "


def _line(tag, name, detail=""):
    print(f"[{tag}] {name:<22s} {detail}")


def check_model(path):
    from pathlib import Path
    p = Path(path)
    if p.exists():
        _line(OK, "hand model", f"{p.stat().st_size / 1e6:.1f} MB  {p}")
        return True
    _line(BAD, "hand model", f"missing: {p}")
    return False


def check_audio(seconds=2.2, device=None):
    eng = Engine(device=device)
    eng.start()
    if eng.silent:
        _line(BAD, "audio", eng.status)
        eng.stop()
        return False
    key = Key()
    step = seconds / (DEGREES + 1)
    start = time.perf_counter()
    played = 0
    while time.perf_counter() - start < seconds:
        want = int((time.perf_counter() - start) / step)
        while played <= want and played < DEGREES:
            freqs, _ = key.chord(played, 3)
            eng.set_chord("test", freqs, 0.7, 0.5 + 0.4 * played / DEGREES, 0.7)
            played += 1
        time.sleep(0.01)
    eng.release("test")
    time.sleep(0.5)
    detail = (f"{eng.status}, {SR} Hz, played {played} chords, "
              f"{eng.underruns} underruns")
    eng.stop()
    good = eng.underruns == 0
    _line(OK if good else BAD, "audio", detail)
    return good


def check_camera(args):
    tracker = Tracker(args.model, camera=args.camera, width=args.width,
                      height=args.height, detect_width=args.detect_width or 0)
    hud = None
    seen, frames, with_hands, last = 0, 0, 0, None
    deadline = time.perf_counter() + args.seconds

    with tracker:
        while time.perf_counter() < deadline:
            seen, frame, hands, stamp = tracker.read(seen, timeout=0.5)
            if frame is None:
                if tracker.error:
                    _line(BAD, "camera", tracker.error)
                    return False, None
                continue
            frames += 1
            with_hands += bool(hands)
            if hud is None:
                key = Key()
                hud = HUD(key, INSTRUMENTS)
                reader = PoseReader(key)
            h, w = frame.shape[:2]
            states, chord, changed, _ = reader.update(hands, w, h, stamp)
            if changed and chord is not None:
                hud.strike(chord.degree, chord.level)
            hud.step(1 / 30.0, states, chord)
            last = hud.draw(frame, hands, states, chord, INSTRUMENTS[0],
                            fps=tracker.detect_fps)

    if not frames:
        _line(BAD, "camera", "no frames arrived")
        return False, None

    _line(OK, "camera", f"{tracker.size[0]}x{tracker.size[1]} @ "
                        f"{tracker.capture_fps:.1f} fps")
    _line(OK, "hand detection", f"{tracker.detect_fps:.1f} fps, "
                                f"{tracker.detect_ms:.1f} ms/frame, "
                                f"hands seen in {with_hands}/{frames} frames")
    if tracker.capture_fps < 20 and tracker.detect_ms < 40:
        # Detection is keeping up, so the camera itself is the slow part:
        # in low light a webcam lengthens its exposure and delivers fewer
        # frames a second, and every one of those is input lag.
        print("       the camera, not the tracking, is the slow part here -- "
              "webcams drop\n       to 10-15 fps in dim light. More light "
              "will make this noticeably tighter.")
    if with_hands == 0:
        print("       (no hands in view -- hold one up to test tracking)")
    return True, last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--detect-width", type=int, default=640)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--device", default=None)
    ap.add_argument("--shot", default=None, help="save a rendered frame here")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args(argv)

    print("air harp self test\n")
    good = check_model(args.model)
    if good:
        cam_ok, frame = check_camera(args)
        good = good and cam_ok
        if args.shot and frame is not None:
            import cv2
            cv2.imwrite(args.shot, frame)
            _line(OK, "screenshot", args.shot)
    if not args.no_audio:
        good = check_audio(device=args.device) and good

    print("\n" + ("all good -- run  python run.py" if good
                  else "something above needs fixing"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
