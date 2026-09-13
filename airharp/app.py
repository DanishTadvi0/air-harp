"""The frame loop: read the hands, decide what they are playing, play it, draw it.

The chord comes from the finger combination and nothing else. Hand height sets
how loud and how bright it is. Nothing here blocks on audio, and nothing in the
audio path blocks on this.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .audio import INSTRUMENTS, Engine
from .music import Key
from .poses import PoseReader
from .tracking import Tracker
from .ui import HUD, waiting_frame

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "hand_landmarker.task"
WINDOW = "Air Harp"

HINT_HOLD = 10.0
HINT_FADE = 2.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="airharp",
        description="Play chords in the air in front of your webcam.",
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--detect-width", type=int, default=640,
                   help="width MediaPipe sees; smaller is faster (0 = full frame)")
    p.add_argument("--root", type=int, default=48, help="MIDI note of the key")
    p.add_argument("--scale", choices=("major", "minor"), default="major")
    p.add_argument("--instrument", type=int, default=1, help="1..%d" % len(INSTRUMENTS))
    p.add_argument("--device", default=None, help="audio output device (name or index)")
    p.add_argument("--silent", action="store_true", help="run with no sound")
    p.add_argument("--no-reverb", action="store_true")
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--debug", action="store_true", help="start with the stats overlay on")
    return p.parse_args(argv)


def _device(spec):
    if spec is None:
        return None
    try:
        return int(spec)
    except (TypeError, ValueError):
        return spec


class App:
    def __init__(self, args):
        self.args = args
        self.key = Key(root=args.root, scale=args.scale)
        self.reader = PoseReader(self.key)
        self.hud = HUD(self.key, INSTRUMENTS)
        self.engine = Engine(device=_device(args.device),
                             reverb=not args.no_reverb,
                             silent=args.silent)
        self.engine.set_instrument(args.instrument - 1)
        self.tracker = Tracker(
            args.model, camera=args.camera, width=args.width, height=args.height,
            detect_width=args.detect_width or 0,
        )
        self.debug = args.debug
        self.fps = 0.0
        self.draw_ms = 0.0
        self.chords = 0

    # -- input -------------------------------------------------------------
    def handle_key(self, code):
        """Returns False to quit."""
        if code in (27, ord("q")):
            return False
        if ord("1") <= code <= ord("9"):
            i = code - ord("1")
            if i < len(INSTRUMENTS):
                self.engine.set_instrument(i)
                self.reader.reset()
        elif code == ord("m"):
            self.key.toggle_scale()
            self._retune()
        elif code == ord("["):
            self.key.transpose(-1)
            self._retune()
        elif code == ord("]"):
            self.key.transpose(1)
            self._retune()
        elif code == ord("d"):
            self.debug = not self.debug
        elif code == ord("r"):
            self.engine.release_all()
            self.reader.reset()
        return True

    def _retune(self):
        self.hud.retune()
        self.reader.reset()
        self.engine.release_all()

    def _debug_lines(self, states, hands=(), chord=None):
        t = self.tracker
        lines = [
            f"capture {t.capture_fps:5.1f} fps   detect {t.detect_fps:5.1f} fps"
            f"   loop {self.fps:5.1f} fps",
            f"detect {t.detect_ms:5.1f} ms   draw {self.draw_ms:5.1f} ms"
            f"   frames dropped {t.frames.dropped}",
            f"audio {self.engine.status}   voices {len(self.engine.voices):2d}"
            f"   held {sum(len(g) for g in self.engine.groups.values()):2d}"
            f"   underruns {self.engine.underruns}",
            f"chords {self.chords}   hands {len(states)}   "
            f"playing {self.key.label(*chord.key) if chord else '-'}",
        ]
        for st in states:
            flag = "settling" if st.settling else "stable  "
            shape = "".join(n[0].upper() if up else "." for n, up
                            in zip("timrp", st.up))
            lines.append(
                f"  {st.label:<6s} [{shape}] {st.fingers} ({st.raw_fingers})  "
                f"{flag}  degree {st.degree if st.degree is not None else '-'}"
                f"  level {st.level:.2f}{'  <- leads' if st.leads else ''}"
            )
        return lines

    # -- one frame ---------------------------------------------------------
    def step(self, frame, hands, stamp, dt, elapsed):
        """Everything a frame needs except the window. Kept separate from `run`
        so the whole loop can be driven from a test with no camera."""
        h, w = frame.shape[:2]
        states, chord, changed, released = self.reader.update(hands, w, h, stamp)
        self._play(chord, changed)

        start = time.perf_counter()
        self.hud.step(dt, states, chord)
        hint = 1.0 - max(0.0, elapsed - HINT_HOLD) / HINT_FADE
        canvas = self.hud.draw(
            frame, hands, states, chord, INSTRUMENTS[self.engine.instrument],
            fps=self.fps,
            debug=self._debug_lines(states, hands, chord) if self.debug else None,
            hint=hint if self.chords == 0 else 0.0,
        )
        self.draw_ms = 0.9 * self.draw_ms + 0.1 * (time.perf_counter() - start) * 1e3
        return canvas

    VOICE = "hands"          # one chord, so one voice group

    def _play(self, chord, changed):
        if chord is None:
            if changed:
                self.engine.release(self.VOICE)
            return
        if changed:
            freqs, _ = self.key.chord(chord.degree, chord.voicing)
            damp = 0.25 + 0.6 * chord.level
            self.engine.set_chord(self.VOICE, freqs, chord.level, chord.bright, damp)
            self.hud.strike(chord.degree, chord.level)
            self.chords += 1
        else:
            # Height keeps moving between chord changes; follow it without
            # restarting anything that is already sounding.
            self.engine.set_level(self.VOICE, chord.level, chord.bright)

    # -- main loop ---------------------------------------------------------
    def run(self):
        args = self.args
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, args.width, args.height)
        cv2.imshow(WINDOW, waiting_frame(args.width, args.height, "starting camera..."))
        cv2.waitKey(1)

        seen = 0
        last_stamp = None
        started = time.perf_counter()

        with self.tracker, self.engine:
            while True:
                seen, frame, hands, stamp = self.tracker.read(seen, timeout=0.25)

                if frame is None:
                    if self.tracker.error:
                        cv2.imshow(WINDOW, waiting_frame(args.width, args.height,
                                                         self.tracker.error))
                    if not self._pump():
                        break
                    continue

                dt = 1.0 / 30.0 if last_stamp is None else max(stamp - last_stamp, 1e-3)
                last_stamp = stamp
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                canvas = self.step(frame, hands, stamp, dt,
                                   time.perf_counter() - started)
                cv2.imshow(WINDOW, canvas)
                if not self._pump():
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

        cv2.destroyAllWindows()

    def _pump(self):
        code = cv2.waitKey(1) & 0xFF
        return True if code == 255 else self.handle_key(code)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.model).exists():
        raise SystemExit(
            f"hand model not found at {args.model}\n"
            "Put MediaPipe's hand_landmarker.task in assets/ or pass --model."
        )
    App(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
