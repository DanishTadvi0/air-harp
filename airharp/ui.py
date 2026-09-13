"""The on-screen instrument.

The camera view is left clear. Nothing is drawn across it except the hands
being tracked and a ribbon of sound that answers them -- how many strands from
how many fingers are up, how tall from how high the hand is held.

Everything you need to know sits around the edges rather than over the picture:
the chord you are playing, large, at the top; and along the foot, all seven
chords with the hand shape that reaches each one drawn above it. So there is
nothing to memorise and nothing in the way.

Drawing runs on uint8 through OpenCV rather than on float32 through NumPy. The
same HUD written the NumPy way costs over ten times as much per frame, and the
whole frame budget is about 16 ms.

The bloom is done on a quarter-size layer: blur 320x180, scale it up, add. A
Gaussian at full resolution would cost more than the rest of the frame put
together and would look no different once it is behind a glow.
"""

from __future__ import annotations

import cv2
import numpy as np

from .music import DEGREES, VOICINGS
from .poses import DEGREE_SHAPES

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe's 21 landmarks, as bones. Drawing the skeleton is not decoration:
# it is how the player sees that tracking is alive and on their hand, so when
# something stops working the screen says so.
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
TIPS = (4, 8, 12, 16, 20)
VOICE_LABELS = tuple(label for _, label in VOICINGS)
STRANDS = (1, 1, 2, 3, 4, 5)      # ribbon strands per finger count


def degree_colours(n=DEGREES):
    """Warm amber at the bottom of the key, cool violet at the top."""
    hsv = np.zeros((1, max(n, 1), 3), dtype=np.uint8)
    hsv[0, :, 0] = np.linspace(14, 140, max(n, 1)).astype(np.uint8)
    hsv[0, :, 1] = 165
    hsv[0, :, 2] = 255
    return [tuple(int(v) for v in c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _scale(c, k):
    return tuple(int(min(255, max(0, v * k))) for v in c)


def _plate(img, x0, y0, x1, y1, strength=0.62):
    """Darken a patch so text on top of a camera picture stays readable."""
    box = img[max(int(y0), 0):max(int(y1), 0), max(int(x0), 0):max(int(x1), 0)]
    if box.size:
        box[:] = (box * (1.0 - strength)).astype(np.uint8)


class HUD:
    GLOW_DIV = 4
    FLASH = 0.45             # seconds for a struck chord to stop glowing
    BAR = 50                 # instrument bar along the foot of the window
    GUIDE = 76               # the seven chords and their hand shapes
    GAMMA = 1.25
    TARGET = 58.0            # mean brightness to leave the camera view at

    def __init__(self, key, instruments, dim=0.62):
        self.key = key
        self.instruments = instruments
        self.dim = dim
        self.energy = np.zeros(DEGREES, dtype=np.float32)
        self.ribbons = {}        # hand label -> how hard it is currently sounding
        self.phase = 0.0
        self.colours = degree_colours()
        self._lut = None
        self._lut_dim = None

    def retune(self):
        self.energy[:] = 0.0
        self.colours = degree_colours()

    # -- state -------------------------------------------------------------
    def strike(self, degree, amount=1.0):
        if degree is not None and 0 <= degree < DEGREES:
            self.energy[degree] = min(1.0, self.energy[degree] * 0.4 + amount)

    def step(self, dt, states, chord=None):
        fade = float(np.exp(-dt / self.FLASH))
        self.phase += dt
        self.energy *= fade
        for label in list(self.ribbons):
            self.ribbons[label] *= fade

        live = set()
        if chord is not None:
            # A held chord stays lit, so holding and ringing both show where
            # the sound is coming from.
            floor = 0.30 + 0.55 * chord.level
            self.energy[chord.degree] = max(self.energy[chord.degree], floor)
            for st in states:
                if st.fingers > 0:
                    live.add(st.label)
                    self.ribbons[st.label] = max(self.ribbons.get(st.label, 0.0),
                                                 floor)

        self.energy[self.energy < 1e-3] = 0.0
        for label, value in list(self.ribbons.items()):
            if value < 1e-3 and label not in live:
                del self.ribbons[label]

    # -- drawing -----------------------------------------------------------
    def _adapt(self, frame):
        """Dim by however much this room actually needs.

        A fixed multiply is tuned for one lighting condition and wrong in every
        other: it washes out a bright room and turns a dim one black, and a dim
        room is exactly where this gets played in the evening. Aiming at a
        target mean instead keeps the view readable either way.
        """
        mean = float(frame[::16, ::16].mean())
        want = (self.TARGET / 255.0) / max(mean / 255.0, 0.004) ** self.GAMMA
        self.dim += (float(np.clip(want, 0.30, 1.0)) - self.dim) * 0.10
        return round(self.dim, 2)

    def _dim_frame(self, frame):
        dim = self._adapt(frame)
        if self._lut is None or self._lut_dim != dim:
            x = np.arange(256, dtype=np.float32) / 255.0
            # Gamma curve rather than a flat multiply: highlights hold their
            # shape, so the player still reads as a person behind the sound.
            self._lut = np.clip(np.power(x, self.GAMMA) * dim * 255.0,
                                0, 255).astype(np.uint8)
            self._lut_dim = dim
        return cv2.LUT(frame, self._lut)

    def draw(self, frame, hands, states, chord, instrument,
             fps=None, debug=None, hint=0.0):
        h, w = frame.shape[:2]
        out = self._dim_frame(frame)
        self._draw_ribbons(out, states, chord, instrument, w, h)
        self._draw_hands(out, hands, states, instrument)
        self._draw_readout(out, chord, states, instrument, w)
        self._draw_guide(out, chord, instrument, w, h)
        self._draw_bar(out, instrument, w, h)
        self._draw_status(out, instrument, w, fps, debug)
        self._draw_hint(out, chord, w, h, hint)
        return out

    # -- the sound itself --------------------------------------------------
    def _draw_ribbons(self, out, states, chord, instrument, w, h):
        """A band of travelling sine waves for each sounding hand.

        It rides at the height of the hand and swells with it, so the one
        control that is continuous rather than stepped has something continuous
        on screen answering it.
        """
        if not self.ribbons or chord is None:
            return

        gw, gh = max(w // self.GLOW_DIV, 1), max(h // self.GLOW_DIV, 1)
        glow = np.zeros((gh, gw, 3), dtype=np.uint8)
        k = 1.0 / self.GLOW_DIV
        xs = np.linspace(0.0, w, 150)
        taper = np.sin(np.pi * xs / max(w, 1)) ** 0.8      # fade out at the edges
        floor = h - self.BAR - self.GUIDE
        base = _mix(self.colours[chord.degree], instrument.colour, 0.45)

        for st in states:
            level = self.ribbons.get(st.label, 0.0)
            if level < 0.02 or st.fingers <= 0:
                continue
            strands = STRANDS[min(st.fingers, len(STRANDS) - 1)]
            cy = float(np.clip(st.y, 80, max(floor - 60, 90)))
            span = 24.0 + 88.0 * level

            for s in range(strands):
                offset = (s - (strands - 1) / 2.0) * (span / max(strands, 1))
                wobble = 0.0055 + 0.0016 * s
                speed = 2.1 + 0.45 * s + 1.8 * st.bright
                ys = (cy + offset + taper * span * 0.55
                      * np.sin(xs * wobble + self.phase * speed + s * 1.1))
                pts = np.stack([xs, ys], axis=1).astype(np.int32)
                cv2.polylines(out, [pts], False,
                              _scale(base, 0.45 + 0.55 * level),
                              2 if s == 0 else 1, cv2.LINE_AA)
                cv2.polylines(glow, [(pts * k).astype(np.int32)], False,
                              _scale(base, 0.30 + 0.55 * level), 1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(glow, (0, 0), 3.0)
        cv2.add(out, cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR), out)

    # -- the hands ---------------------------------------------------------
    def _draw_hands(self, out, hands, states, instrument):
        tint = instrument.colour
        by_label = {st.label: st for st in states}

        for hand in hands:
            pts = hand.points
            if pts is None or len(pts) < 21:
                continue
            st = by_label.get(hand.label)
            xy = pts[:, :2].astype(np.int32)
            up = st.up if st is not None else None

            for a, b in BONES:
                cv2.line(out, tuple(xy[a]), tuple(xy[b]), _scale(tint, 0.60),
                         1, cv2.LINE_AA)
            for j, tip in enumerate(TIPS):
                lit = bool(up[j]) if up is not None else False
                cv2.circle(out, tuple(xy[tip]), 8 if lit else 3,
                           _scale(tint, 1.0 if lit else 0.40), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(xy[0]), 3, _scale(tint, 0.55), -1, cv2.LINE_AA)

    # -- what is sounding --------------------------------------------------
    def _draw_readout(self, out, chord, states, instrument, w):
        """The chord, large, at the top. The one thing worth reading at a
        glance while playing."""
        if chord is None:
            return
        tint = instrument.colour
        settling = any(st.settling for st in states)
        alpha = 0.5 if settling else 1.0
        cx = w // 2

        label = self.key.label(chord.degree, chord.voicing)
        kind = VOICE_LABELS[min(chord.voicing, len(VOICE_LABELS) - 1)]
        roman = self.key.roman(chord.degree)

        (tw, th), _ = cv2.getTextSize(label, FONT, 1.6, 2)
        base = 16 + th                      # keep the cap height on screen
        _plate(out, cx - tw // 2 - 26, base - th - 14, cx + tw // 2 + 26, base + 60)
        cv2.putText(out, label, (cx - tw // 2, base), FONT, 1.6,
                    _scale(tint, alpha), 2, cv2.LINE_AA)

        sub = f"{roman}    {kind}"
        (sw, _), _ = cv2.getTextSize(sub, FONT_S, 0.52, 1)
        cv2.putText(out, sub, (cx - sw // 2, base + 26), FONT_S, 0.52,
                    _scale(tint, 0.68 * alpha), 1, cv2.LINE_AA)

        # Level meter: how high the hands are, which is how loud it is.
        bw = 150
        x0, y0 = cx - bw // 2, base + 40
        cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + 5), (52, 50, 58), -1)
        cv2.rectangle(out, (x0, y0), (x0 + int(bw * chord.level), y0 + 5),
                      _scale(tint, 0.95), -1)

    # -- the guide ---------------------------------------------------------
    def _draw_guide(self, out, chord, instrument, w, h):
        """All seven chords with the hand shape that reaches each one drawn
        above it. This is the whole manual, and it is always on screen."""
        top = h - self.BAR - self.GUIDE
        cv2.rectangle(out, (0, top), (w, h - self.BAR), (14, 13, 17), -1)
        cv2.line(out, (0, top), (w, top), (46, 43, 54), 1)

        slot = w / DEGREES
        here = chord.degree if chord is not None else -1
        for d in range(DEGREES):
            cx = slot * (d + 0.5)
            e = float(self.energy[d])
            on = d == here
            base = _mix(self.colours[d], instrument.colour, 0.40)
            colour = _scale(base, 0.45 + 0.9 * e)

            if on:
                cv2.rectangle(out, (int(cx - slot / 2) + 4, top + 5),
                              (int(cx + slot / 2) - 4, h - self.BAR - 5),
                              _scale(base, 0.26), -1)

            # Five dots, thumb first: the shape that plays this chord.
            shape = DEGREE_SHAPES[d]
            gap = min(13, slot / 7.0)
            x0 = cx - gap * 2
            for i, lit in enumerate(shape):
                cv2.circle(out, (int(x0 + gap * i), top + 20), 4 if lit else 3,
                           colour if lit else (68, 66, 76), -1, cv2.LINE_AA)

            name = self.key.names[d]
            (tw, _), _ = cv2.getTextSize(name, FONT, 0.58, 1)
            cv2.putText(out, name, (int(cx - tw / 2), top + 50), FONT, 0.58,
                        colour, 1, cv2.LINE_AA)
            roman = self.key.roman(d)
            (rw, _), _ = cv2.getTextSize(roman, FONT_S, 0.36, 1)
            cv2.putText(out, roman, (int(cx - rw / 2), top + 66), FONT_S, 0.36,
                        _scale(base, 0.40 + 0.5 * e), 1, cv2.LINE_AA)

    def _draw_bar(self, out, instrument, w, h):
        bar = h - self.BAR
        cv2.rectangle(out, (0, bar), (w, h), (20, 18, 24), -1)
        cv2.line(out, (0, bar), (w, bar), (52, 48, 60), 1)

        pad, x = 13, 14
        for i, inst in enumerate(self.instruments):
            text = f"{i + 1} {inst.name}"
            (tw, _), _ = cv2.getTextSize(text, FONT_S, 0.5, 1)
            box = tw + pad * 2
            on = inst is instrument
            if on:
                cv2.rectangle(out, (x, bar + 10), (x + box, h - 10),
                              _scale(inst.colour, 0.28), -1)
                cv2.rectangle(out, (x, bar + 10), (x + box, h - 10),
                              inst.colour, 1, cv2.LINE_AA)
            cv2.putText(out, text, (x + pad, h - 19), FONT_S, 0.5,
                        inst.colour if on else (124, 122, 132), 1, cv2.LINE_AA)
            x += box + 8

        keys = "[ ] key    m major/minor    d stats    q quit"
        (tw, _), _ = cv2.getTextSize(keys, FONT_S, 0.38, 1)
        if x + tw + 28 < w:
            cv2.putText(out, keys, (w - tw - 16, h - 19), FONT_S, 0.38,
                        (96, 94, 104), 1, cv2.LINE_AA)

    def _draw_status(self, out, instrument, w, fps, debug):
        (nw, _), _ = cv2.getTextSize(instrument.name, FONT, 0.72, 1)
        cv2.putText(out, instrument.name, (16, 36), FONT, 0.72,
                    instrument.colour, 1, cv2.LINE_AA)
        # Say which of the two behaviours this instrument has, in words, rather
        # than leaving the player to work out why the sound will not stop.
        mode = "holds while you hold" if instrument.sustains else "rings out on its own"
        cv2.putText(out, mode, (16, 56), FONT_S, 0.38,
                    _scale(instrument.colour, 0.55), 1, cv2.LINE_AA)
        cv2.putText(out, self.key.key_name, (16, 74), FONT_S, 0.44,
                    (158, 156, 168), 1, cv2.LINE_AA)

        if debug:
            y = 104
            for line in debug:
                cv2.putText(out, line, (16, y), FONT_S, 0.40, (130, 200, 140),
                            1, cv2.LINE_AA)
                y += 18
        elif fps:
            text = f"{fps:.0f} fps"
            (tw, _), _ = cv2.getTextSize(text, FONT_S, 0.42, 1)
            cv2.putText(out, text, (w - tw - 16, 36), FONT_S, 0.42,
                        (112, 110, 120), 1, cv2.LINE_AA)

    def _draw_hint(self, out, chord, w, h, hint):
        if hint <= 0.01 or chord is not None:
            return
        f = min(hint, 1.0)
        msg = "hold up a hand -- the fingers you raise are the chord"
        (tw, th), _ = cv2.getTextSize(msg, FONT, 0.62, 1)
        bx = (w - tw) // 2
        by = h - self.BAR - self.GUIDE - 40
        _plate(out, bx - 20, by - th - 13, bx + tw + 20, by + 15, 0.72 * f)
        c = int(235 * f)
        cv2.putText(out, msg, (bx, by), FONT, 0.62, (c, c, c), 1, cv2.LINE_AA)


def waiting_frame(width, height, message):
    """Shown before the first camera frame arrives, so the window is never blank."""
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(message, FONT, 0.7, 1)
    cv2.putText(img, message, ((width - tw) // 2, height // 2), FONT, 0.7,
                (170, 170, 180), 1, cv2.LINE_AA)
    return img
