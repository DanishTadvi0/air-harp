"""Reading the hand: which fingers are up, what chord that is, and when to
believe it.

Counting fingers by comparing a fingertip's y to its knuckle's y -- the usual
shortcut -- only works while the hand is upright. Tilt it and fingers start
reading as curled. Here the count is taken in the canonical frame instead, from
distances rather than image axes, so it survives the hand being rotated,
tilted, or held at any distance from the lens.

The chord comes from the combination alone. Where the hand is does not matter.

The other half of the job is refusing to believe the detector too quickly. A
raw reading flickers between neighbouring shapes on almost every hand, and a
chord that flickers with it is unplayable. A new shape has to hold for `HOLD`
seconds before it commits, and a hand that blinks out of tracking keeps playing
for `NULL` seconds before it is let go. Together those are the difference
between an instrument and a twitch.
"""

from __future__ import annotations

import numpy as np

from .music import DEFAULT_VOICING, DEGREES, VOICINGS
from .tracking import canonical

# Landmark chains, thumb first: (mcp, pip, dip, tip)
THUMB = (1, 2, 3, 4)
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

EXTEND_RATIO = 1.06     # tip must out-reach its own middle joint by this much
THUMB_CLEAR = 0.58      # palm lengths from the index knuckle to count as out


def extended(canon):
    """Which of the five fingers are extended, thumb first.

    A straight finger puts its tip further from the wrist than its middle
    joint; a curled one folds the tip back toward the palm and reverses that.
    Comparing two distances keeps the test independent of how the hand is
    turned, which comparing y coordinates does not.
    """
    canon = np.asarray(canon, dtype=np.float64)
    reach = np.linalg.norm(canon, axis=1)
    out = np.zeros(5, dtype=bool)

    # The thumb folds across the palm rather than curling along it, so it is
    # measured by how far clear of the index knuckle it sits.
    out[0] = np.linalg.norm(canon[THUMB[3]] - canon[FINGERS[0][0]]) > THUMB_CLEAR
    for i, (_, pip, _, tip) in enumerate(FINGERS):
        out[i + 1] = reach[tip] > reach[pip] * EXTEND_RATIO
    return out


def count_fingers(canon):
    return int(extended(canon).sum())


def degree_of(up):
    """Which chord a finger combination plays, or None for a closed fist.

    One to five fingers give the first five degrees straight off the count, so
    the common cases need no learning at all. The two that are left over get
    the one shape a hand can make that the count cannot reach -- index and
    little finger, with the thumb telling them apart.
    """
    thumb, index, middle, ring, pinky = (bool(v) for v in up)
    if index and pinky and not middle and not ring:
        return 6 if thumb else 5           # VII with the thumb out, else VI
    n = int(thumb) + int(index) + int(middle) + int(ring) + int(pinky)
    if n == 0:
        return None                        # a fist is a deliberate stop
    return min(n, 5) - 1                   # I to V


# The shape that reaches each degree, thumb first. The HUD draws these back at
# the player, so the guide on screen can never drift out of step with
# `degree_of` above -- a test asserts the two agree.
DEGREE_SHAPES = (
    (False, True, False, False, False),    # I    index
    (False, True, True, False, False),     # ii   index middle
    (False, True, True, True, False),      # iii  index middle ring
    (False, True, True, True, True),       # IV   four fingers
    (True, True, True, True, True),        # V    open hand
    (False, True, False, False, True),     # vi   index and little
    (True, True, False, False, True),      # vii  index, little and thumb
)


class HandState:
    """What one hand is doing right now."""

    __slots__ = ("label", "fingers", "up", "degree", "voicing", "x", "y",
                 "level", "bright", "raw_fingers", "settling", "leads")

    def __init__(self, label, fingers, up, degree, voicing, x, y, level,
                 bright, raw_fingers, settling, leads):
        self.label = label
        self.fingers = fingers      # committed finger count
        self.up = up                # committed per-finger flags, for the HUD
        self.degree = degree        # what this hand alone would play, or None
        self.voicing = voicing
        self.x = x
        self.y = y
        self.level = level          # 0..1 from hand height: how loud
        self.bright = bright        # 0..1 from hand height: how open the tone
        self.raw_fingers = raw_fingers
        self.settling = settling    # True while a new shape is being confirmed
        self.leads = leads          # True if this hand is choosing the chord

    def __repr__(self):  # pragma: no cover
        return (f"HandState({self.label}, fingers={self.fingers}, "
                f"degree={self.degree}, voicing={self.voicing})")


class Chord:
    """The one chord currently sounding, pooled from both hands."""

    __slots__ = ("degree", "voicing", "level", "bright")

    def __init__(self, degree, voicing, level, bright):
        self.degree = degree
        self.voicing = voicing
        self.level = level
        self.bright = bright

    @property
    def key(self):
        return (self.degree, self.voicing)

    def __repr__(self):  # pragma: no cover
        return f"Chord(degree={self.degree}, voicing={self.voicing})"


class PoseReader:
    """Turns tracked hands into one stable chord decision.

    The left hand chooses the chord and the right hand chooses how full it is,
    which is the split the reference instrument uses. Either hand on its own
    still plays, though: whichever is up picks the chord, and without a right
    hand the voicing falls back to a plain triad. Needing two hands up before
    anything happens would be a poor first thirty seconds.
    """

    HOLD = 0.09         # seconds a new shape must persist before it is believed
    NULL = 0.16         # seconds of lost tracking before the hand is released
    EDGE = 0.08         # dead margin either side: tracking is poor at the frame edge
    TAPER = 1.8         # amplitude curve across the sweep, so the fade sounds even

    def __init__(self, key=None):
        self.key = key
        self._state = {}
        self._chord = None

    def reset(self):
        self._state.clear()
        self._chord = None

    @property
    def chord(self):
        return self._chord

    def _blank(self, up, now):
        return {"up": tuple(bool(v) for v in up), "cand": None,
                "since": now, "seen": now}

    def update(self, hands, width, height, now):
        """Returns (states, chord, changed, released).

        `chord` is what should be sounding, or None for silence. `changed` is
        True when it just became something different -- what a struck
        instrument plays on.
        """
        states = []
        live = set()

        for hand in hands:
            live.add(hand.label)
            raw_up = extended(canonical(hand.points))
            raw_tuple = tuple(bool(v) for v in raw_up)

            st = self._state.get(hand.label)
            if st is None:
                st = self._state[hand.label] = self._blank(raw_tuple, now)
            st["seen"] = now

            settling = False
            if raw_tuple == st["up"]:
                st["cand"] = None
            elif st["cand"] == raw_tuple:
                if now - st["since"] >= self.HOLD:
                    st["up"] = raw_tuple
                    st["cand"] = None
                else:
                    settling = True
            else:
                st["cand"] = raw_tuple
                st["since"] = now
                settling = True

            up = st["up"]
            x, y = float(hand.tip[0]), float(hand.tip[1])
            # Neither of these is ever stabilised: they are expression, and
            # expression that snaps to steps sounds mechanical.
            #
            # Across is the fader. Right is loud, left fades away, and the far
            # left is silence -- so how slowly you drift left is how long the
            # chord takes to die. A margin at each end keeps the extremes
            # reachable, because tracking is worst at the edge of frame.
            across = (x / max(width, 1) - self.EDGE) / (1.0 - 2.0 * self.EDGE)
            across = float(np.clip(across, 0.0, 1.0))
            high = float(np.clip(1.0 - y / max(height, 1), 0.0, 1.0))
            states.append(HandState(
                label=hand.label,
                fingers=sum(up), up=up,
                degree=degree_of(up),
                voicing=min(sum(up), len(VOICINGS) - 1),
                x=x, y=y,
                level=across ** self.TAPER,
                bright=high ** 0.85,
                raw_fingers=int(raw_up.sum()),
                settling=settling,
                leads=False,
            ))

        released = []
        for label, st in list(self._state.items()):
            if label in live:
                continue
            if now - st["seen"] >= self.NULL:
                del self._state[label]
                released.append(label)

        chord = self._combine(states)
        changed = (chord is None) != (self._chord is None) or (
            chord is not None and self._chord is not None
            and chord.key != self._chord.key)
        self._chord = chord
        return states, chord, changed, released

    def _combine(self, states):
        """Left hand names the chord, right hand says how full. Either alone
        still plays."""
        if not states:
            return None
        by = {st.label: st for st in states}
        lead = by.get("Left") or states[0]
        if lead.degree is None:
            return None                    # a fist stops it
        lead.leads = True

        fuller = by.get("Right")
        if fuller is not None and fuller is not lead and fuller.fingers > 0:
            voicing = fuller.voicing
        elif fuller is lead:
            voicing = lead.voicing
        else:
            voicing = DEFAULT_VOICING

        # The fader belongs to whichever hand is naming the chord, so one hand
        # controls both and the other is free to do nothing at all.
        return Chord(
            degree=int(np.clip(lead.degree, 0, DEGREES - 1)),
            voicing=int(voicing),
            level=float(lead.level),
            bright=float(lead.bright),
        )
