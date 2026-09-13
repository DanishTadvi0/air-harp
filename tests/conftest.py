import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airharp.tracking import Hand  # noqa: E402

# Palm layout in canonical units: wrist at the origin, middle knuckle at (0,-1).
MCP = {
    "thumb":  (-0.62, -0.42),
    "index":  (-0.36, -0.94),
    "middle": (0.00, -1.00),
    "ring":   (0.33, -0.96),
    "pinky":  (0.62, -0.86),
}
BONE = {
    "index":  (0.40, 0.25, 0.19),
    "middle": (0.44, 0.27, 0.20),
    "ring":   (0.40, 0.25, 0.19),
    "pinky":  (0.32, 0.20, 0.16),
}
SLOTS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
         "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _chain(origin, lengths, curl):
    """A finger as three segments, each joint bending by `curl` radians.

    Straight up at curl 0; folded back over the palm by about 1.5 rad a joint,
    which is what puts a fingertip closer to the wrist than its own knuckle.
    """
    pts, p = [], np.asarray(origin, dtype=np.float64)
    for k, length in enumerate(lengths, start=1):
        a = curl * k
        p = p + length * np.array([np.sin(a), -np.cos(a)])
        pts.append(p.copy())
    return pts


def make_points(tip=None, fingers=(True,) * 5, angle=0.0, scale=110.0,
                at=(640.0, 360.0), spread=None):
    """A 21-landmark hand with the chosen fingers extended.

    `fingers` is (thumb, index, middle, ring, pinky). `spread` is the older
    all-or-nothing control kept for the tests that only care about openness.
    """
    if spread is not None:
        fingers = (spread > 0.5,) * 5
        part = float(np.clip(spread, 0.0, 1.0))
    else:
        part = None

    base = np.zeros((21, 2), dtype=np.float64)
    base[0] = (0.0, 0.0)

    for name, slots in SLOTS.items():
        up = fingers[["index", "middle", "ring", "pinky"].index(name) + 1]
        curl = 0.0 if up else 1.55
        if part is not None:
            curl = (1.0 - part) * 1.55
        base[slots[0]] = MCP[name]
        for slot, p in zip(slots[1:], _chain(MCP[name], BONE[name], curl)):
            base[slot] = p

    # The thumb swings across the palm rather than curling along it.
    direction = np.array([-0.80, -0.60]) if fingers[0] else np.array([0.80, -0.60])
    base[1] = np.array(MCP["thumb"]) * 0.55
    base[2] = MCP["thumb"]
    base[3] = base[2] + direction * 0.34
    base[4] = base[2] + direction * 0.62

    c, s = np.cos(angle), np.sin(angle)
    px = (base @ np.array([[c, -s], [s, c]]).T) * scale
    px = px - px[0] + np.asarray(at, dtype=np.float64)
    if tip is not None:
        px = px - px[8] + np.asarray(tip, dtype=np.float64)
    return np.concatenate([px, np.zeros((21, 1))], axis=1)


def make_hand(x, y=360.0, prev=None, dt=1 / 30.0, fingers=(True,) * 5,
              label="Right", spread=None, **kw):
    pts = make_points(tip=(x, y), fingers=fingers, spread=spread, **kw)
    prev_tip = None if prev is None else np.array([prev, y, 0.0])
    return Hand(label, pts, prev_tip, dt, 0.95)


def fist(n):
    """(thumb, index, middle, ring, pinky) for `n` fingers held up."""
    order = [1, 2, 3, 4, 0]       # index, middle, ring, pinky, then thumb
    out = [False] * 5
    for i in order[:n]:
        out[i] = True
    return tuple(out)


@pytest.fixture
def hand_factory():
    return make_hand


@pytest.fixture
def points_factory():
    return make_points


@pytest.fixture
def pose():
    return fist
