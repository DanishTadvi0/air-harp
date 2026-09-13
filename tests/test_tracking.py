"""Smoothing and hand geometry -- the part that decides whether it feels good."""

import threading
import time

import numpy as np
import pytest

from airharp.tracking import INDEX_TIP, Mailbox, OneEuro, canonical

DT = 1 / 30.0
W = 1280.0        # the tracker filters in frame widths; these tests do too
SWEEP = 1800.0    # px/s, a brisk strum


# -- One Euro ---------------------------------------------------------------

def test_one_euro_kills_jitter_on_a_still_hand():
    rng = np.random.default_rng(1)
    truth = np.array([0.5, 0.5])
    f = OneEuro()
    noisy, smooth = [], []
    for _ in range(300):
        x = truth + rng.normal(0, 3.0 / W, 2)      # 3 px of landmark noise
        noisy.append(x)
        smooth.append(f(x, DT))
    raw_err = np.linalg.norm(np.array(noisy[50:]) - truth, axis=1).mean()
    out_err = np.linalg.norm(np.array(smooth[50:]) - truth, axis=1).mean()
    assert out_err < raw_err * 0.45, "a resting fingertip should sit still"


def test_one_euro_keeps_up_with_a_fast_sweep():
    """The point of the adaptive cutoff: heavy damping at rest must not turn
    into heavy lag once the hand moves."""
    speed = SWEEP / W
    xs = [np.array([0.05 + speed * i * DT, 0.5]) for i in range(60)]
    f = OneEuro()
    out = [f(x, DT) for x in xs]
    lag = np.mean([np.linalg.norm(a - b) for a, b in zip(xs[25:], out[25:])])
    assert lag * W < SWEEP * DT, "lag must stay under one frame of travel"
    assert lag / speed < 0.020, "and under 20 ms in absolute terms"


def test_one_euro_defaults_beat_a_fixed_lowpass_at_both_ends():
    """A fixed cutoff has to be tuned for stillness or for speed; the adaptive
    one is asked to win at both."""
    speed = SWEEP / W
    rng = np.random.default_rng(3)
    truth = np.array([0.5, 0.5])

    def jitter(f):
        vals = [f(truth + rng.normal(0, 3.0 / W, 2), DT) for _ in range(300)]
        return np.linalg.norm(np.array(vals[50:]) - truth, axis=1).mean()

    def lag(f):
        xs = [np.array([0.05 + speed * i * DT, 0.5]) for i in range(60)]
        out = [f(x, DT) for x in xs]
        return np.mean([np.linalg.norm(a - b) for a, b in zip(xs[25:], out[25:])])

    adaptive = (jitter(OneEuro()), lag(OneEuro()))
    smooth_fixed = (jitter(OneEuro(beta=0.0)), lag(OneEuro(beta=0.0)))
    fast_fixed = (jitter(OneEuro(min_cutoff=9.0, beta=0.0)),
                  lag(OneEuro(min_cutoff=9.0, beta=0.0)))

    assert adaptive[1] < smooth_fixed[1] * 0.6, "quicker than the smooth setting"
    assert adaptive[0] < fast_fixed[0] * 0.6, "steadier than the fast setting"


def test_one_euro_passes_the_first_sample_through():
    f = OneEuro()
    x = np.array([1.0, 2.0])
    assert np.allclose(f(x, DT), x)


def test_one_euro_reshapes_and_resets():
    f = OneEuro()
    f(np.zeros(2), DT)
    assert np.allclose(f(np.ones(5), DT), np.ones(5))   # shape change restarts
    f.reset()
    assert f.x_prev is None


# -- canonical frame --------------------------------------------------------

def test_canonical_is_invariant_to_where_the_hand_is(points_factory):
    near = canonical(points_factory(at=(200, 200), scale=90))
    far = canonical(points_factory(at=(1000, 600), scale=240))
    assert np.allclose(near, far, atol=1e-9), "position and distance must not matter"


def test_canonical_is_invariant_to_hand_rotation(points_factory):
    upright = canonical(points_factory(angle=0.0))
    tilted = canonical(points_factory(angle=0.9))
    assert np.allclose(upright, tilted, atol=1e-9)


def test_canonical_puts_the_wrist_at_the_origin_and_palm_up(points_factory):
    c = canonical(points_factory())
    assert np.allclose(c[0], (0.0, 0.0), atol=1e-9)
    assert np.allclose(c[9], (0.0, -1.0), atol=1e-9)   # one palm length, upward


def test_canonical_survives_a_degenerate_hand():
    assert np.allclose(canonical(np.zeros((21, 3))), 0.0)


# -- Hand -------------------------------------------------------------------

def test_hand_speed_is_pixels_per_second(hand_factory):
    hand = hand_factory(500.0, prev=400.0, dt=0.1)
    assert hand.speed == pytest.approx(1000.0)


def test_first_frame_of_a_hand_has_no_speed(hand_factory):
    assert hand_factory(500.0).speed == 0.0


def test_hand_tip_is_the_index_finger(hand_factory, points_factory):
    hand = hand_factory(640.0, y=200.0)
    assert np.allclose(hand.tip[:2], (640.0, 200.0))
    assert np.allclose(hand.tip, hand.points[INDEX_TIP])


# -- mailbox ----------------------------------------------------------------

def test_mailbox_hands_over_only_the_newest_item():
    box = Mailbox()
    for i in range(5):
        box.put(i)
    seq, item = box.get(0, timeout=0.01)
    assert item == 4, "stale frames are dropped, not queued"
    assert seq == 5
    assert box.dropped == 4


def test_mailbox_get_returns_nothing_when_nothing_is_new():
    box = Mailbox()
    box.put("a")
    seq, _ = box.get(0, timeout=0.01)
    assert box.get(seq, timeout=0.01) == (seq, None)


def test_mailbox_wakes_a_waiting_consumer():
    box = Mailbox()
    got = []

    def consume():
        got.append(box.get(0, timeout=2.0))

    t = threading.Thread(target=consume)
    t.start()
    time.sleep(0.05)
    box.put("late")
    t.join(timeout=2.0)
    assert got == [(1, "late")]
