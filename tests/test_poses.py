"""Reading the hand, and refusing to believe the detector too quickly."""

import numpy as np
import pytest

from airharp.music import DEFAULT_VOICING, DEGREES, Key
from airharp.poses import (
    DEGREE_SHAPES, PoseReader, count_fingers, degree_of, extended,
)
from airharp.tracking import canonical

from conftest import fist, make_hand, make_points

W, H = 1280, 720
DT = 1 / 30.0


@pytest.fixture
def key():
    return Key(root=48, scale="major")


@pytest.fixture
def reader(key):
    return PoseReader(key)


class Clock:
    """A running wall clock, so a test never accidentally skips past the hold
    time or the null window between one phase and the next."""

    def __init__(self, reader, t0=100.0, dt=DT):
        self.reader, self.t, self.dt = reader, t0, dt

    def feed(self, hands, frames=1):
        """Returns (states, chord, how many times it changed, releases).

        Changes are counted across every frame: a new shape commits a few
        frames after the hand moves, so only looking at the final frame would
        miss it.
        """
        states, chord, changes, released = [], None, 0, []
        for _ in range(frames):
            states, chord, changed, rel = self.reader.update(hands, W, H, self.t)
            changes += int(changed)
            released += rel
            self.t += self.dt
        return states, chord, changes, released


def hand(shape, y=360.0, x=None, label="Left", **kw):
    # Default to the far right, which is full level, so tests that are not
    # about the fader are not accidentally about the fader.
    return [make_hand(W * 0.95 if x is None else x, y=y, fingers=shape,
                      label=label, **kw)]


# -- counting fingers -------------------------------------------------------

@pytest.mark.parametrize("n", range(6))
def test_counts_the_fingers_that_are_up(n):
    assert count_fingers(canonical(make_points(fingers=fist(n)))) == n


@pytest.mark.parametrize("n", range(6))
def test_the_count_survives_the_hand_being_turned(n):
    """Comparing fingertip y to knuckle y -- the usual shortcut -- starts
    miscounting as soon as the hand tilts. Distances in the canonical frame
    do not."""
    for angle in (-1.2, -0.6, 0.0, 0.6, 1.2, 2.4):
        got = count_fingers(canonical(make_points(fingers=fist(n), angle=angle)))
        assert got == n, f"{n} fingers misread as {got} at {angle:.1f} rad"


@pytest.mark.parametrize("n", range(6))
def test_the_count_survives_distance_from_the_camera(n):
    for scale in (55.0, 110.0, 300.0):
        assert count_fingers(canonical(make_points(fingers=fist(n), scale=scale))) == n


def test_which_fingers_are_reported_matches_which_are_up():
    up = extended(canonical(make_points(fingers=(False, True, True, False, False))))
    assert list(up) == [False, True, True, False, False]


def test_a_degenerate_hand_reads_as_a_fist():
    assert count_fingers(np.zeros((21, 2))) == 0


# -- combination to chord ---------------------------------------------------

def test_the_guide_on_screen_matches_what_actually_plays():
    """The HUD draws DEGREE_SHAPES as the manual. If it ever disagreed with
    the classifier the instrument would be lying to the player."""
    assert len(DEGREE_SHAPES) == DEGREES
    for degree, shape in enumerate(DEGREE_SHAPES):
        assert degree_of(shape) == degree, shape


def test_one_to_five_fingers_are_the_first_five_chords():
    for n in range(1, 6):
        assert degree_of(fist(n)) == n - 1


def test_index_and_little_finger_reach_the_last_two():
    assert degree_of((False, True, False, False, True)) == 5
    assert degree_of((True, True, False, False, True)) == 6


def test_a_fist_is_the_only_silence():
    assert degree_of((False,) * 5) is None
    for n in range(1, 6):
        assert degree_of(fist(n)) is not None


def test_every_combination_of_fingers_is_playable():
    """No reachable hand shape may leave the player with nothing."""
    for bits in range(1, 32):
        up = tuple(bool(bits >> i & 1) for i in range(5))
        d = degree_of(up)
        assert d is not None and 0 <= d < DEGREES, up


# -- reading a hand ---------------------------------------------------------

def test_the_shape_of_the_hand_picks_the_chord(reader):
    for degree, shape in enumerate(DEGREE_SHAPES):
        reader.reset()
        _, chord, _, _ = Clock(reader).feed(hand(shape), 6)
        assert chord is not None and chord.degree == degree


def test_where_the_hand_is_does_not_matter(reader):
    """Position is not part of the instrument. The same shape anywhere on
    screen must play the same chord."""
    seen = set()
    for x in (60.0, 400.0, 640.0, 900.0, W - 60.0):
        for y in (80.0, 360.0, H - 80.0):
            reader.reset()
            hands = [make_hand(x, y=y, fingers=DEGREE_SHAPES[2], label="Left")]
            _, chord, _, _ = Clock(reader).feed(hands, 6)
            seen.add(chord.degree)
    assert seen == {2}


def test_a_fist_stops_it(reader):
    clock = Clock(reader)
    _, chord, _, _ = clock.feed(hand(DEGREE_SHAPES[3]), 6)
    assert chord is not None
    _, chord, changes, _ = clock.feed(hand((False,) * 5), 8)
    assert chord is None and changes == 1


def test_across_the_frame_is_the_fader(reader):
    """Right is loud, left is quiet. Nothing to do with height."""
    out = {}
    for name, x in (("right", W * 0.95), ("middle", W * 0.5), ("left", W * 0.2)):
        reader.reset()
        _, chord, _, _ = Clock(reader).feed(hand(DEGREE_SHAPES[0], x=x), 4)
        out[name] = chord.level
    assert out["right"] > out["middle"] > out["left"]
    assert out["right"] > 0.9


def test_the_far_left_is_silence(reader):
    """The whole point of the sweep: keep going left and the chord dies."""
    for x in (0.0, W * 0.02, W * 0.07):
        reader.reset()
        _, chord, _, _ = Clock(reader).feed(hand(DEGREE_SHAPES[0], x=x), 4)
        assert chord is not None, "the chord is still chosen, just silent"
        assert chord.level == 0.0, f"x={x} should be silent"


def test_the_fade_is_monotonic_all_the_way_down(reader):
    """Drifting left must never get louder on the way, or the fade would
    sound like it was wobbling rather than dying."""
    levels = []
    for x in np.linspace(W * 0.95, 0.0, 30):
        reader.reset()
        _, chord, _, _ = Clock(reader).feed(hand(DEGREE_SHAPES[0], x=float(x)), 3)
        levels.append(chord.level)
    assert all(b <= a + 1e-9 for a, b in zip(levels, levels[1:])), levels
    assert levels[0] > 0.9 and levels[-1] == 0.0


def test_level_is_not_quantised(reader):
    """Expression that snaps to steps sounds mechanical."""
    seen = set()
    for x in np.linspace(W * 0.12, W * 0.9, 25):
        reader.reset()
        _, chord, _, _ = Clock(reader).feed(hand(DEGREE_SHAPES[0], x=float(x)), 3)
        seen.add(round(chord.level, 4))
    assert len(seen) >= 20


def test_height_still_sets_brightness(reader):
    """Across and up are independent: one is how loud, the other is how open."""
    out = {}
    for name, y in (("high", 60.0), ("low", H - 90.0)):
        reader.reset()
        _, chord, _, _ = Clock(reader).feed(hand(DEGREE_SHAPES[0], y=y), 4)
        out[name] = chord
    assert out["high"].bright > out["low"].bright + 0.3
    assert out["high"].level == pytest.approx(out["low"].level),         "height must not touch the fader"


def test_the_fader_belongs_to_the_hand_naming_the_chord(reader):
    """One hand does chord and volume, so the other is free to do nothing."""
    hands = [make_hand(W * 0.2, fingers=DEGREE_SHAPES[0], label="Left"),
             make_hand(W * 0.95, fingers=fist(3), label="Right")]
    _, chord, _, _ = Clock(reader).feed(hands, 6)
    assert chord.level < 0.1, "the left hand is low, so it is quiet"


# -- two hands --------------------------------------------------------------

def test_the_right_hand_sets_how_full_the_chord_is(reader):
    hands = [make_hand(400.0, fingers=DEGREE_SHAPES[0], label="Left"),
             make_hand(900.0, fingers=fist(4), label="Right")]
    _, chord, _, _ = Clock(reader).feed(hands, 6)
    assert chord.degree == 0, "the left hand still names the chord"
    assert chord.voicing == 4, "the right hand made it a seventh"


def test_one_hand_alone_plays_a_triad(reader):
    _, chord, _, _ = Clock(reader).feed(hand(DEGREE_SHAPES[2]), 6)
    assert chord.degree == 2
    assert chord.voicing == DEFAULT_VOICING


def test_a_right_hand_alone_still_plays(reader):
    """Needing both hands up before anything happens would be a poor first
    thirty seconds."""
    _, chord, _, _ = Clock(reader).feed(
        hand(DEGREE_SHAPES[4], label="Right"), 6)
    assert chord is not None and chord.degree == 4


# -- stabilisation ----------------------------------------------------------

def test_a_new_shape_is_believed_once_it_has_been_held(reader):
    clock = Clock(reader)
    clock.feed(hand(DEGREE_SHAPES[1]), 6)
    states, chord, changes, _ = clock.feed(hand(DEGREE_SHAPES[4]), 8)
    assert changes == 1 and chord.degree == 4
    assert not states[0].settling


def test_a_flicker_never_commits(reader):
    """A raw reading that bounces between two shapes on alternate frames must
    leave the chord alone -- this is the failure that makes gesture control
    unplayable."""
    clock = Clock(reader)
    clock.feed(hand(DEGREE_SHAPES[2]), 6)
    changes = 0
    for i in range(60):
        shape = DEGREE_SHAPES[2] if i % 2 == 0 else DEGREE_SHAPES[3]
        states, chord, ch, _ = clock.feed(hand(shape), 1)
        changes += ch
    assert changes == 0
    assert chord.degree == 2
    assert states[0].settling, "but the screen should show it is unsure"


def test_a_shape_held_past_the_hold_time_commits_exactly_once(reader):
    clock = Clock(reader)
    clock.feed(hand(DEGREE_SHAPES[0]), 6)
    changes = 0
    for _ in range(20):
        changes += clock.feed(hand(DEGREE_SHAPES[3]), 1)[2]
    assert changes == 1, "committed once, not once a frame"


def test_a_blink_of_lost_tracking_keeps_playing(reader):
    """MediaPipe drops a hand for a frame or two all the time. Letting go of
    the chord every time it does would make the instrument stutter."""
    clock = Clock(reader)
    clock.feed(hand(DEGREE_SHAPES[3]), 6)
    _, _, _, gone = clock.feed([], 3)             # ~100 ms of nothing
    assert gone == []
    _, chord, _, _ = clock.feed(hand(DEGREE_SHAPES[3]), 1)
    assert chord.degree == 3, "and it picks up where it left off"


def test_a_hand_that_really_leaves_is_released(reader):
    clock = Clock(reader)
    clock.feed(hand(DEGREE_SHAPES[3]), 6)
    released = []
    for _ in range(12):
        _, chord, _, gone = clock.feed([], 1)
        released += gone
    assert released == ["Left"]
    assert chord is None
    assert reader._state == {}


def test_reset_forgets_everything(reader):
    Clock(reader).feed(hand(DEGREE_SHAPES[0]), 5)
    reader.reset()
    assert reader._state == {} and reader.chord is None
