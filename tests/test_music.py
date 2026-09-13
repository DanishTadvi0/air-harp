"""Tuning and the promise that no finger combination sounds wrong."""

import numpy as np
import pytest

from airharp.music import (
    DEGREES, MAJOR, MINOR, VOICINGS, Key, midi_to_hz, note_name,
)


@pytest.fixture
def key():
    return Key(root=48, scale="major")


def semitones(freqs):
    return np.round(12.0 * np.log2(np.asarray(freqs) / 440.0) + 69).astype(int)


# -- tuning -----------------------------------------------------------------

def test_a440_is_a440():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(60) == pytest.approx(261.626, abs=1e-3)
    assert note_name(60) == "C4"
    assert note_name(48) == "C3"


def test_c_major_gives_the_chords_you_would_expect(key):
    assert key.names == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    assert key.roman(0) == "I"
    assert key.roman(4) == "V"


def test_the_seventh_degree_is_diminished_not_minor(key):
    """B-D-F has a minor third and a flat fifth. Reading the third alone and
    calling it B minor is the classic way to get this wrong."""
    assert key.quality == ["maj", "min", "min", "maj", "maj", "min", "dim"]
    assert key.names[6] == "Bdim"
    assert key.label(6, 4) == "Bdim7"


def test_a_diminished_degree_is_never_labelled_a_power_chord(key):
    """A '5' chord is a perfect fifth. On the diminished degree that interval
    is a tritone, so the label would be a lie."""
    assert key.label(0, 2) == "C5"
    assert key.label(6, 2) == "Bdim"


def test_minor_key_qualities_are_right():
    k = Key(root=45, scale="minor")
    assert k.names == ["Am", "Bdim", "C", "Dm", "Em", "F", "G"]


def test_a_minor_gives_the_minor_chords():
    k = Key(root=45, scale="minor")
    assert k.names[0] == "Am"
    assert k.roman(0) == "i"
    assert "minor" in k.key_name


def test_there_are_exactly_seven_chords(key):
    assert len(key.names) == len(key.roots) == DEGREES == 7


# -- the "no wrong note" guarantee ------------------------------------------

@pytest.mark.parametrize("scale", ("major", "minor"))
def test_every_chord_belongs_to_the_key(scale):
    """The whole reason it can be played without being learned: nothing you
    can do with your fingers produces a note outside the key."""
    k = Key(root=48, scale=scale)
    parent = MAJOR if scale == "major" else MINOR
    allowed = {(k.root + s) % 12 for s in parent}
    for degree in range(DEGREES):
        for voicing in range(len(VOICINGS)):
            for pitch in semitones(k.chord(degree, voicing)[0]):
                assert pitch % 12 in allowed, f"degree {degree}, voicing {voicing}"


@pytest.mark.parametrize("scale", ("major", "minor"))
def test_no_chord_contains_a_semitone_clash(scale):
    k = Key(root=48, scale=scale)
    for degree in range(DEGREES):
        for voicing in range(len(VOICINGS)):
            pitches = sorted(set(semitones(k.chord(degree, voicing)[0])))
            assert all(g != 1 for g in np.diff(pitches)), (degree, voicing)


def test_more_fingers_is_never_a_smaller_chord(key):
    counts = [len(set(semitones(key.chord(0, n)[0]))) for n in range(len(VOICINGS))]
    assert counts == sorted(counts), counts
    assert counts[0] == 1 and counts[-1] >= 4


def test_a_fist_and_one_finger_both_give_the_single_root(key):
    fist, one = key.chord(3, 0)[0], key.chord(3, 1)[0]
    assert len(fist) == len(one) == 1
    assert fist[0] == pytest.approx(one[0])
    assert key.chord(3, 0)[1] == "muted"


def test_the_triad_is_root_third_fifth(key):
    freqs, label = key.chord(0, 3)
    assert label == "triad"
    assert list(semitones(freqs) - semitones(freqs)[0]) == [0, 4, 7]      # C E G


def test_the_seventh_adds_a_seventh(key):
    pitches = semitones(key.chord(0, 4)[0])
    assert list(pitches - pitches[0]) == [0, 4, 7, 11]                    # Cmaj7


def test_the_open_voicing_spreads_over_two_octaves(key):
    pitches = semitones(key.chord(0, 5)[0])
    assert max(pitches) - min(pitches) > 12


def test_minor_degrees_really_are_minor(key):
    freqs = key.chord(1, 3)[0]                                            # Dm
    assert list(semitones(freqs) - semitones(freqs)[0]) == [0, 3, 7]


def test_degrees_climb(key):
    assert list(key.roots) == sorted(key.roots)
    assert key.root_names[0] == "C3"


# -- labels -----------------------------------------------------------------

def test_labels_say_what_is_actually_sounding(key):
    assert key.label(0, 0) == "C3"
    assert key.label(0, 1) == "C3"
    assert key.label(0, 2) == "C5"          # a bare fifth
    assert key.label(0, 3) == "C"
    assert key.label(0, 4) == "C7"
    assert key.label(1, 4) == "Dm7"


def test_out_of_range_input_is_clamped_not_crashed(key):
    for degree in (-5, 0, 99):
        for voicing in (-2, 0, 99):
            freqs, _ = key.chord(degree, voicing)
            assert len(freqs) >= 1
            assert np.isfinite(freqs).all()
            assert key.label(degree, voicing)


# -- transposing ------------------------------------------------------------

def test_transpose_moves_everything_and_stays_in_range(key):
    before = list(key.roots)
    key.transpose(2)
    assert list(key.roots) == [b + 2 for b in before]
    for _ in range(60):
        key.transpose(1)
    assert key.root <= 64
    for _ in range(60):
        key.transpose(-1)
    assert key.root >= 36


def test_toggling_scale_keeps_the_same_root(key):
    root = key.root
    key.toggle_scale()
    assert key.scale == "minor" and key.root == root
    key.toggle_scale()
    assert key.scale == "major"
