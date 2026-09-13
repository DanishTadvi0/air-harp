"""What a finger combination is worth in notes.

Two things, and only two, decide what sounds:

    which fingers are up   ->  which chord     (degrees I to VII)
    how many are up        ->  how full it is  (one note, a fifth, a triad, ...)

Position does not matter. Nothing is triggered by where a hand is; the shape of
the hand is the whole instrument.

Chords are built by stacking thirds inside the parent major or minor scale, so
every degree gives a chord that belongs to the key. There is no combination of
fingers that sounds wrong, which is what lets it be played without being
learned first.
"""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_MIDI, A4_HZ = 69, 440.0

MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)
DEGREES = 7

ROMAN = {
    "major": ("I", "ii", "iii", "IV", "V", "vi", "vii"),
    "minor": ("i", "ii", "III", "iv", "v", "VI", "VII"),
}

# How many fingers buys how much chord. Indices are into the stacked-thirds
# chord: 0 root, 1 third, 2 fifth, 3 seventh. `+12` means an octave up.
VOICINGS = (
    ((0,), "muted"),                       # fist
    ((0,), "single"),                      # one finger
    ((0, 2), "fifth"),                     # two
    ((0, 1, 2), "triad"),                  # three
    ((0, 1, 2, 3), "seventh"),             # four
    ((0, 1, 2, 12, 13), "open"),           # open hand -- spread over two octaves
)
DEFAULT_VOICING = 3                        # a triad, when only one hand is up


def midi_to_hz(m):
    return A4_HZ * 2.0 ** ((np.asarray(m, dtype=np.float64) - A4_MIDI) / 12.0)


def note_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class Key:
    """A key and the seven chords in it."""

    def __init__(self, root=48, scale="major"):
        self.root = int(root)
        self.scale = scale if scale in ("major", "minor") else "major"
        self._build()

    def _build(self):
        self.parent = MAJOR if self.scale == "major" else MINOR
        self.roots = np.array([self._pitch(d) for d in range(DEGREES)],
                              dtype=np.float64)
        self.names = [self._chord_name(d) for d in range(DEGREES)]
        self.root_names = [note_name(m) for m in self.roots]

    def _pitch(self, degree):
        """Degree of the parent scale, counting past the octave, to MIDI."""
        n = len(self.parent)
        return self.root + self.parent[degree % n] + 12 * (degree // n)

    def _chord_name(self, degree):
        root = note_name(self._pitch(degree))[:-1]
        third = self._pitch(degree + 2) - self._pitch(degree)
        return f"{root}{'' if third >= 4 else 'm'}"

    def roman(self, degree):
        return ROMAN[self.scale][int(degree) % DEGREES]

    # -- tuning ------------------------------------------------------------
    def transpose(self, semitones):
        self.root = int(np.clip(self.root + semitones, 36, 64))
        self._build()

    def toggle_scale(self):
        self.scale = "minor" if self.scale == "major" else "major"
        self._build()

    @property
    def key_name(self):
        return f"{NOTE_NAMES[self.root % 12]} {self.scale}"

    # -- chords ------------------------------------------------------------
    def chord(self, degree, voicing):
        """Frequencies for a degree played at a given fullness."""
        degree = int(np.clip(degree, 0, DEGREES - 1))
        voicing = int(np.clip(voicing, 0, len(VOICINGS) - 1))
        picks, label = VOICINGS[voicing]
        notes = []
        for p in picks:
            octave, step = divmod(p, 12)
            notes.append(self._pitch(degree + 2 * step) + 12 * octave)
        return midi_to_hz(notes), label

    def label(self, degree, voicing):
        """What to print: 'Am7', 'C', 'G5' -- the chord actually sounding."""
        voicing = int(np.clip(voicing, 0, len(VOICINGS) - 1))
        degree = int(np.clip(degree, 0, DEGREES - 1))
        kind = VOICINGS[voicing][1]
        base = self.names[degree]
        if kind in ("muted", "single"):
            return self.root_names[degree]
        if kind == "fifth":
            return base.rstrip("m") + "5"
        if kind == "seventh":
            return base + "7"
        return base
