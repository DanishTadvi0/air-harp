"""Real-time synthesis engine.

Three rules govern this file:

1. The PortAudio callback never synthesises. A producer thread renders blocks
   into a bounded queue; the callback pops one and copies it out. `Queue.put`
   blocking on a full queue is what paces the producer, so nothing here relies
   on `time.sleep` -- Windows rounds a 0.5 ms sleep up to ~15 ms, which would
   be three audio blocks of jitter.

2. Every per-sample loop is expressed as a vector operation. Delay-line
   feedback (Karplus-Strong, reverb combs, allpasses) reads only values written
   a full lap earlier, so a block can be processed in chunks no longer than the
   delay and each chunk collapses to one vector add.

3. Sawtooth-ish waves come from band-limited wavetables. A naive saw aliases
   audibly above a few hundred hertz.
"""

from __future__ import annotations

import queue
import threading
from collections import deque

import numpy as np

SR = 44100
BLOCK = 512          # 11.6 ms -- small enough to feel tight, large enough to be safe
QUEUE_BLOCKS = 4
MAX_VOICES = 24

_TN = 8192           # wavetable length; nearest-neighbour lookup is about -78 dB
_TMASK = _TN - 1
_SINE = np.sin(2.0 * np.pi * np.arange(_TN) / _TN).astype(np.float32)
_RAMP = np.linspace(0.0, 1.0, BLOCK, endpoint=False, dtype=np.float32)
_ARANGE = np.arange(BLOCK, dtype=np.float64)

_rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# band-limited sawtooth bank: [brightness][octave band] -> one cycle
# ---------------------------------------------------------------------------

_BANDS = 10
_BAND_LO = 20.0          # band k covers fundamentals up to _BAND_LO * 2**(k+1)
_BRIGHTS = 5


def _build_saw_bank():
    phase = 2.0 * np.pi * np.arange(_TN) / _TN
    bank = []
    for b in range(_BRIGHTS):
        rolloff = 1.35 - 0.16 * b          # 1/n**rolloff: higher b is brighter
        rows = []
        for k in range(_BANDS):
            f_max = _BAND_LO * (2.0 ** (k + 1))
            n_harm = max(1, min(96, int(SR * 0.45 / f_max)))
            n = np.arange(1, n_harm + 1)
            amp = 1.0 / np.power(n, rolloff)
            wave = (np.sin(np.outer(n, phase)) * amp[:, None]).sum(axis=0)
            peak = np.abs(wave).max()
            rows.append((wave / max(peak, 1e-9)).astype(np.float32))
        bank.append(rows)
    return bank


_SAW = _build_saw_bank()


def _band_of(freq):
    k = int(np.floor(np.log2(max(freq, 1.0) / _BAND_LO)))
    return int(np.clip(k, 0, _BANDS - 1))


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------

class Voice:
    """A single sounding note.

    `delay` lets the scheduler place a strike part-way into a block, which is
    how two notes struck 8 ms apart stay 8 ms apart instead of collapsing onto
    the same camera frame.

    A voice that sustains is held open until `release` is called. One that does
    not simply ignores it and rings out on its own.
    """

    GAIN_K = 0.22          # per block: about a 50 ms glide to a new level

    def __init__(self, delay=0, gain=1.0):
        self.delay = int(delay)
        self.done = False
        self.gain = float(gain)
        self.gain_target = float(gain)

    def add_into(self, out):
        n = out.shape[0]
        if self.delay >= n:
            self.delay -= n
            return
        start, self.delay = self.delay, 0
        self._render(out, start)

    def retarget(self, gain=None, bright=None):
        """Move a held note to a new level without restarting it.

        Restarting is what makes a gesture instrument sound like a machine:
        lift your hand and the chord should get louder, not be struck again.
        """
        if gain is not None:
            self.gain_target = float(gain)

    def release(self):
        """Let go. Voices that decay on their own ignore this."""

    @property
    def held(self):
        return False

    def _ramp(self, n):
        """Gain interpolated across the block, so level changes never step."""
        g0 = self.gain
        self.gain += (self.gain_target - self.gain) * self.GAIN_K
        if abs(self.gain - self.gain_target) < 1e-5:
            self.gain = self.gain_target
        if g0 == self.gain:
            return g0
        return g0 + (self.gain - g0) * _RAMP[:n]

    def _render(self, out, start):  # pragma: no cover
        raise NotImplementedError


class Plucked(Voice):
    """Karplus-Strong: harp and guitar.

    The loop filter reads `buf[i]` and `buf[i-1]` -- both written one full lap
    ago -- so a chunk of up to `L` samples has no intra-chunk dependency and
    becomes a single vector add.
    """

    def __init__(self, freq, amp, bright, t60, pick=0.0, delay=0):
        super().__init__(delay)
        freq = max(freq, 20.0)
        length = max(8, int(round(SR / freq)))
        exc = _rng.standard_normal(length).astype(np.float32)

        # Moving average lowpasses the excitation: dark plucks start dull.
        w = 1 + int(round((1.0 - bright) * 7))
        if w > 1:
            pad = np.concatenate([exc[-w:], exc])
            c = np.cumsum(pad, dtype=np.float32)
            exc = ((c[w:] - c[:-w]) / w).astype(np.float32)

        # Comb notch at the pick position: a string picked near the bridge
        # loses the harmonics that have a node there.
        if pick > 0.0:
            k = max(1, int(length * pick))
            exc = exc - np.roll(exc, k) * 0.75

        peak = float(np.abs(exc).max())
        self.buf = exc * (amp / max(peak, 1e-9))
        self.idx = 0
        self.prev = float(self.buf[-1])
        # One lap of the delay line is one period, so the loop runs `freq` times
        # a second. Solving g**(t60*freq) = 1e-3 for g keeps the decay time the
        # same across the range; with a fixed g, top strings die in a blink.
        g = 10.0 ** (-3.0 / (max(t60, 0.05) * freq))
        self.g = float(np.clip(g, 0.80, 0.99965)) * 0.5
        self.floor = amp * 0.0012 + 2e-5

    def _render(self, out, start):
        buf = self.buf
        length = buf.shape[0]
        i, g, n = self.idx, self.g, out.shape[0]
        pos = start
        while pos < n:
            c = min(n - pos, length - i)
            seg = buf[i:i + c]
            out[pos:pos + c] += seg
            rolled = np.empty(c, dtype=np.float32)
            rolled[0] = self.prev
            if c > 1:
                rolled[1:] = seg[:c - 1]
            self.prev = float(seg[c - 1])
            buf[i:i + c] = (seg + rolled) * g
            i = 0 if i + c >= length else i + c
            pos += c
        self.idx = i
        if float(np.abs(buf).max()) < self.floor:
            self.done = True


class Additive(Voice):
    """Summed partials with per-partial exponential decay: piano, kalimba, bell.

    The decay is evaluated once per block and interpolated across it; over
    11.6 ms the error against a true exponential is far below the noise floor.

    With `sustain > 0` the partials fall toward a floor instead of to nothing
    and the voice holds there until released -- a piano with the pedal down,
    or a bell that keeps singing while the pose is held.
    """

    def __init__(self, freq, amp, ratios, gains, taus, attack, inharm=0.0,
                 delay=0, sustain=0.0, release=0.4):
        # Amplitude belongs to the voice gain, never to the partials. Scaling
        # the partials too would apply it twice the moment anything retargets.
        super().__init__(delay, gain=amp)
        r = np.asarray(ratios, dtype=np.float64)
        if inharm:
            r = r * np.sqrt(1.0 + inharm * r * r)
        f = freq * r
        keep = f < SR * 0.46
        if not keep.any():
            keep = np.zeros_like(f, dtype=bool)
            keep[0] = True
        self.inc = (f[keep] / SR).astype(np.float64)
        self.phase = _rng.random(int(keep.sum())).astype(np.float64)
        self.g = np.asarray(gains, dtype=np.float32)[keep].copy()
        # `taus` are -60 dB times in seconds, which is how the ear hears decay.
        t60 = np.maximum(np.asarray(taus, dtype=np.float64)[keep], 0.02)
        self.dec = np.power(10.0, -3.0 / (t60 * SR)).astype(np.float32)
        self.att = max(1, int(attack * SR))
        self.t = 0
        self.floor = 8e-4          # on the partial envelope, which is unit-scale
        self.rest = (self.g * float(sustain)).astype(np.float32)
        self.sustains = sustain > 0.0
        self.rel_len = max(1, int(release * SR))
        self.rel = -1

    @property
    def held(self):
        return self.sustains and self.rel < 0

    def release(self):
        if self.sustains and self.rel < 0:
            self.rel = 0

    def _render(self, out, start):
        n = out.shape[0] - start
        t = _ARANGE[:n]
        g0 = self.g
        dec = self.dec ** n
        g1 = (self.rest + (g0 - self.rest) * dec).astype(np.float32)
        env = g0[:, None] + (g1 - g0)[:, None] * _RAMP[None, :n]
        ph = self.phase[:, None] + self.inc[:, None] * t[None, :]
        idx = (ph * _TN).astype(np.int32) & _TMASK
        y = np.einsum('ij,ij->j', _SINE[idx], env)

        if self.t < self.att:
            a = (self.t + t) * (1.0 / self.att)
            np.clip(a, 0.0, 1.0, out=a)
            y = y * a.astype(np.float32)
        if self.rel >= 0:
            r = 1.0 - (self.rel + t) * (1.0 / self.rel_len)
            np.clip(r, 0.0, 1.0, out=r)
            y = y * (r * r).astype(np.float32)
            self.rel += n
            if self.rel >= self.rel_len:
                self.done = True

        out[start:] += y * self._ramp(n)
        self.phase = (self.phase + self.inc * n) % 1.0
        self.g = g1
        self.t += n
        if not self.sustains and float(g1.max()) < self.floor:
            self.done = True


class Bowed(Voice):
    """Band-limited sawtooth through a bow-stroke envelope: violin.

    Brightness picks a table with a different harmonic rolloff rather than
    running a filter, so the tone control costs nothing at render time.
    """

    VIB_HZ = 5.3
    VIB_DEPTH = 0.0055
    ATTACK = 0.10          # seconds for the bow to bite
    RELEASE = 0.34         # seconds for it to lift

    def __init__(self, freq, amp, bright, damp, delay=0):
        super().__init__(delay, gain=amp)
        lvl = int(np.clip(round(bright * (_BRIGHTS - 1)), 0, _BRIGHTS - 1))
        self.tab = _SAW[lvl][_band_of(freq)]
        self.inc = freq / SR
        self.phase = 0.0
        self.vph = float(_rng.random())
        self.t = 0
        self.att = max(1, int(self.ATTACK * SR))
        self.rel_len = max(1, int(self.RELEASE * SR))
        self.rel = -1

    @property
    def held(self):
        return self.rel < 0

    def release(self):
        if self.rel < 0:
            self.rel = 0

    def _render(self, out, start):
        n = out.shape[0] - start
        t = _ARANGE[:n]
        abs_t = self.t + t

        # Vibrato fades in, the way a player settles into a held note.
        fade = np.clip(abs_t * (1.0 / (0.35 * SR)), 0.0, 1.0)
        vph = self.vph + (self.VIB_HZ / SR) * t
        vib = 1.0 + self.VIB_DEPTH * fade * _SINE[(vph * _TN).astype(np.int64) & _TMASK]

        ph = self.phase + np.cumsum(self.inc * vib)
        idx = (ph * _TN).astype(np.int64) & _TMASK

        env = np.clip(abs_t * (1.0 / self.att), 0.0, 1.0)
        env = env * env * (3.0 - 2.0 * env)          # smoothstep on the attack
        if self.rel >= 0:
            r = np.clip(1.0 - (self.rel + t) * (1.0 / self.rel_len), 0.0, 1.0)
            env = env * r * r
            self.rel += n
            if self.rel >= self.rel_len:
                self.done = True

        out[start:] += self.tab[idx] * (env * self._ramp(n)).astype(np.float32)
        self.phase = float(ph[-1] % 1.0)
        self.vph = float((vph[-1] + self.VIB_HZ / SR) % 1.0)
        self.t += n


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------

def _harmonics(k, rolloff, tau0, tau_fall):
    n = np.arange(1, k + 1, dtype=np.float64)
    return n, 1.0 / n ** rolloff, tau0 / n ** tau_fall


class Instrument:
    """Turns a note request into a Voice. `bright` and `damp` come from the
    player's hand and are the only per-note timbre controls.

    `sustains` says which of two behaviours the instrument has. A sustaining
    one holds while the pose is held and is released when the hand changes or
    leaves; a struck one is fired once per change and rings out on its own.
    Both are driven by the same gesture -- only the envelope differs.
    """

    def __init__(self, name, colour, build, gain=1.0, sustains=False):
        self.name = name
        self.colour = colour           # BGR, used by the HUD
        self._build = build
        self.gain = gain
        self.sustains = sustains

    def voice(self, freq, amp, bright, damp, delay=0):
        return self._build(freq, amp * self.gain, bright, damp, delay)


_FREF = 261.6      # middle C, the reference for decay tilt


def _tilt(freq, power):
    """Real strings damp faster the higher they are pitched."""
    return (_FREF / max(freq, 20.0)) ** power


def _harp(freq, amp, bright, damp, delay):
    return Plucked(freq, amp, 0.55 + 0.45 * bright,
                   (1.1 + 2.1 * damp) * _tilt(freq, 0.30), 0.0, delay)


def _guitar(freq, amp, bright, damp, delay):
    return Plucked(freq, amp, 0.30 + 0.45 * bright,
                   (0.7 + 1.5 * damp) * _tilt(freq, 0.35), 0.14, delay)


def _piano(freq, amp, bright, damp, delay):
    # Struck, then held near the strike: a piano with the pedal down. The
    # sustain sits high and the decay toward it is slow, so a chord keeps its
    # body for a second or two instead of thinning out straight after the hit,
    # and the long release lets one chord ring on under the next.
    n, g, t60 = _harmonics(8, 1.15 - 0.35 * bright, 2.6 + 3.0 * damp, 0.72)
    g = g * np.exp(-n * 0.045)
    return Additive(freq, amp, n, g, t60 * _tilt(freq, 0.22), 0.004,
                    inharm=0.00028, delay=delay, sustain=0.55, release=1.4)


def _violin(freq, amp, bright, damp, delay):
    return Bowed(freq, amp, 0.25 + 0.75 * bright, damp, delay)


def _kalimba(freq, amp, bright, damp, delay):
    n = np.array([1.0, 2.0, 3.02, 5.1, 7.3])
    g = np.array([1.0, 0.30, 0.16, 0.07, 0.035]) * (0.55 + 0.75 * bright)
    t60 = np.array([1.0, 0.55, 0.33, 0.18, 0.11]) * (0.6 + 1.1 * damp) * _tilt(freq, 0.25)
    return Additive(freq, amp, n, g, t60, 0.002, delay=delay)


def _bell(freq, amp, bright, damp, delay):
    n = np.array([0.50, 1.0, 1.19, 1.56, 2.0, 2.51, 2.66, 3.01])
    g = np.array([0.55, 1.0, 0.42, 0.34, 0.55, 0.22, 0.18, 0.13]) * (0.5 + 0.8 * bright)
    t60 = np.array([1.0, 0.85, 0.60, 0.46, 0.52, 0.32, 0.26, 0.20])
    t60 = t60 * (2.4 + 4.6 * damp) * _tilt(freq, 0.18)
    return Additive(freq, amp, n, g, t60, 0.004, delay=delay,
                    sustain=0.38, release=0.9)


# The gains are loudness-matched, not guesses: `python -m airharp.calibrate`
# renders one note per instrument across three octaves and reports the trim
# needed to bring each to the same one-second RMS.
INSTRUMENTS = [                 # colours are BGR, the order OpenCV draws in
    Instrument("Harp",    (130, 205, 245), _harp,    1.46),                  # gold
    Instrument("Guitar",  (80, 140, 230), _guitar,  1.21),                   # copper
    Instrument("Piano",   (250, 235, 225), _piano,   0.19, sustains=True),   # cool white
    Instrument("Violin",  (250, 150, 150), _violin,  0.24, sustains=True),   # periwinkle
    Instrument("Kalimba", (190, 240, 150), _kalimba, 0.49),                  # mint
    Instrument("Bells",   (225, 170, 245), _bell,    0.21, sustains=True),   # violet
]


# ---------------------------------------------------------------------------
# reverb
# ---------------------------------------------------------------------------

class Reverb:
    """Schroeder reverb. Same chunking trick as the string: each comb and
    allpass reads only taps written a full delay ago."""

    COMBS = (1557, 1617, 1491, 1422)
    ALLPASS = (225, 556)

    def __init__(self, rt60=2.0, mix=0.26):
        self.mix = mix
        self.cb = [np.zeros(d, dtype=np.float32) for d in self.COMBS]
        self.ci = [0] * len(self.COMBS)
        self.cg = [float(10.0 ** (-3.0 * d / (rt60 * SR))) for d in self.COMBS]
        self.ab = [np.zeros(d, dtype=np.float32) for d in self.ALLPASS]
        self.ai = [0] * len(self.ALLPASS)
        self.ag = 0.5

    def process(self, x):
        n = x.shape[0]
        wet = np.zeros(n, dtype=np.float32)
        for k, buf in enumerate(self.cb):
            length, i, g, pos = buf.shape[0], self.ci[k], self.cg[k], 0
            while pos < n:
                c = min(n - pos, length - i)
                seg = buf[i:i + c]
                wet[pos:pos + c] += seg
                buf[i:i + c] = x[pos:pos + c] + seg * g
                i = 0 if i + c >= length else i + c
                pos += c
            self.ci[k] = i
        wet *= 0.25
        for k, buf in enumerate(self.ab):
            length, i, g, pos = buf.shape[0], self.ai[k], self.ag, 0
            while pos < n:
                c = min(n - pos, length - i)
                v_old = buf[i:i + c].copy()
                v_new = wet[pos:pos + c] + v_old * g
                buf[i:i + c] = v_new
                wet[pos:pos + c] = v_old - v_new * g
                i = 0 if i + c >= length else i + c
                pos += c
            self.ai[k] = i
        return x + wet * self.mix


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class Engine:
    """Owns the voice pool, the producer thread and the output stream.

    `silent=True` (or a missing audio device) still keeps every public method
    working, so the rest of the app never has to ask whether sound is available.
    """

    def __init__(self, device=None, reverb=True, silent=False):
        self.voices = []
        self.pending = deque()
        self.groups = {}          # hand label -> {note id: held voice}
        self.instrument = 0
        self.master = 0.62
        self.reverb = Reverb() if reverb else None
        self.q = queue.Queue(maxsize=QUEUE_BLOCKS)
        self.underruns = 0
        self.peak = 0.0
        self._run = False
        self._thread = None
        self._stream = None
        self._device = device
        self.silent = silent
        self.status = "silent" if silent else "stopped"

    # -- control -----------------------------------------------------------
    # Commands are queued and applied on the audio thread, so the frame loop
    # never touches a voice that is mid-render.

    def pluck(self, freq, amp, bright, damp, delay=0, instrument=None):
        idx = self.instrument if instrument is None else instrument
        inst = INSTRUMENTS[idx % len(INSTRUMENTS)]
        self.pending.append(("strike", inst, (float(freq),), float(amp),
                             float(bright), float(damp), int(delay), None))

    def set_chord(self, key, freqs, amp, bright, damp, delay=0):
        """Play these notes for `key`, holding or striking as the instrument
        demands. Notes already sounding for this key keep sounding."""
        inst = INSTRUMENTS[self.instrument]
        kind = "chord" if inst.sustains else "strike"
        self.pending.append((kind, inst, tuple(float(f) for f in freqs),
                             float(amp), float(bright), float(damp),
                             int(delay), key))

    def set_level(self, key, amp, bright):
        """Follow the hand's height without restarting anything."""
        self.pending.append(("level", None, (), float(amp), float(bright),
                             0.0, 0, key))

    def release(self, key):
        self.pending.append(("release", None, (), 0.0, 0.0, 0.0, 0, key))

    def release_all(self):
        # Queued, not done here: `groups` is only ever filled on the audio
        # thread, so releasing by walking it from this side would miss any
        # chord still sitting in `pending` and leave it hanging.
        self.pending.append(("release_all", None, (), 0.0, 0.0, 0.0, 0, None))

    def set_instrument(self, i):
        i = int(i) % len(INSTRUMENTS)
        if i != self.instrument:
            self.release_all()      # never leave the old instrument droning
        self.instrument = i

    @property
    def instrument_name(self):
        return INSTRUMENTS[self.instrument].name

    @property
    def colour(self):
        return INSTRUMENTS[self.instrument].colour

    # -- rendering ---------------------------------------------------------
    @staticmethod
    def _spread(amp, count):
        """Five notes at full level is five times the amplitude. Backing off
        with the square root keeps a big chord about as loud as a small one,
        the way more strings on an instrument do not make it five times louder.
        """
        return amp / max(count, 1) ** 0.55

    def _apply(self, cmd):
        kind, inst, freqs, amp, bright, damp, delay, key = cmd

        if kind == "release":
            for voice in self.groups.pop(key, {}).values():
                voice.release()
            return

        if kind == "release_all":
            for group in self.groups.values():
                for voice in group.values():
                    voice.release()
            self.groups.clear()
            return

        if kind == "level":
            group = self.groups.get(key)
            if not group:
                return
            # Must arrive at exactly the amplitude `set_chord` would have used,
            # or simply following the hand jumps the level.
            inst = INSTRUMENTS[self.instrument]
            each = self._spread(amp, len(group)) * inst.gain
            for voice in group.values():
                voice.retarget(gain=each, bright=bright)
            return

        if kind == "strike":
            each = self._spread(amp, len(freqs))
            for i, f in enumerate(freqs):
                self._add(inst.voice(f, each, bright, damp, delay))
            return

        # kind == "chord": hold the notes, and keep the ones already sounding.
        group = self.groups.setdefault(key, {})
        each = self._spread(amp, len(freqs))
        want = {int(round(f * 4)): f for f in freqs}
        for note in [k for k in group if k not in want]:
            group.pop(note).release()
        for note, f in want.items():
            voice = group.get(note)
            if voice is not None and voice.held:
                voice.retarget(gain=each, bright=bright)
            else:
                voice = inst.voice(f, each, bright, damp, delay)
                group[note] = voice
                self._add(voice)

    def _add(self, voice):
        if len(self.voices) >= MAX_VOICES:
            self._drop_oldest()
        self.voices.append(voice)

    def _drop_oldest(self):
        """Steal a voice. Prefer one already ringing out over one being held,
        so a chord you are holding is the last thing to be sacrificed."""
        for i, v in enumerate(self.voices):
            if not v.held:
                self.voices.pop(i)
                return
        stolen = self.voices.pop(0)
        for group in self.groups.values():
            for note, voice in list(group.items()):
                if voice is stolen:
                    del group[note]

    def render_block(self, n=BLOCK):
        while self.pending:
            self._apply(self.pending.popleft())

        out = np.zeros(n, dtype=np.float32)
        if self.voices:
            for v in self.voices:
                v.add_into(out)
            if any(v.done for v in self.voices):
                self.voices = [v for v in self.voices if not v.done]
                for group in self.groups.values():
                    for note in [k for k, v in group.items() if v.done]:
                        del group[note]

        if self.reverb is not None:
            out = self.reverb.process(out)
        out *= self.master
        # Soft clip: a fistful of simultaneous plucks should compress, not crack.
        np.tanh(out, out=out)
        p = float(np.abs(out).max()) if n else 0.0
        self.peak = max(p, self.peak * 0.86)
        return out

    def _produce(self):
        while self._run:
            blk = self.render_block()
            while self._run:
                try:
                    self.q.put(blk, timeout=0.1)   # blocking here is the clock
                    break
                except queue.Full:
                    continue

    def _callback(self, outdata, frames, time_info, status):
        if status:
            self.underruns += 1
        try:
            blk = self.q.get_nowait()
        except queue.Empty:
            outdata[:] = 0.0
            self.underruns += 1
            return
        if blk.shape[0] != frames:                 # device asked for another size
            blk = np.resize(blk, frames)
        outdata[:, 0] = blk
        outdata[:, 1] = blk

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self.silent:
            return self
        try:
            import sounddevice as sd
        except Exception as exc:                   # pragma: no cover
            self.silent = True
            self.status = "no sounddevice (%s)" % exc.__class__.__name__
            return self
        self._run = True
        self._thread = threading.Thread(target=self._produce, daemon=True,
                                        name="airharp-audio")
        self._thread.start()
        try:
            self._stream = sd.OutputStream(
                samplerate=SR, blocksize=BLOCK, channels=2, dtype="float32",
                device=self._device, latency="low", callback=self._callback,
            )
            self._stream.start()
            self.status = "running"
        except Exception as exc:                   # pragma: no cover
            self._run = False
            self.silent = True
            self.status = "audio unavailable (%s)" % exc
        return self

    def stop(self):
        self._run = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
