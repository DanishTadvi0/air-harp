"""The synthesis engine: does it make sound, stay in range, and stay cheap."""

import numpy as np
import pytest

from airharp.audio import (
    BLOCK, INSTRUMENTS, MAX_VOICES, SR, Additive, Bowed, Engine, Plucked, Reverb,
)

CHORD = (261.6, 329.6, 392.0)


def render(engine, seconds):
    return np.concatenate([engine.render_block()
                           for _ in range(int(seconds * SR / BLOCK))])


def t60(buf):
    """Seconds until the envelope falls 60 dB below its peak."""
    env = np.abs(buf[:len(buf) // BLOCK * BLOCK].reshape(-1, BLOCK)).max(axis=1)
    below = env < env.max() * 1e-3
    return (int(np.argmax(below)) * BLOCK / SR) if below.any() else float("inf")


@pytest.fixture
def engine():
    return Engine(silent=True)


# -- every instrument -------------------------------------------------------

@pytest.mark.parametrize("index,inst", list(enumerate(INSTRUMENTS)))
def test_every_instrument_makes_a_finite_audible_chord(index, inst, engine):
    engine.set_instrument(index)
    engine.set_chord("R", CHORD, 0.8, 0.6, 0.6)
    buf = render(engine, 1.5)
    assert np.isfinite(buf).all(), f"{inst.name} produced NaN or inf"
    assert np.abs(buf).max() > 0.02, f"{inst.name} was silent"
    assert np.abs(buf).max() <= 1.0, f"{inst.name} clipped"


@pytest.mark.parametrize("index,inst", list(enumerate(INSTRUMENTS)))
def test_every_instrument_eventually_frees_its_voices(index, inst, engine):
    """A struck instrument frees itself; a held one frees itself once let go.
    Either way nothing may accumulate, or the voice pool silently fills up."""
    engine.set_instrument(index)
    engine.set_chord("R", CHORD, 0.8, 0.6, 0.6)
    render(engine, 3.0)
    if inst.sustains:
        assert engine.voices, f"{inst.name} should still be holding"
        engine.release("R")
    render(engine, 12.0)
    assert engine.voices == [], f"{inst.name} leaked a voice"
    assert not any(engine.groups.values()), f"{inst.name} leaked a group entry"


def test_the_required_three_are_all_here():
    names = [i.name for i in INSTRUMENTS]
    assert {"Piano", "Violin", "Guitar"} <= set(names)
    assert len(names) == len(set(names)) == 8


def test_both_kinds_of_instrument_exist():
    holds = [i.name for i in INSTRUMENTS if i.sustains]
    rings = [i.name for i in INSTRUMENTS if not i.sustains]
    assert holds and rings
    assert {"Violin", "Piano"} <= set(holds)
    assert {"Harp", "Guitar"} <= set(rings)


def _levels(sustaining):
    """One-second level for struck instruments, held level for sustaining ones.
    Comparing a bell's sustain against a pluck's transient is meaningless."""
    out = {}
    for i, inst in enumerate(INSTRUMENTS):
        if inst.sustains != sustaining:
            continue
        eng = Engine(silent=True, reverb=False)
        eng.set_instrument(i)
        eng.set_chord("R", CHORD, 0.8, 0.6, 0.6)
        buf = render(eng, 2.5)
        window = buf[-SR // 2:] if sustaining else buf[:SR]
        out[inst.name] = float(np.sqrt((window ** 2).mean()))
    return out


def test_struck_instruments_are_loudness_matched():
    """Switching instrument must not jump the volume."""
    levels = _levels(sustaining=False)
    assert max(levels.values()) / min(levels.values()) < 1.35, levels


def test_held_instruments_are_loudness_matched():
    levels = _levels(sustaining=True)
    assert max(levels.values()) / min(levels.values()) < 1.35, levels


def test_a_held_chord_sits_above_a_struck_one():
    """The two families are deliberately not matched to each other.

    A pluck spends its level on a transient -- peaks near full scale, average
    far below it -- while a pad is nearly all average and sits on ten-plus dB
    of unused headroom. Matching them by RMS makes the pad sound faint next to
    the harp even though a meter says they agree. So held sits above struck,
    by enough to be present and not so much that switching is a jump scare.
    """
    ring = np.mean(list(_levels(sustaining=False).values()))
    hold = np.mean(list(_levels(sustaining=True).values()))
    assert 1.4 < hold / ring < 3.2, (ring, hold)


def test_nothing_clips_even_with_both_hands_flat_out():
    """The knee has to hold the worst case a player can actually produce."""
    from airharp.audio import _KNEE
    for i in range(len(INSTRUMENTS)):
        eng = Engine(silent=True)
        eng.set_instrument(i)
        for hand, notes in (("L", (110.0, 138.6, 164.8, 220.0, 277.2)),
                            ("R", (164.8, 207.7, 246.9, 329.6, 415.3))):
            eng.set_chord(hand, notes, 1.0, 1.0, 1.0)
        buf = render(eng, 2.5)
        assert np.isfinite(buf).all(), INSTRUMENTS[i].name
        assert np.abs(buf).max() <= 1.0, INSTRUMENTS[i].name


def test_normal_playing_never_reaches_the_limiter():
    """A bare tanh bends quiet passages to buy headroom they never needed.
    With a knee, ordinary playing should pass through untouched."""
    from airharp.audio import _KNEE
    for i in range(len(INSTRUMENTS)):
        eng = Engine(silent=True)
        eng.set_instrument(i)
        eng.set_chord("h", CHORD, 0.7, 0.6, 0.6)
        buf = render(eng, 2.5)
        touched = float((np.abs(buf) > _KNEE).mean())
        assert touched < 0.005, (
            f"{INSTRUMENTS[i].name}: {touched*100:.2f}% of samples limited "
            f"during ordinary playing")


def test_both_hands_playing_flat_out_stays_in_range(engine):
    for hand in ("Left", "Right"):
        engine.set_chord(hand, (130.8, 164.8, 196.0, 261.6, 329.6), 1.0, 1.0, 1.0)
    buf = render(engine, 2.0)
    assert np.isfinite(buf).all()
    assert np.abs(buf).max() <= 1.0


def test_a_big_chord_is_not_five_times_louder_than_a_small_one(engine):
    levels = []
    for notes in ((261.6,), CHORD, (130.8, 164.8, 196.0, 261.6, 329.6)):
        eng = Engine(silent=True, reverb=False)
        eng.set_chord("R", notes, 0.8, 0.6, 0.6)
        levels.append(float(np.sqrt((render(eng, 1.0) ** 2).mean())))
    assert max(levels) / min(levels) < 2.0, levels


# -- decay behaviour --------------------------------------------------------

def test_high_strings_ring_nearly_as_long_as_low_ones(engine):
    """A fixed Karplus-Strong loop gain makes the top of the range die in a
    blink, because a short delay line laps far more often per second."""
    times = []
    for f in (130.8, 261.6, 523.3):
        eng = Engine(silent=True, reverb=False)
        eng.pluck(f, 0.8, 0.6, 0.8)
        times.append(t60(render(eng, 8.0)))
    assert all(np.isfinite(times))
    assert min(times) > max(times) * 0.45, f"decay too uneven: {times}"


def test_open_hand_rings_longer_than_a_closed_one():
    eng_open = Engine(silent=True, reverb=False)
    eng_open.pluck(261.6, 0.8, 0.6, 1.0)
    eng_shut = Engine(silent=True, reverb=False)
    eng_shut.pluck(261.6, 0.8, 0.6, 0.0)
    assert t60(render(eng_open, 8.0)) > t60(render(eng_shut, 8.0)) * 1.4


def test_additive_decay_matches_the_requested_t60():
    v = Additive(220.0, 0.8, [1.0], [1.0], [1.5], attack=0.001)
    buf = np.zeros(0, dtype=np.float32)
    chunks = []
    for _ in range(int(4.0 * SR / BLOCK)):
        blk = np.zeros(BLOCK, dtype=np.float32)
        v.add_into(blk)
        chunks.append(blk)
    buf = np.concatenate(chunks)
    assert t60(buf) == pytest.approx(1.5, rel=0.12)


# -- sub-frame scheduling ---------------------------------------------------

def test_a_delayed_pluck_starts_where_it_was_told(engine):
    engine.reverb = None
    engine.pluck(261.6, 0.9, 0.6, 0.6, delay=1000)
    buf = render(engine, 0.2)
    assert np.abs(buf[:990]).max() < 1e-6, "sound before its time"
    assert np.abs(buf[1000:1400]).max() > 0.01, "never arrived"


def test_delay_spanning_several_blocks_is_counted_down(engine):
    engine.reverb = None
    engine.pluck(261.6, 0.9, 0.6, 0.6, delay=BLOCK * 3 + 7)
    assert np.abs(np.concatenate([engine.render_block() for _ in range(3)])).max() < 1e-6
    assert np.abs(engine.render_block()).max() > 0.01


# -- band-limited wavetables ------------------------------------------------

def test_the_bowed_voice_does_not_alias():
    """A naive sawtooth at 1.5 kHz folds its upper harmonics back down into
    the audible range as inharmonic junk. The band-limited table must not."""
    f0 = 1500.0
    v = Bowed(f0, 0.9, 1.0, 2.0)
    chunks = []
    for _ in range(int(1.2 * SR / BLOCK)):
        blk = np.zeros(BLOCK, dtype=np.float32)
        v.add_into(blk)
        chunks.append(blk)
    buf = np.concatenate(chunks)[SR // 8: SR // 8 + 32768]

    spec = np.abs(np.fft.rfft(buf * np.hanning(len(buf))))
    freqs = np.fft.rfftfreq(len(buf), 1 / SR)
    total = float((spec ** 2).sum())
    below = float((spec[freqs < f0 * 0.85] ** 2).sum())
    assert below / total < 1e-3, "energy below the fundamental means aliasing"


# -- reverb -----------------------------------------------------------------

def test_reverb_decays_instead_of_running_away():
    rev = Reverb()
    tail = [rev.process(np.zeros(BLOCK, dtype=np.float32))
            if i else rev.process(np.ones(BLOCK, dtype=np.float32) * 0.5)
            for i in range(int(12.0 * SR / BLOCK))]
    tail = np.concatenate(tail)
    assert np.isfinite(tail).all()
    early = np.abs(tail[SR // 2: SR]).max()
    late = np.abs(tail[10 * SR: 11 * SR]).max()
    assert late < early * 0.1, "the tail must die away"


def test_reverb_adds_a_tail_after_the_note_stops(engine):
    dry = Engine(silent=True, reverb=False)
    wet = Engine(silent=True, reverb=True)
    for eng in (dry, wet):
        eng.pluck(261.6, 0.8, 0.6, 0.2)
    d, w = render(dry, 4.0), render(wet, 4.0)
    assert np.abs(w[3 * SR:]).max() > np.abs(d[3 * SR:]).max()


# -- engine plumbing --------------------------------------------------------

def test_voice_count_is_capped(engine):
    for i in range(MAX_VOICES * 3):
        engine.pluck(200.0 + 7 * i, 0.5, 0.6, 0.9)
    engine.render_block()
    assert len(engine.voices) <= MAX_VOICES


def test_silence_when_nothing_is_playing(engine):
    assert np.abs(render(engine, 0.3)).max() == 0.0


def test_instrument_selection_wraps_and_reports(engine):
    engine.set_instrument(len(INSTRUMENTS))
    assert engine.instrument == 0
    engine.set_instrument(3)
    assert engine.instrument_name == INSTRUMENTS[3].name
    assert engine.colour == INSTRUMENTS[3].colour


def test_silent_engine_still_honours_the_whole_api():
    """No audio device must never mean a broken app."""
    eng = Engine(silent=True)
    with eng:
        eng.set_instrument(2)
        eng.pluck(440.0, 0.8, 0.5, 0.5)
        assert np.isfinite(eng.render_block()).all()
    assert eng.status == "silent"


def test_render_block_accepts_an_odd_size(engine):
    engine.pluck(261.6, 0.8, 0.6, 0.6)
    assert engine.render_block(333).shape == (333,)


def test_plucked_handles_absurd_input():
    for freq in (0.0, -5.0, 1e9):
        v = Plucked(freq, 0.8, 0.5, 1.0)
        blk = np.zeros(BLOCK, dtype=np.float32)
        v.add_into(blk)
        assert np.isfinite(blk).all()


def test_render_is_comfortably_faster_than_realtime(engine):
    import time
    for i in range(MAX_VOICES):
        engine.pluck(160.0 + 33.0 * i, 0.6, 0.6, 0.9)
    engine.render_block()
    start = time.perf_counter()
    for _ in range(120):
        engine.render_block()
    per = (time.perf_counter() - start) / 120
    assert per < (BLOCK / SR) * 0.5, f"{per * 1e3:.2f} ms for an 11.6 ms block"


# -- holding and letting go -------------------------------------------------

HOLDERS = [(i, inst) for i, inst in enumerate(INSTRUMENTS) if inst.sustains]
RINGERS = [(i, inst) for i, inst in enumerate(INSTRUMENTS) if not inst.sustains]


@pytest.mark.parametrize("index,inst", HOLDERS)
def test_a_held_chord_is_still_sounding_five_seconds_later(index, inst):
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(index)
    eng.set_chord("R", CHORD, 0.8, 0.6, 0.6)
    buf = render(eng, 5.0)
    late = float(np.sqrt((buf[-SR // 2:] ** 2).mean()))
    assert late > 0.01, f"{inst.name} faded out while the pose was held"


@pytest.mark.parametrize("index,inst", RINGERS)
def test_a_struck_chord_rings_out_on_its_own(index, inst):
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(index)
    eng.set_chord("R", CHORD, 0.8, 0.6, 0.6)
    buf = render(eng, 6.0)
    assert float(np.abs(buf[-SR // 2:]).max()) < 0.005, f"{inst.name} never stopped"


@pytest.mark.parametrize("index,inst", HOLDERS)
def test_releasing_fades_rather_than_cutting(index, inst):
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(index)
    eng.set_chord("R", CHORD, 0.8, 0.6, 0.6)
    render(eng, 1.5)
    eng.release("R")
    tail = render(eng, 0.05)
    assert np.abs(tail).max() > 0.002, f"{inst.name} cut off instantly"
    assert float(np.abs(render(eng, 4.0)[-SR // 2:]).max()) < 1e-3


def test_a_chord_change_keeps_the_notes_that_did_not_change():
    """Two chords a tone apart share notes. Restarting those is what makes a
    gesture instrument sound like a machine gun instead of an instrument."""
    eng = Engine(silent=True)
    eng.set_instrument(3)                       # violin
    eng.set_chord("R", (261.6, 329.6, 392.0), 0.8, 0.6, 0.6)
    render(eng, 0.6)
    before = {id(v) for v in eng.groups["R"].values()}

    eng.set_chord("R", (261.6, 329.6, 440.0), 0.8, 0.6, 0.6)
    render(eng, 0.1)
    kept = sum(1 for v in eng.groups["R"].values() if id(v) in before)
    assert kept == 2, "C and E should not have been re-attacked"
    assert len(eng.groups["R"]) == 3


def test_following_the_hand_height_does_not_restart_anything():
    eng = Engine(silent=True)
    eng.set_instrument(3)
    eng.set_chord("R", CHORD, 0.3, 0.6, 0.6)
    render(eng, 0.4)
    before = {id(v) for v in eng.groups["R"].values()}
    quiet = float(np.sqrt((render(eng, 0.3) ** 2).mean()))

    eng.set_level("R", 1.0, 0.9)
    loud = float(np.sqrt((render(eng, 0.6)[-BLOCK * 20:] ** 2).mean()))
    assert {id(v) for v in eng.groups["R"].values()} == before
    assert loud > quiet * 1.8, "the level should have followed the hand"


def test_a_level_change_glides_instead_of_stepping():
    """A gain that jumps between blocks is an audible click, so the ramp has
    to run inside the block and hand the next one the exact value it ended on."""
    v = Bowed(261.6, 0.2, 0.6, 0.6)
    v.retarget(gain=1.0)
    curve = []
    for _ in range(30):
        r = v._ramp(BLOCK)
        curve.append(np.full(BLOCK, r) if np.isscalar(r) else r)
    curve = np.concatenate(curve)

    assert np.all(np.diff(curve) >= -1e-9), "the glide must not go backwards"
    biggest = float(np.abs(np.diff(curve)).max())
    assert biggest < 0.01, f"gain stepped by {biggest:.4f} between samples"
    assert curve[-1] == pytest.approx(1.0, abs=1e-3), "it must actually arrive"


def test_a_level_change_takes_tens_of_milliseconds():
    """Fast enough to feel like following the hand, slow enough not to zip."""
    v = Bowed(261.6, 0.0, 0.6, 0.6)
    v.retarget(gain=1.0)
    blocks = 0
    while v.gain < 0.632 and blocks < 200:      # one time constant
        v._ramp(BLOCK)
        blocks += 1
    ms = blocks * BLOCK / SR * 1e3
    assert 20.0 < ms < 120.0, f"{ms:.0f} ms"


def test_two_hands_are_held_and_released_separately():
    eng = Engine(silent=True)
    eng.set_instrument(3)
    eng.set_chord("Left", (261.6,), 0.8, 0.6, 0.6)
    eng.set_chord("Right", (392.0,), 0.8, 0.6, 0.6)
    render(eng, 0.5)
    assert set(eng.groups) == {"Left", "Right"}
    eng.release("Left")
    render(eng, 1.0)
    assert set(eng.groups) == {"Right"}
    assert float(np.abs(render(eng, 0.3)).max()) > 0.01, "the right hand stopped too"


def test_switching_instrument_lets_go_of_the_old_one():
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(3)                       # violin, sustains
    eng.set_chord("R", CHORD, 0.8, 0.6, 0.6)
    render(eng, 1.0)
    eng.set_instrument(0)                       # harp, rings
    assert float(np.abs(render(eng, 3.0)[-SR // 2:]).max()) < 1e-3
    assert not any(eng.groups.values())


def test_release_all_lets_go_of_every_hand():
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(3)
    for hand in ("Left", "Right"):
        eng.set_chord(hand, CHORD, 0.8, 0.6, 0.6)
    render(eng, 0.5)
    eng.release_all()
    render(eng, 2.0)
    assert eng.voices == []


def test_releasing_a_hand_that_was_never_playing_is_harmless():
    eng = Engine(silent=True)
    eng.release("nobody")
    eng.set_level("nobody", 0.5, 0.5)
    assert np.abs(render(eng, 0.2)).max() == 0.0


def test_a_held_chord_is_the_last_voice_to_be_stolen():
    """Voice stealing should sacrifice something already dying, not the chord
    the player is holding down."""
    eng = Engine(silent=True)
    eng.set_instrument(3)
    eng.set_chord("R", tuple(200.0 + 30 * i for i in range(6)), 0.5, 0.6, 0.6)
    render(eng, 0.3)
    held = len(eng.groups["R"])
    eng.set_instrument(0)                       # harp: struck, not held
    eng.groups["R"] = eng.groups.get("R", {})
    for i in range(MAX_VOICES * 2):
        eng.pluck(300.0 + 5 * i, 0.4, 0.6, 0.9)
    render(eng, 0.1)
    assert len(eng.voices) <= MAX_VOICES
    assert held == 6


# -- following the hand -----------------------------------------------------

@pytest.mark.parametrize("index,inst", HOLDERS)
def test_following_the_hand_at_one_height_does_not_move_the_level(index, inst):
    """The frame loop calls `set_level` every frame, whether or not the hand
    has moved. If that does not land on exactly the amplitude `set_chord`
    would have used, a held chord lurches the instant you stop changing it --
    and every test that only compares loud against quiet still passes, because
    both ends are wrong by the same factor.
    """
    def run(follow):
        eng = Engine(silent=True, reverb=False)
        eng.set_instrument(index)
        eng.set_chord("h", CHORD, 0.7, 0.6, 0.6)
        render(eng, 0.4)
        if follow:
            for _ in range(30):                 # a second of frames at 30 fps
                eng.set_level("h", 0.7, 0.6)    # same height as the chord
                eng.render_block()
        # Long enough to average out what is deliberately moving -- the
        # detune beating and the synth's slow spectral drift both swing the
        # instantaneous level by several dB on purpose, and a short window
        # cannot tell that apart from the gain fault this test exists to catch.
        buf = render(eng, 6.0)
        return float(np.sqrt((buf[-int(SR * 4.5):] ** 2).mean()))

    alone, followed = run(False), run(True)
    ratio = followed / max(alone, 1e-9)
    assert 0.82 < ratio < 1.22, (
        f"{inst.name}: following the hand changed the level by "
        f"{20 * np.log10(max(ratio, 1e-9)):+.1f} dB")


@pytest.mark.parametrize("index,inst", HOLDERS)
def test_set_level_reaches_the_same_amplitude_as_set_chord(index, inst):
    """Both paths must agree about the instrument's own gain and about how a
    chord's loudness is shared between its notes."""
    eng = Engine(silent=True)
    eng.set_instrument(index)
    eng.set_chord("h", CHORD, 0.55, 0.6, 0.6)
    eng.render_block()
    started = [v.gain_target for v in eng.groups["h"].values()]

    eng.set_level("h", 0.55, 0.6)
    eng.render_block()
    followed = [v.gain_target for v in eng.groups["h"].values()]

    assert started == pytest.approx(followed, rel=1e-6), inst.name


def test_a_quiet_note_is_quiet_because_of_its_gain_not_its_partials():
    """Amplitude lives in one place. If it were baked into the partials as
    well, retargeting would apply it twice."""
    loud = Additive(220.0, 1.0, [1.0, 2.0], [1.0, 0.5], [2.0, 1.0], 0.002, sustain=0.5)
    quiet = Additive(220.0, 0.1, [1.0, 2.0], [1.0, 0.5], [2.0, 1.0], 0.002, sustain=0.5)
    assert loud.g == pytest.approx(quiet.g), "partials must not carry amplitude"
    assert quiet.gain == pytest.approx(0.1)
    assert loud.gain == pytest.approx(1.0)


def test_the_synth_moves_without_drifting_off_level():
    """Its warmth comes from two detuned copies beating and a rolloff that
    swings slowly. Both should colour the sound without walking the level
    somewhere else over time."""
    index = [i for i, x in enumerate(INSTRUMENTS) if x.name == "Synth"][0]
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(index)
    eng.set_chord("h", CHORD, 0.7, 0.6, 0.6)
    buf = render(eng, 16.0)

    win = int(SR * 3.0)
    blocks = [float(np.sqrt((buf[i:i + win] ** 2).mean()))
              for i in range(SR, len(buf) - win, win)]
    assert max(blocks) / min(blocks) < 1.35, f"level wanders: {blocks}"


def test_the_synth_does_not_decay_at_all():
    """Every other instrument models something struck or bowed and dies away.
    The synth is the one where not dying is the point: hold the shape and the
    chord must be as loud eight seconds later as it was at the start."""
    index = [i for i, x in enumerate(INSTRUMENTS) if x.name == "Synth"][0]
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(index)
    eng.set_chord("h", CHORD, 0.7, 0.6, 0.6)
    buf = render(eng, 14.0)

    # Three-second windows: long enough that the detune beating and the slow
    # rolloff swing average out, leaving only a genuine decay if there is one.
    win = int(SR * 3.0)
    early = float(np.sqrt((buf[SR:SR + win] ** 2).mean()))
    late = float(np.sqrt((buf[-win:] ** 2).mean()))
    fall = 20 * np.log10(max(late, 1e-9) / max(early, 1e-9))
    assert abs(fall) < 2.0, f"drifted {fall:+.1f} dB over ten seconds"


def test_the_piano_still_decays_because_it_is_a_piano():
    """The counterpart: adding a pad must not have flattened everything else."""
    index = [i for i, x in enumerate(INSTRUMENTS) if x.name == "Piano"][0]
    eng = Engine(silent=True, reverb=False)
    eng.set_instrument(index)
    eng.set_chord("h", CHORD, 0.7, 0.6, 0.6)
    early = float(np.sqrt((render(eng, 0.5)[-SR // 4:] ** 2).mean()))
    late = float(np.sqrt((render(eng, 7.5)[-SR // 4:] ** 2).mean()))
    assert late < early * 0.85, "a piano chord should settle below its strike"
    assert late > early * 0.3, "but it should not vanish either"
