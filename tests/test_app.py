"""The whole instrument, driven frame by frame with no camera and no window."""

import numpy as np
import pytest

from airharp.app import App, parse_args
from airharp.audio import BLOCK, INSTRUMENTS, SR
from airharp.poses import DEGREE_SHAPES

from conftest import fist, make_hand

W, H = 640, 360
DT = 1 / 30.0


@pytest.fixture
def app():
    args = parse_args(["--silent", "--width", str(W), "--height", str(H)])
    return App(args)


def frame():
    return np.full((H, W, 3), 90, dtype=np.uint8)


class Player:
    """Holds a hand shape in front of the camera for a while."""

    def __init__(self, app, t0=100.0):
        self.app = app
        self.t = t0
        self.started = t0
        self.canvas = None

    def hold(self, hands, frames=8):
        for _ in range(frames):
            self.canvas = self.app.step(frame(), hands, self.t, DT,
                                        self.t - self.started)
            self.t += DT
        return self.canvas

    def shape(self, degree, y=180.0, label="Left", frames=8, right=None, x=None):
        hands = [make_hand(W * 0.95 if x is None else x, y=y,
                           fingers=DEGREE_SHAPES[degree], label=label)]
        if right is not None:
            hands.append(make_hand(500.0, y=y, fingers=fist(right), label="Right"))
        return self.hold(hands, frames)

    def stop(self, frames=10):
        return self.hold([make_hand(320.0, fingers=(False,) * 5, label="Left")],
                         frames)


def audio_of(app, seconds=2.0):
    return np.concatenate([app.engine.render_block()
                           for _ in range(int(seconds * SR / BLOCK))])


# -- playing ----------------------------------------------------------------

def test_holding_a_shape_plays_a_chord(app):
    Player(app).shape(0)
    assert app.chords == 1
    buf = audio_of(app)
    assert np.isfinite(buf).all()
    assert np.abs(buf).max() > 0.05
    assert np.abs(buf).max() <= 1.0


def test_holding_the_same_shape_does_not_replay_it(app):
    Player(app).shape(2, frames=60)
    assert app.chords == 1, "a held shape is one chord, not sixty"


def test_changing_the_shape_plays_a_new_chord(app):
    p = Player(app)
    for degree in range(7):
        p.shape(degree)
    assert app.chords == 7


def test_moving_the_hand_about_does_not_change_the_chord(app):
    """Position is not part of the instrument."""
    p = Player(app)
    for x in (40.0, 200.0, 320.0, 500.0, W - 40.0):
        p.hold([make_hand(x, y=180.0, fingers=DEGREE_SHAPES[3], label="Left")], 6)
    assert app.chords == 1


def test_a_fist_stops_the_sound(app):
    """A closed hand is a deliberate stop -- the only way a gesture is allowed
    to produce silence."""
    app.handle_key(ord("4"))                    # violin, sustains
    p = Player(app)
    p.shape(3)
    audio_of(app, 0.5)
    assert app.engine.groups.get(App.VOICE)

    p.stop()
    tail = audio_of(app, 3.0)
    assert np.abs(tail[-len(tail) // 4:]).max() < 1e-3


def test_drifting_left_quietens_without_restarting(app):
    """The fade has to be the same chord getting quieter, not a new one."""
    app.handle_key(ord("4"))                    # violin
    p = Player(app)
    p.shape(3, x=W * 0.95, frames=10)
    loud = float(np.sqrt((audio_of(app, 0.8)[-8000:] ** 2).mean()))
    voices = {id(v) for v in app.engine.groups[App.VOICE].values()}

    p.shape(3, x=W * 0.30, frames=12)
    quiet = float(np.sqrt((audio_of(app, 0.8)[-8000:] ** 2).mean()))

    assert quiet < loud * 0.5
    assert {id(v) for v in app.engine.groups[App.VOICE].values()} == voices
    assert app.chords == 1, "sliding across is expression, not a new chord"


def test_drifting_to_the_far_left_dies_away(app):
    """Hold the shape, slide left, and it should go silent without ever
    being let go of -- so sliding back brings the same chord in again."""
    app.handle_key(ord("4"))
    p = Player(app)
    p.shape(3, x=W * 0.95, frames=10)
    audio_of(app, 0.5)

    for x in np.linspace(W * 0.95, 0.0, 12):
        p.shape(3, x=float(x), frames=3)
    tail = audio_of(app, 2.0)
    assert np.abs(tail[-len(tail) // 3:]).max() < 5e-3, "should have died away"
    assert app.engine.groups.get(App.VOICE), "but never actually released"
    assert app.chords == 1

    p.shape(3, x=W * 0.95, frames=12)
    back = audio_of(app, 1.0)
    assert np.abs(back).max() > 0.02, "sliding back should bring it in again"


def test_a_low_hand_is_quiet_but_never_silent(app):
    Player(app).shape(4, y=H - 10.0)
    assert np.abs(audio_of(app, 1.5)).max() > 0.01


def test_the_right_hand_makes_the_chord_fuller(app):
    """Left hand names the chord, right hand says how full -- and more fingers
    on the right must actually put more notes in the air."""
    app.handle_key(ord("4"))                    # violin, so the notes are held
    p = Player(app)
    p.shape(0, right=3)
    audio_of(app, 0.3)
    triad = len(app.engine.groups[App.VOICE])

    p.shape(0, right=4)
    audio_of(app, 0.3)
    seventh = len(app.engine.groups[App.VOICE])

    assert triad == 3 and seventh == 4
    assert app.chords == 2, "changing fullness is a new chord"


def test_a_hand_leaving_lets_go_of_the_chord(app):
    app.handle_key(ord("4"))
    p = Player(app)
    p.shape(3)
    p.hold([], 12)
    assert np.abs(audio_of(app, 3.0)[-20000:]).max() < 1e-3


def test_no_hands_at_all_is_silent_and_uneventful(app):
    Player(app).hold([], 20)
    assert app.chords == 0
    assert np.abs(audio_of(app, 0.5)).max() == 0.0


@pytest.mark.parametrize("index,inst", list(enumerate(INSTRUMENTS)))
def test_every_instrument_plays_through_the_whole_loop(index, inst, app):
    app.handle_key(ord(str(index + 1)))
    p = Player(app)
    for degree in (0, 3, 5):
        p.shape(degree)
    assert app.chords == 3, f"{inst.name} played {app.chords} chords"
    buf = audio_of(app, 2.0)
    assert np.isfinite(buf).all()
    assert np.abs(buf).max() > 0.02, f"{inst.name} was silent"
    assert np.abs(buf).max() <= 1.0


# -- drawing ----------------------------------------------------------------

def test_the_canvas_comes_back_the_size_of_the_frame(app):
    canvas = Player(app).shape(2)
    assert canvas.shape == (H, W, 3)
    assert canvas.dtype == np.uint8


def test_the_played_chord_lights_up_and_fades(app):
    Player(app).shape(5)
    assert app.hud.energy[5] > 0.2
    for _ in range(60):
        app.hud.step(0.05, [], None)
    assert app.hud.energy.max() < 0.05


def test_the_stats_overlay_renders(app):
    app.debug = True
    canvas = Player(app).shape(2)
    assert canvas.shape == (H, W, 3)
    assert len(app._debug_lines([])) >= 4


def test_drawing_survives_a_small_window(app):
    """The guide and the bar must still fit rather than indexing off the end."""
    args = parse_args(["--silent", "--width", "320", "--height", "240"])
    small = App(args)
    canvas = small.step(np.full((240, 320, 3), 90, dtype=np.uint8),
                        [make_hand(160.0, y=120.0, fingers=DEGREE_SHAPES[6],
                                   label="Left")],
                        100.0, DT, 0.0)
    assert canvas.shape == (240, 320, 3)


def test_the_hint_goes_away_once_something_is_played(app):
    quiet = Player(app).hold([], 2)
    Player(app).shape(2)
    played = Player(app, t0=200.0).hold([], 2)
    band = slice(H - 190, H - 130)
    assert quiet[band].mean() != pytest.approx(played[band].mean(), abs=0.5)


# -- keys -------------------------------------------------------------------

def test_number_keys_change_instrument(app):
    for i in range(len(INSTRUMENTS)):
        app.handle_key(ord(str(i + 1)))
        assert app.engine.instrument_name == INSTRUMENTS[i].name
    app.handle_key(ord("9"))                    # out of range, ignored
    assert app.engine.instrument_name == INSTRUMENTS[-1].name


def test_transpose_and_scale_keys_retune(app):
    before = list(app.key.roots)
    app.handle_key(ord("]"))
    assert list(app.key.roots) == [b + 1 for b in before]
    app.handle_key(ord("["))
    assert list(app.key.roots) == before
    app.handle_key(ord("m"))
    assert "minor" in app.key.key_name


def test_retuning_mid_chord_does_not_leave_the_old_one_hanging(app):
    app.handle_key(ord("4"))                    # violin
    Player(app).shape(3)
    app.handle_key(ord("m"))
    assert np.abs(audio_of(app, 3.0)[-20000:]).max() < 1e-3


def test_playing_still_works_after_retuning(app):
    app.handle_key(ord("m"))
    Player(app, t0=300.0).shape(2)
    assert app.chords == 1
    assert np.abs(audio_of(app, 1.5)).max() > 0.02


def test_quit_keys_stop_the_loop(app):
    assert app.handle_key(ord("q")) is False
    assert app.handle_key(27) is False
    assert app.handle_key(ord("d")) is True
    assert app.handle_key(ord("r")) is True
