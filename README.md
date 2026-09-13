# Air Harp

A musical instrument you play with your hands in front of a webcam. The fingers
you raise are the chord. Nothing is worn, nothing is clicked, and there is
nothing to read before you start.

**[Play it in your browser](https://danishtadvi0.github.io/air-harp/)** — no install, nothing to download.
Or run the desktop version:

```bash
python run.py
```

Hold up a hand. One finger plays the first chord of the key, two the second, up
to an open hand for the fifth; index and little finger together reach the last
two. Raise your hand to play louder and brighter, close it to a fist to stop.
All seven shapes are drawn along the foot of the screen the whole time you play.

---

## How it is played

| what you do | what happens |
| --- | --- |
| 1–5 fingers | chords I to V of the key |
| index + little finger | chord vi |
| those two + thumb | chord vii |
| right hand, 1–5 fingers | how full the chord is: one note, a fifth, a triad, a seventh, an open voicing |
| raise / lower a hand | louder and brighter / quieter and warmer |
| close to a fist | stop |
| `1`–`7` | harp, guitar, piano, violin, kalimba, bells, synth |
| `[` `]` | change key · `m` major/minor · `d` stats · `q` quit |

Every chord is built by stacking thirds inside the key, so no combination of
fingers can produce a note that does not belong. There is no wrong note to hit,
which is what lets the thing be played rather than learned.

Three of the seven instruments ring out on their own (harp, guitar, kalimba)
and four hold for as long as you hold the shape (piano, violin, bells, synth).
The screen says which.

The synth is the odd one out: it has no decay whatsoever, so a chord stays
exactly as loud as you left it until you change shape or drop your hand.
Everything else here models something struck or bowed and therefore dies away
— this is the one where not dying is the point. Two oscillators sit a few
cents apart and beat slowly against each other, which is all that separates a
warm pad from a test tone.

---

## Two builds

The instrument exists twice, from one design.

**`docs/`** is the browser version: MediaPipe's WASM build for the landmarks,
Canvas for the drawing, and an AudioWorklet for the synthesis. No build step, no
bundler, no npm — three files served straight off GitHub Pages. The camera never
leaves the tab; there is no server and nothing is uploaded.

**`airharp/`** is the Python version. It is the one with the test suite, the
threading, and the measurements, and it is where the design was worked out.

The parts that do not depend on the platform — the One Euro filter, the
canonical hand frame, the finger counting, the chord theory, the hold-time
stabilisation — are the same code translated. Only the camera, the drawing and
the audio output differ, because only those are platform-bound.

Open [the demo](https://danishtadvi0.github.io/air-harp/#demo) with `#demo` on
the end and it plays itself with a synthetic hand, no camera needed.

---

## Running it

Windows, Python 3.10, one webcam, four dependencies.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

If something does not work, this says which part:

```powershell
.\.venv\Scripts\python.exe -m airharp.selftest
```

It reports the camera, the hand model and the audio device separately, with
frame rates and underrun counts, and saves a rendered frame with `--shot`.

`--silent` runs it with no sound; `--debug` starts with the stats overlay on.
The MediaPipe hand model lives in `assets/hand_landmarker.task` (7.8 MB) and
ships with the repository, so there is nothing to download.

---

## How it works

```
capture thread ──frame──▶ detect thread ──landmarks──▶ frame loop ──chord──▶ audio thread
                                                                              │
                                                                      PortAudio callback
```

### Seeing the hand

**MediaPipe** returns 21 landmarks per hand. Everything after that is mine.

**A One Euro filter** smooths each landmark. A fixed low-pass forces a choice
between visible jitter when the hand is still and visible lag when it moves;
One Euro raises its own cutoff in proportion to measured speed, so it can be
heavily damped at rest and nearly transparent during a fast move. Measured on
this pipeline: it rejects 65% of landmark jitter at rest while holding lag to
14.4 ms during a 1800 px/s sweep. A fixed cutoff fast enough to match that lag
leaves twice the jitter. `beta` is a cutoff in hertz per unit of speed, so the filter
is fed frame widths rather than pixels and the tuning holds at any resolution.

**Canonical hand coordinates** — wrist at the origin, scaled by palm length,
rotated onto the palm axis — are what the finger counting actually runs on.
Raw image coordinates conflate hand shape with where the hand happens to be and
how it is turned. Counting fingers by comparing a fingertip's y to its knuckle's
y, the usual shortcut, starts miscounting the moment the hand tilts. Comparing
distances in the canonical frame does not: the tests assert the count is exact
across ±2.4 radians of rotation and a 5× range of distance from the lens.

**Hysteresis in time.** A raw reading flickers between neighbouring shapes on
almost every hand, and a chord that flickers with it is unplayable. A new shape
must persist 90 ms before it commits, and a hand that blinks out of tracking
keeps playing for 160 ms before it is let go. A test feeds a reading that
alternates between two shapes every frame for two seconds and asserts the chord
never moves — while the screen still shows, faintly, that it is unsure.

**Nothing recognised can silence a note.** A fist is the one gesture that stops
the sound, and that is a deliberate control. No confidence score, threshold or
classifier sits between a hand and a sound.

### Threading

`cap.read()` blocks until the camera hands over a frame, and MediaPipe takes
about as long again. Run both on one thread and the costs add: a 30 fps camera
yields about 17 fps of tracking. Split across two threads joined by a
single-slot mailbox and the slower stage sets the rate — measured 30.5 fps
capture, 29 fps detection. The mailbox holds exactly one item and overwrites,
so a stall drops frames rather than building a queue of stale ones. For an
instrument, late input is worse than missing input.

### Making the sound

Seven instruments, all synthesised — no samples.

- **Harp and guitar** are Karplus–Strong. The loop filter reads only taps
  written a full lap earlier, so a block can be processed in chunks no longer
  than the delay and each chunk collapses to a single vector add. The loop gain
  is solved per note from the decay time you want, because a delay line laps
  `f` times a second: with a fixed gain, the top of the range dies in a blink.
- **Piano, kalimba and bells** are summed partials with per-partial decay,
  evaluated once per block and interpolated across it.
- **Violin** reads a band-limited sawtooth. Brightness picks a table with a
  different harmonic rolloff instead of running a filter, so the tone control
  costs nothing at render time. A naive saw at 1.5 kHz folds its upper
  harmonics back into the audible range; a test asserts the energy below the
  fundamental stays under −30 dB.
- **Synth** is the same band-limited table held flat, with a second detuned
  oscillator for warmth and no decay at all.
- **Reverb** is Schroeder — four combs and two allpasses, vectorised the same
  way as the string.

**The PortAudio callback never synthesises.** A producer thread renders blocks
into a bounded queue and the callback copies one out. `Queue.put` blocking on a
full queue is what paces the producer, so nothing relies on `time.sleep` —
Windows rounds a 0.5 ms sleep up to about 15 ms, which would be three audio
blocks of jitter.

**Held notes are not restarted.** Changing from C–E–G to C–E–A keeps the C and
the E sounding and only starts the A; raising your hand moves the gain of what
is already playing rather than striking it again. Gain moves are interpolated
inside the block and handed to the next one at the exact value they ended on,
so a level change glides over about 50 ms instead of clicking.

The instrument gains are measured, not guessed. `python -m airharp.calibrate`
renders each instrument across three octaves and reports the trim needed to
bring them all to the same perceived loudness; two instruments can have
identical peaks and still be far apart, because a pluck is nearly all transient
and a bell is nearly all sustain.

### Drawing

OpenCV on `uint8`, not NumPy on `float32` — the same HUD written the NumPy way
costs over ten times as much per frame, against a 16 ms budget. The bloom is
blurred at quarter resolution and scaled up; at full resolution it would cost
more than the rest of the frame and look no different. The camera view is dimmed
toward a target mean brightness rather than by a fixed multiply, so it stays
readable in a bright room and in a dark one.

---

## Layout

```
airharp/
  tracking.py   threaded capture and detection, One Euro, canonical coordinates
  poses.py      which fingers are up, which chord that is, when to believe it
  music.py      the key, the seven chords, the voicings
  audio.py      voices, instruments, reverb, the producer thread
  ui.py         the on-screen instrument
  app.py        the frame loop
  calibrate.py  loudness matching and render benchmarks
  selftest.py   camera / model / audio diagnostics
docs/
  index.html    the browser build, served by GitHub Pages
  app.js        camera, tracking, chords, canvas HUD
  synth.js      the synthesis engine as an AudioWorklet
tests/          170 tests
```

2,000 lines across the six Python modules that make up the instrument, 350 more
for the two diagnostic tools, 1,350 of tests, and 1,100 of JavaScript for the
browser build.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

170 tests, no camera and no audio device needed — hands are synthesised as
21-landmark skeletons with real finger curl, and the whole frame loop is driven
through a `step()` that takes a frame and returns a canvas.

They are written against the properties that make it playable rather than
against the implementation: that no finger combination produces a note outside
the key, that a flickering detection never moves the chord, that a blink of
lost tracking does not cut the sound off, that raising a hand does not restart
a held note, that the guide drawn on screen always agrees with the classifier,
and that every instrument stays finite and in range with both hands playing
flat out.

Five real bugs came out of them. The One Euro filter was being fed pixels
while tuned for normalised units, so it was barely filtering at all.
`release_all` read engine state from the control thread while that state is
only written by the audio thread, which let a queued chord survive the release
and hang. And the seventh degree was being named from its third alone, which
called B-D-F a B minor when it is diminished — caught by eye in the browser
build, then pinned down with a test in both.

The last two were the worst, because they lived in the code path the frame
loop runs every frame and the tests sailed straight past them. `set_level` --
which follows your hand height -- passed the raw height through as a voice
gain without the instrument's own gain or the chord-size division, so it
landed nowhere near the amplitude `set_chord` had just used: violin jumped
17 dB, bells lost 4.5. And the additive voice carried amplitude twice, baked
into its partials *and* applied again by the gain ramp, so the first
`set_level` after a chord started scaled it by the amplitude a second time
and the piano lost 3 dB the instant you stopped moving.

The tests missed both for the same reason: they compared a loud hand against
a quiet one, and that ratio stays correct when both ends are wrong by the
same factor. The replacements check absolute level -- that following the hand
at an unchanged height does not move the level at all. Reintroducing either
bug now fails seven tests.

## Known limits

- Webcams lengthen their exposure in low light and drop to 10–15 fps, and every
  missing frame is input lag. The self-test says so when it sees it. More light
  is the fix.
- Detection runs at about 30 ms a frame on CPU, which sets the floor on
  responsiveness.
- The thumb is the least reliable of the five, so the two chords that need it
  are the two hardest to hit cleanly.
- The finger thresholds were fitted to one synthetic hand and have not been
  validated against a range of real ones. A hand that cannot extend past about
  55% of its range will not be read, and nothing on screen explains why.
- The browser build has only been checked on desktop Chrome. The layout assumes
  a landscape window.

## Licence

MIT.
