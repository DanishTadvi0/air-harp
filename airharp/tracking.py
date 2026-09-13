"""Hand tracking: threaded capture, landmark detection, and smoothing.

The pipeline is three stages joined by single-slot mailboxes:

    capture thread  --frame-->  detect thread  --result-->  main thread

`cap.read()` blocks until the camera hands over a frame, and MediaPipe takes
about as long again. Run both on one thread and the costs add: a 30 fps camera
yields roughly 17 fps of tracking. Split them and the slower stage sets the
rate. Each mailbox holds exactly one item and overwrites, so a stall in the
detector drops frames rather than building a queue of stale ones -- for an
instrument, late input is worse than missing input.

Smoothing is a One Euro filter per landmark. A fixed low-pass forces a choice
between visible jitter when the hand is still and visible lag when it moves;
One Euro raises its own cutoff in proportion to the measured speed, so it can
be heavily damped at rest and nearly transparent during a sweep.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

# MediaPipe landmark indices
WRIST = 0
INDEX_TIP = 8
MIDDLE_MCP = 9


# ---------------------------------------------------------------------------
# One Euro filter
# ---------------------------------------------------------------------------

def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-3))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro:
    """Casiez, Roussel & Vogel (2012). Works element-wise on any shape.

    `min_cutoff` sets how still a resting hand looks; `beta` sets how quickly
    the filter gets out of the way once the hand starts moving.

    `beta` is a cutoff in hertz per unit of speed, so it is only meaningful
    against a stated unit. Feed this filter frame widths, not pixels: the
    defaults then hold at any camera resolution, and the same `beta` that is
    gentle on a 640 px frame does not go slack on a 1920 px one.
    """

    def __init__(self, min_cutoff=1.0, beta=5.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or self.x_prev.shape != x.shape:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        dx = (x - self.x_prev) / max(dt, 1e-6)
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-3))
        a = 1.0 / (1.0 + tau / max(dt, 1e-6))

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# ---------------------------------------------------------------------------
# hand geometry
# ---------------------------------------------------------------------------

def canonical(px):
    """Landmarks in a hand-local frame: wrist at the origin, palm length one
    unit, palm axis pointing up.

    Raw image coordinates conflate hand shape with where the hand happens to
    be and how it is turned. In this frame a fist is the same set of numbers
    whether it is near the lens or far from it, upright or tilted -- which is
    what any shape measurement, hand-written or learned, actually needs.
    """
    px = np.asarray(px, dtype=np.float64)[:, :2]
    wrist = px[WRIST]
    axis = px[MIDDLE_MCP] - wrist
    scale = float(np.hypot(*axis))
    if scale < 1e-6:
        return np.zeros_like(px)
    c, s = axis[0] / scale, axis[1] / scale
    # Rotate so `axis` lands on (0, -1), i.e. straight up in image coordinates.
    rot = np.array([[-s, c], [-c, -s]])
    return (px - wrist) @ rot.T / scale


class Hand:
    """One tracked hand, in pixels, after smoothing."""

    __slots__ = ("label", "points", "tip", "prev_tip", "score", "dt")

    def __init__(self, label, points, prev_tip, dt, score):
        self.label = label
        self.points = points
        self.tip = points[INDEX_TIP]
        self.prev_tip = prev_tip
        self.dt = dt
        self.score = score

    @property
    def speed(self):
        """Pixels per second of the index fingertip. Reported in the stats
        overlay; useful for telling a jumpy detection from a jumpy hand."""
        if self.prev_tip is None:
            return 0.0
        return float(np.linalg.norm(self.tip - self.prev_tip) / max(self.dt, 1e-6))


# ---------------------------------------------------------------------------
# mailbox
# ---------------------------------------------------------------------------

class Mailbox:
    """One slot, overwritten by the producer, with a sequence number so the
    consumer can tell a fresh item from the one it already handled."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self._seq = 0
        self.dropped = 0

    def put(self, item):
        with self._cv:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._seq += 1
            self._cv.notify_all()

    def get(self, seen, timeout=0.4):
        """Block until something newer than `seen` arrives."""
        with self._cv:
            if self._seq <= seen:
                self._cv.wait(timeout)
            if self._seq <= seen:
                return seen, None
            seq, item = self._seq, self._item
            self._item = None
            return seq, item


# ---------------------------------------------------------------------------
# tracker
# ---------------------------------------------------------------------------

class Tracker:
    """Camera plus hand detection, running on their own threads.

    `detect_width` is the width MediaPipe actually sees. Detecting on a 640 px
    copy of a 1280 px frame roughly halves detection cost and costs nothing in
    accuracy here, because landmarks come back normalised and scale straight
    back up to the full frame.
    """

    def __init__(self, model_path, camera=0, width=1280, height=720, fps=30,
                 detect_width=640, num_hands=2, min_cutoff=1.0, beta=5.0):
        self.model_path = str(model_path)
        self.camera = camera
        self.size = (width, height)
        self.fps = fps
        self.detect_width = detect_width
        self.num_hands = num_hands
        self.filter_cfg = (min_cutoff, beta)

        self.frames = Mailbox()
        self.results = Mailbox()
        self._run = False
        self._threads = []
        self._filters = {}
        self._prev_tip = {}
        self.capture_fps = 0.0
        self.detect_fps = 0.0
        self.detect_ms = 0.0
        self.error = None

    # -- threads -----------------------------------------------------------
    def _capture(self):
        import cv2

        cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error = f"could not open camera {self.camera}"
            self._run = False
            return

        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ema = 0.0
        last = time.perf_counter()
        try:
            while self._run:
                ok, frame = cap.read()
                if not ok:
                    continue
                now = time.perf_counter()
                ema = 0.9 * ema + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now
                self.capture_fps = ema
                # Mirror here so everything downstream -- landmarks, strings,
                # what the player sees -- shares one coordinate system.
                self.frames.put((cv2.flip(frame, 1), now))
        finally:
            cap.release()

    def _detect(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        try:
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            landmarker = vision.HandLandmarker.create_from_options(opts)
        except Exception as exc:
            self.error = f"could not load hand model: {exc}"
            self._run = False
            return

        seen, ema, last = 0, 0.0, time.perf_counter()
        with landmarker:
            while self._run:
                seen, item = self.frames.get(seen)
                if item is None:
                    continue
                frame, stamp = item
                h, w = frame.shape[:2]

                t0 = time.perf_counter()
                small = frame
                if self.detect_width and w > self.detect_width:
                    scale = self.detect_width / w
                    small = cv2.resize(frame, (self.detect_width, int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    res = landmarker.detect_for_video(image, int(stamp * 1000))
                except Exception:
                    continue
                self.detect_ms = 0.9 * self.detect_ms + 0.1 * (time.perf_counter() - t0) * 1e3

                now = time.perf_counter()
                ema = 0.9 * ema + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now
                self.detect_fps = ema
                self.results.put((frame, self._to_hands(res, w, h, stamp), stamp))

    # -- landmark handling -------------------------------------------------
    def _to_hands(self, res, w, h, stamp):
        hands = []
        landmarks = getattr(res, "hand_landmarks", None) or []
        handed = getattr(res, "handedness", None) or []
        seen_labels = set()

        for i, lms in enumerate(landmarks):
            try:
                cat = handed[i][0]
                # The frame was mirrored before detection, so MediaPipe's label
                # is the opposite of the hand the player is actually holding up.
                label = "Right" if cat.category_name == "Left" else "Left"
                score = float(cat.score)
            except Exception:
                label, score = f"hand{i}", 1.0
            if label in seen_labels:
                label = f"{label}{i}"
            seen_labels.add(label)

            px = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float64)

            state = self._filters.get(label)
            if state is None:
                state = {"pos": OneEuro(*self.filter_cfg), "t": stamp}
                self._filters[label] = state
            dt = max(stamp - state["t"], 1e-3)
            state["t"] = stamp
            # Filter in frame widths so the tuning is resolution-independent,
            # then put it back in pixels for everyone downstream.
            px[:, :2] = state["pos"](px[:, :2] / w, dt) * w

            prev = self._prev_tip.get(label)
            hands.append(Hand(label, px, prev, dt, score))
            self._prev_tip[label] = px[INDEX_TIP].copy()

        for label in list(self._filters):
            if label not in seen_labels:
                self._filters[label]["pos"].reset()
                self._prev_tip.pop(label, None)
        return hands

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._run = True
        for fn, name in ((self._capture, "airharp-capture"), (self._detect, "airharp-detect")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self):
        self._run = False
        with self.frames._cv:
            self.frames._cv.notify_all()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []

    def read(self, seen, timeout=0.4):
        """Newest (seq, frame, hands, stamp). `frame` is None if nothing new."""
        seq, item = self.results.get(seen, timeout)
        if item is None:
            return seq, None, [], 0.0
        frame, hands, stamp = item
        return seq, frame, hands, stamp

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
