/**
 * Air Harp in the browser.
 *
 * The logic here is the Python build ported straight across -- the One Euro
 * filter, the canonical hand frame, the finger counting, the chord theory and
 * the hold-time stabilisation are all the same, because none of them depend on
 * the platform. What changed is the three things that do: the camera is
 * getUserMedia, the drawing is Canvas, and the synthesis runs in an
 * AudioWorklet (see synth.js).
 */

import {
  FilesetResolver, HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

// ---------------------------------------------------------------------------
// One Euro filter
// ---------------------------------------------------------------------------

/**
 * Casiez, Roussel & Vogel (2012).
 *
 * A fixed low-pass forces a choice between visible jitter when the hand is
 * still and visible lag when it moves. This raises its own cutoff in
 * proportion to measured speed, so it can be heavily damped at rest and nearly
 * transparent during a sweep. `beta` is hertz per unit of speed, so it is fed
 * frame widths rather than pixels and the tuning holds at any resolution.
 */
class OneEuro {
  constructor(minCutoff = 1.0, beta = 5.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.x = null; this.dx = null;
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * Math.max(cutoff, 1e-3));
    return 1 / (1 + tau / Math.max(dt, 1e-6));
  }
  reset() { this.x = null; this.dx = null; }
  filter(v, dt) {
    if (!this.x || this.x.length !== v.length) {
      this.x = Float64Array.from(v);
      this.dx = new Float64Array(v.length);
      return this.x;
    }
    const ad = OneEuro.alpha(this.dCutoff, dt);
    for (let i = 0; i < v.length; i++) {
      const d = (v[i] - this.x[i]) / Math.max(dt, 1e-6);
      this.dx[i] = ad * d + (1 - ad) * this.dx[i];
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      this.x[i] = a * v[i] + (1 - a) * this.x[i];
    }
    return this.x;
  }
}

// ---------------------------------------------------------------------------
// hand geometry
// ---------------------------------------------------------------------------

const WRIST = 0, MIDDLE_MCP = 9, INDEX_TIP = 8;
const FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const THUMB_TIP = 4, INDEX_MCP = 5;
const EXTEND_RATIO = 1.06, THUMB_CLEAR = 0.58;

/**
 * Landmarks in a hand-local frame: wrist at the origin, palm length one unit,
 * palm axis pointing up.
 *
 * Raw image coordinates conflate hand shape with where the hand happens to be
 * and how it is turned. In this frame a fist is the same set of numbers whether
 * it is near the lens or far from it, upright or tilted.
 */
function canonical(px) {
  const wx = px[WRIST * 2], wy = px[WRIST * 2 + 1];
  const ax = px[MIDDLE_MCP * 2] - wx, ay = px[MIDDLE_MCP * 2 + 1] - wy;
  const scale = Math.hypot(ax, ay);
  const out = new Float64Array(42);
  if (scale < 1e-6) return out;
  const c = ax / scale, s = ay / scale;
  for (let i = 0; i < 21; i++) {
    const dx = px[i * 2] - wx, dy = px[i * 2 + 1] - wy;
    // Rotate so the palm axis lands on (0,-1), i.e. straight up.
    out[i * 2] = (-s * dx + c * dy) / scale;
    out[i * 2 + 1] = (-c * dx - s * dy) / scale;
  }
  return out;
}

/**
 * Which of the five fingers are extended, thumb first.
 *
 * A straight finger puts its tip further from the wrist than its middle joint;
 * a curled one folds the tip back and reverses that. Comparing two distances
 * keeps the test independent of how the hand is turned, which comparing y
 * coordinates does not.
 */
function extended(k) {
  const reach = (i) => Math.hypot(k[i * 2], k[i * 2 + 1]);
  const up = [false, false, false, false, false];
  up[0] = Math.hypot(k[THUMB_TIP * 2] - k[INDEX_MCP * 2],
                     k[THUMB_TIP * 2 + 1] - k[INDEX_MCP * 2 + 1]) > THUMB_CLEAR;
  for (let i = 0; i < 4; i++) {
    const [, pip, , tip] = FINGERS[i];
    up[i + 1] = reach(tip) > reach(pip) * EXTEND_RATIO;
  }
  return up;
}

/**
 * Which chord a finger combination plays, or null for a closed fist.
 *
 * One to five fingers give the first five degrees straight off the count. The
 * two left over get the one shape a count cannot reach -- index and little
 * finger, with the thumb telling them apart.
 */
function degreeOf(up) {
  const [thumb, index, middle, ring, pinky] = up;
  if (index && pinky && !middle && !ring) return thumb ? 6 : 5;
  const n = up.reduce((a, b) => a + (b ? 1 : 0), 0);
  return n === 0 ? null : Math.min(n, 5) - 1;
}

/** The shape that reaches each degree. The guide on screen draws these. */
const DEGREE_SHAPES = [
  [false, true, false, false, false],
  [false, true, true, false, false],
  [false, true, true, true, false],
  [false, true, true, true, true],
  [true, true, true, true, true],
  [false, true, false, false, true],
  [true, true, false, false, true],
];

// ---------------------------------------------------------------------------
// music
// ---------------------------------------------------------------------------

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const MAJOR = [0, 2, 4, 5, 7, 9, 11], MINOR = [0, 2, 3, 5, 7, 8, 10];
const DEGREES = 7, DEFAULT_VOICING = 3;
const ROMAN = {
  major: ["I", "ii", "iii", "IV", "V", "vi", "vii"],
  minor: ["i", "ii", "III", "iv", "v", "VI", "VII"],
};
// Indices into the stacked-thirds chord: 0 root, 1 third, 2 fifth, 3 seventh.
// "+12" means an octave up. More fingers is always a bigger sound.
const VOICINGS = [
  [[0], "muted"], [[0], "single"], [[0, 2], "fifth"],
  [[0, 1, 2], "triad"], [[0, 1, 2, 3], "seventh"], [[0, 1, 2, 12, 13], "open"],
];

const noteName = (m) => NOTE_NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);

class Key {
  constructor(root = 48, scale = "major") { this.root = root; this.scale = scale; this.build(); }
  build() {
    this.parent = this.scale === "major" ? MAJOR : MINOR;
    this.roots = []; this.names = []; this.rootNames = [];
    this.quality = []; this.seventh = [];
    const SUFFIX = { maj: "", min: "m", dim: "dim", aug: "aug" };
    for (let d = 0; d < DEGREES; d++) {
      const p = this.pitch(d);
      const third = this.pitch(d + 2) - p, fifth = this.pitch(d + 4) - p;
      // Naming a chord from its third alone calls B-D-F a B minor. It is
      // diminished -- the fifth is what tells them apart.
      const q = third >= 4 ? (fifth === 7 ? "maj" : "aug")
                           : (fifth === 7 ? "min" : "dim");
      // Appending a bare "7" to every degree is wrong the same way: on the
      // tonic of a major key the stacked seventh is a major seventh, and "A7"
      // names the dominant, which is a different chord.
      const sev = this.pitch(d + 6) - p;
      this.seventh.push(third >= 4 ? (sev === 11 ? "maj7" : "7")
                                   : (fifth === 6 ? "m7b5" : "m7"));
      this.roots.push(p);
      this.quality.push(q);
      this.rootNames.push(noteName(p));
      this.names.push(noteName(p).slice(0, -1) + SUFFIX[q]);
    }
  }
  pitch(d) {
    const n = this.parent.length;
    return this.root + this.parent[((d % n) + n) % n] + 12 * Math.floor(d / n);
  }
  roman(d) { return ROMAN[this.scale][d % DEGREES]; }
  transpose(s) { this.root = Math.min(64, Math.max(36, this.root + s)); this.build(); }
  toggleScale() { this.scale = this.scale === "major" ? "minor" : "major"; this.build(); }
  get keyName() { return `${NOTE_NAMES[this.root % 12]} ${this.scale}`; }

  chord(degree, voicing) {
    degree = Math.min(DEGREES - 1, Math.max(0, degree));
    voicing = Math.min(VOICINGS.length - 1, Math.max(0, voicing));
    return VOICINGS[voicing][0].map((p) => {
      const oct = Math.floor(p / 12), step = p - oct * 12;
      return 440 * Math.pow(2, (this.pitch(degree + 2 * step) + 12 * oct - 69) / 12);
    });
  }
  label(degree, voicing) {
    degree = Math.min(DEGREES - 1, Math.max(0, degree));
    voicing = Math.min(VOICINGS.length - 1, Math.max(0, voicing));
    const kind = VOICINGS[voicing][1], base = this.names[degree];
    if (kind === "muted" || kind === "single") return this.rootNames[degree];
    // A "5" chord means root plus a perfect fifth. On a diminished degree
    // that interval is a tritone, so the name would be a lie.
    if (kind === "fifth")
      return ["maj", "min"].includes(this.quality[degree]) ? base.replace(/m$/, "") + "5" : base;
    if (kind === "seventh")
      return this.rootNames[degree].slice(0, -1) + this.seventh[degree];
    return base;
  }
}

// ---------------------------------------------------------------------------
// pose reading and stabilisation
// ---------------------------------------------------------------------------

/**
 * A raw reading flickers between neighbouring shapes on almost every hand, and
 * a chord that flickers with it is unplayable. A new shape must hold for HOLD
 * seconds before it commits, and a hand that blinks out of tracking keeps
 * playing for NULL seconds before it is let go.
 */
class PoseReader {
  static HOLD = 0.09;
  static NULL = 0.16;
  static FLOOR = 0.18;

  constructor() { this.state = new Map(); this.chord = null; }
  reset() { this.state.clear(); this.chord = null; }

  update(hands, height, now) {
    const states = [], live = new Set();

    for (const h of hands) {
      live.add(h.label);
      const rawUp = extended(canonical(h.px));
      const rawKey = rawUp.map(Number).join("");

      let st = this.state.get(h.label);
      if (!st) { st = { up: rawUp, key: rawKey, cand: null, since: now, seen: now }; this.state.set(h.label, st); }
      st.seen = now;

      let settling = false;
      if (rawKey === st.key) st.cand = null;
      else if (st.cand === rawKey) {
        if (now - st.since >= PoseReader.HOLD) { st.up = rawUp; st.key = rawKey; st.cand = null; }
        else settling = true;
      } else { st.cand = rawKey; st.since = now; settling = true; }

      const fingers = st.up.reduce((a, b) => a + (b ? 1 : 0), 0);
      // Height is continuous and never stabilised: it is expression, and
      // expression that snaps to steps sounds mechanical.
      const high = Math.min(1, Math.max(0, 1 - h.py[INDEX_TIP * 2 + 1] / height));
      states.push({
        label: h.label, up: st.up, fingers,
        degree: degreeOf(st.up),
        voicing: Math.min(fingers, VOICINGS.length - 1),
        x: h.py[INDEX_TIP * 2], y: h.py[INDEX_TIP * 2 + 1],
        level: PoseReader.FLOOR + (1 - PoseReader.FLOOR) * Math.pow(high, 0.75),
        bright: Math.pow(high, 0.85),
        settling, px: h.py,
      });
    }

    const released = [];
    for (const [label, st] of [...this.state]) {
      if (live.has(label)) continue;
      if (now - st.seen >= PoseReader.NULL) { this.state.delete(label); released.push(label); }
    }

    const chord = this.combine(states);
    const changed = (chord === null) !== (this.chord === null)
      || (chord && this.chord && (chord.degree !== this.chord.degree || chord.voicing !== this.chord.voicing));
    this.chord = chord;
    return { states, chord, changed: !!changed, released };
  }

  /** Left hand names the chord, right hand says how full. Either alone plays. */
  combine(states) {
    if (!states.length) return null;
    const by = Object.fromEntries(states.map((s) => [s.label, s]));
    const lead = by.Left || states[0];
    if (lead.degree === null) return null;
    lead.leads = true;

    const fuller = by.Right;
    let voicing = DEFAULT_VOICING;
    if (fuller && fuller !== lead && fuller.fingers > 0) voicing = fuller.voicing;
    else if (fuller === lead) voicing = lead.voicing;

    const loud = states.filter((s) => s.fingers > 0);
    const pool = loud.length ? loud : states;
    return {
      degree: lead.degree, voicing,
      level: pool.reduce((a, s) => a + s.level, 0) / pool.length,
      bright: pool.reduce((a, s) => a + s.bright, 0) / pool.length,
    };
  }
}

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

const INSTRUMENTS = [
  { name: "Harp", rgb: [245, 205, 130], sustains: false },
  { name: "Guitar", rgb: [230, 140, 80], sustains: false },
  { name: "Piano", rgb: [225, 235, 250], sustains: true },
  { name: "Violin", rgb: [150, 150, 250], sustains: true },
  { name: "Kalimba", rgb: [150, 240, 190], sustains: false },
  { name: "Bells", rgb: [245, 170, 225], sustains: true },
  { name: "Synth", rgb: [130, 200, 255], sustains: true },
];
const BONES = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],
  [13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];
const TIPS = [4, 8, 12, 16, 20];
const STRANDS = [1, 1, 2, 3, 4, 5];

/** Warm amber at the bottom of the key, cool violet at the top. */
const DEGREE_COLOURS = Array.from({ length: DEGREES }, (_, i) => {
  const h = 28 + (272 * i) / (DEGREES - 1);
  const f = (n) => {
    const k = (n + h / 60) % 6;
    return Math.round(255 * (1 - 0.65 * Math.max(0, Math.min(k, 4 - k, 1))));
  };
  return [f(5), f(3), f(1)];
});

const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const css = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

class HUD {
  constructor(key) {
    this.key = key;
    this.energy = new Float32Array(DEGREES);
    this.ribbons = new Map();
    this.phase = 0;
    this.dim = 0.55;
  }
  retune() { this.energy.fill(0); }
  strike(d, amt = 1) { if (d != null) this.energy[d] = Math.min(1, this.energy[d] * 0.4 + amt); }

  step(dt, states, chord) {
    const fade = Math.exp(-dt / 0.45);
    this.phase += dt;
    for (let i = 0; i < DEGREES; i++) this.energy[i] *= fade;
    for (const [k, v] of this.ribbons) this.ribbons.set(k, v * fade);

    const live = new Set();
    if (chord) {
      const floor = 0.3 + 0.55 * chord.level;
      this.energy[chord.degree] = Math.max(this.energy[chord.degree], floor);
      for (const s of states) if (s.fingers > 0) {
        live.add(s.label);
        this.ribbons.set(s.label, Math.max(this.ribbons.get(s.label) || 0, floor));
      }
    }
    for (const [k, v] of [...this.ribbons]) if (v < 1e-3 && !live.has(k)) this.ribbons.delete(k);
  }

  draw(ctx, video, states, chord, inst, w, h, fps) {
    const BAR = Math.round(h * 0.072), GUIDE = Math.round(h * 0.108);
    const tint = inst.rgb;

    // Mirror the picture so it reads as a mirror, not a video of someone else.
    if (video && video.readyState >= 2) {
      ctx.save();
      ctx.translate(w, 0); ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, w, h);
      ctx.restore();
      ctx.fillStyle = `rgba(8,7,11,${1 - this.dim})`;
    } else {
      ctx.fillStyle = "#0d0c11";
    }
    ctx.fillRect(0, 0, w, h);

    this.drawRibbons(ctx, states, chord, tint, w, h, BAR + GUIDE);
    this.drawHands(ctx, states, tint);
    this.drawReadout(ctx, chord, states, tint, w, h);
    this.drawGuide(ctx, chord, tint, w, h, BAR, GUIDE);
    this.drawBar(ctx, inst, w, h, BAR);
    this.drawStatus(ctx, inst, w, h, fps, !chord);
  }

  /** A band of travelling sine waves for each sounding hand, riding at its
   *  height -- the one continuous control gets a continuous answer. */
  drawRibbons(ctx, states, chord, tint, w, h, foot) {
    if (!chord || !this.ribbons.size) return;
    const base = mix(DEGREE_COLOURS[chord.degree], tint, 0.45);
    ctx.lineCap = "round";
    for (const s of states) {
      const level = this.ribbons.get(s.label) || 0;
      if (level < 0.02 || s.fingers <= 0) continue;
      const strands = STRANDS[Math.min(s.fingers, STRANDS.length - 1)];
      const cy = Math.min(Math.max(s.y, h * 0.12), h - foot - h * 0.08);
      const span = h * (0.035 + 0.12 * level);

      for (let k = 0; k < strands; k++) {
        const off = (k - (strands - 1) / 2) * (span / strands);
        const wob = 6.4 + 1.9 * k, speed = 2.1 + 0.45 * k + 1.8 * s.bright;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const t = i / 120, x = t * w;
          const taper = Math.pow(Math.sin(Math.PI * t), 0.8);
          const y = cy + off + taper * span * 0.55
            * Math.sin(t * wob + this.phase * speed + k * 1.1);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = css(base, 0.35 + 0.5 * level);
        ctx.lineWidth = k === 0 ? 2.5 : 1.5;
        ctx.shadowColor = css(base, 0.9); ctx.shadowBlur = 14 + 24 * level;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
  }

  /** The skeleton is not decoration: it is how you see tracking is alive and
   *  on your hand, so when something stops working the screen says so. */
  drawHands(ctx, states, tint) {
    for (const s of states) {
      const p = s.px;
      ctx.strokeStyle = css(tint, 0.5); ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (const [a, b] of BONES) {
        ctx.moveTo(p[a * 2], p[a * 2 + 1]);
        ctx.lineTo(p[b * 2], p[b * 2 + 1]);
      }
      ctx.stroke();
      for (let i = 0; i < 5; i++) {
        const t = TIPS[i], lit = s.up[i];
        ctx.beginPath();
        ctx.arc(p[t * 2], p[t * 2 + 1], lit ? 9 : 3.5, 0, 6.284);
        ctx.fillStyle = css(tint, lit ? 1 : 0.4);
        if (lit) { ctx.shadowColor = css(tint, 1); ctx.shadowBlur = 16; }
        ctx.fill(); ctx.shadowBlur = 0;
      }
    }
  }

  /** The chord, large, at the top -- the one thing worth reading while playing. */
  drawReadout(ctx, chord, states, tint, w, h) {
    if (!chord) return;
    const a = states.some((s) => s.settling) ? 0.5 : 1;
    const size = Math.round(h * 0.1);
    const label = this.key.label(chord.degree, chord.voicing);
    const sub = `${this.key.roman(chord.degree)}    ${VOICINGS[chord.voicing][1]}`;

    ctx.textAlign = "center";
    ctx.font = `600 ${size}px ui-sans-serif, system-ui, sans-serif`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(8,7,11,0.62)";
    ctx.fillRect(w / 2 - tw / 2 - 26, h * 0.015, tw + 52, size * 1.62);
    ctx.fillStyle = css(tint, a);
    ctx.fillText(label, w / 2, h * 0.015 + size * 0.9);

    ctx.font = `${Math.round(h * 0.026)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillStyle = css(tint, 0.68 * a);
    ctx.fillText(sub, w / 2, h * 0.015 + size * 1.35);

    const bw = w * 0.13, bx = w / 2 - bw / 2, by = h * 0.015 + size * 1.62 + 6;
    ctx.fillStyle = "rgba(255,255,255,0.16)";
    ctx.fillRect(bx, by, bw, 5);
    ctx.fillStyle = css(tint, 0.95);
    ctx.fillRect(bx, by, bw * chord.level, 5);
  }

  /** All seven chords with the hand shape that reaches each one. This is the
   *  whole manual, and it is always on screen. */
  drawGuide(ctx, chord, tint, w, h, BAR, GUIDE) {
    const top = h - BAR - GUIDE;
    ctx.fillStyle = "rgba(12,11,16,0.92)";
    ctx.fillRect(0, top, w, GUIDE);
    const slot = w / DEGREES, here = chord ? chord.degree : -1;
    ctx.textAlign = "center";

    for (let d = 0; d < DEGREES; d++) {
      const cx = slot * (d + 0.5), e = this.energy[d];
      const base = mix(DEGREE_COLOURS[d], tint, 0.4);
      if (d === here) {
        ctx.fillStyle = css(base, 0.2);
        ctx.fillRect(cx - slot / 2 + 4, top + 4, slot - 8, GUIDE - 8);
      }
      const gap = Math.min(14, slot / 7);
      for (let i = 0; i < 5; i++) {
        const lit = DEGREE_SHAPES[d][i];
        ctx.beginPath();
        ctx.arc(cx - gap * 2 + gap * i, top + GUIDE * 0.25, lit ? 4.5 : 3, 0, 6.284);
        ctx.fillStyle = lit ? css(base, 0.5 + 0.5 * e) : "rgba(255,255,255,0.16)";
        ctx.fill();
      }
      ctx.font = `600 ${Math.round(GUIDE * 0.3)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillStyle = css(base, 0.55 + 0.45 * e);
      ctx.fillText(this.key.names[d], cx, top + GUIDE * 0.66);
      ctx.font = `${Math.round(GUIDE * 0.19)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillStyle = css(base, 0.35 + 0.35 * e);
      ctx.fillText(this.key.roman(d), cx, top + GUIDE * 0.9);
    }
  }

  drawBar(ctx, inst, w, h, BAR) {
    const top = h - BAR;
    ctx.fillStyle = "rgba(16,15,21,0.96)";
    ctx.fillRect(0, top, w, BAR);
    ctx.textAlign = "left";
    ctx.font = `${Math.round(BAR * 0.34)}px ui-sans-serif, system-ui, sans-serif`;
    let x = w * 0.012;
    for (let i = 0; i < INSTRUMENTS.length; i++) {
      const it = INSTRUMENTS[i], text = `${i + 1} ${it.name}`;
      const tw = ctx.measureText(text).width, box = tw + BAR * 0.5;
      if (it === inst) {
        ctx.fillStyle = css(it.rgb, 0.22);
        ctx.fillRect(x, top + BAR * 0.2, box, BAR * 0.6);
        ctx.strokeStyle = css(it.rgb, 0.8); ctx.lineWidth = 1;
        ctx.strokeRect(x, top + BAR * 0.2, box, BAR * 0.6);
      }
      ctx.fillStyle = it === inst ? css(it.rgb) : "rgba(200,198,212,0.5)";
      ctx.fillText(text, x + BAR * 0.25, top + BAR * 0.62);
      x += box + BAR * 0.2;
    }
    const keys = "[ ] key    m major/minor    q stop";
    ctx.font = `${Math.round(BAR * 0.26)}px ui-sans-serif, system-ui, sans-serif`;
    const kw = ctx.measureText(keys).width;
    if (x + kw + 20 < w) {
      ctx.fillStyle = "rgba(160,157,175,0.5)";
      ctx.fillText(keys, w - kw - w * 0.012, top + BAR * 0.62);
    }
  }

  drawStatus(ctx, inst, w, h, fps, idle) {
    ctx.textAlign = "left";
    const s = Math.round(h * 0.034);
    ctx.font = `600 ${s}px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillStyle = css(inst.rgb);
    ctx.fillText(inst.name, w * 0.014, h * 0.055);
    ctx.font = `${Math.round(s * 0.55)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillStyle = css(inst.rgb, 0.55);
    ctx.fillText(inst.sustains ? "holds while you hold" : "rings out on its own",
                 w * 0.014, h * 0.082);
    ctx.fillStyle = "rgba(168,164,184,0.75)";
    ctx.fillText(this.key.keyName, w * 0.014, h * 0.107);

    if (fps) {
      ctx.textAlign = "right";
      ctx.fillStyle = "rgba(150,147,165,0.45)";
      ctx.fillText(`${fps.toFixed(0)} fps`, w - w * 0.014, h * 0.055);
    }
    if (idle) {
      ctx.textAlign = "center";
      ctx.font = `${Math.round(h * 0.036)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillStyle = "rgba(232,230,239,0.82)";
      ctx.fillText("hold up a hand — the fingers you raise are the chord",
                   w / 2, h * 0.72);
    }
  }
}

// ---------------------------------------------------------------------------
// demo mode
// ---------------------------------------------------------------------------

/**
 * A synthetic hand, so the instrument can be seen and heard without a camera.
 *
 * Open `#demo` and it cycles the seven shapes on its own. That covers the two
 * people this page will meet who will not grant camera access, and it is also
 * how the drawing gets checked without a webcam in the loop.
 */
const HAND_MCP = { thumb: [-0.62, -0.42], index: [-0.36, -0.94],
                   middle: [0, -1], ring: [0.33, -0.96], pinky: [0.62, -0.86] };
const HAND_BONE = { index: [0.4, 0.25, 0.19], middle: [0.44, 0.27, 0.2],
                    ring: [0.4, 0.25, 0.19], pinky: [0.32, 0.2, 0.16] };
const HAND_SLOTS = { index: [5, 6, 7, 8], middle: [9, 10, 11, 12],
                     ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] };

function syntheticHand(up, cx, cy, scale, angle = 0) {
  const b = new Float64Array(42);
  const names = ["index", "middle", "ring", "pinky"];
  names.forEach((name, fi) => {
    const curl = up[fi + 1] ? 0 : 1.55;
    const slots = HAND_SLOTS[name];
    let [x, y] = HAND_MCP[name];
    b[slots[0] * 2] = x; b[slots[0] * 2 + 1] = y;
    HAND_BONE[name].forEach((len, k) => {
      const a = curl * (k + 1);
      x += len * Math.sin(a); y += len * -Math.cos(a);
      b[slots[k + 1] * 2] = x; b[slots[k + 1] * 2 + 1] = y;
    });
  });
  // The thumb swings across the palm rather than curling along it.
  const dir = up[0] ? [-0.8, -0.6] : [0.8, -0.6];
  b[2] = HAND_MCP.thumb[0] * 0.55; b[3] = HAND_MCP.thumb[1] * 0.55;
  b[4] = HAND_MCP.thumb[0]; b[5] = HAND_MCP.thumb[1];
  b[6] = b[4] + dir[0] * 0.34; b[7] = b[5] + dir[1] * 0.34;
  b[8] = b[4] + dir[0] * 0.62; b[9] = b[5] + dir[1] * 0.62;

  const c = Math.cos(angle), s = Math.sin(angle), px = new Float64Array(42);
  for (let i = 0; i < 21; i++) {
    px[i * 2] = (b[i * 2] * c - b[i * 2 + 1] * s) * scale + cx;
    px[i * 2 + 1] = (b[i * 2] * s + b[i * 2 + 1] * c) * scale + cy;
  }
  return px;
}

function demoHands(t, w, h) {
  const d = Math.floor(t / 1.6) % DEGREES;
  const wobble = Math.sin(t * 1.1) * 0.12;
  return [{
    label: "Left",
    px: syntheticHand(DEGREE_SHAPES[d], w * 0.5, h * 0.55 + Math.sin(t * 0.7) * h * 0.08,
                      h * 0.22, wobble),
    py: syntheticHand(DEGREE_SHAPES[d], w * 0.5, h * 0.55 + Math.sin(t * 0.7) * h * 0.08,
                      h * 0.22, wobble),
  }];
}

// ---------------------------------------------------------------------------
// the app
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const video = $("v"), canvas = $("c"), ctx = canvas.getContext("2d", { alpha: false });

// Build the intro guide from the same table the HUD uses, so the page and the
// instrument can never disagree about how a chord is played.
(function introGuide() {
  const key = new Key();
  $("guide").innerHTML = DEGREE_SHAPES.map((shape, d) => {
    const dots = shape.map((on) => `<span class="dot${on ? " on" : ""}"></span>`).join("");
    return `<tr><td><span class="dots">${dots}</span></td>` +
           `<td><strong>${key.names[d]}</strong></td>` +
           `<td>${["one finger", "two fingers", "three fingers", "four fingers",
                   "open hand", "index + little", "those + thumb"][d]}</td></tr>`;
  }).join("") +
  `<tr><td colspan="2">fist</td><td>stop</td></tr>` +
  `<tr><td colspan="2">raise / lower a hand</td><td>louder / quieter</td></tr>` +
  `<tr><td colspan="2">right hand, 1–5 fingers</td><td>how full the chord is</td></tr>`;
})();

const key = new Key();
const hud = new HUD(key);
const reader = new PoseReader();
const filters = new Map();
let node = null, landmarker = null, instrument = 0, fps = 0, running = false;
let demoMode = false;
let lastStamp = 0, lastVideoTime = -1, cached = { states: [], chord: null };

const send = (m) => node && node.port.postMessage(m);

async function startAudio() {
  const ac = new AudioContext({ latencyHint: "interactive" });
  await ac.audioWorklet.addModule("synth.js");
  node = new AudioWorkletNode(ac, "airharp", { outputChannelCount: [2] });
  node.connect(ac.destination);
  if (ac.state === "suspended") await ac.resume();
  return ac;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  await new Promise((r) => (video.videoWidth ? r() : (video.onloadedmetadata = r)));
}

async function startVision() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "hand_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });
}

function resize() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const scale = Math.min(window.innerWidth / vw, window.innerHeight / vh, 1.5);
  canvas.width = Math.round(vw);
  canvas.height = Math.round(vh);
  canvas.style.width = `${Math.round(vw * scale)}px`;
  canvas.style.height = `${Math.round(vh * scale)}px`;
}

/** Turn a MediaPipe result into mirrored pixel coordinates, smoothed. */
function readHands(result, w, h, dt) {
  const out = [];
  const lms = result.landmarks || [];
  const handed = result.handednesses || result.handedness || [];
  const seen = new Set();

  for (let i = 0; i < lms.length; i++) {
    let label = "Right", cat = handed[i] && handed[i][0];
    // The picture is mirrored, so MediaPipe's label is the opposite of the
    // hand actually being held up.
    if (cat) label = cat.categoryName === "Left" ? "Right" : "Left";
    if (seen.has(label)) label += i;
    seen.add(label);

    const raw = new Float64Array(42);
    for (let j = 0; j < 21; j++) {
      raw[j * 2] = (1 - lms[i][j].x) * w;      // mirror x
      raw[j * 2 + 1] = lms[i][j].y * h;
    }
    let f = filters.get(label);
    if (!f) { f = new OneEuro(); filters.set(label, f); }
    // Filter in frame widths so the tuning is resolution-independent.
    const norm = new Float64Array(42);
    for (let k = 0; k < 42; k++) norm[k] = raw[k] / w;
    const sm = f.filter(norm, dt);
    const py = new Float64Array(42);
    for (let k = 0; k < 42; k++) py[k] = sm[k] * w;

    out.push({ label, px: py, py });
  }
  for (const [label, f] of filters) if (!seen.has(label)) f.reset();
  return out;
}

function play(chord, changed) {
  if (!chord) { if (changed) send({ type: "release", key: "hands" }); return; }
  if (changed) {
    send({
      type: "chord", key: "hands",
      freqs: key.chord(chord.degree, chord.voicing),
      amp: chord.level, bright: chord.bright, damp: 0.25 + 0.6 * chord.level,
    });
    hud.strike(chord.degree, chord.level);
  } else {
    // Height keeps moving between chord changes; follow it without restarting.
    send({ type: "level", key: "hands", amp: chord.level });
  }
}

function frame(nowMs) {
  if (!running) return;
  requestAnimationFrame(frame);
  const w = canvas.width, h = canvas.height;
  const now = nowMs / 1000;
  const dt = lastStamp ? Math.min(Math.max(now - lastStamp, 1e-3), 0.2) : 1 / 30;
  lastStamp = now;
  fps = fps ? fps * 0.9 + 0.1 / dt : 1 / dt;

  if (demoMode) {
    const { states, chord, changed } = reader.update(demoHands(now, w, h), h, now);
    play(chord, changed);
    cached = { states, chord };
  } else if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    // Only run detection on a genuinely new camera frame; the display refreshes
    // faster than the camera does and re-detecting the same picture is waste.
    lastVideoTime = video.currentTime;
    try {
      const result = landmarker.detectForVideo(video, nowMs);
      const hands = readHands(result, w, h, dt);
      const { states, chord, changed } = reader.update(hands, h, now);
      play(chord, changed);
      cached = { states, chord };
    } catch (e) { /* a dropped frame is not worth stopping for */ }
  }

  hud.step(dt, cached.states, cached.chord);
  hud.draw(ctx, video, cached.states, cached.chord, INSTRUMENTS[instrument], w, h, fps);
}

addEventListener("keydown", (e) => {
  if (!running) return;
  const k = e.key;
  if (k >= "1" && k <= "7") { instrument = +k - 1; send({ type: "instrument", index: instrument }); reader.reset(); }
  else if (k === "m") { key.toggleScale(); retune(); }
  else if (k === "[") { key.transpose(-1); retune(); }
  else if (k === "]") { key.transpose(1); retune(); }
  else if (k === "q") { send({ type: "releaseAll" }); reader.reset(); }
});
const retune = () => { hud.retune(); reader.reset(); send({ type: "releaseAll" }); };
addEventListener("resize", () => running && !demoMode && resize());

if (location.hash === "#demo") {
  demoMode = true;
  $("start").textContent = "Start demo — no camera";
}

$("start").addEventListener("click", async () => {
  const btn = $("start"), err = $("err");
  btn.disabled = true; err.style.display = "none";
  try {
    // Load everything that can fail quietly first, and ask for the camera
    // last -- so the permission prompt arrives the moment it is needed rather
    // than leaving the page sitting on a granted camera doing nothing.
    btn.textContent = "starting audio…";
    await startAudio();
    if (demoMode) {
      canvas.width = 1280; canvas.height = 720;
      const scale = Math.min(innerWidth / 1280, innerHeight / 720, 1.5);
      canvas.style.width = `${Math.round(1280 * scale)}px`;
      canvas.style.height = `${Math.round(720 * scale)}px`;
    } else {
      btn.textContent = "loading hand tracking…";
      await startVision();
      btn.textContent = "waiting for camera…";
      await startCamera();
      resize();
    }
    running = true;
    $("intro").classList.add("hidden");
    requestAnimationFrame(frame);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = "Start — allow camera";
    err.style.display = "block";
    err.textContent =
      e && e.name === "NotAllowedError"
        ? "Camera permission was refused. It is needed to see your hands — nothing is recorded or sent anywhere."
        : `Could not start: ${e && e.message ? e.message : e}`;
  }
});
