/**
 * Air Harp synthesis, as an AudioWorklet.
 *
 * This is the Python engine ported to run on the browser's audio thread. The
 * structure is the same and so is the reasoning behind it:
 *
 *  - Nothing allocates inside `process`. Voices are built when a chord starts,
 *    never per quantum, because the audio thread has 2.9 ms to fill 128 samples
 *    and a garbage collection pause inside that is an audible dropout.
 *  - Delay-line feedback (Karplus-Strong, reverb combs, allpasses) reads only
 *    taps written a full lap earlier, so the loops stay simple and in order.
 *  - Sawtooth waves come from band-limited tables. A naive saw aliases audibly
 *    above a few hundred hertz.
 *  - Held notes are retuned and re-levelled, never restarted. Changing C-E-G to
 *    C-E-A keeps the C and the E sounding.
 */

const TN = 8192;                 // sine table length
const TMASK = TN - 1;
const SAW_N = 4096;              // one cycle of each band-limited saw
const SAW_MASK = SAW_N - 1;
const BRIGHTS = 4;
const BANDS = 9;
const BAND_LO = 20;
const MAX_VOICES = 24;
const GAIN_K = 0.055;            // per 128-sample quantum: about a 50 ms glide

const SINE = new Float32Array(TN);
for (let i = 0; i < TN; i++) SINE[i] = Math.sin((2 * Math.PI * i) / TN);

/**
 * Band-limited sawtooth bank, [brightness][octave band].
 * Built from sine-table lookups rather than Math.sin so start-up stays under
 * a frame or two instead of taking most of a second.
 */
const SAW = [];
(function buildSaw(sr) {
  for (let b = 0; b < BRIGHTS; b++) {
    const rolloff = 2.40 - 0.427 * b;          // higher b is brighter
    const row = [];
    for (let k = 0; k < BANDS; k++) {
      const fMax = BAND_LO * Math.pow(2, k + 1);
      const nHarm = Math.max(1, Math.min(96, Math.floor((sr * 0.45) / fMax)));
      const wave = new Float32Array(SAW_N);
      for (let n = 1; n <= nHarm; n++) {
        const amp = 1 / Math.pow(n, rolloff);
        const step = (n * TN) / SAW_N;
        let idx = 0;
        for (let i = 0; i < SAW_N; i++) {
          wave[i] += amp * SINE[(idx | 0) & TMASK];
          idx += step;
        }
      }
      let peak = 0;
      for (let i = 0; i < SAW_N; i++) peak = Math.max(peak, Math.abs(wave[i]));
      if (peak > 0) for (let i = 0; i < SAW_N; i++) wave[i] /= peak;
      row.push(wave);
    }
    SAW.push(row);
  }
})(sampleRate);

const bandOf = (f) =>
  Math.max(0, Math.min(BANDS - 1, Math.floor(Math.log2(Math.max(f, 1) / BAND_LO))));

// ---------------------------------------------------------------------------
// voices
// ---------------------------------------------------------------------------

class Voice {
  constructor(gain = 1) {
    this.done = false;
    this.gain = gain;
    this.target = gain;
  }
  get held() { return false; }
  retarget(gain) { this.target = gain; }
  release() {}
  /** Gain stepped once per quantum; 128 samples is short enough not to click. */
  stepGain() {
    this.gain += (this.target - this.gain) * GAIN_K;
    if (Math.abs(this.gain - this.target) < 1e-5) this.gain = this.target;
    return this.gain;
  }
}

/** Karplus-Strong: harp and guitar. */
class Plucked extends Voice {
  constructor(freq, amp, bright, t60, pick = 0) {
    super(1);
    freq = Math.max(freq, 20);
    const len = Math.max(8, Math.round(sampleRate / freq));
    const buf = new Float32Array(len);
    for (let i = 0; i < len; i++) buf[i] = Math.random() * 2 - 1;

    // Moving average lowpasses the excitation: dark plucks start dull.
    const w = 1 + Math.round((1 - bright) * 7);
    if (w > 1) {
      const src = Float32Array.from(buf);
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let j = 0; j < w; j++) s += src[(i - j + len) % len];
        buf[i] = s / w;
      }
    }
    // Comb notch at the pick position, as on a string picked near the bridge.
    if (pick > 0) {
      const src = Float32Array.from(buf);
      const k = Math.max(1, Math.floor(len * pick));
      for (let i = 0; i < len; i++) buf[i] = src[i] - src[(i - k + len) % len] * 0.75;
    }
    let peak = 0;
    for (let i = 0; i < len; i++) peak = Math.max(peak, Math.abs(buf[i]));
    const scale = amp / Math.max(peak, 1e-9);
    for (let i = 0; i < len; i++) buf[i] *= scale;

    this.buf = buf;
    this.idx = 0;
    this.prev = buf[len - 1];
    // A delay line laps `freq` times a second, so solving g**(t60*freq)=1e-3
    // keeps the decay even across the range. With a fixed g the top dies fast.
    const g = Math.pow(10, -3 / (Math.max(t60, 0.05) * freq));
    this.g = Math.min(Math.max(g, 0.8), 0.99965) * 0.5;
    this.floor = amp * 0.0012 + 2e-5;
  }

  render(out, n) {
    const buf = this.buf, len = buf.length, g = this.g;
    let i = this.idx, prev = this.prev, peak = 0;
    for (let s = 0; s < n; s++) {
      const cur = buf[i];
      out[s] += cur;
      buf[i] = (cur + prev) * g;
      prev = cur;
      if (Math.abs(buf[i]) > peak) peak = Math.abs(buf[i]);
      i = i + 1 >= len ? 0 : i + 1;
    }
    this.idx = i;
    this.prev = prev;
    if (peak < this.floor) this.done = true;
  }
}

/** Summed partials with per-partial decay: piano, kalimba, bells. */
class Additive extends Voice {
  constructor(freq, amp, ratios, gains, t60s, attack, inharm = 0, sustain = 0,
              release = 0.4, drift = 0, driftHz = 0.17) {
    // Amplitude belongs to the voice gain, never to the partials. Scaling the
    // partials too would apply it twice the moment anything retargets.
    super(amp);
    const inc = [], g = [], dec = [], rest = [], phase = [], harm = [];
    for (let i = 0; i < ratios.length; i++) {
      let r = ratios[i];
      if (inharm) r *= Math.sqrt(1 + inharm * r * r);
      const f = freq * r;
      if (f >= sampleRate * 0.46) continue;
      inc.push(f / sampleRate);
      phase.push(Math.random());
      harm.push(Math.max(r, 1));
      const g0 = gains[i];
      g.push(g0);
      rest.push(g0 * sustain);
      dec.push(Math.pow(10, -3 / (Math.max(t60s[i], 0.02) * sampleRate)));
    }
    if (!inc.length) { inc.push(freq / sampleRate); phase.push(0); g.push(1);
                       rest.push(0); dec.push(0.9999); harm.push(1); }
    this.inc = Float64Array.from(inc);
    this.phase = Float64Array.from(phase);
    this.g = Float32Array.from(g);
    this.rest = Float32Array.from(rest);
    this.dec = Float32Array.from(dec);
    this.att = Math.max(1, Math.floor(attack * sampleRate));
    this.t = 0;
    this.sustains = sustain > 0;
    this.relLen = Math.max(1, Math.floor(release * sampleRate));
    this.rel = -1;
    // A spectrum that never moves is most of what the ear hears as
    // switched-on rather than alive. Swinging the harmonic rolloff slowly is
    // what a filter opening and closing does. `dmul` is preallocated because
    // nothing may allocate inside process().
    this.harm = Float32Array.from(harm);
    this.drift = drift;
    this.driftHz = driftHz;
    this.dphase = Math.random();
    this.dmul = new Float32Array(this.harm.length).fill(1);
    this.floor = 8e-4;          // on the partial envelope, which is unit-scale
  }

  get held() { return this.sustains && this.rel < 0; }
  release() { if (this.sustains && this.rel < 0) this.rel = 0; }

  render(out, n) {
    const k = this.inc.length;
    const gain = this.stepGain();
    if (this.drift) {
      this.dphase = (this.dphase + (this.driftHz * n) / sampleRate) % 1;
      const off = -this.drift * Math.sin(2 * Math.PI * this.dphase);
      for (let p = 0; p < k; p++) this.dmul[p] = Math.pow(this.harm[p], off);
    }
    for (let p = 0; p < k; p++) {
      const dm = this.drift ? this.dmul[p] : 1;
      const inc = this.inc[p], dec = this.dec[p], rest = this.rest[p];
      let ph = this.phase[p], g = this.g[p];
      let t = this.t, rel = this.rel;
      for (let s = 0; s < n; s++) {
        let e = g * dm;
        if (t < this.att) e *= t / this.att;
        if (rel >= 0) { const r = Math.max(0, 1 - rel / this.relLen); e *= r * r; rel++; }
        out[s] += SINE[((ph * TN) | 0) & TMASK] * e * gain;
        ph += inc; if (ph >= 1) ph -= 1;
        g = rest + (g - rest) * dec;
        t++;
      }
      this.phase[p] = ph;
      this.g[p] = g;
      if (p === k - 1) { this.t = t; if (this.rel >= 0) this.rel = rel; }
    }
    if (this.rel >= this.relLen) this.done = true;
    if (!this.sustains) {
      let peak = 0;
      for (let p = 0; p < k; p++) peak = Math.max(peak, this.g[p]);
      if (peak < this.floor) this.done = true;
    }
  }
}

/** Band-limited sawtooth under a bow-stroke envelope: violin. */
class Bowed extends Voice {
  constructor(freq, amp, bright, opts = {}) {
    super(amp);
    const { attack = 0.1, release = 0.34, vibDepth = 0.0055, vibHz = 5.3,
            detune = 0 } = opts;
    const lvl = Math.min(BRIGHTS - 1, Math.max(0, Math.round(bright * (BRIGHTS - 1))));
    this.tab = SAW[lvl][bandOf(freq)];
    this.inc = freq / sampleRate;
    // Two oscillators a few cents apart beat slowly against each other. That
    // slow drift is the whole of what makes a pad sound warm rather than like
    // a test tone.
    this.ratios = detune > 0 ? [1 - detune * 0.5, 1 + detune * 0.5] : [1];
    this.phases = this.ratios.map((_, i) => (i ? Math.random() : 0));
    this.norm = 1 / this.ratios.length;
    this.vibDepth = vibDepth;
    this.vibHz = vibHz;
    this.vph = Math.random();
    this.t = 0;
    this.att = Math.max(1, Math.floor(attack * sampleRate));
    this.relLen = Math.max(1, Math.floor(release * sampleRate));
    this.rel = -1;
  }

  get held() { return this.rel < 0; }
  release() { if (this.rel < 0) this.rel = 0; }

  render(out, n) {
    const gain = this.stepGain();
    const vibInc = this.vibHz / sampleRate;
    const ph = this.phases.slice();
    let vph = this.vph, t = this.t, rel = this.rel;
    for (let s = 0; s < n; s++) {
      // Vibrato fades in, the way a player settles into a held note.
      const fade = Math.min(1, t / (0.35 * sampleRate));
      const vib = 1 + this.vibDepth * fade * SINE[((vph * TN) | 0) & TMASK];
      let e = Math.min(1, t / this.att);
      e = e * e * (3 - 2 * e);                      // smoothstep attack
      if (rel >= 0) { const r = Math.max(0, 1 - rel / this.relLen); e *= r * r; rel++; }
      let sig = 0;
      for (let k = 0; k < ph.length; k++) {
        sig += this.tab[((ph[k] * SAW_N) | 0) & SAW_MASK];
        ph[k] += this.inc * this.ratios[k] * vib;
        if (ph[k] >= 1) ph[k] -= 1;
      }
      out[s] += sig * this.norm * e * gain;
      vph += vibInc; if (vph >= 1) vph -= 1;
      t++;
    }
    this.phases = ph; this.vph = vph; this.t = t;
    if (this.rel >= 0) { this.rel = rel; if (rel >= this.relLen) this.done = true; }
  }
}

// ---------------------------------------------------------------------------
// instruments -- gains loudness-matched, carried over from the Python build
// ---------------------------------------------------------------------------

const FREF = 261.6;
const tilt = (f, p) => Math.pow(FREF / Math.max(f, 20), p);
const harmonics = (k, rolloff, t0, fall) => {
  const n = [], g = [], t = [];
  for (let i = 1; i <= k; i++) {
    n.push(i);
    g.push(Math.pow(i, -rolloff) * Math.exp(-i * 0.045));
    t.push(t0 / Math.pow(i, fall));
  }
  return [n, g, t];
};

const INSTRUMENTS = [
  { name: "Harp", gain: 1.46, sustains: false,
    make: (f, a, b, d) => new Plucked(f, a, 0.55 + 0.45 * b, (1.1 + 2.1 * d) * tilt(f, 0.3), 0) },
  { name: "Guitar", gain: 1.21, sustains: false,
    make: (f, a, b, d) => new Plucked(f, a, 0.3 + 0.45 * b, (0.7 + 1.5 * d) * tilt(f, 0.35), 0.14) },
  // Sustain sits high and the decay toward it is slow, so a chord keeps its
  // body for a second or two instead of thinning out straight after the hit;
  // the long release lets one chord ring on under the next.
  { name: "Piano", gain: 0.19, sustains: true,
    make: (f, a, b, d) => {
      const [n, g, t] = harmonics(8, 1.15 - 0.35 * b, 2.6 + 3.0 * d, 0.72);
      return new Additive(f, a, n, g, t.map((x) => x * tilt(f, 0.22)), 0.004, 0.00028, 0.55, 1.4);
    } },
  { name: "Violin", gain: 0.12, sustains: true,
    make: (f, a, b) => new Bowed(f, a, 0.25 + 0.75 * b) },
  { name: "Kalimba", gain: 0.49, sustains: false,
    make: (f, a, b, d) => {
      const n = [1, 2, 3.02, 5.1, 7.3];
      const g = [1, 0.3, 0.16, 0.07, 0.035].map((x) => x * (0.55 + 0.75 * b));
      const t = [1, 0.55, 0.33, 0.18, 0.11].map((x) => x * (0.6 + 1.1 * d) * tilt(f, 0.25));
      return new Additive(f, a, n, g, t, 0.002);
    } },
  { name: "Bells", gain: 0.21, sustains: true,
    make: (f, a, b, d) => {
      const n = [0.5, 1, 1.19, 1.56, 2, 2.51, 2.66, 3.01];
      const g = [0.55, 1, 0.42, 0.34, 0.55, 0.22, 0.18, 0.13].map((x) => x * (0.5 + 0.8 * b));
      const t = [1, 0.85, 0.6, 0.46, 0.52, 0.32, 0.26, 0.2]
        .map((x) => x * (2.4 + 4.6 * d) * tilt(f, 0.18));
      return new Additive(f, a, n, g, t, 0.004, 0, 0.38, 0.9);
    } },
  // A soft pad that holds flat. Six partials falling away steeply, which is
  // why it is gentle rather than buzzy -- a sawtooth carries every harmonic at
  // 1/n and that brightness is what made the first attempt sound robotic. The
  // rolloff drifts slowly while the note is held, and the partials are doubled
  // a few cents apart so the two copies beat against each other.
  { name: "Synth", gain: 0.14, sustains: true,
    make: (f, a, b) => {
      const det = 0.004, ratios = [], gains = [], t60 = [];
      for (const side of [1 - det * 0.5, 1 + det * 0.5])
        for (let n = 1; n <= 6; n++) {
          ratios.push(n * side);
          gains.push(0.5 * Math.pow(n, -(2.6 - 0.5 * b)));
          t60.push(9);                       // unused: sustain holds it flat
        }
      return new Additive(f, a, ratios, gains, t60, 0.18, 0, 1.0, 1.3, 0.30, 0.17);
    } },
];

// ---------------------------------------------------------------------------
// reverb
// ---------------------------------------------------------------------------

class Reverb {
  constructor(rt60 = 2, mix = 0.26) {
    this.mix = mix;
    this.combs = [1557, 1617, 1491, 1422].map((d) => ({
      buf: new Float32Array(d), i: 0, g: Math.pow(10, (-3 * d) / (rt60 * sampleRate)),
    }));
    this.aps = [225, 556].map((d) => ({ buf: new Float32Array(d), i: 0, g: 0.5 }));
    this.wet = new Float32Array(256);
  }
  process(x, n) {
    const wet = this.wet;
    wet.fill(0, 0, n);
    for (const c of this.combs) {
      const { buf, g } = c; const len = buf.length; let i = c.i;
      for (let s = 0; s < n; s++) {
        const old = buf[i];
        wet[s] += old;
        buf[i] = x[s] + old * g;
        i = i + 1 >= len ? 0 : i + 1;
      }
      c.i = i;
    }
    for (let s = 0; s < n; s++) wet[s] *= 0.25;
    for (const a of this.aps) {
      const { buf, g } = a; const len = buf.length; let i = a.i;
      for (let s = 0; s < n; s++) {
        const vOld = buf[i];
        const vNew = wet[s] + vOld * g;
        buf[i] = vNew;
        wet[s] = vOld - vNew * g;
        i = i + 1 >= len ? 0 : i + 1;
      }
      a.i = i;
    }
    for (let s = 0; s < n; s++) x[s] += wet[s] * this.mix;
  }
}

// ---------------------------------------------------------------------------
// the processor
// ---------------------------------------------------------------------------

class AirHarpProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.voices = [];
    this.groups = new Map();       // key -> Map(noteId -> voice)
    this.instrument = 0;
    this.master = 0.62;
    this.reverb = new Reverb();
    this.buf = new Float32Array(256);
    this.peak = 0;
    this.port.onmessage = (e) => this.command(e.data);
  }

  command(m) {
    switch (m.type) {
      case "instrument":
        if (m.index !== this.instrument) this.releaseAll();
        this.instrument = m.index % INSTRUMENTS.length;
        break;
      case "chord": this.setChord(m.key, m.freqs, m.amp, m.bright, m.damp); break;
      case "level": this.setLevel(m.key, m.amp); break;
      case "release": this.release(m.key); break;
      case "releaseAll": this.releaseAll(); break;
    }
  }

  add(v) {
    if (this.voices.length >= MAX_VOICES) {
      // Steal something already ringing out before anything being held.
      let i = this.voices.findIndex((x) => !x.held);
      if (i < 0) i = 0;
      const stolen = this.voices.splice(i, 1)[0];
      for (const g of this.groups.values())
        for (const [id, v] of g) if (v === stolen) g.delete(id);
    }
    this.voices.push(v);
  }

  setChord(key, freqs, amp, bright, damp) {
    const inst = INSTRUMENTS[this.instrument];
    // Five notes at full level is five times the amplitude; back off so a big
    // chord is about as loud as a small one.
    const each = (amp * inst.gain) / Math.pow(Math.max(freqs.length, 1), 0.55);

    if (!inst.sustains) {
      for (const f of freqs) this.add(inst.make(f, each, bright, damp));
      return;
    }
    let g = this.groups.get(key);
    if (!g) { g = new Map(); this.groups.set(key, g); }
    const want = new Map(freqs.map((f) => [Math.round(f * 4), f]));
    for (const [id, v] of [...g]) if (!want.has(id)) { v.release(); g.delete(id); }
    for (const [id, f] of want) {
      const v = g.get(id);
      if (v && v.held) v.retarget(each);
      else { const nv = inst.make(f, each, bright, damp); g.set(id, nv); this.add(nv); }
    }
  }

  setLevel(key, amp) {
    const g = this.groups.get(key);
    if (!g) return;
    const inst = INSTRUMENTS[this.instrument];
    const each = (amp * inst.gain) / Math.pow(Math.max(g.size, 1), 0.55);
    for (const v of g.values()) v.retarget(each);
  }

  release(key) {
    const g = this.groups.get(key);
    if (!g) return;
    for (const v of g.values()) v.release();
    this.groups.delete(key);
  }

  releaseAll() {
    for (const g of this.groups.values()) for (const v of g.values()) v.release();
    this.groups.clear();
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    const n = out[0].length;
    const buf = this.buf;
    buf.fill(0, 0, n);

    if (this.voices.length) {
      for (const v of this.voices) v.render(buf, n);
      if (this.voices.some((v) => v.done)) {
        this.voices = this.voices.filter((v) => !v.done);
        for (const g of this.groups.values())
          for (const [id, v] of [...g]) if (v.done) g.delete(id);
      }
    }
    this.reverb.process(buf, n);

    let peak = 0;
    for (let s = 0; s < n; s++) {
      // Soft clip: a fistful of notes at once should compress, not crack.
      const y = Math.tanh(buf[s] * this.master);
      out[0][s] = y;
      if (out.length > 1) out[1][s] = y;
      const a = y < 0 ? -y : y;
      if (a > peak) peak = a;
    }
    this.peak = Math.max(peak, this.peak * 0.86);
    return true;
  }
}

registerProcessor("airharp", AirHarpProcessor);
