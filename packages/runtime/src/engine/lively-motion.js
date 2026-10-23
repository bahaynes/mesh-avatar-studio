// Optional coupling state. The classic path never constructs or advances this helper.
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ease = t => t * t * (3 - 2 * t);
const follow = (a, b, dt, period) => a + (b - a) * (1 - Math.exp(-dt / period));

export class LivelyMotion {
  constructor(motion, immediate = false) {
    this.mix = immediate ? 1 : 0;
    this.from = this.mix;
    this.target = 1;
    this.elapsed = 0;
    this.gaze = [motion.cur.gazeX, motion.cur.gazeY];
    this.start = [...this.gaze];
    this.goal = [...this.gaze];
    this.jumpTime = 0.05;
    this.previous = null;
    this.micro = [0, 0];
    this.nextMicro = motion.t + 0.6;
    this.head = null;
    this.body = [0, 0];
    this.lastBlink = motion.t;
    this.nod = 0;
    this.brow = 0;
    this.lineActive = false;
    this.breathTime = 2;
    this.kana = null;
    this.mora = -1;
    this.kanaTime = 0;
  }

  setEnabled(on) {
    this.from = this.mix;
    this.target = on ? 1 : 0;
    this.elapsed = 0;
  }

  syllable(level) {
    // A bounded impulse prevents a long string of fast morae accumulating a large tilt.
    this.nod = Math.min(3, this.nod + 2.5 * level);
    if (level > 0.6) this.brow = Math.min(0.12, this.brow + 0.08 * level);
  }

  prepare(motion, T, dt) {
    this.elapsed += dt;
    this.mix = this.from + (this.target - this.from) * ease(Math.min(1, this.elapsed / 0.3));
    const w = this.mix;
    this.wasArmed = motion.syllableArmed;
    this.nod *= Math.exp(-dt / 0.16);
    this.brow *= Math.exp(-dt / 0.12);
    this.breathTime += dt;

    if (motion.t >= this.nextMicro) {
      const angle = Math.random() * Math.PI * 2, amplitude = 0.02 + Math.random() * 0.02;
      this.micro = [Math.cos(angle) * amplitude, Math.sin(angle) * amplitude];
      this.nextMicro = motion.t + 0.4 + Math.random() * 0.8;
    }
    const raw = [T.gazeX, T.gazeY];
    if (w >= 0.5 && this.previous && Math.hypot(raw[0] - this.previous[0], raw[1] - this.previous[1]) > 0.35
      && motion.t - this.lastBlink >= 0.8 && motion.blink.t < motion.blink.start && Math.random() < 0.5) {
      motion.blink.start = motion.blink.t;
      motion.blink.double = false;
      this.lastBlink = motion.t;
    }
    this.previous = raw;
    const goal = raw.map((v, i) => clamp(v + this.micro[i], -1, 1));
    if (Math.hypot(goal[0] - this.goal[0], goal[1] - this.goal[1]) > 0.005) {
      this.start = [...this.gaze]; this.goal = goal; this.jumpTime = 0;
    }
    this.jumpTime = Math.min(0.05, this.jumpTime + dt);
    this.gaze = this.goal.map((v, i) => this.start[i] + (v - this.start[i]) * ease(this.jumpTime / 0.05));

    let head;
    if (motion.mode === 'auto') {
      const gaze = [motion.gaze.x + this.micro[0], motion.gaze.y + this.micro[1]];
      // Keep wandering, speech and authored tracks; delay only the gaze contribution.
      if (!this.head) this.head = [this.classicGaze[0] * 11, this.classicGaze[1] * 7, 0];
      this.head[0] = follow(this.head[0], gaze[0] * 11, dt, 0.2);
      this.head[1] = follow(this.head[1], gaze[1] * 7, dt, 0.2);
      head = [this.head[0], this.head[1]];
      // The classic auto target used the preceding glance before picking a new one.
      T.angleX += (head[0] - this.classicGaze[0] * 11) * w;
      T.angleY += (head[1] - this.classicGaze[1] * 7) * w;
    } else if (motion.mode === 'mouse' && motion.pointer) {
      if (!this.head) this.head = [motion.cur.angleX, motion.cur.angleY, motion.cur.angleZ];
      for (const [i, id] of ['angleX', 'angleY', 'angleZ'].entries()) {
        this.head[i] = follow(this.head[i], T[id], dt, 0.2);
        T[id] += (this.head[i] - T[id]) * w;
      }
    }
    this.body[0] = follow(this.body[0], motion.cur.angleX * 0.25, dt, 1);
    this.body[1] = follow(this.body[1], motion.cur.angleZ * 0.25, dt, 1);
    T.bodyAngleX += this.body[0] * w;
    T.bodyAngleZ += this.body[1] * w;
    T.angleY -= this.nod * motion.talkGain * w;
    if (this.breathTime < 1.6) T.breath = clamp(T.breath + Math.sin(this.breathTime / 1.6 * Math.PI) * 0.2 * motion.talkGain * w, 0, 1.2);
  }

  finish(motion, out) {
    if (motion.blink.t >= motion.blink.start) this.lastBlink = motion.t;
    for (const [i, id, expression] of [[0, 'gazeX', 'gx'], [1, 'gazeY', 'gy']]) {
      const value = clamp(this.gaze[i] + motion.exprMix[expression], -1, 1);
      out[id] += (value - out[id]) * this.mix;
    }
    out.browY += this.brow * motion.talkGain * this.mix;
    const kana = motion.kana;
    if (kana) {
      let start = 0, index = 0;
      while (index < kana.moras.length && kana.t >= start + kana.moras[index].dur) start += kana.moras[index++].dur;
      if (kana !== this.kana || index !== this.mora || kana.t < this.kanaTime) {
        const vowel = kana.moras[index]?.vowel;
        if (vowel && vowel !== 'n') this.syllable(vowel === 'a' || vowel === 'o' ? 1 : 0.75);
      }
      this.mora = index;
      this.kanaTime = kana.t;
    } else if (motion.speaking && this.wasArmed && !motion.syllableArmed) this.syllable(motion.voice);
    this.kana = kana;
    const active = motion.speaking || kana !== null;
    if (this.lineActive && !active) this.breathTime = 0;
    this.lineActive = active;
  }
}
