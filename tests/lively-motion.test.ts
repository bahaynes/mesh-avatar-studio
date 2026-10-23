import { expect, test, vi } from 'vitest';
import { Motion } from '../packages/runtime/src/engine/motion.js';
import { LivelyMotion } from '../packages/runtime/src/engine/lively-motion.js';
import { seededRandom } from '../tools/motion-scenarios.mjs';

type Style = 'classic' | 'lively';
function seeded<T>(run: () => T) {
  const random = vi.spyOn(Math, 'random').mockImplementation(seededRandom(1));
  try { return run(); } finally { random.mockRestore(); }
}
function mouse(style: Style = 'classic') {
  const motion = new Motion([0, 0], style);
  motion.mode = 'mouse'; motion.pointer = [0, 0];
  motion.autoIdle = false; motion.autoMotion = false;
  return motion;
}
const snapshot = (motion: Motion, dt: number): Record<string, number> => ({ ...motion.update(dt) });
const step = (motion: Motion, frames: number, fps = 60) => Array.from({ length: frames }, () => snapshot(motion, 1 / fps));

test.each([
  ['in', 0.149, false], ['in', 0.15, true], ['in', 0.151, true],
  ['out', 0.149, true], ['out', 0.15, true], ['out', 0.151, false],
] as const)('gaze blinks respect the half-weight boundary on fade-%s at %ss', (direction, seconds, expected) => {
  const random = vi.spyOn(Math, 'random').mockReturnValue(0);
  try {
    const motion = mouse(), lively = new LivelyMotion(motion, direction === 'out');
    lively.prepare(motion, { ...motion.P }, 0);
    lively.setEnabled(direction === 'in');
    // An old enough blink allows this test to isolate blending from the cooldown.
    motion.t = 1;
    const scheduled = motion.blink.start;
    random.mockClear();
    lively.prepare(motion, { ...motion.P, gazeX: 1 }, seconds);
    expect(motion.blink.start).toBe(expected ? motion.blink.t : scheduled);
    // Three micro-saccade draws still precede the optional blink-probability draw.
    expect(random).toHaveBeenCalledTimes(expected ? 4 : 3);
  } finally { random.mockRestore(); }
});

test('lively is deterministic across idle, kana and audio syllables with a seed', () => {
  const run = () => seeded(() => {
    const motion = new Motion([0, 0], 'lively');
    const trace = step(motion, 300);
    motion.speakKana('あいうえお。こんにちは。', { loop: true });
    trace.push(...step(motion, 180));
    motion.stopLipSync(); motion.setSpeaking(true);
    for (let i = 0; i < 180; i++) {
      motion.setVoiceLevel(i % 12 < 6 ? 0.9 : 0.1);
      trace.push(snapshot(motion, 1 / 60));
    }
    motion.setSpeaking(false); trace.push(...step(motion, 120));
    expect(trace.every(p => Object.values(p).every(Number.isFinite))).toBe(true);
    return trace;
  });
  expect(run()).toEqual(run());
});

test.each([30, 60, 120])('a lively gaze jump settles without overshoot before head and body at %i fps', fps => seeded(() => {
  const motion = mouse('lively');
  step(motion, fps * 2, fps);
  const initial = motion.P;
  motion.pointer = [500, 0];
  const trace = step(motion, fps * 5, fps);
  const eye = trace.findIndex(p => p.gazeX >= 0.9);
  const head = trace.findIndex(p => p.angleX >= 27);
  const body = trace.findIndex(p => p.bodyAngleX >= 10);
  expect((eye + 1) / fps).toBeLessThanOrEqual(0.067);
  expect((head - eye) / fps).toBeGreaterThan(0.15);
  expect(body).toBeGreaterThan(head);
  expect(trace.slice(0, Math.ceil(fps * 0.1)).every(p => p.gazeX <= 1 && p.gazeX >= initial.gazeX)).toBe(true);
}));

test('style switching in either direction stays within classic per-frame movement', () => {
  const baseline = seeded(() => {
    const motion = mouse();
    step(motion, 120);
    motion.pointer = [500, -350];
    const trace = [motion.P, ...step(motion, 240)];
    // Mouse mode keeps body roll fixed; include classic idle's natural roll as its baseline.
    const idle = new Motion([0, 0]);
    return [trace, step(idle, 1800)];
  });
  const ids = ['gazeX', 'gazeY', 'angleX', 'angleY', 'angleZ', 'bodyAngleX', 'bodyAngleZ', 'breath'];
  const maxima = Object.fromEntries(ids.map(id => [id, Math.max(...baseline.flatMap(trace => trace.slice(1).map((p, i) => Math.abs(p[id] - trace[i][id]))))]));
  seeded(() => {
    const motion = mouse(); motion.pointer = [500, -350]; step(motion, 240);
    for (const style of ['lively', 'classic', 'lively', 'classic'] as const) {
      const before = { ...motion.P };
      motion.setMotionStyle(style);
      expect(motion.update(0)).toEqual(before);
      const trace = [before, ...step(motion, 18)];
      for (const id of ids) {
        const max = Math.max(...trace.slice(1).map((p, i) => Math.abs(p[id] - trace[i][id])));
        expect(max, `${style}: ${id}`).toBeLessThanOrEqual(maxima[id] + 1e-12);
      }
      step(motion, 120);
    }
  });
});

test('manual poses ignore style, including switches while held', () => {
  const run = (style: Style) => seeded(() => {
    const motion = new Motion([0, 0], style);
    motion.mode = 'manual'; motion.manual.angleX = 15; motion.manual.gazeX = 0.6;
    motion.speakKana('こんにちは');
    const trace = step(motion, 60);
    motion.setMotionStyle(style === 'classic' ? 'lively' : 'classic');
    return [...trace, ...step(motion, 60)];
  });
  expect(run('classic')).toEqual(run('lively'));
});

test.each(['kana', 'audio'])('%s onsets add small nods and a breath after the line, scaled by talkGain', source => {
  const run = (gain: number) => seeded(() => {
    const motion = mouse('lively'); motion.talkGain = gain;
    step(motion, 120);
    if (source === 'kana') motion.speakKana('あいうえお');
    else motion.setSpeaking(true);
    const trace = [];
    for (let i = 0; i < 48; i++) {
      if (source === 'audio') motion.setVoiceLevel(i % 12 < 6 ? 0.9 : 0.05);
      trace.push(snapshot(motion, 1 / 60));
    }
    motion.setSpeaking(false);
    trace.push(...step(motion, 120));
    return trace;
  });
  const active = run(1), quiet = run(0);
  expect(Math.min(...active.slice(0, 48).map((p, i) => p.angleY - quiet[i].angleY))).toBeLessThan(-0.3);
  expect(Math.max(...active.map((p, i) => p.browY - quiet[i].browY))).toBeGreaterThan(0.02);
  expect(Math.max(...active.slice(48).map((p, i) => p.breath - quiet[i + 48].breath))).toBeGreaterThan(0.08);
});
