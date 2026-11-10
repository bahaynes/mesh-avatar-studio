// Homeserver-only: a /__picker page in the Studio dev server for drawn-variant generation and picking
// (homeserver/comfy-variants.py), so nobody has to type --pick lines. Same Caddy + Authelia gate as Studio.
// ponytail: one job at a time, in memory; a dev-server restart forgets a running job (its log file stays).
import { spawn, execFile } from 'node:child_process';
import { copyFile, readdir, readFile, writeFile, stat, mkdir, appendFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { Plugin } from 'vite';
import type { IncomingMessage, ServerResponse } from 'node:http';

const NAME = /^[A-Za-z0-9_-]+$/;
const UV = ['run', '--no-project', '--no-python-downloads', '--python', '>=3.10,<3.14', '--with', 'numpy', '--with', 'pillow', '--with', 'opencv-python-headless'];
let job: { project: string; label: string; done: boolean; ok: boolean; log: string } | null = null;

const json = (res: ServerResponse, code: number, body: unknown) => { res.statusCode = code; res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(body)); };
const body = (req: IncomingMessage) => new Promise<any>((ok, bad) => { let s = ''; req.on('data', c => { s += c; if (s.length > 1e5) bad(new Error('too big')); }); req.on('end', () => { try { ok(JSON.parse(s || '{}')); } catch (e) { bad(e); } }); });
const ls = async (dir: string) => { try { return await readdir(dir); } catch { return []; } };

function run(root: string, project: string, label: string, steps: string[][], env: Record<string, string> = {}) {
  job = { project, label, done: false, ok: false, log: '' };
  const j = job;
  const logFile = join(root, 'projects', project, 'work', 'picker.log');
  const next = (i: number) => {
    if (i >= steps.length) { j.done = true; j.ok = true; return; }
    const p = spawn('uv', [...UV, ...steps[i]], { cwd: root, env: { ...process.env, ...env, PYTHONUNBUFFERED: '1' } });
    const add = (c: Buffer) => { const s = c.toString(); j.log = (j.log + s).slice(-20000); appendFile(logFile, s).catch(() => {}); };
    p.stdout.on('data', add); p.stderr.on('data', add);
    p.on('error', e => { j.log += `\n${e.message}`; j.done = true; });
    p.on('close', code => { if (code === 0) next(i + 1); else { j.log += `\nexit ${code}`; j.done = true; } });
  };
  mkdir(join(root, 'projects', project, 'work'), { recursive: true }).then(() => next(0));
}

async function projectState(root: string, project: string) {
  const dir = join(root, 'projects', project);
  const picks = JSON.parse(await readFile(join(dir, 'work', 'picks.json'), 'utf8').catch(() => '{}'));
  const tries = await ls(join(dir, 'work', 'tries'));
  const crops = new Set(await ls(join(dir, 'work', 'crops')));
  const variants = [];
  for (const v of (await ls(join(dir, 'variant-requests'))).filter(v => NAME.test(v)).sort()) {
    const n = tries.filter(f => f.startsWith(`${v}-`) && /^\d+\.png$/.test(f.slice(v.length + 1))).length;
    variants.push({ name: v, tries: n, crops: crops.has(`${v}-0.png`), picked: picks[v] ?? null, hasVariant: (await ls(join(dir, 'variants'))).includes(`${v}.png`) });
  }
  return { project, variants };
}

export function pickerPlugin(root: string): Plugin {
  const script = 'homeserver/comfy-variants.py';
  return {
    name: 'homeserver-picker',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://x');
        if (!url.pathname.startsWith('/__picker')) { next(); return; }
        try {
          const parts = url.pathname.split('/').filter(Boolean).slice(1);
          if (parts.length === 0) { res.setHeader('content-type', 'text/html'); res.end(await readFile(join(root, 'homeserver', 'picker.html'))); return; }
          const project = parts[1];
          if (project !== undefined && !NAME.test(project)) { json(res, 400, { error: 'bad project' }); return; }
          if (parts[0] === 'projects') {
            const all = [];
            for (const p of (await ls(join(root, 'projects'))).filter(p => NAME.test(p)).sort())
              if ((await ls(join(root, 'projects', p, 'variant-requests'))).length) all.push(p);
            json(res, 200, all); return;
          }
          if (parts[0] === 'state' && project) { json(res, 200, { ...(await projectState(root, project)), job }); return; }
          if (parts[0] === 'prompts' && project) {
            execFile('uv', [...UV, script, `projects/${project}`, '--prompts'], { cwd: root, timeout: 60_000 }, (e, out, err) =>
              e ? json(res, 500, { error: err.slice(-2000) || e.message }) : (res.setHeader('content-type', 'application/json'), res.end(out)));
            return;
          }
          if (parts[0] === 'img' && project) {
            const file = parts[2];
            if (!/^[A-Za-z0-9_-]+\.png$/.test(file ?? '')) { json(res, 400, { error: 'bad file' }); return; }
            const path = join(root, 'projects', project, 'work', 'crops', file);
            await stat(path);
            res.setHeader('content-type', 'image/png'); res.setHeader('cache-control', 'no-cache'); res.end(await readFile(path)); return;
          }
          if (req.method !== 'POST' || !project) { json(res, 404, { error: 'not found' }); return; }
          const b = await body(req);
          const dir = join(root, 'projects', project);
          if (parts[0] === 'pick') {
            if (!NAME.test(b.variant) || !Number.isInteger(b.try) || b.try < 1) { json(res, 400, { error: 'bad pick' }); return; }
            await mkdir(join(dir, 'variants'), { recursive: true });
            await copyFile(join(dir, 'work', 'tries', `${b.variant}-${b.try}.png`), join(dir, 'variants', `${b.variant}.png`));
            const picksPath = join(dir, 'work', 'picks.json');
            const picks = JSON.parse(await readFile(picksPath, 'utf8').catch(() => '{}'));
            picks[b.variant] = b.try;
            await writeFile(picksPath, JSON.stringify(picks, null, 2));
            json(res, 200, { ok: true }); return;
          }
          if (parts[0] === 'prompt') {
            if (!NAME.test(b.variant) || (b.text !== null && (typeof b.text !== 'string' || b.text.length > 4000))) { json(res, 400, { error: 'bad prompt' }); return; }
            const path = join(dir, 'work', 'prompt-overrides.json');
            const all = JSON.parse(await readFile(path, 'utf8').catch(() => '{}'));
            if (b.text === null || !b.text.trim()) delete all[b.variant]; else all[b.variant] = b.text.trim();
            await mkdir(join(dir, 'work'), { recursive: true });
            await writeFile(path, JSON.stringify(all, null, 2));
            json(res, 200, { ok: true }); return;
          }
          if (parts[0] === 'run') {
            if (job && !job.done) { json(res, 409, { error: `busy: ${job.label} on ${job.project}` }); return; }
            const variants: string[] = Array.isArray(b.variants) ? b.variants : [];
            if (!variants.every(v => NAME.test(v))) { json(res, 400, { error: 'bad variants' }); return; }
            const p = `projects/${project}`;
            if (b.action === 'generate') {
              const tries = Number(b.tries) || 3;
              if (tries < 1 || tries > 8) { json(res, 400, { error: 'tries 1-8' }); return; }
              // A fresh seed each run, or a rerun would redraw the same tries. Regenerated variants lose their pick.
              const seed = String(Math.floor(Math.random() * 1e9));
              const picksPath = join(dir, 'work', 'picks.json');
              const picks = JSON.parse(await readFile(picksPath, 'utf8').catch(() => '{}'));
              for (const v of variants.length ? variants : Object.keys(picks)) delete picks[v];
              await mkdir(join(dir, 'work'), { recursive: true });
              await writeFile(picksPath, JSON.stringify(picks, null, 2));
              run(root, project, `generate ${variants.join(' ') || 'all'} (seed ${seed})`, [[script, p, ...variants]], { TRIES: String(tries), SEED: seed });
            } else if (b.action === 'sheets') run(root, project, 'rebuild sheets', [[script, p, '--sheets', ...variants]]);
            else if (b.action === 'sprites') run(root, project, 'build sprites', [['tools/build-sprites.py', p]]);
            else { json(res, 400, { error: 'bad action' }); return; }
            json(res, 200, { ok: true }); return;
          }
          json(res, 404, { error: 'not found' });
        } catch (e) { json(res, 500, { error: (e as Error).message }); }
      });
    },
  };
}
