#!/usr/bin/env node
import { spawn } from 'child_process';
import fetch from 'node-fetch';
import path from 'path';
import fs from 'fs';

const ROOT = process.cwd();
const smoke = path.join(ROOT, 'scripts', 'smoke.ts');

function spawnDevFull() {
  console.log('▶️ Starting dev:full (coordinated full stack)...');
  // On Windows use the shell 'node' invocation to avoid issues with paths that contain spaces
  const isWin = process.platform === 'win32';
  const cmd = isWin ? 'node' : process.execPath;
  const spawnOpts = isWin ? { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], shell: true } : { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] };
  let p;
  try {
    p = spawn(cmd, ['scripts/start-coordinated-full-stack.mjs'], spawnOpts);
  } catch (e) {
    console.error('Failed to spawn dev:full process:', e);
    // return a minimal fake object that mimics the needed parts of ChildProcess
    const fake = {
      stdout: { on: () => {} },
      stderr: { on: () => {} },
      on: (ev, cb) => { if (ev === 'close') setImmediate(() => cb(1)); }
    };
    return fake;
  }

  p.stdout.on('data', (d) => process.stdout.write(`[dev:full] ${d}`));
  p.stderr.on('data', (d) => process.stderr.write(`[dev:full ERR] ${d}`));
  p.on('error', (err) => {
    console.error('[dev:full] child process error event:', err && err.message ? err.message : err);
    // allow wrapper to continue; log and rely on 'close' if it follows
  });
  p.on('close', (code) => console.log('dev:full exited with', code));
  return p;
}

async function waitForUrl(url, timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url, { method: 'GET' });
      if (r.ok) return true;
    } catch (e) {
      // ignore
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  return false;
}

async function run() {
  const dev = spawnDevFull();
  console.log('⏳ Waiting for frontend at http://localhost:5173 (120s timeout)...');

  // Race frontend readiness against the dev process exiting.
  const devExit = new Promise((resolve) => dev.on('close', (code) => resolve({ exited: true, code })));
  const frontendReady = waitForUrl('http://localhost:5173/', 120000);

  const race = await Promise.race([devExit, frontendReady]);

  let ok = false;
  if (race === true) {
    ok = true;
  } else if (race && race.exited) {
    // dev exited before frontend became reachable — check quickly if frontend is reachable now
    console.warn('⚠ dev:full process exited early with code', race.code);
    ok = await waitForUrl('http://localhost:5173/', 5000);
  }

  if (!ok) {
    console.error('❌ Frontend did not become healthy in time. Check dev:full logs.');
    process.exit(2);
  }

  console.log('✅ Frontend reachable — running smoke test (health + routes)');
  // run smoke via npx tsx to ensure TypeScript ESM runs
  const smokeProc = spawn('npx', ['-y', 'tsx', smoke, '--crawl-routes', '--host', 'http://localhost:5173', '--routes-file', '../scripts/routes-to-test.json'], { cwd: ROOT, stdio: 'inherit', shell: true });
  smokeProc.on('close', (code) => {
    console.log('Smoke test exited with', code);
    if (code === 0) console.log('🎉 Smoke checks passed');
    else console.error('❌ Smoke checks failed');
    // leave dev:full running for debugging
  });
}

run().catch((e) => { console.error(e); process.exit(1); });
