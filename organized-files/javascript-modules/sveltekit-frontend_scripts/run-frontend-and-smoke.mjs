#!/usr/bin/env node
import { spawn } from 'child_process';
import fetch from 'node-fetch';
import path from 'path';

const ROOT = process.cwd();
const smoke = path.join(ROOT, 'scripts', 'smoke.ts');

function spawnFrontend() {
  console.log('▶️ Starting SvelteKit frontend (vite dev)...');
  // Use npm script 'dev' so environment from package.json is respected
  const p = spawn(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'dev'], { cwd: ROOT, shell: true, stdio: ['ignore', 'pipe', 'pipe'] });
  p.stdout.on('data', (d) => process.stdout.write(`[frontend] ${d}`));
  p.stderr.on('data', (d) => process.stderr.write(`[frontend ERR] ${d}`));
  p.on('close', (code) => console.log('frontend exited with', code));
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
  const frontend = spawnFrontend();
  console.log('⏳ Waiting for frontend at http://localhost:5173 (120s timeout)...');
  const ok = await waitForUrl('http://localhost:5173/', 120000);
  if (!ok) {
    console.error('❌ Frontend did not become healthy in time. Check frontend logs.');
    process.exit(2);
  }
  console.log('✅ Frontend reachable — running smoke test (health + routes)');
  const smokeProc = spawn('npx', ['-y', 'tsx', smoke, '--crawl-routes', '--host', 'http://localhost:5173', '--routes-file', '../scripts/routes-to-test.json'], { cwd: ROOT, stdio: 'inherit', shell: true });
  smokeProc.on('close', (code) => {
    console.log('Smoke test exited with', code);
    if (code === 0) console.log('🎉 Smoke checks passed');
    else console.error('❌ Smoke checks failed');
    // keep frontend running for debugging
  });
}

run().catch((e) => { console.error(e); process.exit(1); });
