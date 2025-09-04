#!/usr/bin/env node
/**
 * Capture Svelte + TS errors into .vscode/svelte-check.json
 */
import { spawn } from 'node:child_process';
import { writeFile, mkdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const OUT_DIR = '.vscode';
const OUT_FILE = `${OUT_DIR}/svelte-check.json`;

async function run() {
  if (!existsSync(OUT_DIR)) await mkdir(OUT_DIR, { recursive: true });
  console.log('▶ Running svelte-check (JSON capture)...');
  // Prefer local binary to avoid npx dependency issues on some Windows dev shells
  const cmd = process.platform === 'win32' ? 'node' : 'node';
  const localBin = 'node_modules/svelte-check/bin/svelte-check';
  const args = [localBin, '--tsconfig', 'sveltekit-frontend/tsconfig.json'];
  const proc = spawn(cmd, args, { stdio: ['ignore','pipe','pipe'] });
  let stdout = '';
  let stderr = '';
  proc.stdout.on('data', d => stdout += d.toString());
  proc.stderr.on('data', d => stderr += d.toString());
  const code = await new Promise(res => proc.on('close', res));

  // Fallback: svelte-check may not support JSON yet; store raw output
  await writeFile(OUT_FILE, JSON.stringify({ code, capturedAt: new Date().toISOString(), raw: stdout + stderr }, null, 2));
  console.log(`✅ Captured output (${code}) -> ${OUT_FILE}`);
  if (code !== 0) process.exitCode = 0; // do not fail pipeline here
}

run().catch(e => { console.error('capture failed', e); process.exit(1); });
