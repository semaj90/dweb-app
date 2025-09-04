#!/usr/bin/env node
/**
 * Lightweight readiness probe & flag generator.
 *
 * Features:
 *  - Checks required TCP ports & HTTP health endpoints.
 *  - Optional wait / retry loop until ready.
 *  - Writes a platform-ready.flag file (JSON metadata) when all required checks pass.
 *  - Exit code 0 when ready, 1 otherwise.
 *
 * Usage:
 *    node scripts/is-ready.mjs                # immediate check
 *    node scripts/is-ready.mjs --wait 90       # wait up to 90s
 *    node scripts/is-ready.mjs --wait 120 --interval 5 --json
 *    node scripts/is-ready.mjs --flag ../platform-ready.flag
 */
import fs from 'fs';
import path from 'path';
import net from 'net';
import { exec as _exec } from 'child_process';
import { promisify } from 'util';

const exec = promisify(_exec);

// -------- Configuration --------
const REQUIRED = [
  { name: 'Frontend', port: 5173, url: 'http://localhost:5173' },
  { name: 'Main Service', port: 8080, url: 'http://localhost:8080/health', optional: true }, // health fallback if exists
  { name: 'Upload Service', port: 8093, url: 'http://localhost:8093/health', optional: true },
  { name: 'PostgreSQL', port: 5432 },
  { name: 'Redis', port: 6379 },
  { name: 'Ollama', port: 11434, url: 'http://localhost:11434/api/tags' }
];
const OPTIONAL = [
  { name: 'MinIO', port: 9000, url: 'http://localhost:9000/minio/health', optional: true },
  { name: 'Qdrant', port: 6333, url: 'http://localhost:6333' }
];

// Command line args
const args = process.argv.slice(2);
function getArg(flag, def) {
  const idx = args.indexOf(flag);
  if (idx === -1) return def;
  const val = args[idx + 1];
  if (!val || val.startsWith('--')) return true; // boolean flag
  return val;
}

const WAIT_SECONDS = parseInt(getArg('--wait', '0'), 10) || 0;
const INTERVAL_SECONDS = parseInt(getArg('--interval', '3'), 10) || 3;
const OUTPUT_JSON = Boolean(getArg('--json', false));
const FLAG_PATH = path.resolve(getArg('--flag', path.join(process.cwd(), '..', 'platform-ready.flag')));
const MINIMAL = args.includes('--minimal');

// -------- Helpers --------
function checkPort(port, host = '127.0.0.1', timeoutMs = 1200) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let done = false;
    const onResult = (ok) => { if (!done) { done = true; socket.destroy(); resolve(ok); } };
    socket.setTimeout(timeoutMs);
    socket.once('error', () => onResult(false));
    socket.once('timeout', () => onResult(false));
    socket.connect(port, host, () => onResult(true));
  });
}

async function checkHttp(url, timeoutMs = 3000) {
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

async function runChecks() {
  const timestamp = new Date().toISOString();
  const results = [];
  for (const svc of [...REQUIRED, ...OPTIONAL]) {
    const portOk = await checkPort(svc.port);
    let httpOk = true;
    if (svc.url) {
      httpOk = await checkHttp(svc.url).catch(() => false);
    }
    const ok = portOk && httpOk;
    const required = !svc.optional && !MINIMAL; // minimal mode treats everything as optional
    results.push({ name: svc.name, port: svc.port, portOk, httpOk, ok, required });
  }
  const allRequired = results.filter(r => r.required).every(r => r.ok);
  return { timestamp, results, allRequired };
}

function writeFlag(meta) {
  try {
    const payload = {
      generatedAt: meta.timestamp,
      allRequired: meta.allRequired,
      services: meta.results.reduce((acc, r) => { acc[r.name] = r.ok; return acc; }, {}),
      details: meta.results
    };
    fs.writeFileSync(FLAG_PATH, JSON.stringify(payload, null, 2));
    return true;
  } catch (e) {
    console.error('Failed to write readiness flag:', e.message);
    return false;
  }
}

async function waitLoop() {
  const deadline = Date.now() + WAIT_SECONDS * 1000;
  while (true) {
    const meta = await runChecks();
    if (OUTPUT_JSON) {
      console.log(JSON.stringify(meta));
    } else {
      console.log(`[is-ready] ${meta.timestamp} required-ok=${meta.allRequired}`);
      for (const r of meta.results) {
        const status = r.ok ? '✅' : (r.required ? '❌' : '⚠️');
        console.log(`  ${status} ${r.name} (port:${r.port}) port=${r.portOk?'ok':'fail'} http=${r.httpOk?'ok':'fail'}${r.required?'':' (optional)'}`);
      }
    }
    if (meta.allRequired) {
      writeFlag(meta);
      return 0;
    }
    if (Date.now() > deadline) {
      return 1;
    }
    await new Promise(r => setTimeout(r, INTERVAL_SECONDS * 1000));
  }
}

waitLoop().then(code => process.exit(code));
