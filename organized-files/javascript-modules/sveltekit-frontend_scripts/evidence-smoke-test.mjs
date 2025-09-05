#!/usr/bin/env node
/**
 * Evidence System Smoke Test
 * Sequential lightweight validation of core evidence endpoints.
 *
 * Steps:
 * 1. Health check
 * 2. Upload test PDF
 * 3. Trigger processing
 * 4. Open WebSocket stream until processing-complete (or timeout)
 * 5. Analyze
 * 6. Validate
 * 7. Hash verify
 * 8. Summarize PASS/FAIL
 *
 * Usage:
 *   node scripts/evidence-smoke-test.mjs --base http://localhost:5173 \
 *       --file ../lawpdfs/test-document.pdf --timeout 20000
 *
 * Exit codes: 0 success, 1 failure
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { WebSocket } from 'ws';

// ----------- Arg parsing -----------
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith('--')) {
    const key = cur.replace(/^--/, '');
    const next = arr[i + 1];
    if (!next || next.startsWith('--')) acc.push([key, 'true']); else acc.push([key, next]);
  }
  return acc;
}, []));

const BASE = args.base || process.env.EVIDENCE_BASE || 'http://localhost:5173';
const FILE_PATH = args.file || '../lawpdfs/test-document.pdf';
const TIMEOUT_MS = parseInt(args.timeout || '30000', 10);

// ----------- Helpers -----------
const results = [];
const record = (name, ok, info = '') => {
  results.push({ name, ok, info });
  const sym = ok ? '✅' : '❌';
  console.log(`${sym} ${name}${info ? ' - ' + info : ''}`);
};

const abortController = new AbortController();
const timeout = setTimeout(() => {
  abortController.abort();
}, TIMEOUT_MS);

function summarizeAndExit() {
  clearTimeout(timeout);
  const failed = results.filter(r => !r.ok);
  console.log('\n--- Summary ---');
  results.forEach(r => console.log(`${r.ok ? 'PASS' : 'FAIL'} | ${r.name} | ${r.info}`));
  console.log(`\nOverall: ${failed.length === 0 ? 'PASS' : 'FAIL'} (${results.length - failed.length}/${results.length} passed)`);
  process.exit(failed.length === 0 ? 0 : 1);
}

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });
  let body = null;
  const txt = await res.text();
  try { body = txt ? JSON.parse(txt) : null; } catch { body = txt; }
  return { ok: res.ok, status: res.status, body, text: txt };
}

// ----------- Main Flow -----------
(async () => {
  console.log(`🔍 Evidence Smoke Test starting (base=${BASE}, file=${FILE_PATH}, timeout=${TIMEOUT_MS}ms)`);

  // 1. Health
  try {
    const r = await fetch(`${BASE}/api/health`);
    record('Health Check', r.ok, `status=${r.status}`);
  } catch (e) {
    record('Health Check', false, e.message);
  }

  // 2. Upload
  let fileId = null;
  let sessionId = null;
  try {
    const abs = path.resolve(FILE_PATH);
    const data = fs.readFileSync(abs);
    const form = new FormData();
    form.append('file', new Blob([data]), path.basename(abs));
    const r = await fetch(`${BASE}/api/evidence/upload`, { method: 'POST', body: form });
    const txt = await r.text();
    let parsed; try { parsed = JSON.parse(txt); } catch { parsed = { raw: txt }; }
    fileId = parsed.fileId || parsed.id || parsed.file?.id || null;
    sessionId = parsed.sessionId || parsed.streamSessionId || null;
    record('Upload File', r.ok && !!fileId, `status=${r.status} fileId=${fileId || 'n/a'}`);
  } catch (e) {
    record('Upload File', false, e.message);
  }

  // 3. Trigger processing if needed
  if (fileId) {
    try {
      const { ok, status, body } = await jsonFetch(`${BASE}/api/evidence/process`, { method: 'POST', body: JSON.stringify({ fileId }) });
      sessionId = sessionId || body?.sessionId || body?.id || body?.streamSessionId || sessionId;
      record('Start Processing', ok, `status=${status} sessionId=${sessionId || 'n/a'}`);
    } catch (e) {
      record('Start Processing', false, e.message);
    }
  }

  // 4. WebSocket stream (optional if sessionId present)
  if (sessionId) {
    await new Promise(resolve => {
      let done = false;
      const ws = new WebSocket(`${BASE.replace(/^http/, 'ws')}/api/evidence/stream/${sessionId}`);
      const timer = setTimeout(() => { if (!done) { record('WebSocket Stream', false, 'timeout'); ws.close(); resolve(); } }, 12000);
      ws.on('open', () => { /* noop */ });
      ws.on('message', raw => {
        try {
          const msg = JSON.parse(raw.toString());
          if (msg.type === 'processing-step') {
            // first observable step counts success
            if (!done) {
              record('WebSocket Stream', true, `step=${msg.step}`);
              done = true;
            }
          }
          if (msg.type === 'processing-complete') {
            if (!done) record('WebSocket Stream', true, 'complete immediate');
            clearTimeout(timer); ws.close(); resolve();
          }
        } catch { /* ignore */ }
      });
      ws.on('error', err => {
        if (!done) record('WebSocket Stream', false, err.message);
        clearTimeout(timer); resolve();
      });
      ws.on('close', () => { if (!done) { record('WebSocket Stream', false, 'closed without data'); } clearTimeout(timer); resolve(); });
    });
  } else {
    record('WebSocket Stream', false, 'no sessionId');
  }

  // 5. Analyze
  if (fileId) {
    try {
      const { ok, status } = await jsonFetch(`${BASE}/api/evidence/analyze`, { method: 'POST', body: JSON.stringify({ fileId }) });
      record('Analyze Endpoint', ok, `status=${status}`);
    } catch (e) { record('Analyze Endpoint', false, e.message); }
  } else {
    record('Analyze Endpoint', false, 'no fileId');
  }

  // 6. Validate
  if (fileId) {
    try {
      const { ok, status } = await jsonFetch(`${BASE}/api/evidence/validate`, { method: 'POST', body: JSON.stringify({ fileId }) });
      record('Validate Endpoint', ok, `status=${status}`);
    } catch (e) { record('Validate Endpoint', false, e.message); }
  } else {
    record('Validate Endpoint', false, 'no fileId');
  }

  // 7. Hash verify
  if (fileId) {
    try {
      // Compute local hash
      const abs = path.resolve(FILE_PATH);
      const data = fs.readFileSync(abs);
      const localHash = crypto.createHash('sha256').update(data).digest('hex');
      const { ok, status, body } = await jsonFetch(`${BASE}/api/evidence/hash`, { method: 'POST', body: JSON.stringify({ fileId }) });
      const remoteHash = body?.hash || body?.sha256 || body?.data?.hash;
      record('Hash Verify', ok && !!remoteHash && remoteHash === localHash, `status=${status} match=${remoteHash === localHash}`);
    } catch (e) { record('Hash Verify', false, e.message); }
  } else {
    record('Hash Verify', false, 'no fileId');
  }

  summarizeAndExit();
})().catch(e => { record('Unhandled', false, e.message); summarizeAndExit(); });
