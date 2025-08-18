// CJS runtime script to update rolling performance baseline
const { readFileSync, writeFileSync } = require('node:fs');
const { join } = require('node:path');
const BASELINE_PATH = join(__dirname, 'perf-baseline.json');
const MAX_HISTORY = 20;

function loadBaseline() {
  try { return JSON.parse(readFileSync(BASELINE_PATH, 'utf8')); } catch { return { schemaVersion: 1, metric: 'perDoc', history: [], rollingMean: null, updated: null }; }
}

function saveBaseline(perDoc) {
  const data = loadBaseline();
  data.history.push(perDoc);
  if (data.history.length > MAX_HISTORY) data.history.splice(0, data.history.length - MAX_HISTORY);
  const sum = data.history.reduce((a, b) => a + b, 0);
  data.rollingMean = +(sum / data.history.length).toFixed(4);
  data.updated = new Date().toISOString();
  writeFileSync(BASELINE_PATH, JSON.stringify(data, null, 2));
  return data;
}

function runSplitterBenchmark(iterations = 50) {
  // Try to leverage the real TS benchmark for parity.
  try {
    // Attempt lightweight ts-node registration if available.
    try { require('ts-node/register/transpile-only'); } catch {}
    const real = require('./benchmark-splitter.ts');
    if (real && typeof real.runSplitterBenchmark === 'function') {
      return { perDoc: real.runSplitterBenchmark(iterations).perDoc };
    }
  } catch (e) {
    // Fallback below
  }
  const { performance } = require('node:perf_hooks');
  const re = /(?<=[.!?])\s+(?=[A-Z])/g;
  const sample = ['Section', 'Article', 'Exhibit', 'WHEREAS,', 'the', 'Party', 'Agreement', 'obligation', 'shall', 'be', 'binding', 'hereunder', 'Recital', 'A.', 'provided', 'however', 'that', 'pursuant', 'to', 'Sec.', '12.'];
  const texts = [];
  for (let i = 0; i < iterations; i++) {
    const words = 80 + Math.floor(Math.random() * 40);
    texts.push(Array.from({ length: words }, () => sample[Math.floor(Math.random() * sample.length)]).join(' ') + '.');
  }
  const start = performance.now();
  for (const t of texts) t.split(re);
  const ms = performance.now() - start;
  return { perDoc: ms / iterations };
}

(function main(){
  const allow = process.env.ALLOW_BASELINE_UPDATE === 'true';
  const branch = process.env.GIT_BRANCH || process.env.BRANCH_NAME || process.env.CI_COMMIT_REF_NAME;
  if (!allow) { console.log('[perf-baseline] Skipped (ALLOW_BASELINE_UPDATE!=true)'); return; }
  if (branch && !['main', 'master'].includes(branch)) { console.log(`[perf-baseline] Skipped (branch ${branch} not main/master)`); return; }
  const iterations = process.env.PERF_ITERS ? parseInt(process.env.PERF_ITERS,10) : 40;
  const first = runSplitterBenchmark(iterations).perDoc;
  const second = runSplitterBenchmark(iterations).perDoc;
  const best = Math.min(first, second);
  const updated = saveBaseline(best);
  console.log('[perf-baseline] Updated baseline', { best: +best.toFixed(3), rollingMean: updated.rollingMean });
})();
