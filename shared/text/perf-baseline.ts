import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { runSplitterBenchmark } from './benchmark-splitter';

interface BaselineFile {
  schemaVersion: 1;
  metric: 'perDoc';
  history: number[]; // last N perDoc values
  rollingMean: number | null;
  updated: string | null;
}

const BASELINE_PATH = join(__dirname, 'perf-baseline.json');
const MAX_HISTORY = 20;

export function loadBaseline(): BaselineFile {
  try {
    const raw = readFileSync(BASELINE_PATH, 'utf8');
    return JSON.parse(raw);
  } catch {
    return { schemaVersion: 1, metric: 'perDoc', history: [], rollingMean: null, updated: null };
  }
}

export function updateBaseline(perDoc: number): BaselineFile {
  const data = loadBaseline();
  data.history.push(perDoc);
  if (data.history.length > MAX_HISTORY) data.history.splice(0, data.history.length - MAX_HISTORY);
  const sum = data.history.reduce((a, b) => a + b, 0);
  data.rollingMean = +(sum / data.history.length).toFixed(4);
  data.updated = new Date().toISOString();
  writeFileSync(BASELINE_PATH, JSON.stringify(data, null, 2));
  return data;
}

export function checkRegression(perDoc: number, opts: { threshold: number; tolerancePct?: number; allowDeltaPct?: number; }): { pass: boolean; reason?: string; baseline?: BaselineFile } {
  const baseline = loadBaseline();
  const tolerancePct = opts.tolerancePct ?? 0.1; // 10% headroom default
  const allowDeltaPct = opts.allowDeltaPct ?? 0.15; // 15% over rolling mean allowed

  if (perDoc > opts.threshold * (1 + tolerancePct)) {
    return { pass: false, reason: `Exceeded hard threshold (+${tolerancePct * 100}% headroom)`, baseline };
  }

  if (baseline.rollingMean != null) {
    const deltaPct = (perDoc - baseline.rollingMean) / baseline.rollingMean;
    if (deltaPct > allowDeltaPct) {
      return { pass: false, reason: `Regression vs rolling mean: ${(deltaPct * 100).toFixed(1)}% > ${(allowDeltaPct * 100)}%`, baseline };
    }
  }
  return { pass: true, baseline };
}

if (require.main === module) {
  const iterations = process.env.PERF_ITERS ? parseInt(process.env.PERF_ITERS, 10) : 40;
  const { perDoc } = runSplitterBenchmark(iterations);
  const updated = updateBaseline(perDoc);
  // eslint-disable-next-line no-console
  console.log(JSON.stringify({ updated, newPerDoc: perDoc }, null, 2));
}
