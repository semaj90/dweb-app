import { describe, it, expect } from 'vitest';
import { runBestOfTwoBenchmark } from '$text/best-of-two-benchmark';
import { loadBaseline, checkRegression } from '$text/perf-baseline';

describe('Sentence splitter performance regression', () => {
  it('does not regress vs hard threshold & rolling mean', async () => {
    const iterations = process.env.PERF_REG_ITERS ? parseInt(process.env.PERF_REG_ITERS, 10) : 24;
    const best = await runBestOfTwoBenchmark(iterations);
    const threshold = process.env.SPLITTER_MAX_MS_PER_DOC ? parseFloat(process.env.SPLITTER_MAX_MS_PER_DOC) : 4.0;
    const result = checkRegression(best.best, { threshold });
    if (!result.pass) {
      const base = result.baseline;
      expect.fail(`Regression: ${result.reason} | baselineMean=${base?.rollingMean} | best=${best.best.toFixed(3)} (first=${best.first.toFixed(3)} second=${best.second.toFixed(3)})`);
    }
    expect(result.pass).toBe(true);
  }, 30_000);
});
