import { describe, it, expect } from 'vitest';
import { runSplitterBenchmark } from '$text/benchmark-splitter';

const MAX_MS_PER_DOC = parseFloat(process.env.SPLITTER_MAX_MS_PER_DOC || '3.5');

describe('LegalSentenceSplitter Performance', () => {
  it('stays within performance budget', () => {
    const { perDoc, iterations, totalSentences } = runSplitterBenchmark(30);
    expect(iterations).toBeGreaterThan(0);
    expect(totalSentences).toBeGreaterThan(0);
    expect(perDoc).toBeLessThanOrEqual(MAX_MS_PER_DOC);
  });
});
