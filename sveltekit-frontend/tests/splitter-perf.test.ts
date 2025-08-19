import { describe, it, expect } from 'vitest';
import { benchmarkSplitter } from "../shared/text/benchmark-splitter.js";;

// CI performance guard: adjust threshold if environment differs.
// Threshold chosen to allow moderate variance; failing test signals regression.
const MAX_MS_PER_DOC = parseFloat(import.meta.env.SPLITTER_MAX_MS_PER_DOC || '3.5');

describe('LegalSentenceSplitter Performance', () => {
  it('should stay within performance budget', () => {
    const { perDoc, iterations, totalSentences } = runSplitterBenchmark(40);
    // Basic sanity
    expect(iterations).toBeGreaterThan(0);
    expect(totalSentences).toBeGreaterThan(0);
    // Performance budget
    expect(perDoc).toBeLessThanOrEqual(MAX_MS_PER_DOC);
  });
});
