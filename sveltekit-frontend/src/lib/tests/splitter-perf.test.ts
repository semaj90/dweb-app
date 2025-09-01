/// <reference types="vite/client" />
/// <reference types="vitest" />
// TODO: Fix import - // Orphaned content: import {  const MAX_MS_PER_DOC = parseFloat(import.meta.env.SPLITTER_MAX_MS_PER_DOC || '3.5');

describe('LegalSentenceSplitter Performance', () => {
  it('stays within performance budget', () => {
    const { perDoc, iterations, totalSentences } = runSplitterBenchmark(30);
    expect(iterations).toBeGreaterThan(0);
    expect(totalSentences).toBeGreaterThan(0);
    expect(perDoc).toBeLessThanOrEqual(MAX_MS_PER_DOC);
  });
});
