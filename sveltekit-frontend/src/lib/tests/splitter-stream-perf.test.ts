/// <reference types="vite/client" />
import { describe, it, expect } from "vitest";
// TODO: Fix import - // Orphaned content: import {  const MAX_MS_STREAM_TOTAL = parseFloat(import.meta.env.SPLITTER_MAX_MS_STREAM_TOTAL || '30');

function simulateStream(full: string, chunkSize = 48) {
  const splitter = new EnhancedSentenceSplitter();
  const context = { buffer: '', processedSentences: [] as string[] } as any;
  const start = performance.now();
  for (let i = 0; i < full.length; i += chunkSize) {
    splitter.processStreamingChunk(full.slice(i, i + chunkSize), context);
  }
  splitter.finalizeStreaming(context);
  const ms = performance.now() - start;
  return { ms, sentences: context.processedSentences };
}

describe('EnhancedSentenceSplitter Streaming Performance', () => {
  it('processes chunks within budget', () => {
    const synthetic = Array.from({ length: 60 }, (_, i) => `Section ${i + 1}. This clause establishes obligations and terms for the agreement parties involved.`).join(' ');
    const { ms, sentences } = simulateStream(synthetic);
    expect(sentences.length).toBeGreaterThan(20);
    expect(ms).toBeLessThanOrEqual(MAX_MS_STREAM_TOTAL);
  });
});
