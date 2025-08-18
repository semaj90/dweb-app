// Lightweight benchmark harness for the legal sentence splitter
import { performance } from 'node:perf_hooks';
import { getSharedLegalSentenceSplitter } from './legal-sentence-splitter';

function randomParagraph(words: number): string {
  const sample = ['Section', 'Article', 'Exhibit', 'WHEREAS,', 'the', 'Party', 'Agreement', 'obligation', 'shall', 'be', 'binding', 'hereunder', 'Recital', 'A.', 'provided', 'however', 'that', 'pursuant', 'to', 'Sec.', '12.'];
  return Array.from({ length: words }, () => sample[Math.floor(Math.random() * sample.length)]).join(' ') + '.';
}

export function runSplitterBenchmark(iterations = 50) {
  const splitter = getSharedLegalSentenceSplitter();
  const texts: string[] = [];
  for (let i = 0; i < iterations; i++) {
    texts.push(randomParagraph(80 + Math.floor(Math.random() * 40)));
  }
  const start = performance.now();
  let totalSentences = 0;
  for (const t of texts) {
    totalSentences += splitter.split(t).length;
  }
  const ms = performance.now() - start;
  const perDoc = ms / iterations;
  const perSentence = ms / totalSentences;
  return { iterations, totalSentences, ms, perDoc, perSentence };
}

if (require.main === module) {
  const result = runSplitterBenchmark(100);
  // eslint-disable-next-line no-console
  console.log('[splitter-benchmark]', result);
}
