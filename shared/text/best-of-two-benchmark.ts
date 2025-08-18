import { runSplitterBenchmark } from './benchmark-splitter';

export interface BestOfTwoResult {
  first: number; // first perDoc ms
  second: number; // second perDoc ms
  best: number; // min
  took: 1 | 2; // which run produced best
}

/**
 * Run the splitter benchmark twice and return the best (minimum) per-doc time.
 * This mitigates noisy first-run JIT / GC spikes while staying inexpensive.
 */
export async function runBestOfTwoBenchmark(iterations = 30): Promise<BestOfTwoResult> {
  const r1 = runSplitterBenchmark(iterations);
  const first = r1.perDoc;
  const r2 = runSplitterBenchmark(iterations);
  const second = r2.perDoc;
  const best = Math.min(first, second);
  return { first, second, best, took: best === first ? 1 : 2 };
}

export default runBestOfTwoBenchmark;
