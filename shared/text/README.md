# Shared Text Utilities

This directory provides shared sentence splitting utilities consumed by both frontend and backend services.

## Modules

- `enhanced-sentence-splitter.ts` – Core implementation handling:
  - Legal & business abbreviations (precompiled for performance)
  - Heading detection & merge (e.g., `Section 1.` with following sentence)
  - Fragment merging (drops or merges ultra-short fragments)
  - Streaming processing helpers

- `legal-sentence-splitter.ts` – Factory + lazy singleton adding legal-specific defaults (extra headings & abbreviations) for consistent use across services.

- `benchmark-splitter.ts` – Lightweight performance harness used in CI to detect regressions.

## Performance Guard

`src/lib/tests/splitter-perf.test.ts` runs a micro-benchmark gating average milliseconds per synthetic document.

Environment variables:

`SPLITTER_MAX_MS_PER_DOC` (default: 3.5)
`SPLITTER_MAX_MS_STREAM_TOTAL` (default: 30 for streaming test)

To run locally:

```bash
npm run test:perf:splitter
```

### Rolling Baseline & Regression Guard

Files:
- `perf-baseline.json` – Rolling history (last 20 samples) + rolling mean.
- `perf-baseline.ts` – Baseline utilities (TypeScript, used in tests).
- `best-of-two-benchmark.ts` – Runs benchmark twice and picks the faster (reduces noise).
- `update-perf-baseline.cjs` – CI-safe CommonJS updater (no TS build needed).

Test Guard:
- `sveltekit-frontend/src/lib/tests/splitter-regression.test.ts` uses best-of-two result + `checkRegression`.

Regression logic:
1. Fail if `perDoc > threshold * (1 + 0.10)` (hard threshold +10% headroom)
2. If baseline mean exists, fail if `(perDoc - mean)/mean > 0.15` ( >15% slower )

Scripts:
```bash
npm run perf:regression        # run regression guard test only
npm run perf:update-baseline   # (main branch + ALLOW_BASELINE_UPDATE=true) update rolling baseline
```

Manual benchmark (raw):
```bash
npm run bench:splitter
```

CI Example (GitHub Actions fragment):
```yaml
- name: Splitter regression
  run: npm run perf:regression

- name: Update baseline (main only)
  if: github.ref == 'refs/heads/main'
  env:
    ALLOW_BASELINE_UPDATE: 'true'
    GIT_BRANCH: 'main'
  run: npm run perf:update-baseline
```

Why best-of-two? Warm-up / transient GC spikes often affect the first run. Taking the minimum of two short runs yields a stable conservative measure of attainable performance without long multi-iteration warmups.

### Streaming Performance

`src/lib/tests/splitter-stream-perf.test.ts` validates total processing time for a synthetic streaming scenario, ensuring incremental overhead remains bounded.

### Flakiness Controls

Implemented:
- Rolling mean & delta check
- Best-of-two selection

Potential Future Enhancements:
1. Auto-relaxed headroom when `CI=true` and runners are cold
2. Median-of-three when variance > configured threshold
3. Percentile-based (P75) regression guard when history >= 10 samples

## Usage Patterns

Direct import (shared core):

```ts
import { EnhancedSentenceSplitter, splitSentencesEnhanced } from '../../shared/text/enhanced-sentence-splitter';
```

Singleton legal splitter:

```ts
import { getSharedLegalSentenceSplitter } from '../../shared/text/legal-sentence-splitter';
const splitter = getSharedLegalSentenceSplitter();
```

## Modification Guidelines

- When adding abbreviations prefer the factory (`legal-sentence-splitter`) so all consumers stay in sync.
- Keep regex changes benchmarked (`npm run bench:splitter`). Aim not to increase `perDoc` > 10%.
- If adding async behavior, preserve a synchronous path (many callers expect sync split).

## Benchmark Example

```bash
npm run bench:splitter
```

Outputs object: `{ iterations, totalSentences, ms, perDoc, perSentence }`.

## License / Attribution

Internal project utility – no external license headers required. Keep this README updated when semantics change.
