# Staged Backup Restoration Plan

This document tracks the phased application of analyzed backups to reduce TypeScript/Svelte errors quickly and safely.

## Goals
- Rapidly lower error count by restoring higher-quality historical versions (fewer @ts-nocheck, better typing, fewer syntax issues).
- Preserve ability to bisect regressions by staging restorations (P5 → re-check → P4 → re-check → P3 optional).
- Avoid redundant overwrites where multiple timestamped backups target the same file—final timestamp wins but we may choose earliest clean version if later reintroduces issues.

## Phase Sequence
1. Priority 5 (High Impact Core UI, Stores, Services, Routes)  — restore now using `restore-priority5.bat`.
2. TypeScript + Svelte Check → record new baseline (expect substantial drop in suppressed diagnostics if @ts-nocheck removed).
3. Diff & Deduplicate remaining restore list for Priority 4.
4. Apply Priority 4 subset (editor enhancements, auxiliary modals, dashboard components).
5. Re-run checks, update baseline tables (README + tracking doc).
6. Optionally apply Priority 3 (only if still > target thresholds or specific defects persist).
7. Archive unused backup variants to `legacy-backups/`.

## Metrics to Capture After Each Phase
| Phase | TS Errors | TS Warnings | Svelte Errors | Files with @ts-nocheck | Notable Changes |
|-------|-----------|-------------|---------------|-------------------------|-----------------|
| Baseline (pre-P5) | TBD (record before run) | TBD | TBD | TBD | Reference snapshot |
| After P5 |  |  |  |  |  |
| After P4 |  |  |  |  |  |
| After P3 (optional) |  |  |  |  |  |

## Verification Steps
1. `npm run check:typescript` (capture counts)
2. `npm run check:svelte` (capture counts)
3. Spot compile: open a few key restored components to ensure no new syntax regressions.
4. Run minimal dev server smoke (if needed) to validate runtime imports.

## Risk Mitigation
- Staging isolates root cause if a restored file introduces new errors.
- Maintain original current versions in git history; each restore is a plain copy so revert is trivial via git checkout.
- Deduplication pass planned before P4 to eliminate redundant copy ops.

## Next Actions
- [ ] Run `restore-priority5.bat`
- [ ] Run TS & Svelte checks; fill After P5 row
- [ ] Generate deduped Priority 4 script
- [ ] Apply P4 and re-measure
- [ ] Decide on P3 necessity

## Notes
Discrepancy: Original mention of 306 candidates vs. 274 in generated script. Action: generate a listing of unmatched expected files after P5 metrics capture.
