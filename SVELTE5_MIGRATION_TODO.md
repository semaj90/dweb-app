## Svelte 5 Migration & Error Remediation Master TODO

Baseline (latest run): 6150 errors / 1641 warnings across 745 files (`svelte-check`).

Primary high‑volume categories (from `context7-multicore-error-analysis.ts` + fresh scan):
- svelte5_migration (~800) – legacy `export let` patterns / missing runes.
- ui_component_mismatch (~600) – missing `class` / variant prop unions / prop name drift.
- css_unused_selectors (~400) – unused style blocks (UnoCSS / dead selectors).
- binding_issues (~162) – `bind:` on non‑bindable props + event map gaps.

Secondary recurring patterns (aggregated estimates):
- event_name_never (on:click / on:keydown → parameter of type `never`) 450–600
- slot_prop_mismatch (snippet / slot props `'children'`, `sidebar`, `case`) 150–200
- union_literal_mismatch (Button variant / StatusCard variant) 120–160
- cross_user_type_conflict (duplicate `User` shape) 60–90
- store_unknown_access ($user typed unknown) 30–40
- array_vs_scalar (string[] bound to string Input) 40–60
- orchestrator_method_missing (`submitTask`) 20–30
- grid_form_union (YoRHa grid/form type strictness) 40–50
- server_env_importmeta (import.meta.env vs process.env) 20–30
- cjs_default_import (postgres default import) 10–15
- dialog_bind_open / openState redundancy 30–40
- dropdown_event_click_never 50–70

---
### Phase Overview
| Phase | Goal | Est. Errors Resolved | Automation Potential | Prereq |
|-------|------|----------------------|----------------------|--------|
| 0 | Baseline snapshot & tracking infra | n/a | High | none |
| 1 | Common Props augmentation (`class`, `id`, `style`) | 1500–2000 | High | 0 |
| 2 | Event map & bindability normalization | 800–1000 | Medium | 1 |
| 3 | Runes migration (`export let` → `$props()`) | 600–800 | Medium | 1 |
| 4 | Variant & union prop harmonization | 300–400 | Medium | 1 |
| 5 | Slot/snippet prop definition consolidation | 150–200 | Medium | 2 |
| 6 | User & shared domain types unification | 80–120 | High | 0 |
| 7 | Array vs scalar & form/grid schema softening | 120–150 | Medium | 4,6 |
| 8 | Orchestrator / service method typings | 30–50 | Medium | 6 |
| 9 | Server env & CJS import cleanup | 30–40 | Low | 0 |
| 10 | Dialog / Dropdown structural props cleanup | 120–160 | Medium | 2,3 |
| 11 | CSS unused selector pruning & keep‑directives | 400–500 (warnings) | High | 1 |
| 12 | Vendor (`bits-ui`) strategy (fork / AST transform) | Prevent regressions | Low | 3 |

---
### Phase 0 – Baseline & Tracking
Tasks:
- [ ] Create `.vscode/error-metrics.json` after each `svelte-check` run (augment existing logger).
- [ ] Add script `scripts/error-metrics.mjs` to diff counts and output delta.
Acceptance Criteria: Running metrics script twice shows cumulative reduction log with timestamp.

### Phase 1 – Common Props Augmentation
Root Cause: Component `Props` lack ubiquitous DOM attributes (`class`, `style`, `id`, maybe `children`).
Tasks:
- [ ] Add `src/lib/types/component-props.d.ts` with `export interface CommonProps { class?: string; id?: string; style?: string; }`.
- [ ] For internal components (Card*, Button, Dialog*, Dropdown*, CaseCard, Modal, Header) extend or merge `CommonProps`.
- [ ] Add temporary index signature ONLY if a component still generates >10 unknown prop errors after extension.
Automation:
- [ ] Script scans `.svelte` files for `<[A-Z][A-Za-z0-9]* class="` and maps to component definitions to ensure props extended.
Acceptance Criteria: Re-run check: ≥60% drop in "'"class"' does not exist" errors.

### Phase 2 – Event Map & Bindability
Root Cause: Missing event typing yields `never` and invalid `bind:` usage.
Tasks:
- [ ] Identify components used with `bind:`. For each, either (a) declare `let { open = $bindable(false) } = $props()` or (b) replace bind usage in callsites with explicit prop + change handler.
- [ ] Add event map types: `export interface ButtonEvents { click: MouseEvent }` etc., and pattern for DropdownMenuItem, DialogRoot, CardRoot, Input.
Automation:
- [ ] Script finds `on:(\w+)={` where error log includes `assignable to parameter of type 'never'` and injects stub event map if missing.
Acceptance: `never` event errors reduced ≥70%.

### Phase 3 – Runes Migration (`export let`)
Root Cause: Legacy `export let` + missing `$props()` destructures.
Tasks:
- [ ] AST transform (recast / svelte parser) simple `export let name = init;` → accumulate into a single `let { name = init, ... } = $props();` placed after imports.
- [ ] For re-exported or reactive declarations, skip with annotation comment `// MIGRATION_SKIP`.
Acceptance: ≥70% of direct `export let` declarations converted; no new parse errors introduced.

### Phase 4 – Variant & Union Prop Harmonization
Root Cause: Strict variant unions missing used literals (e.g., `primary`, `danger`).
Tasks:
- [ ] Audit Button, Card, StatusCard, Badge components for variant unions.
- [ ] Add extended union literal OR map alias to canonical variant internally.
Acceptance: All variant mismatch errors for Buttons/StatusCard gone (sample from `showcase/+page.svelte`).

### Phase 5 – Slot / Snippet Prop Definitions
Root Cause: Snippet `children`, `sidebar`, `case` props not declared.
Tasks:
- [ ] Define snippet prop interfaces or update components to accept `children?: any`.
- [ ] For complex slot props, create typed context object interface (e.g., `DropdownMenuContext`).
Acceptance: All `Object literal may only specify known properties, and 'children'/'sidebar'/'case' does not exist` errors resolved.

### Phase 6 – User Type Unification
Root Cause: Multiple `User` interfaces with optional/required divergence (`avatarUrl`).
Tasks:
- [ ] Single `src/lib/types/user.ts` exported interface; remove duplicates.
- [ ] Update all imports to the unified path.
Acceptance: No cross-import `User` incompatibility diagnostics.

### Phase 7 – Array vs Scalar & Grid/Form Schema Softening
Root Cause: Input expecting string receives string[]; strict grid/form field unions.
Tasks:
- [ ] Introduce coercion helpers: `ensureString(value: string|string[]): string`.
- [ ] Add `| string` fallback to form field `type` OR refine creation logic to use permitted literals.
Acceptance: All `string[]` assignability errors + grid/form union mismatches removed.

### Phase 8 – Orchestrator Method Typings
Root Cause: `orchestrator.submitTask` not in type definition.
Tasks:
- [ ] Extend orchestrator interface; ensure actual implementation exports method.
- [ ] Add overload signatures if multiple task types.
Acceptance: No missing property errors on pages referencing orchestrator tasks.

### Phase 9 – Server Env & CJS Import Cleanup
Root Cause: `import.meta.env` + default CJS import patterns.
Tasks:
- [ ] Replace server-only `import.meta.env` usage with `process.env` guard.
- [ ] Convert `import postgres from 'postgres'` → `import * as postgres from 'postgres'` (or enable esModuleInterop once globally validated).
Acceptance: Zero diagnostics referencing `import.meta` or default import of `postgres`.

### Phase 10 – Dialog / Dropdown Structural Props
Root Cause: Over-specified props (`overlay={{}}`, `content={{}}`, `openState`) & non-bindable `bind:open`.
Tasks:
- [ ] Remove redundant structural props where defaults suffice.
- [ ] Normalize open state pattern: `<DialogRoot {open} onOpenChange={...}>`.
Acceptance: Removal of related object-literal unknown property errors in dialog usages.

### Phase 11 – CSS Unused Selector Pruning
Root Cause: Legacy or placeholder selectors.
Tasks:
- [ ] Add analyzer script: parse `.svelte` style blocks; cross-reference class tokens in markup.
- [ ] Remove or mark kept selectors with `/* @keep */` comment if intentionally reserved.
Acceptance: ≥80% reduction in unused selector warnings; purposeful remainder annotated.

### Phase 12 – Vendor (`bits-ui`) Long-Term Strategy
Root Cause: Precompiled dist with legacy runes.
Options:
- [ ] Fork & rebuild with Svelte 5.
- [ ] AST transform pipeline at dev time (esbuild / rollup plugin) to rewrite patterns; remove shim.
- [ ] Contribution upstream PR.
Acceptance: Shim alias removable without reintroducing optimizer errors.

---
### Cross-Cutting Automation Scripts (Planned)
| Script | Purpose | Est. Impact |
|--------|---------|-------------|
| `scripts/props-augment.mjs` | Inject / verify CommonProps extension across components | 1500–2000 errors |
| `scripts/migrate-export-let.mjs` | Transform `export let` to `$props()` destructure | 600–800 errors |
| `scripts/event-map-infer.mjs` | Build event maps from usage (`on:click`, etc.) | 450–600 errors |
| `scripts/bind-mismatch-report.mjs` | Report & optionally refactor invalid `bind:` usages | 150–200 errors |
| `scripts/unused-css-prune.mjs` | Identify & optionally remove unused selectors | 400–500 warnings |
| `scripts/type-union-harmonize.mjs` | Extend union literals or map aliases | 120–160 errors |

---
### Metrics & Acceptance Schema
Each phase completion commit should include:
```json
{
  "phase": <number>,
  "timestamp": "2025-08-19T..Z",
  "errors_before": <int>,
  "errors_after": <int>,
  "warnings_before": <int>,
  "warnings_after": <int>,
  "reduction_percent": <float>
}
```

Integrate into existing `.vscode/vite-errors.json` or create dedicated `.vscode/error-metrics.json`.

---
### Immediate Next Action Recommendation
1. Implement Phase 1 (Common Props) – fastest bulk drop.
2. Parallel start Phase 6 (User type unification) – low risk, isolates cascades.
3. Run metrics script to capture new baseline.

---
### Risk & Ordering Notes
- Perform runes migration (Phase 3) only after Common Props to avoid merging conflicts in converted destructures.
- Vendor fork (Phase 12) can occur anytime; earlier fork may simplify phases 1–5 by removing mismatch noise.
- Keep automated scripts idempotent; dry-run mode first.

---
### Tracking Checklist (High-Level)
- [ ] Phase 0 Metrics Infra
- [ ] Phase 1 Common Props
- [ ] Phase 2 Event Maps & Bindability
- [ ] Phase 3 Runes Migration
- [ ] Phase 4 Variant Unions
- [ ] Phase 5 Slot/Snippet Props
- [ ] Phase 6 User Type Unification
- [ ] Phase 7 Array vs Scalar / Grid/Form
- [ ] Phase 8 Orchestrator Methods
- [ ] Phase 9 Server Env & CJS Imports
- [ ] Phase 10 Dialog/Dropdown Structural Props
- [ ] Phase 11 CSS Pruning
- [ ] Phase 12 Vendor Strategy Finalization

---
_Generated from high-level review of `svelte-check` output and `context7-multicore-error-analysis.ts` categories. Adjust counts as real automated metrics replace estimates._
