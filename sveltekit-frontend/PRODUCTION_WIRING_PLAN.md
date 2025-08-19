# Production Wiring & Implementation Plan (Frontend)

Goal: Reduce 3460 Svelte errors + 1141 warnings to <50 blocking issues; establish consistent, production-safe component & event model.

## 0. Guiding Principles
- Eliminate duplicate UI paradigms (pick ONE dialog/modal system).
- Forward native HTML attributes (`class`, events) via `...$$restProps` where appropriate.
- Replace migration artefact attributes (`transitionfly`, `transitionfade`) with canonical Svelte syntax.
- Normalize component prop + event naming to Svelte 5 conventions (`bind:`, `on:`).
- Add missing union members (e.g. Button `variant`) OR refactor usages to supported set.

## 1. High-Impact Error Categories
| Category | Symptom | Root Cause | Batch Fix Method |
|----------|---------|-----------|------------------|
| Missing prop typing (`class`) on UI components (Card*, Progress, etc.) | "'class' does not exist in type 'Props'" | Component interfaces omit `class?: string` & rest forwarding | Add `export let class: string = ''` + apply to wrapper element; or use `...$$restProps` |
| Invalid custom event props (`onresponse`, `onupload`) | Unknown prop errors | Should be DOM/custom events using `on:response` | Global regex replace `onresponse=` -> `on:response=` etc. |
| Wrong attribute names (`transitionfly`, `transitionfade`) | Unknown prop | Migration artifact; Svelte transition directive syntax changed | Regex: `transitionfly` -> `transition:fly`; `transitionfade` -> `transition:fade` |
| Button variant values not in union (`primary`, `danger`) | Type errors | Variant union excludes values | (A) Extend variant union & styles OR (B) Map to existing (`primary`->`legal`,`danger`->`destructive`) |
| Binding to non-bindable props (`bind:open` errors) | Bind failure | Components not using `$bindable()` | Add `open = $bindable(false)` and expose in `$props()` |
| Event parameter type `never` (`on:click` in DropdownMenuItem) | `'click' not assignable to never` | Component lacks event generics or incorrect export pattern | Patch component TS definitions to include `declare namespace svelteHTML { interface HTMLAttributes<T> { 'on:click'?: (e:any)=>void } }` or proper `createEventDispatcher` typings |
| Form field config arrays type mismatch | Arrays of `{id,label,type}` vs union expects constrained `type` | Overly narrow union or wrong runtime strings | Broaden `type` union OR map unsupported strings to allowed values |
| Attributify utility props (`mb-8`, `rounded-lg`) flagged | Unknown HTML props | Using UnoCSS attributify without TS ambient types | Add `unocss-attributify.d.ts` with interface augmentations OR shift to class strings |
| Unused CSS selectors flood warnings | Dead selectors | Legacy / merged styles | Remove or scope; optionally tolerate via svelte-check threshold |
| Duplicate dialog implementations (custom vs Bits vs enhanced) | Multiple APIs & error noise | Partial migrations | Pick one (recommend Bits + thin wrapper) and delete / deprecate others |

## 2. Execution Order (Batches)
1. Syntax / mechanical fixes (regex-safe): transitions, `onresponse` → `on:response`, `className` → `class`.
2. Add `class?: string` + `...$$restProps` to UI primitives (Card*, Progress, Dialog wrappers, DataGrid, Modal).
3. Normalize Button variants (extend union + style map OR rename usages).
4. Dialog consolidation: keep `ui/enhanced-bits/Dialog.svelte` + remove legacy `lib/components/Dialog.svelte` & partially migrated `ui/dialog/*.svelte` duplicates after extracting any unique features (vibe selection, accessibility props).
5. Event typings: augment dropdown / menu / advanced upload components to declare emitted events so `on:filesAdded` etc. resolve.
6. Form model typing: extend `FormField['type']` union or map unsupported types (e.g. `'string'` → `'text'`).
7. Attributify typing: add ambient declaration or convert high-error pages (upload-test, optimization dashboards) to standard `class="..."`.
8. Clean dead CSS selectors; run svelte-check with reduced noise, re-measure.
9. Remaining semantic fixes (data shape conversions, keyEntities mapping logic, `similarity` property design) finalize.

## 3. Concrete Patch Patterns
```txt
Find: /(on)(response|upload|close)=/g  → Replace: on:$2=
Find: transitionfly=  → transition:fly=
Find: transitionfade= → transition:fade=
Find: className=      → class=
```

Add rest forwarding example:
```svelte
<script lang="ts">
  export let class: string = '';
  let $$restProps: any;
</script>
<div class={`card-root ${class}`} {...$$restProps}>
  <slot />
</div>
```

UNO attributify ambient (create `src/types/unocss-attributify.d.ts`):
```ts
declare namespace svelteHTML { interface HTMLAttributes<T> { [attr: string]: any } }
```
(Short-term broad allowance; tighten later with specific attrib whitelist.)

## 4. Dialog Consolidation Steps
- Retain: `src/lib/components/ui/enhanced-bits/Dialog.svelte`.
- Migrate unique features from legacy `components/Dialog.svelte` (vibe selection, history) into feature-specific composable or slot.
- Delete / archive: `lib/components/Dialog.svelte`, `ui/dialog/Dialog*.svelte` (after ensuring no unique exports required) to eliminate prop/event mismatch duplication.

## 5. Data Shape & Model Fixes
- `local-ai-demo`: Convert `keyEntities: string[]` to required object array earlier; define union type `KeyEntity = { text:string; type:'entity'; confidence:number };` and coerce once.
- `AnalysisResults`: Either add `similarity?: number` or remove assignment.
- `saved-citations`: Replace manual dropdown snippet with Bits UI pattern; ensure events typed.
- `yorha-demo` form config: Replace free-form `type: string` with mapped union or extend `FormField['type']` union.

## 6. Button Variant Strategy
Option A (preferred): Extend variant map & CSS tokens (`primary`, `danger`).
Option B: Replace usages with existing `default` / `destructive`.

## 7. Metrics & Acceptance
After batches 1–4 expect >60% error reduction (prop + attr). Track svelte-check counts after each batch; fail CI if >500 errors remain post-batch 4.

## 8. CI Integration (Follow-up)
Add GitHub Action job:
- Install deps
- Run `npm run check:all` (frontend)
- Run perf regression
- Fail on remaining error budget threshold.

## 9. Deferred / Nice-to-have
- Introduce ESLint rule to ban legacy dialog imports.
- Generate codemod script for variant + event attribute rewrites.
- Tighten attributify ambient types (replace `[attr: string]: any`).

## 10. Immediate Next Commands (Optional)
(1) Mechanical rewrites: apply regex replacements.
(2) Add attributify ambient d.ts or convert attributes to class.
(3) Patch UI primitives with `class` + rest forwarding.
(4) Remove duplicate dialogs; update imports.

---
This plan targets rapid structural error reduction while converging on a single component architecture suitable for production hardening.
