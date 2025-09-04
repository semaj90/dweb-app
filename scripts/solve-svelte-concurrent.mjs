#!/usr/bin/env zx
/**
 * Orchestrated concurrent pipeline using zx:
 * 1. capture raw svelte-check
 * 2. extract patterns
 * 3. semantic solve
 * 4. concurrent suggestions
 */
import { $ } from 'zx';

$.verbose = false;

async function step(cmd, label){
  console.log(`\n▶ ${label}`);
  try { await $`${cmd}`; } catch(e){ console.error(`❌ ${label} failed`); process.exit(1); }
}

await step('node scripts/capture-svelte-errors.mjs', 'Capture Errors');
await step('node scripts/extract-svelte-errors.mjs', 'Extract Patterns');
await step('node scripts/semantic-solve-svelte-errors.mjs', 'Semantic Solve');
await step('node scripts/generate-suggestions-concurrent.mjs', 'Concurrent Suggestions');

console.log('\n✅ Pipeline complete. See .vscode/svelte-fix-suggestions.json');
