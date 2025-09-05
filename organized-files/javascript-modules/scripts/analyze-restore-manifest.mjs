#!/usr/bin/env node
/**
 * analyze-restore-manifest.mjs
 * Utility to slice a large git diff manifest (name-only or name-status) into
 * targeted subsets (YoRHa, Svelte components, routes, infrastructure, scripts, etc.)
 * and emit summary statistics.
 *
 * Usage:
 *   node scripts/analyze-restore-manifest.mjs RESTORED_FILES_MANIFEST_a55884a8_to_519c6b70.txt [--status]
 *
 * If --status is passed the input is expected to be a name-status file (A/M/D<TAB>path).
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

const args = process.argv.slice(2);
if (!args.length) {
  console.error('Manifest path required.');
  process.exit(1);
}
const manifestPath = args[0];
const isStatus = args.includes('--status');

const raw = readFileSync(manifestPath, 'utf8').replace(/\r\n/g, '\n');
const lines = raw.split('\n').filter(Boolean);

// Parse entries
const entries = lines.map(l => {
  if (isStatus) {
    const tab = l.indexOf('\t');
    if (tab > 0) {
      return { status: l.slice(0, tab), path: l.slice(tab + 1) };
    }
  }
  return { status: null, path: l };
});

// Helper filters
const hasExt = (p, ext) => p.toLowerCase().endsWith(ext.toLowerCase());

// Category definitions (order matters for reporting)
const categories = [
  { name: 'yorha_components', test: p => p.includes('/yorha/') && hasExt(p, '.svelte') },
  { name: 'svelte_components', test: p => hasExt(p, '.svelte') && p.includes('/src/lib/components/') && !p.includes('/yorha/') },
  { name: 'routes', test: p => p.includes('/src/routes/') && hasExt(p, '.svelte') },
  { name: 'typescript_sources', test: p => hasExt(p, '.ts') || hasExt(p, '.tsx') },
  { name: 'scripts', test: p => /(^|\/)scripts\//.test(p) && /\.(mjs|cjs|js|ps1|sh|bat)$/.test(p) },
  { name: 'infrastructure', test: p => /(docker|compose|\.vscode\/|infrastructure|terraform|k8s)/i.test(p) },
  { name: 'wasm_gpu', test: p => /(wasm|webgpu|gpu|accelerator|cuda)/i.test(p) },
  { name: 'documentation', test: p => hasExt(p, '.md') || /README/i.test(basename(p)) },
];

const catBuckets = Object.fromEntries(categories.map(c => [c.name, []]));

for (const e of entries) {
  for (const cat of categories) {
    if (cat.test(e.path)) {
      catBuckets[cat.name].push(e);
    }
  }
}

// Status counts
const statusCounts = entries.reduce((acc, e) => {
  const k = e.status || 'NA';
  acc[k] = (acc[k] || 0) + 1;
  return acc;
}, {});

// Write subset files
for (const cat of categories) {
  const outPath = `${manifestPath}.${cat.name}.txt`;
  const body = catBuckets[cat.name]
    .map(e => (isStatus && e.status ? `${e.status}\t${e.path}` : e.path))
    .join('\n');
  writeFileSync(outPath, body + (body ? '\n' : ''));
}

// Summary JSON
const summary = {
  manifest: manifestPath,
  totalEntries: entries.length,
  statusCounts,
  categories: Object.fromEntries(categories.map(c => [c.name, catBuckets[c.name].length])),
  generatedAt: new Date().toISOString(),
  mode: isStatus ? 'name-status' : 'name-only'
};

const summaryPath = `${manifestPath}.summary.json`;
writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

console.log('Restore Manifest Summary');
console.log('------------------------');
console.log(JSON.stringify(summary, null, 2));
console.log('\nSubset files written alongside manifest.');

// Quick ranking of largest categories
const ranked = Object.entries(summary.categories).sort((a,b)=>b[1]-a[1]);
console.log('\nCategory ranking (count desc):');
for (const [name, count] of ranked) {
  console.log(name.padEnd(22), count.toString().padStart(6));
}

if (ranked.length) {
  const top = ranked[0];
  console.log(`\nTop category '${top[0]}' represents ${(top[1] / summary.totalEntries * 100).toFixed(2)}% of entries.`);
}

console.log('\nNext suggestions:');
console.log('- Review yorha_components subset first to ensure UI integrity.');
console.log('- Check routes subset for any misaligned +layout/+page migrations.');
console.log('- Use statusCounts to focus on Added (A) items for license/compliance review.');
