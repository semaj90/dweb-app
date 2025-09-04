#!/usr/bin/env node
/**
 * Regenerate .vscode/tasks.json from fragment files in .vscode/tasks/
 * Ensures monolith count matches fragments so verify script passes.
 * Deterministic ordering: tasks sorted by label (case-insensitive).
 * Fails on duplicate labels unless --allow-duplicates passed.
 */
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const fragDir = path.join(root, '.vscode', 'tasks');
const monolithPath = path.join(root, '.vscode', 'tasks.json');
const allowDuplicates = process.argv.includes('--allow-duplicates');

if (!fs.existsSync(fragDir)) {
  console.error('❌ Fragment directory not found:', fragDir);
  process.exit(1);
}

const fragFiles = fs.readdirSync(fragDir).filter(f => /^tasks-.*\.json$/.test(f));
if (!fragFiles.length) {
  console.error('❌ No fragment files (tasks-*.json) found in', fragDir);
  process.exit(1);
}

let version = '2.0.0';
const tasks = [];
for (const file of fragFiles) {
  const full = path.join(fragDir, file);
  try {
    const obj = JSON.parse(fs.readFileSync(full, 'utf8'));
    if (obj.version) version = obj.version; // last one wins, they should be same
    if (Array.isArray(obj.tasks)) tasks.push(...obj.tasks);
    else console.warn(`⚠️  Fragment ${file} missing tasks array`);
  } catch (e) {
    console.error(`❌ Failed to parse fragment ${file}: ${e.message}`);
    process.exit(1);
  }
}

// Detect duplicates
const labelCounts = new Map();
for (const t of tasks) {
  const lbl = t.label || '__undefined__';
  labelCounts.set(lbl, (labelCounts.get(lbl) || 0) + 1);
}
const duplicates = [...labelCounts.entries()].filter(([, c]) => c > 1);
if (duplicates.length && !allowDuplicates) {
  console.error('❌ Duplicate task labels detected:');
  for (const [lbl, c] of duplicates) console.error(`  ${lbl} (${c})`);
  console.error('Re-run with --allow-duplicates to force regeneration.');
  process.exit(1);
}

// Sort tasks by label for deterministic output
tasks.sort((a, b) => (a.label || '').toLowerCase().localeCompare((b.label || '').toLowerCase()));

const monolith = { version, tasks };
fs.writeFileSync(monolithPath, JSON.stringify(monolith, null, 2));

console.log(`✅ Regenerated monolith: ${tasks.length} tasks from ${fragFiles.length} fragments → .vscode/tasks.json`);
if (duplicates.length) {
  console.log(`⚠️  Duplicates kept (count=${duplicates.length}) due to --allow-duplicates flag.`);
}
console.log('Run: npm run tasks:verify  (or pre-commit) to confirm parity.');
