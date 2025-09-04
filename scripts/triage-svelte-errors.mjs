#!/usr/bin/env node
// Triage Svelte/TypeScript errors and output categorized summary.
// Produces svelte-errors.ndjson and SVELTE_ERROR_TRIAGE.md
import { spawn } from 'node:child_process';
import { createWriteStream, existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Derive repository root from script location instead of current working directory
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(scriptDir); // scripts/ -> repo root
const svelteFrontendDir = path.join(root, 'sveltekit-frontend');
console.log('[triage] repoRoot=', root);
console.log('[triage] svelteFrontendDir=', svelteFrontendDir);
const tsconfigCandidates = [
  path.join(svelteFrontendDir, 'tsconfig.json'),
  path.join(root, 'tsconfig.json'),
];
const tsconfig = tsconfigCandidates.find((p) => existsSync(p));
if (!tsconfig) {
  console.error('No tsconfig.json found for svelte-check triage');
  process.exit(1);
}

const outNdjson = path.join(root, 'svelte-errors.ndjson');
const outMd = path.join(root, 'SVELTE_ERROR_TRIAGE.md');

function runSvelteCheck() {
  return new Promise((resolve, reject) => {
    // Always run from repo root so devDependencies (svelte-check) are resolvable
    const cwd = root;
  // Use package script context inside sveltekit-frontend to ensure local deps resolution
  const args = ['npm', 'run', 'check:svelte:machine', '--silent'];
  console.log('[triage] running (frontend script):', args.join(' '), 'cwd=', svelteFrontendDir);
  const child = spawn(args[0], args.slice(1), { cwd: svelteFrontendDir, shell: process.platform === 'win32', env: { ...process.env, FORCE_COLOR: '0', NODE_OPTIONS: '' } });
  const ws = createWriteStream(outNdjson, { encoding: 'utf8' });
  const rawPath = path.join(root, 'svelte-errors-raw.log');
  const raw = createWriteStream(rawPath, { encoding: 'utf8' });
  child.stdout.on('data', (d) => { ws.write(d); raw.write(d); });
    child.stderr.on('data', (d) => process.stderr.write(d));
    const timeout = setTimeout(() => {
      console.error('[triage] ERROR: svelte-check timeout (120s) – killing');
      child.kill('SIGKILL');
    }, 120000);
    child.on('error', reject);
    child.on('close', (code) => {
      clearTimeout(timeout);
  ws.end(); raw.end();
      if (code === 0 || code === 1) return resolve();
      reject(new Error('svelte-check failed with code ' + code));
    });
  });
}

function classify(e) {
  const code = e.code || '';
  const msg = e.message || '';
  if (/parse|parser|unexpected token/i.test(msg)) return 'syntax';
  if (/Cannot find module|Module not found|Cannot import/i.test(msg)) return 'missing-module';
  if (/is not a valid prop|Unknown prop|not assignable|Type '(.*)' is not assignable/i.test(msg)) return 'prop-or-type';
  if (/implicitly has an 'any' type|implicitly has type 'any'/i.test(msg)) return 'implicit-any';
  if (/unused|declared but its value is never read/i.test(msg)) return 'unused';
  if (/JSX/i.test(msg)) return 'legacy-syntax';
  return code.startsWith('ts') ? 'ts-other' : 'other';
}

function summarize() {
  if (!existsSync(outNdjson)) {
    console.error('NDJSON output not found:', outNdjson);
    process.exit(1);
  }
  const lines = readFileSync(outNdjson, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .filter((l) => l.trim().startsWith('{'));

  if (!lines.length) {
    console.error('[triage] No machine-output lines parsed. Possible causes:');
    console.error(' - svelte-check produced no diagnostics');
    console.error(' - Build cache prevented analysis');
    console.error(' - Wrong tsconfig path');
    console.error('tsconfig used:', tsconfig);
    console.error('Exiting with code 2.');
    process.exit(2);
  }

  const entries = [];
  for (const line of lines) {
    try {
      const obj = JSON.parse(line);
      if (obj.diagnostic) {
        entries.push({
          file: obj.file || obj.diagnostic.file || 'unknown',
          code: obj.diagnostic.code || obj.code || '',
          message: obj.diagnostic.message || obj.message || '',
          start: obj.diagnostic.start || obj.start || null,
          end: obj.diagnostic.end || obj.end || null,
        });
      } else if (obj.file || obj.message) {
        entries.push(obj);
      }
    } catch {}
  }

  const buckets = new Map();
  for (const e of entries) {
    const cat = classify(e);
    if (!buckets.has(cat)) buckets.set(cat, []);
    buckets.get(cat).push(e);
  }

  const byFile = new Map();
  for (const e of entries) {
    const f = e.file;
    if (!byFile.has(f)) byFile.set(f, 0);
    byFile.set(f, byFile.get(f) + 1);
  }
  const topFiles = [...byFile.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25);

  const order = [
    'syntax',
    'missing-module',
    'prop-or-type',
    'implicit-any',
    'ts-other',
    'unused',
    'legacy-syntax',
    'other',
  ];

  let md = '# Svelte Error Triage\n\n';
  md += `Total diagnostics: ${entries.length}\n\n`;
  md += '## Category Counts (priority order)\n\n';
  for (const cat of order) {
    if (buckets.has(cat)) {
      md += `- ${cat}: ${buckets.get(cat).length}\n`;
    }
  }
  md += '\n## Top Affected Files\n\n';
  for (const [file, count] of topFiles) {
    md += `- ${file} (${count})\n`;
  }

  function sample(cat, limit = 5) {
    const arr = (buckets.get(cat) || []).slice(0, limit);
    if (!arr.length) return '';
    let sec = `\n### Examples: ${cat}\n\n`;
    for (const e of arr) {
      sec += `- ${e.file}: ${e.message.replace(/\s+/g, ' ').slice(0, 160)}\n`;
    }
    return sec;
  }

  for (const cat of order) md += sample(cat);

  md += '\n## Remediation Plan\n\n';
  md += '1. Fix ALL syntax & missing-module first (stop build blockers).\n';
  md += '2. Address top 10 files by error density to collapse cascades.\n';
  md += '3. Standardize component prop definitions (export let ...) & shared types.\n';
  md += '4. Gradually add strict types where implicit-any clusters appear.\n';
  md += '5. Clean unused via eslint/prettier or suppress.\n';

  writeFileSync(outMd, md, 'utf8');
  console.log('Generated triage summary at', outMd);
}

try {
  await runSvelteCheck();
  summarize();
} catch (e) {
  console.error('Triage failed:', e.message);
  process.exit(1);
}
