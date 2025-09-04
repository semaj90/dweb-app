import fs from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const glob = require('glob');

const root = process.cwd();
const pattern = 'src/**/*.svelte';

const files = glob.sync(pattern, { cwd: root, absolute: true });

const results = [];

for (const fp of files) {
  const raw = fs.readFileSync(fp, 'utf8');
  const lines = raw.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.includes('$:')) {
      // Record 3-line snippet: prev, current, next
      const start = Math.max(0, i - 1);
      const end = Math.min(lines.length - 1, i + 1);
      const snippet = lines.slice(start, end + 1).join('\n');

      // Classify pattern
      const trimmed = line.trim();
      let patternType = 'other';
      if (/^\$:\s*\w+\s*=/.test(trimmed)) patternType = 'simple_assignment';
      else if (/^\$:\s*if\s*\(/.test(trimmed) || /^\$:\s*if\s*/.test(trimmed)) patternType = 'conditional_block';
      else if (/^\$:\s*\{/.test(trimmed)) patternType = 'side_effect_block';
      else if (/console\.(log|warn|error)\(/.test(trimmed)) patternType = 'debug_side_effect';
      else if (/\$state\(|\$derived|\$effect|on:|onclick|on\:/.test(trimmed)) patternType = 'mixed_runes_usage';

      results.push({ file: fp, line: i + 1, snippet, pattern: patternType });
    }
  }
}

// Also ensure we include the fatal file explicitly if it wasn't found
const fatal = `${root.replace(/\\/g, '/')}/src/routes/cases/[caseId]/rag/+page.svelte`.replace(/\//g, require('path').sep);
if (!results.some(r => r.file === fatal) && fs.existsSync(fatal)) {
  const raw = fs.readFileSync(fatal,'utf8');
  const lines = raw.split(/\r?\n/);
  for (let i=0;i<lines.length;i++){
    const line = lines[i];
    if (line.includes('$:')) {
      const start = Math.max(0, i - 1);
      const end = Math.min(lines.length - 1, i + 1);
      const snippet = lines.slice(start, end + 1).join('\n');
      const trimmed = line.trim();
      let patternType = 'other';
      if (/^\$:\s*\w+\s*=/.test(trimmed)) patternType = 'simple_assignment';
      else if (/^\$:\s*if\s*\(/.test(trimmed) || /^\$:\s*if\s*/.test(trimmed)) patternType = 'conditional_block';
      else if (/^\$:\s*\{/.test(trimmed)) patternType = 'side_effect_block';
      else if (/console\.(log|warn|error)\(/.test(trimmed)) patternType = 'debug_side_effect';
      else if (/\$state\(|\$derived|\$effect|on:|onclick|on\:/.test(trimmed)) patternType = 'mixed_runes_usage';
      results.push({ file: fatal, line: i+1, snippet, pattern: patternType });
    }
  }
}

// Deduplicate by file+line
const uniq = [];
const seen = new Set();
for (const r of results) {
  const key = `${r.file}:${r.line}`;
  if (!seen.has(key)) { seen.add(key); uniq.push(r); }
}

// Print a concise report to stdout
console.log(JSON.stringify({ generatedAt: new Date().toISOString(), totalMatches: uniq.length, matches: uniq }, null, 2));
