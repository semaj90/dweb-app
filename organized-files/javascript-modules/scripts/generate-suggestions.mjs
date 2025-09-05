#!/usr/bin/env node
/**
 * Generate simple patch suggestions from semantic clustering results.
 * Focus: parse & import categories first.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

// Allow running from any subdirectory by walking upward to locate files
function findUpwards(relPath) {
  let dir = process.cwd();
  while (true) {
    const candidate = path.join(dir, relPath);
    if (existsSync(candidate)) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir) return null; // hit filesystem root
    dir = parent;
  }
}

const SEM_REL = path.join('.vscode','svelte-semantic-results.json');
const PATTERN_REL = path.join('.vscode','svelte-error-patterns.json');
let semFile = findUpwards(SEM_REL);
let patternFile = findUpwards(PATTERN_REL);

if(!semFile){
  console.error('Semantic results missing; could not locate', SEM_REL, 'from any parent directory.');
  process.exit(2);
}

// Output file sits beside semantic results for consistency
const OUT_FILE = path.join(path.dirname(semFile), 'svelte-fix-suggestions.json');

function buildSuggestion(problem){
  if(problem.category === 'import'){
    return {
      file: problem.filePath,
      action: 'normalize-imports',
      note: 'Check module specifiers; ensure relative paths start with ./ or ../ and remove duplicate type keyword.'
    };
  }
  if(problem.category === 'parse'){
    return {
      file: problem.filePath,
      action: 'syntax-repair',
      note: 'Likely stray characters or truncated block. Reconstruct import header & ensure balanced braces.'
    };
  }
  if(problem.category === 'migration'){
    return {
      file: problem.filePath,
      action: 'event-directive',
      note: 'Replace on:click with onclick or proper <button on:click> syntax per Svelte5 migration.'
    };
  }
  if(problem.category === 'binding'){
    return {
      file: problem.filePath,
      action: 'binding-fix',
      note: 'Normalize runes ($state/$props) and ensure exported props & correct bind: usage.'
    };
  }
  return {
    file: problem.filePath,
    action: 'type-fix',
    note: 'Investigate TS type mismatch; add explicit types or generic parameters.'
  };
}

(async function(){
  const data = JSON.parse(await readFile(semFile,'utf8'));
  // Our semantic output structure uses 'results' array
  let problems = Array.isArray(data.results) ? data.results : (data.problems || []);
  if(!problems.length){
    console.error('No semantic results entries found (results array empty).');
    process.exit(3);
  }
  // Load pattern report for category mapping by file if category missing
  let categoryByFile = new Map();
  if (patternFile && existsSync(patternFile)) {
    try {
      const pr = JSON.parse(await readFile(patternFile,'utf8'));
      for(const grp of pr.top||[]){
        categoryByFile.set(grp.file, grp.category);
      }
    } catch {}
  }
  const dedup = new Map();
  for(const p of problems){
    if(!p.filePath) continue;
    // Attach category if missing using pattern mapping
    if(!p.category && categoryByFile.has(p.filePath)){
      p.category = categoryByFile.get(p.filePath);
    }
    if(!dedup.has(p.filePath)) dedup.set(p.filePath, buildSuggestion(p));
  }
  const suggestions = Array.from(dedup.values());
  await writeFile(OUT_FILE, JSON.stringify({ generatedAt: new Date().toISOString(), suggestions }, null, 2));
  console.log('✅ Wrote suggestions ->', OUT_FILE, 'count=', suggestions.length);
})();
