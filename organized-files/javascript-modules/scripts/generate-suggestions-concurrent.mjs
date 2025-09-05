#!/usr/bin/env node
/**
 * Concurrent suggestion generator:
 * 1. Loads semantic results
 * 2. Chunks problems
 * 3. Processes chunks in parallel using p-limit
 * 4. Aggregates suggestions (dedup by file)
 */
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import pLimit from 'p-limit';
import { readFile as fsReadFile } from 'node:fs/promises';

const CONCURRENCY = parseInt(process.env.SUGGEST_CONCURRENCY||'8',10);

function findUpwards(relPath){
  let dir = process.cwd();
  while(true){
    const cand = path.join(dir, relPath);
    if (existsSync(cand)) return cand;
    const parent = path.dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

const SEM_FILE = findUpwards(path.join('.vscode','svelte-semantic-results.json'));
if(!SEM_FILE){
  console.error('❌ Cannot locate semantic results');
  process.exit(2);
}
const PATTERN_FILE = findUpwards(path.join('.vscode','svelte-error-patterns.json'));
const OUT_FILE = path.join(path.dirname(SEM_FILE), 'svelte-fix-suggestions.json');

async function inferCategory(p, categoryByFile, content){
  if (p.category) return p.category;
  if (p.problemTypes?.includes('import-error')) return 'import';
  if (p.problemTypes?.some(t=>/parse|syntax/i.test(t))) return 'parse';
  if (categoryByFile.has(p.filePath)) return categoryByFile.get(p.filePath);
  if (content){
    // Simple heuristics
  // Event directive migration (Svelte5 runes adaptation)
  if (/on:[a-zA-Z]+=/m.test(content) && /onclick=/.test(content) === false) return 'migration';
  // Rune / binding issues (experimental) categorize as 'binding'
  if (/(\$state|\$derived|\$props|\$effect)/.test(content) || /bind:[a-zA-Z]+=/m.test(content)) return 'binding';
  // Store misuse (reactive store incorrectly de-structured) treat as type for now
    const importStmts = content.match(/^import[^;]+;?$/mg) || [];
    if (importStmts.some(l=>!/from\s+['"].+['"];?$/.test(l.trim()) || l.includes('@/'))) return 'import';
    const openBraces = (content.match(/\{/g)||[]).length;
    const closeBraces = (content.match(/\}/g)||[]).length;
    if (Math.abs(openBraces - closeBraces) > 5) return 'parse';
  }
  return 'type';
}

function buildSuggestion(problem){
  switch(problem.category){
  case 'binding': return { file: problem.filePath, action: 'binding-fix', note: 'Normalize Svelte5 runes ($state, $props) and ensure proper bind: usage / exported props.' };
    case 'import': return { file: problem.filePath, action: 'normalize-imports', note: 'Normalize/verify module specifiers (prefix ./, remove duplicate type, merge imports).' };
    case 'parse': return { file: problem.filePath, action: 'syntax-repair', note: 'Repair unbalanced braces/tags and incomplete statements.' };
    case 'migration': return { file: problem.filePath, action: 'svelte5-event', note: 'Adjust deprecated event directive (on: to onclick or proper binding).' };
    default: return { file: problem.filePath, action: 'type-fix', note: 'Add explicit types, correct generics, or split overly broad unions.' };
  }
}

(async function(){
  const raw = JSON.parse(await readFile(SEM_FILE,'utf8'));
  let problems = Array.isArray(raw.results)? raw.results : [];
  if(!problems.length){
    console.error('No semantic results found.');
    process.exit(3);
  }
  // category mapping from pattern file
  const categoryByFile = new Map();
  if (PATTERN_FILE && existsSync(PATTERN_FILE)) {
    try {
      const pr = JSON.parse(await readFile(PATTERN_FILE,'utf8'));
      for(const g of pr.top||[]) categoryByFile.set(g.file, g.category);
    } catch {}
  }
  // Basic filtering & cap
  problems = problems.filter(p=>p.filePath).slice(0, 500);
  const limit = pLimit(CONCURRENCY);
  const ENRICH = process.env.SUGGEST_ENRICH === '1';
  const tasks = problems.map(p=> limit(async () => {
    let content = null;
    if (ENRICH) {
      try { content = await fsReadFile(p.filePath, 'utf8'); } catch {}
    }
    const category = await inferCategory(p, categoryByFile, content);
    return buildSuggestion({...p, category});
  }));
  const suggestionsRaw = await Promise.all(tasks);
  // Dedup by file keeping earliest import/type priority order
  const priority = { 'normalize-imports':1, 'syntax-repair':2, 'type-fix':3, 'svelte5-event':2 };
  const best = new Map();
  for(const s of suggestionsRaw){
    const existing = best.get(s.file);
    if(!existing || priority[s.action] < priority[existing.action]){
      best.set(s.file, s);
    }
  }
  const suggestions = Array.from(best.values()).sort((a,b)=> priority[a.action]-priority[b.action]);
  await writeFile(OUT_FILE, JSON.stringify({ generatedAt: new Date().toISOString(), concurrency: CONCURRENCY, totalInput: problems.length, suggestions }, null, 2));
  console.log(`✅ Concurrent suggestions written (${suggestions.length}) -> ${OUT_FILE}`);
})();
