#!/usr/bin/env node
/**
 * Adapter: Feed extracted Svelte/TS error patterns into MultiCoreClusterManager for semantic clustering.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

// Force worker-thread mode (avoid cluster forking recursion when adapter is entrypoint)
process.env.ENABLE_CLUSTERING = 'false';
// If somehow invoked inside a cluster worker context, abort early to avoid duplicate processing
if (process.env.WORKER_TYPE === 'cluster') {
  process.exit(0);
}

async function dynamicImportSolver(){
  const solverPath = path.resolve('vscode-auto-solver/core/multi-core-solver.js');
  // Ensure Windows absolute path converted to file:// URL for ESM import
  const url = process.platform === 'win32' && !solverPath.startsWith('file:')
    ? new URL('file://' + solverPath.replace(/\\/g,'/'))
    : solverPath;
  return await import(url.href || url);
}

async function fileContentOrMessage(file, fallback){
  try {
    if (existsSync(file)) {
      const raw = await readFile(file,'utf8');
      return raw.slice(0, 8000); // cap to avoid huge payloads
    }
  } catch {}
  return fallback;
}

async function expandToProblemObjects(patternReport){
  const problems = [];
  const MAX_GROUPS = 200; // cap to avoid explosion
  let groupCount = 0;
  for(const group of (patternReport.top || [])){
    if(!group || !Array.isArray(group.sample)) continue;
    for(const sample of group.sample){
      if(!sample || !sample.file || !sample.msg) continue;
      const content = await fileContentOrMessage(sample.file, sample.msg);
      problems.push({
        filePath: sample.file,
        content: content || sample.msg,
        message: sample.msg,
        category: group.category,
        line: sample.line,
        column: sample.col
      });
    }
    if(++groupCount >= MAX_GROUPS) break;
  }
  return problems;
}

(async function(){
  const reportFile = '.vscode/svelte-error-patterns.json';
  if(!existsSync(reportFile)){
    console.error('❌ Pattern report missing. Run npm run errors:patterns first.');
    process.exit(2);
  }
  const report = JSON.parse(await readFile(reportFile,'utf8'));
  const problems = await expandToProblemObjects(report);
  if(!problems.length){
    console.log('✅ No problems to process.');
    return;
  }
  // Filter duplicates by file+line+message hash
  const seen = new Set();
  const deduped = [];
  for(const p of problems){
    const key = `${p.filePath}:${p.line}:${p.message}`;
    if(!seen.has(key)) { seen.add(key); deduped.push(p); }
  }
  if(deduped.length !== problems.length){
    console.log(`🔁 Deduplicated problems ${problems.length} -> ${deduped.length}`);
  }
  const { MultiCoreClusterManager } = await dynamicImportSolver();
  const mgr = new MultiCoreClusterManager();
  await mgr.initializeCluster();
  const batch = await mgr.processProblemBatch(deduped);
  await writeFile('.vscode/svelte-semantic-results.json', JSON.stringify(batch, null, 2));
  console.log('✅ Semantic clustering complete -> .vscode/svelte-semantic-results.json');
})();
