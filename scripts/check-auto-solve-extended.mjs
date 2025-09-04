#!/usr/bin/env node
/**
 * Extended AutoSolve pipeline:
 * 1. Capture raw svelte-check output
 * 2. Extract grouped patterns
 * 3. Semantic solve (problem typing & categorization)
 * 4. Concurrent suggestion generation (+ optional enrichment SUGGEST_ENRICH=1)
 * 5. Apply suggestions (codemods) with dry-run option
 * 6. Re-run svelte-check to measure delta
 * 7. Append metrics JSONL entry for CI gating (.vscode/autosolve-metrics.jsonl)
 */
import { spawn } from 'node:child_process';
import { readFile, writeFile, appendFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

// Flexible DRY_RUN parsing: accepts 1/true/yes/on (case-insensitive)
const DRY_RUN = (() => {
  const v = process.env.DRY_RUN;
  if(!v) return false;
  return ['1','true','yes','on'].includes(v.toLowerCase());
})();
const FORCE_CONTINUE = process.argv.includes('--force-continue');
const ROOT = process.cwd();
const METRICS_FILE = path.join(ROOT, '.vscode', 'autosolve-metrics.jsonl');

async function run(cmd, args, label, { allowNonZero=false } = {}){
  return new Promise((resolve,reject)=>{
    const start = Date.now();
    const child = spawn(cmd, args, { stdio: 'pipe', shell: process.platform === 'win32' });
    let out=''; let err='';
    child.stdout.on('data',d=> out+=d.toString());
    child.stderr.on('data',d=> err+=d.toString());
    child.on('close', code => {
      const dur = Date.now()-start;
      if(code!==0 && !allowNonZero){
        console.error(`❌ ${label} failed (code ${code})`);
        console.error(err||out);
        return reject(new Error(label));
      }
      if(code!==0 && allowNonZero){
        console.warn(`⚠️ ${label} exited with code ${code} (continuing due to allowNonZero)`);
      } else {
        console.log(`✅ ${label} (${dur}ms)`);
      }
      resolve({ out, err, dur, exitCode: code });
    });
  });
}

function parseSvelteCheckOutput(text){
  // Looks for "svelte-check found X errors and Y warnings"
  const m = text.match(/svelte-check found (\d+) errors? and (\d+) warnings?/i);
  if(!m) return null;
  return { errors: parseInt(m[1],10), warnings: parseInt(m[2],10) };
}

async function ensureVscodeDir(){
  const dir = path.join(ROOT,'.vscode');
  if(!existsSync(dir)){
    await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir,'README.autosolve.md'),'Autosolve artifacts directory');
  }
}

async function captureBaseline(){
  console.log('\n▶ Baseline svelte-check');
  const res = await run('npx',['svelte-check'], 'svelte-check (baseline)', { allowNonZero: true });
  const metrics = parseSvelteCheckOutput(res.out+res.err) || { errors: -1, warnings: -1 };
  return { raw: res.out+res.err, ...metrics };
}

async function pipeline(){
  console.log('🔧 Autosolve pipeline starting...');
  console.log(`   DRY_RUN=${DRY_RUN} FORCE_CONTINUE=${FORCE_CONTINUE} AUTOSOLVE_MIN_REDUCTION=${process.env.AUTOSOLVE_MIN_REDUCTION ?? '∅'}`);
  await ensureVscodeDir();
  const started = new Date().toISOString();
  const baseline = await captureBaseline();

  console.log('\n▶ Capture errors');
  await run('node',['scripts/capture-svelte-errors.mjs'],'Capture Svelte Errors');

  console.log('\n▶ Extract patterns');
  await run('node',['scripts/extract-svelte-errors.mjs'],'Extract Patterns');

  console.log('\n▶ Semantic solve');
  await run('node',['scripts/semantic-solve-svelte-errors.mjs'],'Semantic Solve');

  console.log('\n▶ Generate concurrent suggestions');
  await run('node',['scripts/generate-suggestions-concurrent.mjs'],'Concurrent Suggestions');

  console.log(`\n▶ Apply suggestions${DRY_RUN ? ' (dry-run)' : ''}`);
  await run('node',['scripts/apply-suggestions.mjs', DRY_RUN ? '--dry-run':''].filter(Boolean),'Apply Suggestions');

  console.log('\n▶ Post-fix svelte-check');
  const post = await run('npx',['svelte-check'],'svelte-check (post-fix)', { allowNonZero: true });
  const postMetrics = parseSvelteCheckOutput(post.out+post.err) || { errors: -1, warnings: -1 };

  const deltaErrors = baseline.errors - postMetrics.errors;
  const deltaWarnings = baseline.warnings - postMetrics.warnings;

  const record = { timestamp: started, completedAt: new Date().toISOString(), baseline, post: postMetrics, delta: { errors: deltaErrors, warnings: deltaWarnings }, dryRun: DRY_RUN };
  const line = JSON.stringify(record);
  try {
    await appendFile(METRICS_FILE, line + '\n');
  } catch (e) {
    // Fallback – ensure file exists
    await writeFile(METRICS_FILE, line + '\n');
  }
  // Verification & reliability retry if size still zero
  try {
    const txt = await readFile(METRICS_FILE, 'utf8');
    if(!txt.trim()) {
      await writeFile(METRICS_FILE, line + '\n');
    } else if(!txt.includes(line)) {
      // Append again if missing (race / flush issue)
      await appendFile(METRICS_FILE, line + '\n');
    }
  } catch {}
  console.log('\n📈 Metrics line:');
  console.log(line);
  console.log(`📁 Metrics stored at ${METRICS_FILE}`);

  // Simple gate output
  if(deltaErrors>0){
    console.log(`✅ Error reduction: ${deltaErrors} errors resolved, warnings Δ ${deltaWarnings}`);
  } else if (deltaErrors===0) {
    console.log('⚠️ No net error change this cycle.');
  } else {
    console.log(`❌ Errors increased by ${Math.abs(deltaErrors)} (baseline ${baseline.errors} → post ${postMetrics.errors}).`);
  }

  // ENV gate: AUTOSOLVE_MIN_REDUCTION = percent threshold (e.g. 1.5 for 1.5%)
  const gate = process.env.AUTOSOLVE_MIN_REDUCTION ? parseFloat(process.env.AUTOSOLVE_MIN_REDUCTION) : null;
  if(gate !== null && baseline.errors > 0) {
    const percentReduction = ((baseline.errors - postMetrics.errors) / baseline.errors) * 100;
    const pass = percentReduction >= gate;
    console.log(`🔒 Gate: baseline=${baseline.errors} post=${postMetrics.errors} reduction=${percentReduction.toFixed(2)}% (threshold ${gate}%) -> ${pass ? 'PASS' : 'FAIL'}`);
    if(!pass) process.exitCode = 1;
  }
}

pipeline().catch(e=>{ if(FORCE_CONTINUE){ console.warn('Force-continue enabled; attempting partial metrics append.'); } else { console.error('Pipeline failed:', e.message); process.exit(1); } });
