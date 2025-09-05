#!/usr/bin/env zx
/**
 * Iterative autosolve loop:
 * - Applies event handler codemod
 * - Runs svelte-check (machine output)
 * - Logs high-impact pattern counts (YoRHa module, default export, Button variant, FormField)
 * - Stops when no error reduction or max cycles reached
 */
import { $ } from 'zx';
import fs from 'node:fs';
import path from 'node:path';

const projectRoot = path.resolve(process.cwd());
const logDir = path.join(projectRoot,'logs');
if(!fs.existsSync(logDir)) fs.mkdirSync(logDir,{recursive:true});

const MAX_CYCLES = parseInt(process.env.AUTOSOLVE_MAX_CYCLES||'4',10);
let lastErrorCount = Infinity;

async function runCheck(){
  try {
    const { stdout } = await $`npx svelte-check --tsconfig ./tsconfig.json --output machine`;
    fs.writeFileSync(path.join(logDir,'last-svelte-check-machine.txt'), stdout, 'utf8');
    const lines = stdout.trim().split(/\r?\n/).filter(l=>l.includes('|'));
    let errors = 0, warnings = 0;
    for(const l of lines){
      const parts = l.split('|');
      if(parts[1]==='error') errors++; else if(parts[1]==='warning') warnings++;
    }
    return { errors, warnings, lines };
  } catch(e){
    console.error('svelte-check failed', e.exitCode||'', e.stderr||e.message);
    return { errors: lastErrorCount, warnings: 0, lines: [] };
  }
}

function detectPatterns(lines){
  const patterns = { yorhaDataGridModule:0, missingDefaultExport:0, buttonVariant:0, formFieldType:0, deprecatedEventResidual:0 };
  for(const l of lines){
    if(l.includes('YoRHaDataGrid') && l.includes('not a module')) patterns.yorhaDataGridModule++;
    if(/has no default export/.test(l)) patterns.missingDefaultExport++;
    if(/variant=/.test(l) && /not assignable to type/.test(l) && /Button/.test(l)) patterns.buttonVariant++;
    if(/FormField/.test(l) && /not assignable/.test(l)) patterns.formFieldType++;
    if(/deprecated/.test(l) && /event/.test(l)) patterns.deprecatedEventResidual++;
  }
  return patterns;
}

async function runEventCodemod(){
  await $`node ./scripts/codemod-events.mjs`;
}

async function main(){
  for(let cycle=1; cycle<=MAX_CYCLES; cycle++){
    console.log(`\n🔄 Autosolve Cycle ${cycle}`);
    await runEventCodemod();
    const result = await runCheck();
    const delta = lastErrorCount - result.errors;
    const patterns = detectPatterns(result.lines);
    const snapshot = { cycle, errors: result.errors, warnings: result.warnings, delta, patterns, timestamp: new Date().toISOString() };
    fs.appendFileSync(path.join(logDir,'autosolve-iterative-history.jsonl'), JSON.stringify(snapshot)+'\n');
    console.log('📊 Snapshot:', snapshot);
    if(delta <= 0){
      console.log('⏹️  No further error reduction; stopping.');
      break;
    }
    lastErrorCount = result.errors;
  }
  console.log('✅ Iterative autosolve complete. See logs/autosolve-iterative-history.jsonl');
}

main();
