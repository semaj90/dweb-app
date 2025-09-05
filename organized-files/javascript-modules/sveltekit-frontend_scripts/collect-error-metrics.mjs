#!/usr/bin/env node
/*
 Phase 0 – Error Metrics Collector
 ---------------------------------
 Captures TypeScript + Svelte diagnostic counts and top patterns, writes diffable JSON.
*/
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, '.vscode');
const METRICS_FILE = path.join(OUT_DIR, 'error-metrics.json');
const HISTORY_FILE = path.join(OUT_DIR, 'error-metrics-history.jsonl');
const isBaseline = process.argv.includes('--baseline');

function run(cmd){
  try { return execSync(cmd, { stdio: 'pipe', encoding: 'utf8' }); } catch (e){ return e.stdout || e.message; }
}

function parseTS(output){
  const lines = output.split(/\r?\n/).filter(Boolean);
  const fileErrors = {};
  const patternCounts = {};
  let total = 0;
  for(const l of lines){
    const m = l.match(/^(.*?\.\w+):(\d+):(\d+) - error (TS\d+): (.*)$/);
    if(m){
      total++; const [,file,, , code, msg] = m;
      fileErrors[file] = (fileErrors[file]||0)+1;
      const key = `${code} ${msg.split(/[,.]/)[0]}`.slice(0,140);
      patternCounts[key] = (patternCounts[key]||0)+1;
    }
  }
  return { total, fileErrors, patternCounts };
}

function parseSvelte(output){
  const lines = output.split(/\r?\n/).filter(l=>/Error:/i.test(l));
  const total = lines.length; // simple approximation
  const patternCounts = {};
  for(const l of lines){
    const key = l.replace(/.*Error:/,'Error:').trim().slice(0,140);
    patternCounts[key] = (patternCounts[key]||0)+1;
  }
  return { total, patternCounts };
}

function top(obj, n=15){
  return Object.entries(obj).sort((a,b)=>b[1]-a[1]).slice(0,n).map(([k,v])=>({ pattern:k, count:v }));
}

if(!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive:true });

console.log('[metrics] Running TypeScript check...');
const tsRaw = run('npx tsc --noEmit --skipLibCheck');
console.log('[metrics] Running Svelte check...');
const svRaw = run('npx svelte-check --tsconfig ./tsconfig.json --threshold warning');

const ts = parseTS(tsRaw);
const sv = parseSvelte(svRaw);
const timestamp = new Date().toISOString();

let previous = null;
if(fs.existsSync(METRICS_FILE)){
  try { previous = JSON.parse(fs.readFileSync(METRICS_FILE,'utf8')); } catch {}
}

const data = {
  timestamp,
  baseline: isBaseline && !previous,
  totals: {
    typescript: ts.total,
    svelte: sv.total,
    combined: ts.total + sv.total
  },
  topPatterns: {
    typescript: top(ts.patternCounts),
    svelte: top(sv.patternCounts)
  },
  topFiles: top(ts.fileErrors, 20),
  diff: previous ? {
    typescript: ts.total - previous.totals.typescript,
    svelte: sv.total - previous.totals.svelte,
    combined: (ts.total + sv.total) - previous.totals.combined
  } : null
};

fs.writeFileSync(METRICS_FILE, JSON.stringify(data,null,2));
fs.appendFileSync(HISTORY_FILE, JSON.stringify(data)+"\n");
console.log(`[metrics] Written ${METRICS_FILE}`);
if(data.diff) console.log('[metrics] Diff:', data.diff);
if(isBaseline) console.log('[metrics] Baseline recorded.');
