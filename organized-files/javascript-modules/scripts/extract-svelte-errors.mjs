#!/usr/bin/env node
/**
 * Extract dominant Svelte / TS error patterns from .vscode/svelte-check.json
 * Output top 20 patterns grouped by (inferredCategory, file)
 */
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const INPUT = '.vscode/svelte-check.json';
const REPORT = '.vscode/svelte-error-patterns.json';

function inferCategory(line){
  if (/Unexpected token|Unterminated|TS1005|TS1128|TS1109/.test(line)) return 'parse';
  if (/Cannot find module|Cannot find name|TS2307|TS2304/.test(line)) return 'import';
  if (/is not assignable|TS2322|TS2345|TS2769|TS2559/.test(line)) return 'type';
  if (/on:click|onclick|\$props\(/.test(line)) return 'migration';
  return 'other';
}

async function run(){
  if(!existsSync(INPUT)) { console.error('No capture file found'); process.exit(1); }
  const raw = JSON.parse(await readFile(INPUT,'utf8'));
  const stripAnsi = s => s.replace(/\u001b\[[0-9;]*m/g,'');
  const lines = (raw.raw||'').split(/\r?\n/).map(l=>stripAnsi(l.trim())).filter(Boolean);

  // New parser targets lines like: path/to/file.svelte:LINE:COL then next non-empty line starting with Error: or Warn:
  const buckets = new Map();
  for(let i=0;i<lines.length;i++){
    const header = lines[i];
    const mh = header.match(/^(.+?(?:\.svelte|\.ts|\.js)):(\d+):(\d+)$/i);
    if(!mh) continue;
    const file = mh[1];
    const lineNo = +mh[2];
    const colNo = +mh[3];
    // Locate message line before next header
    let j = i+1, messageLine='';
    while(j<lines.length){
      const nxt = lines[j];
      if(/^(.+?(?:\.svelte|\.ts|\.js)):(\d+):(\d+)$/.test(nxt)) break; // encountered next header without a message
      if(nxt.startsWith('Error:') || nxt.startsWith('Warn:')) { messageLine = nxt; break; }
      j++;
    }
    if(!messageLine) continue;
    const mm = messageLine.match(/^(Error|Warn):\s+(.+?)(?:\s+\((ts|svelte|css)\))?$/i);
    if(!mm) continue;
    const sev = mm[1].toLowerCase()==='error' ? 'error':'warning';
    const msg = mm[2];
    const codeMatch = msg.match(/TS(\d{3,5})/i);
    const code = codeMatch ? 'TS'+codeMatch[1] : (mm[3] ? mm[3].toUpperCase() : 'GEN');
    const category = inferCategory(msg + ' ' + file);
    const key = `${category}::${file}`;
    const arr = buckets.get(key)||[];
    arr.push({ file, line: lineNo, col: colNo, sev, code, msg });
    buckets.set(key, arr);
  }

  const summary = [];
  for(const [key, arr] of buckets.entries()){
    const [category, file] = key.split('::');
    summary.push({ category, file, count: arr.length, sample: arr.slice(0,3) });
  }
  summary.sort((a,b)=>b.count-a.count);
  const top = summary.slice(0,20);
  await writeFile(REPORT, JSON.stringify({ generatedAt: new Date().toISOString(), totalGroups: summary.length, top }, null, 2));
  console.log('✅ Wrote pattern report ->', REPORT);
  if(top.length){
    console.log(top.map(t=>`${t.category.padEnd(9)} ${t.count.toString().padStart(4)} ${t.file}` ).join('\n'));
  } else {
    console.log('ℹ No patterns extracted (parser produced zero groups)');
  }
}

run().catch(e=>{console.error(e);process.exit(1);});
