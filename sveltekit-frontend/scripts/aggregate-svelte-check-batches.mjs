#!/usr/bin/env node
// Aggregate per-batch svelte-check stdout logs into a summarized JSON report.
// Usage: node scripts/aggregate-svelte-check-batches.mjs --inputs logs/batch-*.stdout --out logs/svelte-check-aggregate.json
import fs from 'fs';
import path from 'path';

function parseArgs(){
  const args = process.argv.slice(2);
  const inputs = [];
  let out = 'logs/svelte-check-aggregate.json';
  for(let i=0;i<args.length;i++){
    if(args[i]==='--inputs'){
      const pattern = args[++i];
      // simple glob: expand * at end
      if(pattern.includes('*')){
        const dir = path.dirname(pattern);
        const base = path.basename(pattern).replace('*','');
        fs.readdirSync(dir).forEach(f=>{ if(f.startsWith(base)) inputs.push(path.join(dir,f)); });
      } else { inputs.push(pattern); }
    } else if(args[i]==='--out') { out = args[++i]; }
  }
  return { inputs, out };
}

function extractEntries(text){
  const lines = text.split(/\r?\n/);
  const issues = [];
  let currentFile = null;
  let buffer = [];
  const flush = ()=>{ if(buffer.length && currentFile){ issues.push({ file: currentFile, message: buffer.join('\n') }); buffer=[]; } };
  for(let i=0;i<lines.length;i++){
    const line = lines[i];
    const fileMatch = line.match(/^(.*\.(svelte|ts|js|d\.ts)):(\d+):(\d+)/);
    if(fileMatch){ flush(); currentFile = fileMatch[1]; continue; }
    if(/^(Error|TypeError|ReferenceError):/.test(line) || line.includes('Type ') || line.includes('is not assignable')){
      buffer.push(line.trim());
      // capture following indented context lines
      let j=i+1;
      while(j<lines.length && /^\s{2,}\S/.test(lines[j])){ buffer.push(lines[j].trim()); j++; }
      i = j-1;
      flush();
    }
  }
  flush();
  return issues;
}

function main(){
  const { inputs, out } = parseArgs();
  if(!inputs.length){ console.error('No input logs specified'); process.exit(1); }
  const aggregate = { generatedAt: new Date().toISOString(), totalIssues: 0, byFile: {}, sample: [] };
  for(const f of inputs){
    if(!fs.existsSync(f)) continue;
    const txt = fs.readFileSync(f,'utf8');
    let issues = [];
    if(f.endsWith('.json')){
      try {
        const parsed = JSON.parse(txt);
        if(Array.isArray(parsed.diagnostics)){
          issues = parsed.diagnostics.map(d=>({ file: d.file, message: d.message || d.code }));
        } else if(Array.isArray(parsed)) {
          issues = parsed.flatMap(entry => (entry?.diagnostics||[]).map(d=>({ file: d.file, message: d.message||d.code })));
        }
      } catch(e){
        issues = extractEntries(txt); // fallback to heuristic
      }
    } else {
      issues = extractEntries(txt);
    }
    for(const issue of issues){
      if(!issue.file) continue;
      aggregate.totalIssues++;
      aggregate.byFile[issue.file] = (aggregate.byFile[issue.file] || 0) + 1;
      if(aggregate.sample.length < 50) aggregate.sample.push(issue);
    }
  }
  const sorted = Object.entries(aggregate.byFile).sort((a,b)=>b[1]-a[1]).slice(0,50);
  aggregate.topFiles = sorted.map(([file,count])=>({file,count}));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(aggregate, null, 2), 'utf8');
  console.log('Aggregate written to', out, 'Total issues:', aggregate.totalIssues, 'Top file:', aggregate.topFiles[0]);
}

main();
