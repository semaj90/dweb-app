#!/usr/bin/env node
/**
 * Split monolithic .vscode/tasks.json into categorized fragments under .vscode/tasks/
 * Categories: ai, autosolve, db, backend, frontend, docs, health, misc
 */
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const tasksPath = path.join(root,'.vscode','tasks.json');
const outDir = path.join(root,'.vscode','tasks');
if(!fs.existsSync(tasksPath)){
  console.error('tasks.json not found');
  process.exit(1);
}
let text = fs.readFileSync(tasksPath,'utf8');
// Robust comment stripper: remove // and /* */ only when outside string literals
function stripComments(input){
  let out = '';
  let inStr = false; let esc = false; let i=0; const n=input.length;
  while(i<n){
    const ch = input[i];
    if(inStr){
      out += ch;
      if(esc){ esc=false; }
      else if(ch==='\\'){ esc=true; }
      else if(ch==='"'){ inStr=false; }
      i++; continue;
    }
    if(ch==='"'){ inStr=true; out+=ch; i++; continue; }
    if(ch==='/' && i+1<n){
      const nxt = input[i+1];
      if(nxt==='/' ){ // line comment
        // skip until newline
        i+=2;
        while(i<n && input[i] !== '\n'){ i++; }
        continue; // keep newline (will be appended below if present)
      } else if(nxt==='*'){ // block comment
        i+=2;
        while(i+1<n && !(input[i]==='*' && input[i+1]==='/')){ i++; }
        i+=2; // skip closing */
        continue;
      }
    }
    out+=ch; i++;
  }
  return out;
}
text = stripComments(text);

let tasks = [];
let version = '2.0.0';
let skipped = 0; // unified skipped counter across strategies
// First attempt: parse whole JSON after comment stripping (fast + lossless)
try {
  const rootObj = JSON.parse(text);
  if(Array.isArray(rootObj.tasks)) tasks = rootObj.tasks; else throw new Error('tasks not array');
  if(rootObj.version) version = rootObj.version;
  console.log(`Full JSON parse succeeded: ${tasks.length} tasks`);
} catch(parseErr){
  console.warn('Full JSON parse failed, falling back to manual extraction:', parseErr.message);
  // Locate tasks array
  const tasksKeyIdx = text.indexOf('"tasks"');
  if(tasksKeyIdx === -1){
    console.error('No "tasks" key found');
    process.exit(1);
  }
  const bracketStart = text.indexOf('[', tasksKeyIdx);
  if(bracketStart === -1){
    console.error('No tasks array start');
    process.exit(1);
  }
  const len = text.length;
  let i = bracketStart + 1;
  skipped = 0;
  while(i < len){
    while(i < len && /[\s,]/.test(text[i])) i++;
    if(i >= len) break;
    if(text[i] === ']') break;
    if(text[i] === '{'){
      let depth = 0; let inString = false; let esc = false; let start = i; let bufArr = [];
      for(; i < len; i++){
        const ch = text[i];
        if(inString){
          if(esc){ bufArr.push(ch); esc = false; continue; }
          if(ch === '\\'){ bufArr.push(ch); esc = true; continue; }
          if(ch === '"'){ inString = false; bufArr.push(ch); continue; }
          if(ch === '\n'){ bufArr.push('\\n'); continue; }
          if(ch === '\r'){ bufArr.push('\\r'); continue; }
          const code = ch.charCodeAt(0);
          if(code < 0x20){ bufArr.push('\\u'+code.toString(16).padStart(4,'0')); continue; }
          bufArr.push(ch);
        } else {
          if(ch === '"'){ inString = true; bufArr.push(ch); continue; }
          if(ch === '{'){ depth++; bufArr.push(ch); continue; }
          if(ch === '}'){ depth--; bufArr.push(ch); if(depth === 0){ i++; break; } continue; }
          bufArr.push(ch);
        }
      }
      const buf = bufArr.join('');
      try { const obj = JSON.parse(buf); tasks.push(obj); }
      catch(e){
        skipped++;
        console.warn('Skipped unparsable task fragment starting at char', start, e.message);
        try {
          const debugDir = path.join(outDir,'_skipped');
          if(!fs.existsSync(debugDir)) fs.mkdirSync(debugDir,{recursive:true});
          fs.writeFileSync(path.join(debugDir, `fragment-${start}.jsonc`), buf);
        } catch(writeErr){ console.warn('Failed to write debug fragment', writeErr.message); }
      }
    } else {
      i++;
    }
  }
  const m = text.match(/"version"\s*:\s*"([^"]+)"/);
  if(m) version = m[1];
  if(skipped) console.warn(`Manual parse summary: parsed ${tasks.length}, skipped ${skipped}`);
}

const buckets = { ai: [], autosolve: [], db: [], backend: [], frontend: [], docs: [], health: [], misc: [] };
function classify(label){
  const l = (label||'').toLowerCase();
  if(l.includes('gemma')||l.includes('ollama')||l.includes('vllm')) return 'ai';
  if(l.includes('autosolve')) return 'autosolve';
  if(l.includes('pgvector')||l.includes('redis')||l.includes('minio')) return 'db';
  if(l.startsWith('go')||l.includes('grpc')||l.includes('nats')||l.includes('quic')||l.includes('elk')||l.includes('kratos')||l.includes('cluster')||l.includes('summarizer')||l.includes('nvidia')||l.includes('rag')) return 'backend';
  if(l.includes('svelte')||l.includes('frontend')||l.includes('vite')) return 'frontend';
  if(l.startsWith('docs')||l.includes('readme')||l.includes('best practices')) return 'docs';
  if(l.includes('health')||l.includes('status')||l.includes('check')) return 'health';
  return 'misc';
}
for(const t of tasks){ buckets[classify(t.label)].push(t); }
if(!fs.existsSync(outDir)) fs.mkdirSync(outDir,{recursive:true});
for(const [name,list] of Object.entries(buckets)){
  const fragment = { version, tasks: list };
  const fp = path.join(outDir, `tasks-${name}.json`);
  fs.writeFileSync(fp, JSON.stringify(fragment,null,2));
  console.log(`Wrote ${list.length} tasks → ${fp}`);
}
fs.writeFileSync(path.join(outDir,'manifest.json'), JSON.stringify({ generated: new Date().toISOString(), counts: Object.fromEntries(Object.entries(buckets).map(([k,v])=>[k,v.length])), totalParsed: tasks.length, skipped }, null, 2));
console.log(`Split complete: ${tasks.length} tasks parsed, ${skipped} skipped.`);
