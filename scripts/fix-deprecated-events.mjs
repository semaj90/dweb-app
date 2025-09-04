#!/usr/bin/env node
// Scan repo for Svelte old event directives on:click etc and produce a report + optional in-place fix.
import { promises as fs } from 'fs';
import path from 'path';

const ROOT = process.cwd();
const exts = ['.svelte'];
const oldEvents = ['on:click','on:change','on:submit','on:keydown','on:keyup','on:focus','on:blur'];
const newMap = Object.fromEntries(oldEvents.map(e=>[e, e.replace('on:','on')]));
const doWrite = process.argv.includes('--write');
let totalFiles=0, modified=0;

async function walk(dir){
  const entries = await fs.readdir(dir,{withFileTypes:true});
  for(const ent of entries){
    if(ent.name.startsWith('.') || ent.name==='node_modules') continue;
    const full = path.join(dir, ent.name);
    if(ent.isDirectory()) await walk(full); else if(exts.some(ext=>ent.name.endsWith(ext))) await processFile(full);
  }
}

async function processFile(file){
  let txt = await fs.readFile(file,'utf8');
  let changed = false;
  for(const ev of oldEvents){
    if(txt.includes(ev)){
      const regex = new RegExp(ev, 'g');
      txt = txt.replace(regex, newMap[ev]);
      changed = true;
    }
  }
  if(changed){
    totalFiles++;
    if(doWrite){
      await fs.writeFile(file, txt,'utf8');
      modified++;
    }
  }
}

(async()=>{
  await walk(ROOT);
  console.log(`Scan complete. Files needing change: ${totalFiles}. ${doWrite? 'Modified:'+modified: 'Run with --write to apply fixes.'}`);
})();
