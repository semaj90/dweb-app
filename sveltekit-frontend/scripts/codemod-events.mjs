#!/usr/bin/env node
/**
 * SAFER Event Codemod (v2)
 * - Only transforms DOM elements (tag starts with lowercase letter)
 * - Leaves component/custom events (capitalized tags) untouched
 * - Only converts whitelisted DOM events (native)
 * - Skips files containing @codemod-skip-events marker
 *
 * Regression from v1: converting component events like <DropdownMenuItem onclick=> introduced type errors.
 * This version prevents that by requiring a preceding '<tag' where tag[0] is lowercase.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const glob = require('glob');

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const LOG_DIR = path.join(projectRoot, 'logs');
if(!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR,{recursive:true});

// Native DOM event allow list
const DOM_EVENTS = [
  'click','submit','keydown','keyup','keypress','change','input','dragover','dragleave','drop','dragenter','dragstart','dragend','mousedown','mouseup','mousemove','mouseover','mouseout','focus','blur'
];
const eventPattern = new RegExp(`(<[a-z][\\w:-]*[^>]*?)\\bon:(${DOM_EVENTS.join('|')})(=)`, 'g');

function transform(content){
  let changed = false;
  const updated = content.replace(eventPattern,(full, before, ev, eq)=>{
    changed = true;
    return `${before}on${ev}${eq}`; // onclick=
  });
  return { updated, changed };
}

const files = glob.sync('src/**/*.svelte', { cwd: projectRoot, nodir: true, absolute: true });
let changedFiles = [];

for(const abs of files){
  let txt = fs.readFileSync(abs,'utf8');
  if(!txt.includes('on:') || txt.includes('@codemod-skip-events')) continue;
  const { updated, changed } = transform(txt);
  if(changed){
    fs.writeFileSync(abs, updated,'utf8');
    changedFiles.push(abs.replace(projectRoot+path.sep,''));
  }
}

const log = { timestamp:new Date().toISOString(), changed: changedFiles.length, changedFiles };
fs.writeFileSync(path.join(LOG_DIR,'codemod-events.json'), JSON.stringify(log,null,2));
console.log(`⚙️  Event codemod (safe) complete. Updated ${changedFiles.length} files.`);
if(changedFiles.length){
  console.log(changedFiles.slice(0,12).map(f=>' - '+f).join('\n'));
  if(changedFiles.length>12) console.log(` ... (+${changedFiles.length-12} more)`);
}
