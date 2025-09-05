#!/usr/bin/env node
/**
 * Revert erroneous onclick/onchange/etc on Svelte component tags (capitalized) back to directive form on:click
 * Pattern: <CapitalTag ... onclick= -> <CapitalTag ... on:click=
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const glob = require('glob');

const projectRoot = process.cwd();
const files = glob.sync('src/**/*.svelte',{cwd:projectRoot, absolute:true});
const EVENTS = ['click','change','keydown','keyup','submit','input'];
const compEventRegex = new RegExp(`(<[A-Z][\\w:-]*[^>]*?)\\bon(${EVENTS.join('|')})=`,`g`);

let touched=0, list=[];
for(const f of files){
  let txt = fs.readFileSync(f,'utf8');
  if(!/on(click|change|keydown|keyup|submit|input)=/.test(txt)) continue;
  const updated = txt.replace(compEventRegex,(full,before,ev)=>`${before}on:${ev}=`);
  if(updated!==txt){
    fs.writeFileSync(f,updated,'utf8');
    touched++; list.push(f.replace(projectRoot+path.sep,''));
  }
}
console.log(`♻️  Reverted component native event attributes in ${touched} files.`);
if(list.length) console.log(list.slice(0,15).map(f=>' - '+f).join('\n'));
