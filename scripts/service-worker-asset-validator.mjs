#!/usr/bin/env node
// Validate that assets referenced in service worker precache exist.
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const SW_CANDIDATES = [
  'sveltekit-frontend/src/service-worker.js',
  'sveltekit-frontend/src/service-worker.ts'
];

function extractAssets(code){
  const assets = new Set();
  const block = code.match(/PRECACHE[^=]*=\s*\[(.*?)\]/s);
  if (block){
    for (const m of block[1].matchAll(/['"]([^'"?]+)['"]/g)) assets.add(m[1]);
  }
  return [...assets];
}

function assetExists(rel){
  const full = resolve('sveltekit-frontend/static', rel);
  return existsSync(full);
}

let code='', swFile=null;
for (const c of SW_CANDIDATES){ if (existsSync(c)){ swFile=c; code=readFileSync(c,'utf8'); break; } }
if (!swFile){ console.warn('No service worker found'); process.exit(0); }

const assets = extractAssets(code);
const report = assets.map(a=>({ asset:a, exists: assetExists(a) }));
const missing = report.filter(r=>!r.exists);
console.log(JSON.stringify({ ts: new Date().toISOString(), swFile, total: assets.length, missing: missing.length, report }, null, 2));
if (missing.length) process.exitCode=1;
