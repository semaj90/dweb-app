#!/usr/bin/env node
// Composite check + autosolve + delta summary
import { execSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

function run(cmd){
  console.log(`\n▶ ${cmd}`)
  try { return execSync(cmd,{stdio:'pipe'}).toString() } catch(e){ return e.stdout?.toString()||'' }
}

function captureErrors(tag){
  const out = run('npx tsc --noEmit --skipLibCheck')
  const lines = out.split(/\r?\n/)
  const errorLines = lines.filter(l=>/error TS\d+:/.test(l))
  return { tag, count: errorLines.length }
}

const start = Date.now()
const baseline = captureErrors('before')
console.log(`Baseline TypeScript errors: ${baseline.count}`)

// Run autosolve scripts if threshold exceeded
const THRESHOLD = parseInt(process.env.AUTOSOLVE_THRESHOLD||'100')
if(baseline.count > THRESHOLD){
  console.log(`Errors > ${THRESHOLD}. Running autosolve agents...`)
  run('npm run autosolve:ai-synthesis || true')
  if(fs.existsSync('scripts/autosolve-svelte5-bitsui.mjs')){
    run('npm run autosolve:svelte5-bitsui || true')
  }
} else {
  console.log(`Errors (=${baseline.count}) <= threshold (${THRESHOLD}); skipping autosolve.`)
}

const after = captureErrors('after')
const delta = baseline.count - after.count
console.log(`\nΔ Error reduction: ${delta} ( ${baseline.count} -> ${after.count} )`)

const stamp = new Date().toISOString().replace(/[:T]/g,'-').slice(0,19)
const reportDir = '.vscode'
if(!fs.existsSync(reportDir)) fs.mkdirSync(reportDir,{recursive:true})
const mdName = `AUTOSOLVE-${stamp}.md`
const mdPath = path.join(reportDir, mdName)
const minutes = ((Date.now()-start)/60000).toFixed(2)
const md = `# Autosolve Delta Report\n\nTimestamp: ${new Date().toISOString()}\nRuntime: ${minutes} min\n\n| Phase | Errors |\n|-------|--------|\n| Before | ${baseline.count} |\n| After | ${after.count} |\n| Reduction | ${delta} |\n\nThreshold: ${THRESHOLD}\nSkipped Autosolve: ${baseline.count <= THRESHOLD}\n`
fs.writeFileSync(mdPath, md)
console.log(`Report written: ${mdPath}`)

// Simple JSON summary for tooling
const json = { timestamp: new Date().toISOString(), baseline: baseline.count, after: after.count, reduction: delta, threshold: THRESHOLD }
fs.writeFileSync(path.join(reportDir,'autosolve-latest.json'), JSON.stringify(json,null,2))
console.log('JSON summary: .vscode/autosolve-latest.json')

process.exit(0)
