#!/usr/bin/env node

import { spawn } from 'child_process'

const EXCLUDES = [
  /\bsrc[\\/]+routes[\\/]+demo[\\/]+/i,
  /\bsrc[\\/]+routes[\\/]+dev[\\/]+/i,
  /\bsrc[\\/]+routes[\\/]+showcase/i,
  /\bsrc[\\/]+routes[\\/]+gaming-demo/i,
  /\bsrc[\\/]+routes[\\/]+test/i
]

function isExcluded(file) {
  if (!file) return false
  return EXCLUDES.some((re) => re.test(file))
}

function run(cmd, args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { shell: true })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (d) => (stdout += d.toString()))
    child.stderr.on('data', (d) => (stderr += d.toString()))
    child.on('error', reject)
    child.on('close', (code) => resolve({ code, stdout, stderr }))
  })
}

(async () => {
  console.log('🔎 Running svelte-check (filtered) ...')
  const include = (process.env.INCLUDE_GLOBS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)

  const args = ['svelte-check', '--output', 'machine']
  if (include.length) {
    console.log('Including globs:', include.join(', '))
    args.push(...include)
  }
  const result = await run('npx', args)

  // Try to parse machine output (one JSON per line). Fallback to human parsing.
  const lines = result.stdout.split(/\r?\n/).filter(Boolean)
  const entries = []
  for (const line of lines) {
    try {
      const obj = JSON.parse(line)
      entries.push(obj)
    } catch {
      // best-effort parse for human lines: "path:line:col level message"
      const m = line.match(/^(.*?):(\d+):(\d+):\s+(warning|error):\s+(.*)$/i)
      if (m) {
        entries.push({ file: m[1], level: m[4].toLowerCase(), message: m[5] })
      }
    }
  }

  const kept = []
  const ignored = []
  for (const e of entries) {
    const file = e.file || e.filename || e.path || ''
    if (isExcluded(file)) ignored.push(e)
    else kept.push(e)
  }

  const keptErrors = kept.filter((e) => (e.level || e.severity || '').toString().toLowerCase() === 'error')
  console.log(`Total diagnostics: ${entries.length} | kept: ${kept.length} | ignored: ${ignored.length}`)
  if (keptErrors.length > 0) {
    console.error(`❌ Errors after filtering: ${keptErrors.length}`)
    // echo a few examples
    for (const e of keptErrors.slice(0, 10)) {
      console.error('-', (e.file || e.filename || ''), '-', e.message || e.text || '')
    }
    process.exit(1)
  } else {
    console.log('✅ No blocking errors after filtering (demo/dev excluded).')
    process.exit(0)
  }
})().catch((err) => {
  console.error('check-filtered failed:', err.message)
  process.exit(1)
})
