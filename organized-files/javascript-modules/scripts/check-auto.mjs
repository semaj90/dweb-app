#!/usr/bin/env node
/**
 * Unified automated check + log + recommendation trigger
 * Steps:
 * 1. Run full frontend checks (TS + Svelte + ESLint) capturing JSON & text
 * 2. If errors, write consolidated log file with timestamp
 * 3. Optionally call recommendation endpoint to process new log
 * 4. Output concise summary & next actions
 */
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

const LOG_DIR = path.resolve('error-logs');
const RECOMMENDATION_ENDPOINT = process.env.RECOMMENDATION_ENDPOINT || 'http://localhost:8099/api/process-error-log';
const TRIGGER_RECOMMENDATIONS = process.env.TRIGGER_RECOMMENDATIONS !== 'false';
const ADV_AST = process.env.ADV_AST_FIX === 'true';

async function ensureDir(dir) { await fs.mkdir(dir, { recursive: true }); }

function run(cmd, args, opts={}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { shell: true, ...opts });
    let stdout='', stderr='';
    child.stdout.on('data', d => stdout += d.toString());
    child.stderr.on('data', d => stderr += d.toString());
    child.on('close', code => resolve({ code, stdout, stderr }));
  });
}

async function main() {
  const start = Date.now();
  await ensureDir(LOG_DIR);
  console.log('🔍 Running comprehensive checks (auto) ...');

  // 1. Run full check (delegates to frontend package)
  const check = await run('npm', ['run','check:full']);
  const hadErrors = /error/i.test(check.stdout) || /error/i.test(check.stderr) || check.code !== 0;

  // 2. If errors: attempt targeted auto-fixes (Svelte5 compliance + eslint --fix)
  let logFile = null;
  if (hadErrors) {
    console.log('🛠  Attempting targeted auto-fix passes (Svelte5 compliance + ESLint)...');
    // Svelte5 compliance with autofix
    await run('npm', ['--prefix','sveltekit-frontend','run','validate:svelte5','--','--fix']);
    // ESLint fix
    await run('npm', ['--prefix','sveltekit-frontend','run','lint:fix']);
    if (ADV_AST) {
      console.log('🧠 Running advanced AST fix pass...');
      await run('node', ['scripts/advanced-ast-fix.mjs']);
    }
    // Re-run fast check after fixes
    const postFix = await run('npm', ['run','check:fast']);
    const stillErrors = /error/i.test(postFix.stdout) || /error/i.test(postFix.stderr) || postFix.code !== 0;
    const ts = new Date().toISOString().replace(/[:.]/g,'-');
    logFile = path.join(LOG_DIR, `auto-check-${ts}.log`);
    await fs.writeFile(logFile, check.stdout + '\n--- STDERR ---\n' + check.stderr + '\n--- AFTER AUTOFIX RE-CHECK ---\n' + postFix.stdout + '\n' + postFix.stderr);
    if (!stillErrors) {
      console.log('✅ Auto-fix passes cleared errors');
    } else {
      console.log('⚠️  Some errors remain after auto-fix');
    }
    console.log(`📄 Log saved: ${logFile}`);
  }

  // 3. Trigger recommendations if requested
  if (hadErrors && TRIGGER_RECOMMENDATIONS) {
    try {
      const res = await fetch(RECOMMENDATION_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          logFile: logFile && path.basename(logFile),
          content: (check.stdout + '\n' + check.stderr).slice(0, 500000),
          timestamp: new Date().toISOString()
        })
      });
      if (res.ok) {
        const data = await res.json();
        console.log(`💡 Recommendations: ${(data.recommendations||[]).length}`);
      } else {
        console.log(`⚠️ Recommendation service HTTP ${res.status}`);
      }
    } catch (e) {
      console.log('⚠️ Recommendation trigger failed:', e.message);
    }
  }

  const duration = Date.now() - start;
  if (!hadErrors) {
    console.log(`✅ All checks passed in ${duration}ms`);
  } else {
    console.log(`❌ Checks found issues (code=${check.code}) in ${duration}ms`);
    console.log('Next: review log, auto-fix suggestions will appear if daemon running');
  }
  process.exit(hadErrors ? 1 : 0);
}

main();
