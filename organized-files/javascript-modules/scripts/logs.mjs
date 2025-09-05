#!/usr/bin/env node

import chalk from 'chalk';
import fs from 'node:fs';
import path from 'node:path';
import 'zx/globals';

const log = {
  info: (m) => console.log(chalk.blue('ℹ'), m),
  ok: (m) => console.log(chalk.green('✓'), m),
  warn: (m) => console.log(chalk.yellow('⚠'), m),
  err: (m) => console.log(chalk.red('✗'), m),
};

function tailFile(file, lines = 100) {
  try {
    const data = fs.readFileSync(file, 'utf8').split(/\r?\n/);
    const out = data.slice(-lines).join('\n');
    console.log(chalk.gray(`\n— ${file} (last ${lines} lines) —`));
    console.log(out);
  } catch (e) {
    log.warn(`No readable log at ${file}`);
  }
}

async function main() {
  console.log(chalk.cyan.bold('📜 YoRHa Legal AI — Aggregated Logs'));

  // Go microservice logs
  const goDir = path.join(process.cwd(), 'go-microservice');
  if (fs.existsSync(goDir)) {
    const files = fs.readdirSync(goDir)
      .filter(f => f.endsWith('.log'))
      .map(f => ({ f, t: fs.statSync(path.join(goDir, f)).mtimeMs }))
      .sort((a, b) => b.t - a.t);
    if (files.length) {
      tailFile(path.join(goDir, files[0].f), 150);
    } else {
      log.warn('No .log files found for Go service');
    }
  }

  // SvelteKit recent dev output (vite)
  const viteDir = path.join(process.cwd(), '.vite');
  if (fs.existsSync(viteDir)) {
    const entries = fs.readdirSync(viteDir).filter(f => f.endsWith('.log'));
    for (const f of entries) tailFile(path.join(viteDir, f), 100);
  }

  // Service ports summary
  log.info('Active service ports (netstat snapshot):');
  try {
    const res = await $`netstat -ano | findstr LISTENING | findstr :5173 :3000 :8080 :50051 :11434 :6333 :5432 :6379`;
    console.log(res.stdout);
  } catch {}

  // Optional Redis INFO snapshot
  if (fs.existsSync(path.join(process.cwd(), 'redis-windows', 'redis-cli.exe'))) {
    try {
      const info = await $`echo INFO memory | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
      console.log(chalk.gray('\n— Redis memory —'));
      console.log(info.stdout);
    } catch {}
  }

  log.ok('Done.');
}

main().catch(e => {
  log.err(e.message);
  process.exit(1);
});
