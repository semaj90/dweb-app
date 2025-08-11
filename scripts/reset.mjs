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

async function removeDirSafe(dir) {
  try {
    if (fs.existsSync(dir)) {
      await fs.promises.rm(dir, { recursive: true, force: true });
      log.ok(`Removed ${dir}`);
    }
  } catch (e) {
    log.warn(`Failed to remove ${dir}: ${e.message}`);
  }
}

async function main() {
  log.info('Resetting local build artifacts and temp state...');

  // Frontend artifacts
  await removeDirSafe(path.join(process.cwd(), 'sveltekit-frontend', '.svelte-kit'));
  await removeDirSafe(path.join(process.cwd(), 'sveltekit-frontend', 'build'));
  await removeDirSafe(path.join(process.cwd(), 'sveltekit-frontend', 'node_modules', '.cache'));

  // Root caches
  await removeDirSafe(path.join(process.cwd(), '.vite'));

  // Redis dump (optional)
  const dump = path.join(process.cwd(), 'dump.rdb');
  if (fs.existsSync(dump)) {
    try { await fs.promises.unlink(dump); log.ok('Removed Redis dump.rdb'); } catch {}
  }

  // Qdrant storage (optional)
  await removeDirSafe(path.join(process.cwd(), 'qdrant_storage'));

  log.ok('Reset complete.');
}

main().catch(e => { log.err(e.message); process.exit(1); });
