#!/usr/bin/env node
/**
 * Periodic log maintenance loop.
 * Runs the existing log-maintenance.mjs script at an interval.
 * Intended for dev/CI environments where a lightweight scheduler is desired.
 */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const INTERVAL_SEC = parseInt(process.env.LOG_MAINTENANCE_INTERVAL_SEC || '300', 10); // default 5m
const WARN_MB = process.env.LOG_WARN_MB ? ['--warn', process.env.LOG_WARN_MB] : [];
const MAX_MB = process.env.LOG_MAX_MB ? ['--max', process.env.LOG_MAX_MB] : [];
const HISTORY = process.env.LOG_HISTORY ? ['--history', process.env.LOG_HISTORY] : [];
const COMPRESS = process.env.LOG_COMPRESS === 'true' ? ['--compress'] : [];

function runOnce() {
  const scriptPath = resolve(__dirname, 'log-maintenance.mjs');
  const args = [scriptPath, ...WARN_MB, ...MAX_MB, ...HISTORY, ...COMPRESS];
  const child = spawn(process.execPath, args, { stdio: 'inherit' });
  child.on('exit', (code) => {
    if (code !== 0) {
      console.error(`[log-maintenance-loop] run exited with code ${code}`);
    }
  });
}

console.log(`[log-maintenance-loop] Starting; interval=${INTERVAL_SEC}s`);
runOnce();
setInterval(runOnce, INTERVAL_SEC * 1000);
