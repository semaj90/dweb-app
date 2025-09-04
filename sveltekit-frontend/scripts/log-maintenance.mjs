#!/usr/bin/env node
/**
 * Log rotation & size guard utility.
 *
 * Strategy:
 *  - Scan ./logs for *.log and *.jsonl files.
 *  - If file size > MAX_SIZE_MB, rotate: file -> file.YYYYMMDD-HHMMSS
 *  - Keep at most MAX_HISTORY rotated siblings per base (oldest pruned).
 *  - Warn (no rotation) when size > WARN_SIZE_MB but < MAX_SIZE_MB.
 *  - Optional gzip compression for rotated files (disabled by default for speed on Windows).
 *
 * Flags:
 *   --dry-run          : Show actions only.
 *   --compress         : Gzip rotated files.
 *   --warn <mb>        : Override warning threshold.
 *   --max <mb>         : Override rotation size threshold.
 *   --history <n>      : Override history count.
 */
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';

const LOG_DIR = path.join(process.cwd(), 'logs');
const args = process.argv.slice(2);
function getArg(flag, def) {
  const idx = args.indexOf(flag);
  if (idx === -1) return def;
  const next = args[idx + 1];
  if (!next || next.startsWith('--')) return true;
  return next;
}

let WARN_SIZE_MB = parseInt(getArg('--warn', '10'), 10);      // warn if exceeded
let MAX_SIZE_MB = parseInt(getArg('--max', '20'), 10);        // rotate if exceeded
let MAX_HISTORY = parseInt(getArg('--history', '5'), 10);     // per base file
const DRY_RUN = Boolean(getArg('--dry-run', false));
const COMPRESS = Boolean(getArg('--compress', false));

if (!fs.existsSync(LOG_DIR)) {
  console.log('[log-maintenance] No logs directory, nothing to do.');
  process.exit(0);
}

function human(bytes) { return (bytes / (1024 * 1024)).toFixed(2) + 'MB'; }

function rotate(file) {
  const ts = new Date().toISOString().replace(/[-:T]/g,'').slice(0, 13); // YYYYMMDDHHMM
  const dir = path.dirname(file);
  const base = path.basename(file);
  const rotated = path.join(dir, `${base}.${ts}`);
  if (DRY_RUN) {
    console.log(`[log-maintenance] ROTATE ${base} -> ${path.basename(rotated)}`);
    return;
  }
  fs.renameSync(file, rotated);
  if (COMPRESS) {
    const gzPath = rotated + '.gz';
    const input = fs.createReadStream(rotated);
    const output = fs.createWriteStream(gzPath);
    const gzip = zlib.createGzip();
    input.pipe(gzip).pipe(output).on('finish', () => {
      fs.unlinkSync(rotated);
      console.log(`[log-maintenance] Compressed ${path.basename(gzPath)}`);
    });
  }
  console.log(`[log-maintenance] Rotated ${base}`);
}

function pruneHistory(fileBase) {
  const dir = path.dirname(fileBase);
  const base = path.basename(fileBase);
  const siblings = fs.readdirSync(dir)
    .filter(f => f.startsWith(base + '.'))
    .sort(); // chronological due to timestamp prefix style
  const excess = siblings.length - MAX_HISTORY;
  if (excess > 0) {
    for (let i = 0; i < excess; i++) {
      const target = path.join(dir, siblings[i]);
      if (DRY_RUN) {
        console.log(`[log-maintenance] PRUNE ${siblings[i]}`);
      } else {
        fs.unlinkSync(target);
        console.log(`[log-maintenance] Pruned ${siblings[i]}`);
      }
    }
  }
}

const files = fs.readdirSync(LOG_DIR).filter(f => /(\.log|\.jsonl)$/i.test(f));
if (files.length === 0) {
  console.log('[log-maintenance] No log files to process.');
  process.exit(0);
}

for (const f of files) {
  const full = path.join(LOG_DIR, f);
  try {
    const stat = fs.statSync(full);
    const sizeMb = stat.size / (1024 * 1024);
    if (sizeMb >= MAX_SIZE_MB) {
      rotate(full);
      pruneHistory(full);
    } else if (sizeMb >= WARN_SIZE_MB) {
      console.warn(`[log-maintenance] WARNING: ${f} size ${human(stat.size)} exceeds ${WARN_SIZE_MB}MB threshold`);
    }
  } catch (e) {
    console.error('[log-maintenance] Error processing', f, e.message);
  }
}

console.log('[log-maintenance] Done.');
