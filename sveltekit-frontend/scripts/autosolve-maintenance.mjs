#!/usr/bin/env node
/* Autosolve Maintenance Cycle
 * 1. Run ultra-fast TS check
 * 2. If errors > threshold env AUTOSOLVE_THRESHOLD (default 5), trigger force_cycle
 * 3. Fetch status & health, append concise log line to .vscode/autosolve-maintenance.log
 */
import { execSync } from 'node:child_process';
import fs from 'node:fs';

const threshold = Number(process.env.AUTOSOLVE_THRESHOLD || 5);
let errors = 0;
let tsOutput = '';
try {
  tsOutput = execSync('npx tsc --noEmit --skipLibCheck', { stdio: 'pipe' }).toString();
} catch (e){
  // tsc exits non-zero on errors but we still capture output
  tsOutput = e.stdout?.toString() || '';
}
const match = tsOutput.match(/Found (\d+) error/);
if(match) errors = Number(match[1]);
let triggered = false;
let cycleResult = null;
if(errors > threshold){
  try {
  cycleResult = execSync("curl -s -X POST -H 'Content-Type: application/json' -d '{\"action\":\"force_cycle\"}' http://localhost:5173/api/context7-autosolve", { stdio: 'pipe' }).toString();
    triggered = true;
  } catch {}
}
let status = '';
try { status = execSync("curl -s 'http://localhost:5173/api/context7-autosolve?action=status'").toString(); } catch {}
let health = '';
try { health = execSync("curl -s 'http://localhost:5173/api/context7-autosolve?action=health'").toString(); } catch {}
const line = JSON.stringify({ timestamp: new Date().toISOString(), errors, threshold, triggered, cycleResult: cycleResult? '[captured]' : null });
if(!fs.existsSync('.vscode')) fs.mkdirSync('.vscode');
fs.appendFileSync('.vscode/autosolve-maintenance.log', line + '\n');
console.log(`Maintenance cycle complete. Errors=${errors} threshold=${threshold} triggered=${triggered}`);
