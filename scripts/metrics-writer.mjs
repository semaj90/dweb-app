#!/usr/bin/env node
// Write NATS health & metrics periodically to logs/nats-metrics.log
import { mkdirSync, appendFileSync, statSync, renameSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { renderMetrics, healthSnapshot } from '../microservices/node-api/src/lib/services/nats-messaging-service.js';

const INTERVAL_MS = parseInt(process.env.METRICS_INTERVAL_MS || '5000',10);
const MAX_SIZE = 512*1024; // 512KB rotation
const FILE = resolve('logs/nats-metrics.log');

function rotateIfNeeded(){
  try { const st = statSync(FILE); if (st.size > MAX_SIZE){ const rotated = FILE + '.' + Date.now(); renameSync(FILE, rotated); } } catch {}
}
function writeOnce(){
  rotateIfNeeded();
  const line = JSON.stringify({ ts: new Date().toISOString(), health: healthSnapshot(), metrics: renderMetrics() })+'\n';
  appendFileSync(FILE, line, 'utf8');
  process.stdout.write('.');
}
mkdirSync(dirname(FILE), { recursive: true });
writeOnce();
if (INTERVAL_MS>0){ setInterval(writeOnce, INTERVAL_MS); }
