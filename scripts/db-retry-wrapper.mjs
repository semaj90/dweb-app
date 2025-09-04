#!/usr/bin/env node
// Simple retry wrapper for initial Postgres connectivity.
// Usage: node scripts/db-retry-wrapper.mjs "node some-script.js"
import { spawn } from 'node:child_process';
import process from 'node:process';

const MAX_ATTEMPTS = parseInt(process.env.DB_RETRY_ATTEMPTS || '12', 10); // ~60s if 5s interval
const DELAY_MS = parseInt(process.env.DB_RETRY_DELAY_MS || '5000', 10);
const HEALTH_QUERY = process.env.DB_HEALTH_QUERY || 'SELECT 1';
const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/legal_ai_db';

import pg from 'pg';
const { Client } = pg;

async function wait(ms){ return new Promise(r=>setTimeout(r, ms)); }

async function tryConnect(attempt){
  const client = new Client({ connectionString });
  try {
    await client.connect();
    await client.query(HEALTH_QUERY);
    await client.end();
    console.log(`[db-retry] success on attempt ${attempt}`);
    return true;
  } catch (e){
    console.log(`[db-retry] attempt ${attempt} failed: ${e.message}`);
    try { await client.end(); } catch {}
    return false;
  }
}

async function main(){
  for (let a=1; a<=MAX_ATTEMPTS; a++){
    if (await tryConnect(a)) break;
    if (a === MAX_ATTEMPTS){
      console.error('[db-retry] exhausted attempts');
      process.exit(1);
    }
    await wait(DELAY_MS);
  }
  const cmd = process.argv.slice(2);
  if (cmd.length){
    console.log('[db-retry] launching command:', cmd.join(' '));
    const child = spawn(cmd[0], cmd.slice(1), { stdio: 'inherit', shell: false });
    child.on('exit', code => process.exit(code ?? 0));
  }
}
main();
// (removed duplicate placeholder block)
