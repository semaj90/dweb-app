#!/usr/bin/env node
// Cluster manager launcher with auto port resolution, graceful shutdown of existing instance,
// and support for passing worker counts via CLI flags or environment variables.

import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

async function tryShutdownExisting(port){
  try {
    const res = await fetch(`http://localhost:${port}/shutdown`, { method:'POST' });
    if(res.ok){
      console.log(`[cluster:manager] Requested shutdown of existing manager on port ${port}`);
      await new Promise(r=>setTimeout(r,1500));
    }
  } catch { /* ignore */ }
}

const args = process.argv.slice(2);

// Resolve counts (environment overrides or defaults)
const managerPortBase = parseInt(process.env.MANAGER_PORT || '3000', 10);
process.env.MANAGER_PORT = String(managerPortBase);
process.env.MANAGER_PORT_AUTO = process.env.MANAGER_PORT_AUTO || '1';

const legal = process.env.LEGAL_COUNT || '1';
const ai = process.env.AI_COUNT || '1';
const vector = process.env.VECTOR_COUNT || '1';
const database = process.env.DATABASE_COUNT || '1';

const clusterPath = path.resolve('node-cluster/cluster-manager.cjs');

const finalArgs = [
  `--manager-port=${process.env.MANAGER_PORT}`,
  `--legal-count=${legal}`,
  `--ai-count=${ai}`,
  `--vector-count=${vector}`,
  `--database-count=${database}`,
  ...args
];

(async () => {
  await tryShutdownExisting(managerPortBase);
  console.log('[cluster:manager] starting', { managerPort: process.env.MANAGER_PORT, legal, ai, vector, database });
  const child = spawn(process.execPath, [clusterPath, ...finalArgs], { stdio: 'inherit', env: process.env });
  child.on('exit', (code) => {
    console.log('[cluster:manager] exited', code);
    process.exit(code ?? 0);
  });
})();
