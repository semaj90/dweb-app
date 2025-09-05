import cluster from 'node:cluster';
import os from 'node:os';
import process from 'node:process';
import { createServer } from './server.js';

const isPrimary = cluster.isPrimary;
const CPU_COUNT = Math.max(1, Math.min(os.cpus()?.length || 1, parseInt(process.env.WORKERS || '2', 10)));

if (isPrimary) {
  console.log(`[gateway] primary ${process.pid} starting with ${CPU_COUNT} workers`);
  for (let i = 0; i < CPU_COUNT; i++) cluster.fork();

  cluster.on('exit', (worker, code, signal) => {
    console.warn(`[gateway] worker ${worker.process.pid} exited (${code||''} ${signal||''}), forking...`);
    cluster.fork();
  });
} else {
  createServer();
}
