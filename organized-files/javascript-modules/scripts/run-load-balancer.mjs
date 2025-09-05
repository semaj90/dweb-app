#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

// Simple runner for Go load balancer using build tag `loadbalancer`.
// Falls back to `go run` if binary not built, otherwise reuses existing binary.
// Environment variables honored:
//   LB_PORT, LOAD_BALANCER_STRATEGY, UPSTREAM_SERVICES, HEALTH_CHECK_INTERVAL,
//   QUARANTINE_BASE, ENABLE_PROMETHEUS, LB_ADMIN_TOKEN

const root = process.cwd();
const goDir = path.join(root, 'go-microservice');
const binDir = path.join(goDir, 'bin');
const binPath = path.join(binDir, 'load-balancer.exe');

const env = { ...process.env };
if (!env.LB_PORT) env.LB_PORT = '8099';
if (!env.LOAD_BALANCER_STRATEGY) env.LOAD_BALANCER_STRATEGY = 'gpu_aware';
if (!env.UPSTREAM_SERVICES) env.UPSTREAM_SERVICES = 'http://localhost:8094,http://localhost:8095';
if (!env.HEALTH_CHECK_INTERVAL) env.HEALTH_CHECK_INTERVAL = '30s';
if (!env.QUARANTINE_BASE) env.QUARANTINE_BASE = '30s';
if (!env.ENABLE_PROMETHEUS) env.ENABLE_PROMETHEUS = 'true';

async function buildIfNeeded() {
  if (existsSync(binPath)) return binPath;
  await run('go', ['build', '-tags', 'loadbalancer', '-o', binPath, './go-microservice/load-balancer.go']);
  return binPath;
}

function run(cmd, args, opts={}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit', ...opts });
    p.on('exit', code => code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`)) );
  });
}

const start = async () => {
  try {
    const binary = await buildIfNeeded();
    await gateUpstreams();
    console.log(`▶️ Starting load balancer (${binary}) on :${env.LB_PORT} strategy=${env.LOAD_BALANCER_STRATEGY}`);
    const proc = spawn(binary, [], { env, stdio: 'inherit' });
    proc.on('exit', code => console.log(`⏹ Load balancer exited with code ${code}`));
  } catch (e) {
    console.error('Failed to start load balancer:', e.message);
    process.exit(1);
  }
};

async function gateUpstreams() {
  const services = env.UPSTREAM_SERVICES.split(',').map(s => s.trim()).filter(Boolean);
  if (!services.length) return;
  const timeoutMs = parseInt(env.LB_READY_TIMEOUT_MS || '60000', 10);
  const start = Date.now();
  const interval = 2000;
  console.log(`⏳ Health gate: waiting for ${services.length} upstream(s) (timeout ${timeoutMs}ms)...`);
  const healthy = new Set();
  while (Date.now() - start < timeoutMs) {
    await Promise.all(services.map(async (u) => {
      if (healthy.has(u)) return;
      try {
        const url = u.replace(/\/$/, '') + '/health';
        const res = await fetch(url, { method: 'GET' });
        if (res.ok) healthy.add(u);
      } catch (_) { /* ignore */ }
    }));
    const statusLine = services.map(s => healthy.has(s) ? '✅' : '…').join(' ');
    process.stdout.write(`\r   Upstreams: ${statusLine}`);
    if (healthy.size === services.length) {
      console.log(`\n✅ All upstreams healthy (${healthy.size}/${services.length})`);
      return;
    }
    await new Promise(r => setTimeout(r, interval));
  }
  console.log(`\n⚠️ Proceeding with partial readiness: ${healthy.size}/${services.length} healthy after ${timeoutMs}ms`);
}

start();
