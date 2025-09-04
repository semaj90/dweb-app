#!/usr/bin/env node
/**
 * Evidence System - Service Detection
 * Quickly checks availability & versions of required external services.
 */
import { $ } from 'zx';
import os from 'os';
import fs from 'fs';
import chalk from 'chalk';
import boxen from 'boxen';
import { createSpinner } from 'nanospinner';
import pLimit from 'p-limit';
import pRetry from 'p-retry';

const SERVICES = [
  { name: 'PostgreSQL',  test: async () => await fetchJson('http://localhost:5173/api/health', 'postgres') },
  { name: 'Qdrant',      test: async () => await fetchJson('http://localhost:6333/health') },
  { name: 'Ollama',      test: async () => await fetchJson('http://localhost:11434/api/tags') },
  { name: 'Redis',       test: async () => await tcpCheck(6379) },
  { name: 'MinIO',       test: async () => await tcpCheck(9000) },
  { name: 'RabbitMQ',    test: async () => await tcpCheck(5672) },
  { name: 'OCR Service', test: async () => await fetchText('http://localhost:8601/health') },
];

const limit = pLimit(4);

async function fetchJson(url, key) {
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) throw new Error(`${res.status}`);
  const data = await res.json().catch(()=>({}));
  return key ? data[key] || 'ok' : 'ok';
}
async function fetchText(url){
  const res = await fetch(url);
  if(!res.ok) throw new Error(`${res.status}`);
  return 'ok';
}
async function tcpCheck(port){
  // lightweight port probe using PowerShell / bash depending on platform
  if (os.platform() === 'win32') {
    const { exitCode } = await $`powershell -Command "(New-Object Net.Sockets.TcpClient).Connect('localhost',${port})"`.nothrow();
    if(exitCode!==0) throw new Error('unreachable');
  } else {
    const { exitCode } = await $`bash -lc "</dev/tcp/127.0.0.1/${port}"`.nothrow();
    if(exitCode!==0) throw new Error('unreachable');
  }
  return 'ok';
}

async function main(){
  console.log(boxen(chalk.cyan.bold('Evidence System - Service Detection'), { padding: 1, borderColor: 'cyan' }));
  const spinner = createSpinner('Checking services...').start();
  const results = [];
  await Promise.all(SERVICES.map(s => limit(async () => {
    const start = Date.now();
    try {
      const val = await pRetry(() => s.test(), { retries: 2 });
      results.push({ name: s.name, status: 'UP', latency: Date.now()-start });
    } catch (e) {
      results.push({ name: s.name, status: 'DOWN', error: e.message });
    }
  })));
  spinner.success({ text: 'Scan complete' });
  const up = results.filter(r=>r.status==='UP').length;
  const down = results.length - up;
  for(const r of results){
    const line = r.status==='UP'
      ? `${chalk.green('✔')} ${chalk.white(r.name)} ${chalk.gray(`${r.latency}ms`)}`
      : `${chalk.red('✖')} ${chalk.white(r.name)} ${chalk.red(r.error||'error')}`;
    console.log(line);
  }
  console.log('\nSummary:', chalk.green(`${up} up`), '/', chalk.red(`${down} down`));
  const missing = results.filter(r=>r.status==='DOWN').map(r=>r.name);
  if(missing.length){
    console.log(chalk.yellow('\nMissing services:'), missing.join(', '));
    process.exitCode = 1;
  }
}

main().catch(e=>{ console.error(chalk.red('Fatal:'), e); process.exit(1); });
