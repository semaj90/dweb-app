#!/usr/bin/env zx

import { $ } from 'zx';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const argv = yargs(hideBin(process.argv)).argv;

async function start() {
  console.log('Starting Legal AI Platform Services...');

  await $`net start postgresql-x64-17`.catch(() => console.log('PostgreSQL already running'));
  console.log('PostgreSQL started');

  await $`start /min redis-server`.catch(() => $`start /min .\redis-windows\redis-server.exe`);
  console.log('Redis started');

  await $`tasklist | findstr \"ollama\"
`.catch(() => $`start /min ollama serve`);
  console.log('Ollama started');

  if (!await $.fs.exists('./minio-data')) {
    await $.fs.mkdir('./minio-data');
  }
  await $`tasklist | findstr \"minio\"
`.catch(() => $`start /min minio.exe server ./minio-data --address :9000 --console-address :9001`);
  console.log('MinIO started');

  await $`tasklist | findstr \"qdrant\"
`.catch(() => $`start /min .\qdrant-windows\qdrant.exe`);
  console.log('Qdrant Vector Database started');

  await $`powershell -Command "Start-Service neo4j"`.catch(() => console.log('Neo4j manual start required'));
  console.log('Neo4j started');

  cd('./go-services/cmd/enhanced-rag');
  await $`start /min cmd /c "go run main.go"`;
  cd('../../../');
  console.log('Go Enhanced RAG Service started');

  cd('./go-microservice');
  await $`start /min cmd /c "go run main.go"`;
  cd('..');
  console.log('Go Upload Service started');

  cd('./go-services/cmd/xstate-manager');
  await $`start /min cmd /c "go run main.go"`;
  cd('../../../');
  console.log('Go XState Manager started');

  cd('./sveltekit-frontend');
  await $`start cmd /k "npm run dev -- --host 0.0.0.0"`;
  cd('..');
  console.log('SvelteKit Frontend started');

  await $.sleep(8000);

  console.log('\nLEGAL AI PLATFORM STARTED SUCCESSFULLY!');
}

async function stop() {
  console.log('Stopping Legal AI Platform Services...');
  await $`taskkill /F /IM node.exe /T`.catch(() => {});
  await $`taskkill /F /IM go.exe /T`.catch(() => {});
  await $`taskkill /F /IM ollama.exe /T`.catch(() => {});
  await $`taskkill /F /IM minio.exe /T`.catch(() => {});
  await $`taskkill /F /IM qdrant.exe /T`.catch(() => {});
  await $`taskkill /F /IM redis-server.exe /T`.catch(() => {});
  console.log('All services stopped');
}

async function status() {
  console.log('Legal AI Platform Service Status');

  const services = [
    { name: 'PostgreSQL', port: 5432 },
    { name: 'Redis', port: 6379 },
    { name: 'Ollama', port: 11434 },
    { name: 'MinIO', port: 9000 },
    { name: 'Qdrant', port: 6333 },
    { name: 'Neo4j', port: 7474 },
    { name: 'SvelteKit', port: 5173 },
    { name: 'Enhanced RAG', port: 8094 },
    { name: 'Upload Service', port: 8093 },
  ];

  for (const service of services) {
    try {
      await $`netstat -ano | findstr :${service.port}`;
      console.log(`[OK] ${service.name}: Running`);
    } catch (e) {
      console.log(`[ERR] ${service.name}: Not running`);
    }
  }
}

async function install() {
  console.log('Installing Legal AI Platform Dependencies...');
  await $`choco install redis-64 -y`;
  await $`choco install nodejs -y`;
  await $`choco install golang -y`;
  await $`choco install postgresql -y`;

  cd('./sveltekit-frontend');
  await $`npm install`;
  cd('..');

  cd('./go-services');
  await $`go mod tidy`;
  cd('..');

  console.log('Dependencies installed');
}

async function test() {
  console.log('Testing Legal AI Platform Integration...');
  try {
    const result = await fetch('http://localhost:11434/api/tags');
    const data = await result.json();
    console.log(`[OK] Ollama API responsive with ${data.models.length} models`);
  } catch (e) {
    console.log('[ERR] Ollama API not responding');
  }

  try {
    await $`redis-cli ping`;
    console.log('[OK] Redis connection successful');
  } catch (e) {
    console.log('[ERR] Redis connection failed');
  }
}

if (argv.start) {
  await start();
} else if (argv.stop) {
  await stop();
} else if (argv.status) {
  await status();
} else if (argv.install) {
  await install();
} else if (argv.test) {
  await test();
} else {
  console.log('Usage:');
  console.log('  ./scripts/wire-up.mjs --start    # Start all services');
  console.log('  ./scripts/wire-up.mjs --stop     # Stop all services');
  console.log('  ./scripts/wire-up.mjs --status   # Check service status');
  console.log('  ./scripts/wire-up.mjs --install  # Install dependencies');
  console.log('  ./scripts/wire-up.mjs --test     # Test integration');
}