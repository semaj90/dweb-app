#!/usr/bin/env node

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const services = [
  { name: 'MinIO', url: 'http://localhost:9000', description: 'Object Storage' },
  { name: 'Redis', url: 'redis://localhost:6379', description: 'Cache Server', command: 'redis-cli ping' },
  { name: 'PostgreSQL', url: 'postgresql://localhost:5432', description: 'Database', command: 'pg_isready -h localhost -p 5432' },
  { name: 'Neo4j', url: 'http://localhost:7474', description: 'Graph Database' },
  { name: 'Ollama', url: 'http://localhost:11434', description: 'AI Models' },
];

console.log('🔍 Checking External Services Status...\n');

async function checkService(service) {
  try {
    if (service.command) {
      const { stdout } = await execAsync(service.command);
      return { ...service, status: '✅ Running', details: stdout.trim() };
    } else {
      const response = await fetch(service.url, { method: 'HEAD', timeout: 3000 });
      return { ...service, status: response.ok ? '✅ Running' : '❌ Not Responding' };
    }
  } catch (error) {
    return { ...service, status: '❌ Not Running', error: error.message };
  }
}

async function checkAllServices() {
  const results = await Promise.allSettled(services.map(checkService));
  
  console.log('📊 Service Status Report:');
  console.log('=' .repeat(50));
  
  results.forEach((result, index) => {
    const service = result.status === 'fulfilled' ? result.value : services[index];
    const status = result.status === 'fulfilled' ? service.status : '❌ Check Failed';
    
    console.log(`${status.padEnd(15)} ${service.name.padEnd(12)} ${service.description}`);
    if (service.error && !service.error.includes('fetch')) {
      console.log(`                   ⚠️  ${service.error}`);
    }
  });
  
  console.log('\n🚀 To start services:');
  console.log('   MinIO:      minio.exe server ./data');
  console.log('   Redis:      redis-server');
  console.log('   PostgreSQL: pg_ctl start (or service postgresql start)');
  console.log('   Neo4j:      neo4j console');
  console.log('   Ollama:     ollama serve');
}

checkAllServices().catch(console.error);