#!/usr/bin/env node

import fetch from 'node-fetch';

const services = [
  { name: 'PostgreSQL', url: 'http://localhost:5432', description: 'Database' },
  { name: 'Ollama', url: 'http://localhost:11434/api/version', description: 'AI Service' },
  { name: 'Qdrant', url: 'http://localhost:6333/health', description: 'Vector Database' },
  { name: 'Neo4j', url: 'http://localhost:7474', description: 'Graph Database' },
  { name: 'RabbitMQ', url: 'http://localhost:15672', description: 'Message Queue' },
  { name: 'Redis', url: 'http://localhost:6379', description: 'Cache' },
  { name: 'SvelteKit', url: 'http://localhost:5173', description: 'Frontend' }
];

async function checkService(service) {
  try {
    const response = await fetch(service.url, { 
      timeout: 5000,
      method: 'GET'
    });
    return { ...service, status: 'UP', statusCode: response.status };
  } catch (error) {
    return { ...service, status: 'DOWN', error: error.message };
  }
}

async function checkAllServices() {
  console.log('🔍 Legal AI Stack Health Check');
  console.log('================================\n');

  const results = await Promise.all(services.map(checkService));
  
  results.forEach(result => {
    const status = result.status === 'UP' ? '✅' : '❌';
    const statusText = result.status === 'UP' ? 'UP' : 'DOWN';
    
    console.log(`${status} ${result.name.padEnd(12)} | ${statusText.padEnd(5)} | ${result.description}`);
    
    if (result.error) {
      console.log(`   └─ Error: ${result.error}`);
    }
  });

  const upCount = results.filter(r => r.status === 'UP').length;
  const totalCount = results.length;

  console.log(`\n📊 Summary: ${upCount}/${totalCount} services running`);
  
  if (upCount === totalCount) {
    console.log('🎉 All services are healthy!');
    process.exit(0);
  } else {
    console.log('⚠️  Some services are down. Check logs with: npm run docker:logs');
    process.exit(1);
  }
}

checkAllServices();
