#!/usr/bin/env node
/**
 * Quick Legal AI System Status Check
 * Fast overview of running services
 */

const services = [
  { name: 'Frontend (SvelteKit)', port: 5183, url: 'http://localhost:5183' },
  { name: 'PostgreSQL Database', port: 5432, test: 'PGPASSWORD=123456 psql -U postgres -h localhost -d legal_ai_db -c "SELECT 1;" 2>/dev/null' },
  { name: 'Ollama AI Service', port: 11434, url: 'http://localhost:11434/api/tags' },
  { name: 'Enhanced RAG Service', port: 8094, url: 'http://localhost:8094/api/health' },
  { name: 'Upload Service', port: 8093, url: 'http://localhost:8093/health' },
  { name: 'MinIO Storage', port: 9000, url: 'http://localhost:9000/minio/health/live' },
  { name: 'Qdrant Vector DB', port: 6333, url: 'http://localhost:6333/collections' },
  { name: 'Neo4j Graph DB', port: 7474, url: 'http://localhost:7474/browser' }
];

const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

async function checkUrl(url) {
  try {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 3000);
    
    const response = await fetch(url, { 
      signal: controller.signal,
      headers: { 'User-Agent': 'Legal-AI-Health-Check' }
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function checkCommand(cmd) {
  return new Promise(async (resolve) => {
    try {
      const { spawn } = await import('child_process');
      const process = spawn.spawn('bash', ['-c', cmd], { stdio: 'ignore' });
      process.on('close', (code) => resolve(code === 0));
      setTimeout(() => { process.kill(); resolve(false); }, 5000);
    } catch (error) {
      resolve(false);
    }
  });
}

async function checkServices() {
  console.log(`${colors.bold}${colors.blue}🔍 Legal AI System Quick Status${colors.reset}\n`);
  
  let running = 0;
  const total = services.length;
  
  for (const service of services) {
    let isRunning = false;
    
    if (service.url) {
      isRunning = await checkUrl(service.url);
    } else if (service.test) {
      isRunning = await checkCommand(service.test);
    }
    
    const status = isRunning ? 
      `${colors.green}✅ RUNNING${colors.reset}` : 
      `${colors.red}❌ OFFLINE${colors.reset}`;
    
    console.log(`${service.name.padEnd(25)} (${service.port}) ${status}`);
    
    if (isRunning) running++;
  }
  
  console.log(`\n${colors.bold}System Status: ${running}/${total} services running${colors.reset}`);
  
  const percentage = Math.round((running / total) * 100);
  let healthStatus;
  
  if (percentage >= 80) {
    healthStatus = `${colors.green}EXCELLENT${colors.reset}`;
  } else if (percentage >= 60) {
    healthStatus = `${colors.yellow}GOOD${colors.reset}`;
  } else {
    healthStatus = `${colors.red}NEEDS ATTENTION${colors.reset}`;
  }
  
  console.log(`Overall Health: ${healthStatus} (${percentage}%)\n`);
  
  // Show access URLs for running services
  console.log(`${colors.cyan}Access Points:${colors.reset}`);
  if (running > 0) {
    console.log(`🌐 Frontend: http://localhost:5183`);
    if (await checkUrl('http://localhost:9001')) {
      console.log(`📊 MinIO Console: http://localhost:9001`);
    }
    if (await checkUrl('http://localhost:7474')) {
      console.log(`🗄️  Neo4j Browser: http://localhost:7474`);
    }
  }
  
  console.log(`\n${colors.yellow}💡 To start missing services:${colors.reset}`);
  console.log(`   npm run dev:full         # Start all services`);
  console.log(`   START-LEGAL-AI.bat       # Windows batch startup`);
}

checkServices().catch(console.error);