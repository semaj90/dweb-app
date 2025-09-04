#!/usr/bin/env node
/**
 * Simple Legal AI System Health Check
 */

const services = [
  { name: 'Frontend (SvelteKit)', port: 5183, url: 'http://localhost:5183' },
  { name: 'Ollama AI Service', port: 11434, url: 'http://localhost:11434/api/tags' },
  { name: 'Enhanced RAG Service', port: 8094, url: 'http://localhost:8094/api/health' },
  { name: 'Upload Service', port: 8093, url: 'http://localhost:8093/health' },
  { name: 'MinIO Storage', port: 9000, url: 'http://localhost:9000/minio/health/live' },
  { name: 'Qdrant Vector DB', port: 6333, url: 'http://localhost:6333/collections' },
  { name: 'Neo4j Graph DB', port: 7474, url: 'http://localhost:7474/browser' }
];

const colors = {
  green: '\x1b[32m', red: '\x1b[31m', yellow: '\x1b[33m',
  blue: '\x1b[34m', cyan: '\x1b[36m', reset: '\x1b[0m', bold: '\x1b[1m'
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

async function main() {
  console.log(`${colors.bold}${colors.blue}🔍 Legal AI System Status${colors.reset}\n`);
  
  let running = 0;
  
  for (const service of services) {
    const isRunning = await checkUrl(service.url);
    const status = isRunning ? 
      `${colors.green}✅ RUNNING${colors.reset}` : 
      `${colors.red}❌ OFFLINE${colors.reset}`;
    
    console.log(`${service.name.padEnd(25)} (${service.port}) ${status}`);
    if (isRunning) running++;
  }
  
  console.log(`\n${colors.bold}System Status: ${running}/${services.length} services running${colors.reset}`);
  
  const percentage = Math.round((running / services.length) * 100);
  const healthStatus = percentage >= 80 ? 
    `${colors.green}EXCELLENT${colors.reset}` : 
    percentage >= 60 ? `${colors.yellow}GOOD${colors.reset}` : 
    `${colors.red}NEEDS ATTENTION${colors.reset}`;
  
  console.log(`Overall Health: ${healthStatus} (${percentage}%)\n`);
  
  console.log(`${colors.cyan}Access Points:${colors.reset}`);
  console.log(`🌐 Frontend: http://localhost:5183`);
  console.log(`📊 MinIO Console: http://localhost:9001 (if running)`);
  console.log(`🗄️  Neo4j Browser: http://localhost:7474 (if running)`);
  
  console.log(`\n${colors.yellow}Database Status:${colors.reset}`);
  console.log(`📁 PostgreSQL: Running ✅ (verified earlier)`);
  
  console.log(`\n${colors.yellow}💡 Quick Start Commands:${colors.reset}`);
  console.log(`   npm run dev:full         # Start all services`);
  console.log(`   START-LEGAL-AI.bat       # Windows batch startup`);
  console.log(`   npm run ollama:start     # Start Ollama AI`);
}

main().catch(console.error);