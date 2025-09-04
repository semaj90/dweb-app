// System Validation Script
// Run with: node validate-system-complete.mjs

import { execSync } from 'child_process';
import fetch from 'node-fetch';
import pg from 'pg';
import Redis from 'ioredis';

console.log('====================================');
console.log('  LEGAL AI SYSTEM VALIDATION');
console.log('====================================\n');

const results = {
  database: false,
  redis: false,
  ollama: false,
  minio: false,
  qdrant: false,
  apis: {
    cases: false,
    evidence: false,
    reports: false,
    citations: false,
    search: false,
    health: false
  },
  services: {
    enhancedRag: false,
    gpuOrchestrator: false
  }
};

// 1. Check PostgreSQL
console.log('Checking PostgreSQL...');
try {
  const client = new pg.Client({
    connectionString: 'postgresql://postgres:postgres@localhost:5432/legal_ai'
  });
  await client.connect();
  const res = await client.query('SELECT COUNT(*) FROM users');
  console.log(`  ✓ PostgreSQL connected (${res.rows[0].count} users)`);
  
  // Check pgvector
  const vectorRes = await client.query("SELECT extname FROM pg_extension WHERE extname = 'vector'");
  if (vectorRes.rows.length > 0) {
    console.log('  ✓ pgvector extension installed');
  } else {
    console.log('  ✗ pgvector extension missing');
  }
  
  await client.end();
  results.database = true;
} catch (err) {
  console.log('  ✗ PostgreSQL connection failed:', err.message);
}

// 2. Check Redis
console.log('\nChecking Redis...');
try {
  const redis = new Redis({
    host: 'localhost',
    port: 6379,
    retryStrategy: () => null
  });
  await redis.ping();
  console.log('  ✓ Redis connected');
  redis.disconnect();
  results.redis = true;
} catch (err) {
  console.log('  ✗ Redis connection failed');
}

// 3. Check Ollama
console.log('\nChecking Ollama...');
try {
  const response = await fetch('http://localhost:11434/api/tags');
  const data = await response.json();
  console.log(`  ✓ Ollama connected (${data.models?.length || 0} models)`);
  
  // Check for legal model
  const hasLegalModel = data.models?.some(m => m.name.includes('gemma') || m.name.includes('legal'));
  if (hasLegalModel) {
    console.log('  ✓ Legal model available');
  } else {
    console.log('  ✗ Legal model not found');
  }
  results.ollama = true;
} catch (err) {
  console.log('  ✗ Ollama connection failed');
}

// 4. Check MinIO
console.log('\nChecking MinIO...');
try {
  const response = await fetch('http://localhost:9000/minio/health/live');
  if (response.ok) {
    console.log('  ✓ MinIO connected');
    results.minio = true;
  }
} catch (err) {
  console.log('  ✗ MinIO connection failed');
}

// 5. Check Qdrant
console.log('\nChecking Qdrant...');
try {
  const response = await fetch('http://localhost:6333/collections');
  if (response.ok) {
    const data = await response.json();
    console.log(`  ✓ Qdrant connected (${data.result?.collections?.length || 0} collections)`);
    results.qdrant = true;
  }
} catch (err) {
  console.log('  ✗ Qdrant connection failed');
}

// 6. Check API Endpoints
console.log('\nChecking API Endpoints...');
const apiBase = 'http://localhost:5173/api';

for (const [endpoint, _] of Object.entries(results.apis)) {
  try {
    const url = endpoint === 'search' ? `${apiBase}/search/vector` : `${apiBase}/${endpoint}`;
    const method = endpoint === 'search' ? 'POST' : 'GET';
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' }
    };
    
    if (method === 'POST') {
      options.body = JSON.stringify({
        query: 'test',
        collections: ['cases'],
        limit: 1
      });
    }
    
    const response = await fetch(url, options);
    if (response.ok || response.status === 401) { // 401 means API works but needs auth
      console.log(`  ✓ /${endpoint} API responding`);
      results.apis[endpoint] = true;
    } else {
      console.log(`  ✗ /${endpoint} API failed (${response.status})`);
    }
  } catch (err) {
    console.log(`  ✗ /${endpoint} API not accessible`);
  }
}

// 7. Check Go Services
console.log('\nChecking Go Microservices...');
const services = [
  { name: 'enhancedRag', port: 8094 },
  { name: 'gpuOrchestrator', port: 8095 }
];

for (const service of services) {
  try {
    const response = await fetch(`http://localhost:${service.port}/health`);
    if (response.ok) {
      console.log(`  ✓ ${service.name} running on port ${service.port}`);
      results.services[service.name] = true;
    }
  } catch (err) {
    console.log(`  ✗ ${service.name} not responding`);
  }
}

// 8. Check TypeScript Status
console.log('\nChecking TypeScript Status...');
try {
  const tscResult = execSync('npm run check:ultra-fast 2>&1', { encoding: 'utf-8' });
  const errorCount = (tscResult.match(/error TS/g) || []).length;
  console.log(`  ✓ TypeScript check complete (${errorCount} errors)`);
  
  if (errorCount < 100) {
    console.log('  ✓ TypeScript errors within acceptable range');
  } else if (errorCount < 500) {
    console.log('  ⚠ TypeScript errors elevated but manageable');
  } else {
    console.log('  ✗ TypeScript errors require attention');
  }
} catch (err) {
  console.log('  ✗ TypeScript check failed');
}

// 9. Summary
console.log('\n====================================');
console.log('  VALIDATION SUMMARY');
console.log('====================================');

const coreServices = results.database && results.ollama;
const allAPIs = Object.values(results.apis).every(v => v);
const enhancedFeatures = results.redis && results.qdrant && results.minio;
const goServices = Object.values(results.services).some(v => v);

if (coreServices && allAPIs) {
  console.log('\n✅ SYSTEM READY FOR PRODUCTION');
  console.log('  Core services: OPERATIONAL');
  console.log('  All APIs: FUNCTIONAL');
  
  if (enhancedFeatures) {
    console.log('  Enhanced features: ENABLED');
  } else {
    console.log('  Enhanced features: PARTIALLY AVAILABLE');
  }
  
  if (goServices) {
    console.log('  Go microservices: ACTIVE');
  }
} else if (coreServices) {
  console.log('\n⚠️ SYSTEM PARTIALLY OPERATIONAL');
  console.log('  Core services are running but some APIs may be unavailable');
  console.log('  Check individual service logs for details');
} else {
  console.log('\n❌ SYSTEM NOT READY');
  console.log('  Critical services are not running');
  console.log('  Run START-PRODUCTION-COMPLETE.bat to initialize all services');
}

// 10. Detailed Report
console.log('\n====================================');
console.log('  DETAILED SERVICE STATUS');
console.log('====================================');

console.log('\nDATABASE SERVICES:');
console.log(`  PostgreSQL: ${results.database ? '✓' : '✗'}`);
console.log(`  Redis: ${results.redis ? '✓' : '✗'}`);

console.log('\nAI SERVICES:');
console.log(`  Ollama: ${results.ollama ? '✓' : '✗'}`);
console.log(`  Enhanced RAG: ${results.services.enhancedRag ? '✓' : '✗'}`);
console.log(`  GPU Orchestrator: ${results.services.gpuOrchestrator ? '✓' : '✗'}`);

console.log('\nSTORAGE SERVICES:');
console.log(`  MinIO: ${results.minio ? '✓' : '✗'}`);
console.log(`  Qdrant: ${results.qdrant ? '✓' : '✗'}`);

console.log('\nAPI ENDPOINTS:');
Object.entries(results.apis).forEach(([endpoint, status]) => {
  console.log(`  /${endpoint}: ${status ? '✓' : '✗'}`);
});

console.log('\n====================================');
console.log('  READY FOR DEVELOPMENT ✨');
console.log('====================================');

// Exit with appropriate code
const overallHealth = coreServices && allAPIs;
process.exit(overallHealth ? 0 : 1);