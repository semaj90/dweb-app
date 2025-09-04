// System Validation Script
// Run with: node validate-system.mjs

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

// 1. Check PostgreSQL (using your existing legal_ai_db)
console.log('Checking PostgreSQL...');
try {
  const client = new pg.Client({
    connectionString: 'postgresql://postgres:postgres@localhost:5432/legal_ai_db'
  });
  await client.connect();
  
  // Check if tables exist
  const tablesRes = await client.query(`
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
  `);
  console.log(`  ✓ PostgreSQL connected (${tablesRes.rows.length} tables)`);
  
  // Check pgvector
  const vectorRes = await client.query("SELECT extname FROM pg_extension WHERE extname = 'vector'");
  if (vectorRes.rows.length > 0) {
    console.log('  ✓ pgvector extension installed');
  } else {
    console.log('  ✗ pgvector extension missing');
  }
  
  // Check if users table exists
  const userRes = await client.query(`
    SELECT COUNT(*) as count FROM information_schema.tables 
    WHERE table_name = 'users' AND table_schema = 'public'
  `);
  
  if (userRes.rows[0].count > 0) {
    const userCount = await client.query('SELECT COUNT(*) FROM users');
    console.log(`  ✓ Users table exists (${userCount.rows[0].count} users)`);
  } else {
    console.log('  ⚠ Users table not found - migration needed');
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
  const hasLegalModel = data.models?.some(m => 
    m.name.includes('gemma3:legal') || 
    m.name.includes('gemma3-legal')
  );
  if (hasLegalModel) {
    console.log('  ✓ Gemma3:legal model available');
  } else {
    console.log('  ✗ Gemma3:legal model not found - run: ollama pull gemma3:legal');
  }
  
  // Check for embedding model
  const hasEmbedModel = data.models?.some(m => 
    m.name.includes('nomic-embed-text')
  );
  if (hasEmbedModel) {
    console.log('  ✓ nomic-embed-text model available');
  } else {
    console.log('  ✗ nomic-embed-text not found - run: ollama pull nomic-embed-text');
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

// 8. Summary
console.log('\n====================================');
console.log('  VALIDATION SUMMARY');
console.log('====================================');

const coreServices = results.database && results.ollama;
const allAPIs = Object.values(results.apis).every(v => v);
const enhancedFeatures = results.redis && results.qdrant && results.minio;

if (coreServices && allAPIs) {
  console.log('\n✅ SYSTEM READY FOR PRODUCTION');
  console.log('  Core services: OPERATIONAL');
  console.log('  All APIs: FUNCTIONAL');
  
  if (enhancedFeatures) {
    console.log('  Enhanced features: ENABLED');
  } else {
    console.log('  Enhanced features: PARTIALLY AVAILABLE');
  }
} else if (coreServices) {
  console.log('\n⚠️ SYSTEM PARTIALLY OPERATIONAL');
  console.log('  Core services are running but some APIs may be offline');
} else {
  console.log('\n❌ SYSTEM NOT READY');
  console.log('  Core services need to be started');
}

console.log('\nNext steps:');
if (!results.database) {
  console.log('  1. Start PostgreSQL and ensure legal_ai_db is accessible');
}
if (!results.ollama) {
  console.log('  2. Start Ollama: ollama serve');
  console.log('     Install models: ollama pull gemma3:legal');
  console.log('                     ollama pull nomic-embed-text');
}
if (!allAPIs) {
  console.log('  3. Start the application: npm run dev');
}
if (!enhancedFeatures) {
  console.log('  4. Start optional services:');
  console.log('     - Redis: redis-server --port 6379');
  console.log('     - Qdrant: qdrant');
  console.log('     - MinIO: minio server ./minio-data --console-address :9001');
}

console.log('\nRun: npm run test:validate again after starting services');
