#!/usr/bin/env node
// Vector Search Integration Setup for Legal AI System

import { readFile, writeFile } from 'fs/promises';
import { join } from 'path';

const vectorIntegrationScript = `
// src/lib/server/vector/setup-vector-integration.ts
import { vectorService } from './vectorService.js';
import { db } from '../db/index.js';
import { cases, evidence, criminals } from '../db/schema-postgres-enhanced.js';

export async function setupVectorIntegration() {
  console.log('🚀 Setting up vector search integration...');
  
  try {
    // 1. Initialize Qdrant collection
    await vectorService.initializeCollection();
    console.log('✅ Qdrant collection initialized');
    
    // 2. Check health of all services
    const health = await vectorService.healthCheck();
    console.log('🔍 Service health:', health);
    
    if (!health.qdrant || !health.redis) {
      throw new Error('Vector services not ready. Please start Docker containers.');
    }
    
    // 3. Sync existing data to vector database
    await syncExistingData();
    console.log('✅ Existing data synced to vector database');
    
    // 4. Test embeddings generation
    await testEmbeddings();
    console.log('✅ Embeddings test passed');
    
    console.log('🎉 Vector integration setup complete!');
    
  } catch (error) {
    console.error('❌ Vector integration setup failed:', error);
    throw error;
  }
}

async function syncExistingData() {
  // Sync cases
  const allCases = await db.select().from(cases);
  for (const caseItem of allCases) {
    const content = \`\${caseItem.title} \${caseItem.description} \${caseItem.category}\`;
    await vectorService.storeDocument(caseItem.id, content, {
      type: 'case',
      case_id: caseItem.id,
      title: caseItem.title,
      created_at: caseItem.createdAt.toISOString()
    });
  }
  console.log(\`📁 Synced \${allCases.length} cases\`);
  
  // Sync evidence
  const allEvidence = await db.select().from(evidence);
  for (const evidenceItem of allEvidence) {
    const content = \`\${evidenceItem.title} \${evidenceItem.description || ''} \${evidenceItem.summary || ''}\`;
    await vectorService.storeDocument(evidenceItem.id, content, {
      type: 'evidence',
      case_id: evidenceItem.caseId,
      title: evidenceItem.title,
      created_at: evidenceItem.createdAt.toISOString()
    });
  }
  console.log(\`🔍 Synced \${allEvidence.length} evidence items\`);
  
  // Sync criminals
  const allCriminals = await db.select().from(criminals);
  for (const criminal of allCriminals) {
    const content = \`\${criminal.firstName} \${criminal.lastName} \${criminal.notes || ''}\`;
    await vectorService.storeDocument(criminal.id, content, {
      type: 'criminal',
      title: \`\${criminal.firstName} \${criminal.lastName}\`,
      created_at: criminal.createdAt.toISOString()
    });
  }
  console.log(\`👤 Synced \${allCriminals.length} criminal profiles\`);
}

async function testEmbeddings() {
  const testText = "Legal case involving evidence analysis and prosecution strategy";
  const embedding = await vectorService.generateEmbedding(testText);
  
  if (!embedding || embedding.length !== 768) {
    throw new Error('Invalid embedding generated');
  }
  
  console.log(\`🧠 Generated embedding with \${embedding.length} dimensions\`);
}

if (import.meta.url === \`file://\${process.argv[1]}\`) {
  setupVectorIntegration().catch(console.error);
}
`;

const apiEndpoint = `
// src/routes/api/search/semantic/+server.ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { vectorService } from '$lib/server/vector/vectorService.js';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options = {} } = await request.json();
    
    if (!query || typeof query !== 'string') {
      throw error(400, 'Query parameter is required');
    }
    
    // Perform hybrid vector + keyword search
    const results = await vectorService.hybridSearch(query, {
      limit: options.limit || 10,
      threshold: options.threshold || 0.7,
      filter: options.filter || {},
      keywordWeight: 0.3,
      vectorWeight: 0.7
    });
    
    return json({
      success: true,
      query,
      results,
      count: results.length,
      timestamp: new Date().toISOString()
    });
    
  } catch (err) {
    console.error('Semantic search error:', err);
    throw error(500, \`Search failed: \${err instanceof Error ? err.message : 'Unknown error'}\`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const query = url.searchParams.get('q');
  const limit = parseInt(url.searchParams.get('limit') || '10');
  
  if (!query) {
    return json({
      message: 'Semantic search endpoint',
      usage: 'POST with { query: string, options?: { limit, threshold, filter } }',
      example: '/api/search/semantic?q=witness+statement&limit=5'
    });
  }
  
  try {
    const results = await vectorService.hybridSearch(query, { limit });
    return json({ success: true, query, results, count: results.length });
  } catch (err) {
    throw error(500, \`Search failed: \${err instanceof Error ? err.message : 'Unknown error'}\`);
  }
};
`;

const enhancedBatchScript = `
@echo off
echo ========================================
echo ENHANCED VECTOR SEARCH SETUP
echo ========================================

echo.
echo Checking Docker services...
docker ps --filter "name=postgres" --filter "name=redis" --filter "name=qdrant" --format "table {{.Names}}\\t{{.Status}}"

echo.
echo Testing Ollama nomic-embed model...
curl -s http://localhost:11434/api/tags | findstr "nomic-embed"

echo.
echo Setting up vector integration...
cd sveltekit-frontend
node src/lib/server/vector/setup-vector-integration.ts

echo.
echo Testing vector search API...
curl -X POST http://localhost:5173/api/search/semantic -H "Content-Type: application/json" -d "{\\"query\\":\\"legal case evidence\\"}"

echo.
echo ========================================
echo VECTOR SEARCH SETUP COMPLETE
echo ========================================
echo.
echo Available endpoints:
echo - POST /api/search/semantic
echo - GET  /api/vector/search  
echo - POST /api/embeddings/generate
echo.
echo Test with: curl -X POST http://localhost:5173/api/search/semantic -H "Content-Type: application/json" -d "{\\"query\\":\\"witness statement\\"}"
echo.
`;

  console.log('📝 Writing vector integration files...');
  
  await writeFile('setup-vector-integration.ts', vectorIntegrationScript);
  await writeFile('semantic-search-endpoint.ts', apiEndpoint);
  await writeFile('test-vector-setup.bat', enhancedBatchScript);
  
  console.log('✅ Vector integration files created:');
  console.log('- setup-vector-integration.ts');
  console.log('- semantic-search-endpoint.ts'); 
  console.log('- test-vector-setup.bat');
}