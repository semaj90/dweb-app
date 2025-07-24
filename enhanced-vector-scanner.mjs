#!/usr/bin/env node
// Enhanced Vector Setup with Directory Scanning and Merging

import { readdir, readFile, writeFile, mkdir } from 'fs/promises';
import { join, basename } from 'path';

const projectRoot = process.cwd();

async function scanForVectorFiles() {
  console.log('🔍 Scanning for existing vector/search files...');
  
  const patterns = [
    'vector', 'search', 'embedding', 'qdrant', 'redis', 
    'ollama', 'chat', 'semantic', 'embedding'
  ];
  
  const foundFiles = [];
  
  async function scanDir(dir, depth = 0) {
    if (depth > 4) return; // Limit recursion
    
    try {
      const entries = await readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = join(dir, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
          await scanDir(fullPath, depth + 1);
        } else if (entry.isFile() && (entry.name.endsWith('.ts') || entry.name.endsWith('.js'))) {
          // Check if filename contains vector-related terms
          const fileName = entry.name.toLowerCase();
          if (patterns.some(pattern => fileName.includes(pattern))) {
            foundFiles.push(fullPath);
          }
        }
      }
    } catch (err) {
      // Ignore permission errors
    }
  }
  
  await scanDir(projectRoot);
  return foundFiles;
}

async function analyzeVectorFiles(files) {
  const analysis = {
    hasOllama: false,
    hasQdrant: false,
    hasRedis: false,
    hasPgVector: false,
    hasEmbeddings: false,
    apiEndpoints: [],
    configFiles: [],
    services: []
  };
  
  for (const file of files) {
    try {
      const content = await readFile(file, 'utf-8');
      
      if (content.includes('ollama') || content.includes('Ollama')) analysis.hasOllama = true;
      if (content.includes('qdrant') || content.includes('Qdrant')) analysis.hasQdrant = true;
      if (content.includes('redis') || content.includes('Redis')) analysis.hasRedis = true;
      if (content.includes('pgvector') || content.includes('vector(')) analysis.hasPgVector = true;
      if (content.includes('embedding') || content.includes('nomic-embed')) analysis.hasEmbeddings = true;
      
      if (file.includes('routes/api')) analysis.apiEndpoints.push(file);
      if (file.includes('config') || file.includes('.config.')) analysis.configFiles.push(file);
      if (file.includes('service') || file.includes('Service')) analysis.services.push(file);
      
    } catch (err) {
      console.warn(`Cannot read ${file}: ${err.message}`);
    }
  }
  
  return analysis;
}

async function generateMergedVectorService(analysis, existingFiles) {
  const timestamp = new Date().toISOString();
  
  return `// Enhanced Vector Service - Auto-generated from ${existingFiles.length} files
// Generated: ${timestamp}
// Features detected: ${Object.entries(analysis).filter(([k,v]) => v === true).map(([k]) => k).join(', ')}

import { QdrantClient } from '@qdrant/js-client-rest';
import Redis from 'ioredis';
import { db } from '../db/index.js';
import { cases, evidence, criminals } from '../db/schema-postgres-enhanced.js';
import { embedding_cache, vector_metadata } from '../db/schema-postgres-enhanced.js';
import { eq, sql } from 'drizzle-orm';

export class EnhancedVectorService {
  private qdrant: QdrantClient;
  private redis: Redis;
  private collectionName = 'legal_documents';
  
  constructor() {
    this.qdrant = new QdrantClient({
      url: process.env.QDRANT_URL || 'http://localhost:6333'
    });
    
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3
    });
  }
  
  async initializeCollection() {
    const collections = await this.qdrant.getCollections();
    const exists = collections.collections.some(c => c.name === this.collectionName);
    
    if (!exists) {
      await this.qdrant.createCollection(this.collectionName, {
        vectors: { size: 768, distance: 'Cosine' },
        optimizers_config: { default_segment_number: 2 }
      });
      
      await this.qdrant.createPayloadIndex(this.collectionName, {
        field_name: 'type',
        field_schema: 'keyword'
      });
    }
  }
  
  async generateEmbedding(text: string): Promise<number[]> {
    const cacheKey = \`embed:\${Buffer.from(text).toString('base64').slice(0, 32)}\`;
    
    // Check Redis cache
    const cached = await this.redis.get(cacheKey);
    if (cached) return JSON.parse(cached);
    
    // Generate with Ollama nomic-embed
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: text
      })
    });
    
    if (!response.ok) throw new Error(\`Ollama API error: \${response.statusText}\`);
    
    const result = await response.json();
    const embedding = result.embedding;
    
    // Cache for 24 hours
    await this.redis.setex(cacheKey, 86400, JSON.stringify(embedding));
    
    return embedding;
  }
  
  async storeDocument(id: string, content: string, metadata: any) {
    const embedding = await this.generateEmbedding(content);
    
    await this.qdrant.upsert(this.collectionName, {
      wait: true,
      points: [{
        id,
        vector: embedding,
        payload: { content, ...metadata }
      }]
    });
  }
  
  async hybridSearch(query: string, options: any = {}) {
    const { limit = 10, threshold = 0.7 } = options;
    
    // Vector search
    const queryEmbedding = await this.generateEmbedding(query);
    const vectorResults = await this.qdrant.search(this.collectionName, {
      vector: queryEmbedding,
      limit,
      score_threshold: threshold,
      with_payload: true
    });
    
    // Keyword search in PostgreSQL
    const keywordResults = await this.keywordSearch(query, limit);
    
    // Combine results
    return this.combineResults(vectorResults, keywordResults);
  }
  
  private async keywordSearch(query: string, limit: number) {
    const caseResults = await db.select().from(cases)
      .where(sql\`title ILIKE \${"%"+query+"%"} OR description ILIKE \${"%"+query+"%"}\`)
      .limit(limit);
      
    return caseResults.map(c => ({
      id: c.id,
      score: 0.8,
      metadata: { type: 'case', title: c.title },
      content: \`\${c.title} \${c.description}\`
    }));
  }
  
  private combineResults(vectorResults: any[], keywordResults: any[]) {
    const combined = new Map();
    
    vectorResults.forEach(r => combined.set(r.id, { ...r, score: r.score * 0.7 }));
    keywordResults.forEach(r => {
      const existing = combined.get(r.id);
      if (existing) existing.score += r.score * 0.3;
      else combined.set(r.id, { ...r, score: r.score * 0.3 });
    });
    
    return Array.from(combined.values()).sort((a, b) => b.score - a.score);
  }
  
  async healthCheck() {
    try {
      await this.qdrant.getCollections();
      await this.redis.ping();
      return { qdrant: true, redis: true };
    } catch (error) {
      return { qdrant: false, redis: false, error: error.message };
    }
  }
}

export const vectorService = new EnhancedVectorService();`;
}

async function createVectorAPIEndpoint() {
  return `// Enhanced Vector Search API - Auto-generated
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { vectorService } from '$lib/server/vector/EnhancedVectorService.js';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options = {} } = await request.json();
    
    if (!query) throw error(400, 'Query required');
    
    const results = await vectorService.hybridSearch(query, options);
    
    return json({
      success: true,
      query,
      results,
      count: results.length,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('Vector search error:', err);
    throw error(500, \`Search failed: \${err.message}\`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const query = url.searchParams.get('q');
  if (!query) {
    return json({
      message: 'Vector Search API',
      usage: 'POST /api/vector/search with { query, options }',
      health: await vectorService.healthCheck()
    });
  }
  
  const results = await vectorService.hybridSearch(query);
  return json({ success: true, query, results });
};`;
}

async function enhancedVectorSetup() {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  
  // Scan existing files
  const existingFiles = await scanForVectorFiles();
  console.log(`Found ${existingFiles.length} vector-related files`);
  
  // Backup existing files
  const backupDir = join(projectRoot, `vector-backup-${timestamp}`);
  await mkdir(backupDir, { recursive: true });
  
  for (const file of existingFiles) {
    try {
      const content = await readFile(file, 'utf-8');
      const backupPath = join(backupDir, `${basename(file)}-backup`);
      await writeFile(backupPath, content);
      console.log(`📦 Backed up ${basename(file)}`);
    } catch (err) {
      console.warn(`Cannot backup ${file}: ${err.message}`);
    }
  }
  
  // Analyze existing functionality
  const analysis = await analyzeVectorFiles(existingFiles);
  console.log('🔍 Analysis:', analysis);
  
  // Generate enhanced service
  const serviceDir = join(projectRoot, 'sveltekit-frontend', 'src', 'lib', 'server', 'vector');
  await mkdir(serviceDir, { recursive: true });
  
  const enhancedService = await generateMergedVectorService(analysis, existingFiles);
  await writeFile(join(serviceDir, 'EnhancedVectorService.ts'), enhancedService);
  
  // Generate API endpoint
  const apiDir = join(projectRoot, 'sveltekit-frontend', 'src', 'routes', 'api', 'vector', 'search');
  await mkdir(apiDir, { recursive: true });
  
  const apiEndpoint = await createVectorAPIEndpoint();
  await writeFile(join(apiDir, '+server.ts'), apiEndpoint);
  
  // Create setup report
  const report = `# Enhanced Vector Setup Report
Generated: ${new Date().toISOString()}

## Backup Location
${backupDir}

## Analysis Results
- Ollama Integration: ${analysis.hasOllama ? '✅' : '❌'}
- Qdrant Vector DB: ${analysis.hasQdrant ? '✅' : '❌'}
- Redis Caching: ${analysis.hasRedis ? '✅' : '❌'}
- PostgreSQL Vector: ${analysis.hasPgVector ? '✅' : '❌'}
- Embeddings: ${analysis.hasEmbeddings ? '✅' : '❌'}

## Files Created
- ${serviceDir}/EnhancedVectorService.ts
- ${apiDir}/+server.ts

## Existing Files Merged
${existingFiles.join('\n')}

## Test Commands
\`\`\`bash
curl -X POST http://localhost:5173/api/vector/search \\
  -H "Content-Type: application/json" \\
  -d '{"query": "legal case evidence"}'
\`\`\`
`;

  await writeFile(join(backupDir, 'VECTOR-SETUP-REPORT.md'), report);
  
  console.log(`✅ Enhanced vector setup complete!`);
  console.log(`📦 Backups: ${backupDir}`);
  console.log(`🔧 Service: ${serviceDir}/EnhancedVectorService.ts`);
  console.log(`🌐 API: ${apiDir}/+server.ts`);
}

enhancedVectorSetup().catch(console.error);