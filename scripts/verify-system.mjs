#!/usr/bin/env node

/**
 * System Verification Script
 * Verifies all components of the Enhanced File Upload system
 */

import { promises as fs } from 'fs';
import { join } from 'path';

const config = {
  services: [
    { name: 'SvelteKit', url: 'http://localhost:5175', path: '/upload-test' },
    { name: 'Ollama', url: 'http://localhost:11434', path: '/api/tags' },
    { name: 'PostgreSQL', host: 'localhost', port: 5432 },
    { name: 'Qdrant', url: 'http://localhost:6333', path: '/collections' },
    { name: 'MinIO', url: 'http://localhost:9000', path: '/minio/health/live' },
    { name: 'Enhanced RAG', url: 'http://localhost:8094', path: '/api/rag' },
    { name: 'Upload Service', url: 'http://localhost:8093', path: '/health' }
  ],
  apis: [
    { name: 'LawPDFs API', url: 'http://localhost:5175/api/ai/lawpdfs' },
    { name: 'Document Upload', url: 'http://localhost:5175/api/documents/upload-embed' },
    { name: 'Evidence Upload', url: 'http://localhost:5175/api/evidence/upload' },
    { name: 'AI Search', url: 'http://localhost:5175/api/ai/search' }
  ]
};

async function checkUrl(url, path = '') {
  try {
    const response = await fetch(url + path, { 
      method: 'GET',
      timeout: 5000 
    });
    return { status: response.status, ok: response.ok };
  } catch (error) {
    return { status: 'ERROR', ok: false, error: error.message };
  }
}

async function testLawPdfsApi() {
  console.log('\n🧪 Testing LawPDFs API...');
  
  const testPayload = {
    content: "This is a sample legal document for testing. It contains contract terms, liability clauses, and warranty provisions.",
    fileName: "test-document.pdf",
    analysisType: "comprehensive",
    useLocalModels: true,
    modelPreferences: {
      summaryModel: "gemma3-legal",
      embeddingModel: "nomic-embed-text"
    }
  };

  try {
    const response = await fetch('http://localhost:5175/api/ai/lawpdfs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testPayload)
    });

    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ LawPDFs API Test PASSED');
      console.log(`   📄 Summary: ${result.summary?.substring(0, 100)}...`);
      console.log(`   🏷️  Entities: ${result.entities?.length || 0}`);
      console.log(`   ⚖️  Legal Concepts: ${result.legalConcepts?.length || 0}`);
      console.log(`   🔍 Key Terms: ${result.keyTerms?.length || 0}`);
      console.log(`   ⚠️  Risk Level: ${result.riskAssessment?.riskLevel}`);
      console.log(`   📊 Embedding Size: ${result.embedding?.length || 'N/A'}`);
      console.log(`   ⏱️  Processing Time: ${result.metadata?.processingTime}ms`);
      console.log(`   🤖 Model Used: ${result.metadata?.modelUsed}`);
    } else {
      console.log('❌ LawPDFs API Test FAILED');
      console.log(`   Error: ${result.error || 'Unknown error'}`);
      console.log(`   Details: ${result.details || 'No details'}`);
    }
  } catch (error) {
    console.log('❌ LawPDFs API Test FAILED');
    console.log(`   Error: ${error.message}`);
  }
}

async function verifySystem() {
  console.log('🔍 Legal AI System Verification\n');
  console.log('=' .repeat(50));

  // Check services
  console.log('\n📡 Checking Services...');
  for (const service of config.services) {
    const result = await checkUrl(service.url, service.path);
    const status = result.ok ? '✅' : '❌';
    console.log(`${status} ${service.name}: ${service.url}${service.path || ''}`);
    if (!result.ok && result.error) {
      console.log(`   Error: ${result.error}`);
    }
  }

  // Check API endpoints
  console.log('\n🔌 Checking API Endpoints...');
  for (const api of config.apis) {
    const result = await checkUrl(api.url);
    const status = result.status === 405 ? '✅' : (result.ok ? '✅' : '❌'); // 405 = Method Not Allowed is OK for GET on POST endpoints
    console.log(`${status} ${api.name}: ${api.url}`);
  }

  // Test LawPDFs API functionality
  await testLawPdfsApi();

  // Check file structure
  console.log('\n📁 Checking File Structure...');
  const criticalFiles = [
    'src/routes/api/ai/lawpdfs/+server.ts',
    'src/routes/upload-test/+page.svelte',
    'src/lib/components/forms/EnhancedFileUpload.svelte',
    'src/lib/db/schema.ts',
    'src/lib/db/schema/vectors.ts'
  ];

  for (const file of criticalFiles) {
    try {
      await fs.access(join(process.cwd(), 'sveltekit-frontend', file));
      console.log(`✅ ${file}`);
    } catch {
      console.log(`❌ ${file} - NOT FOUND`);
    }
  }

  console.log('\n' + '=' .repeat(50));
  console.log('✅ System verification complete!');
  console.log('🌐 Access the upload test at: http://localhost:5175/upload-test');
}

verifySystem().catch(console.error);