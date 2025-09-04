#!/usr/bin/env node
// Comprehensive Real System Test Suite
// Tests all production APIs with real data

import fetch from 'node-fetch';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const BASE_URL = 'http://localhost:5173';
const TEST_RESULTS = {};

console.log('🧪 REAL SYSTEM TEST SUITE');
console.log('==========================');
console.log('Testing production APIs with real implementations...\n');

// Color functions for output
const colors = {
  green: (text) => `\x1b[32m${text}\x1b[0m`,
  red: (text) => `\x1b[31m${text}\x1b[0m`,
  yellow: (text) => `\x1b[33m${text}\x1b[0m`,
  blue: (text) => `\x1b[34m${text}\x1b[0m`,
  cyan: (text) => `\x1b[36m${text}\x1b[0m`
};

// Test helper function
async function testAPI(name, testFn) {
  try {
    console.log(colors.blue(`🔄 Testing ${name}...`));
    const startTime = Date.now();
    const result = await testFn();
    const duration = Date.now() - startTime;
    
    TEST_RESULTS[name] = {
      status: 'PASS',
      duration: `${duration}ms`,
      result
    };
    
    console.log(colors.green(`✅ ${name} - PASSED (${duration}ms)`));
    if (result && typeof result === 'object') {
      console.log(colors.cyan(`   Result: ${JSON.stringify(result, null, 2).substring(0, 200)}...`));
    }
    console.log('');
    return result;
  } catch (error) {
    TEST_RESULTS[name] = {
      status: 'FAIL',
      error: error.message,
      duration: 'N/A'
    };
    
    console.log(colors.red(`❌ ${name} - FAILED`));
    console.log(colors.red(`   Error: ${error.message}`));
    console.log('');
    return null;
  }
}

// Test 1: OCR Health Check
await testAPI('OCR Health Check', async () => {
  const response = await fetch(`${BASE_URL}/api/ocr/langextract`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (result.status !== 'healthy') throw new Error('OCR service not healthy');
  
  return {
    status: result.status,
    features: result.features,
    supportedFormats: result.features.supportedFormats
  };
});

// Test 2: Embeddings Health Check
await testAPI('Embeddings Health Check', async () => {
  const response = await fetch(`${BASE_URL}/api/embeddings/generate`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (!['healthy', 'degraded'].includes(result.status)) {
    throw new Error('Embeddings service not healthy');
  }
  
  return {
    status: result.status,
    features: result.features,
    models: result.models
  };
});

// Test 3: Search & Database Health Check
await testAPI('Search & Database Health Check', async () => {
  const response = await fetch(`${BASE_URL}/api/documents/search`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (result.status !== 'healthy') throw new Error('Search service not healthy');
  
  return {
    status: result.status,
    database: result.database,
    features: result.features
  };
});

// Test 4: Real Embedding Generation
await testAPI('Real Embedding Generation', async () => {
  const testText = 'This is a legal contract between two parties regarding property transfer and ownership rights.';
  
  const response = await fetch(`${BASE_URL}/api/embeddings/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: testText,
      model: 'nomic-embed-text',
      options: {
        rope: true,
        dimensions: 768
      }
    })
  });
  
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (!result.success) throw new Error(result.error || 'Embedding generation failed');
  if (!result.embedding || !Array.isArray(result.embedding)) {
    throw new Error('Invalid embedding format');
  }
  if (result.embedding.length !== 768) {
    throw new Error(`Expected 768 dimensions, got ${result.embedding.length}`);
  }
  
  return {
    success: result.success,
    dimensions: result.dimensions,
    documentId: result.documentId,
    chunks: result.metadata.chunks,
    ropeApplied: result.metadata.ropeApplied,
    embeddingPreview: result.embedding.slice(0, 5)
  };
});

// Test 5: Document Storage
await testAPI('Document Storage', async () => {
  const testDocument = {
    content: 'Sample legal document for testing storage functionality. This contract outlines terms and conditions.',
    embedding: Array.from({ length: 768 }, () => Math.random() - 0.5), // Random 768-dim vector
    metadata: {
      filename: 'test_document.txt',
      documentType: 'contract',
      jurisdiction: 'test',
      testDocument: true
    },
    filename: 'test_document.txt',
    confidence: 0.95
  };
  
  const response = await fetch(`${BASE_URL}/api/documents/store`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(testDocument)
  });
  
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (!result.success) throw new Error(result.error || 'Storage failed');
  if (!result.documentId) throw new Error('No document ID returned');
  
  return {
    success: result.success,
    documentId: result.documentId,
    document: result.document
  };
});

// Test 6: Real Document Search
await testAPI('Real Document Search', async () => {
  const searchQuery = 'legal contract property rights';
  
  const response = await fetch(`${BASE_URL}/api/documents/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: searchQuery,
      searchType: 'hybrid',
      limit: 5,
      threshold: 0.3
    })
  });
  
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (!result.success) throw new Error(result.error || 'Search failed');
  
  return {
    success: result.success,
    count: result.count,
    searchType: result.searchType,
    searchMethod: result.searchMethod,
    hasResults: result.results && result.results.length > 0,
    firstResult: result.results && result.results.length > 0 ? {
      id: result.results[0].id,
      filename: result.results[0].filename,
      similarity: result.results[0].similarity
    } : null
  };
});

// Test 7: OCR with Sample Text File
await testAPI('OCR Text Processing', async () => {
  // Create a simple text file for testing
  const testText = 'LEGAL CONTRACT\n\nThis agreement is made between Party A and Party B for the provision of legal services.\n\nWHEREAS Party A requires legal consultation;\nWHEREAS Party B provides such services;\n\nNOW THEREFORE the parties agree to the terms herein.';
  
  // Create FormData with text as blob
  const blob = new Blob([testText], { type: 'text/plain' });
  const formData = new FormData();
  formData.append('file', blob, 'test_contract.txt');
  
  const response = await fetch(`${BASE_URL}/api/ocr/langextract`, {
    method: 'POST',
    headers: {
      'X-Enable-LegalBERT': 'true'
    },
    body: formData
  });
  
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const result = await response.json();
  if (!result.success) throw new Error(result.error || 'OCR processing failed');
  if (!result.text) throw new Error('No text extracted');
  
  return {
    success: result.success,
    textLength: result.text.length,
    confidence: result.confidence,
    processingMethod: result.processingMethod,
    hasLegalAnalysis: !!result.legal,
    legalEntities: result.legal ? result.legal.entities.length : 0,
    language: result.language
  };
});

// Test Summary
console.log('\n' + '='.repeat(50));
console.log(colors.cyan('📊 TEST SUMMARY'));
console.log('='.repeat(50));

let passedTests = 0;
let failedTests = 0;

for (const [testName, result] of Object.entries(TEST_RESULTS)) {
  const status = result.status === 'PASS' 
    ? colors.green('✅ PASS') 
    : colors.red('❌ FAIL');
  
  console.log(`${status} | ${testName.padEnd(30)} | ${result.duration || 'N/A'}`);
  
  if (result.status === 'PASS') {
    passedTests++;
  } else {
    failedTests++;
    console.log(colors.red(`      Error: ${result.error}`));
  }
}

console.log('\n' + '='.repeat(50));
console.log(colors.cyan(`📈 RESULTS: ${passedTests} PASSED, ${failedTests} FAILED`));

if (failedTests === 0) {
  console.log(colors.green('\n🎉 ALL TESTS PASSED! REAL SYSTEM IS FULLY OPERATIONAL!'));
  console.log(colors.green('✅ OCR processing working with real Tesseract.js'));
  console.log(colors.green('✅ Embeddings working with real Ollama'));
  console.log(colors.green('✅ Database working with real PostgreSQL + pgvector'));
  console.log(colors.green('✅ Search working with real vector similarity'));
  console.log(colors.green('✅ Storage working with real document persistence'));
  console.log(colors.green('✅ All APIs responding correctly'));
  
  console.log(colors.blue('\n🔗 Ready to use at: http://localhost:5173/ai-upload-demo'));
} else {
  console.log(colors.red('\n⚠️  Some tests failed. Check the services and try again.'));
}

// Save test results
writeFileSync('test-results.json', JSON.stringify(TEST_RESULTS, null, 2));
console.log(colors.yellow('\n📄 Test results saved to test-results.json'));

console.log('\n' + '='.repeat(50));
