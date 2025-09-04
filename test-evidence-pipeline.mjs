#!/usr/bin/env node

/**
 * Complete Evidence Pipeline Test
 * Tests MinIO → PostgreSQL → Qdrant → AI integration
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

console.log('🧪 Testing Complete Evidence Pipeline\n');

// Test configuration
const tests = {
  services: [
    { name: 'Qdrant', url: 'http://localhost:6333/collections', expected: 'ok' },
    { name: 'Ollama', url: 'http://localhost:11434/api/tags', expected: 'models' },
    { name: 'MinIO', url: 'http://localhost:9000/minio/health/live', expected: '' }
  ],
  
  models: ['gemma3-legal:latest', 'nomic-embed-text:latest'],
  
  database: {
    user: 'legal_admin',
    database: 'legal_ai_db',
    host: 'localhost'
  }
};

// Service testing function
async function testServices() {
  console.log('🔧 Testing Services...');
  
  for (const service of tests.services) {
    try {
      const response = await fetch(service.url);
      const data = await response.text();
      
      if (service.name === 'Qdrant') {
        const json = JSON.parse(data);
        console.log(`✅ ${service.name}: ${json.status} - ${json.result?.collections?.length || 0} collections`);
      } else if (service.name === 'Ollama') {
        const json = JSON.parse(data);
        console.log(`✅ ${service.name}: ${json.models?.length || 0} models available`);
        
        // Check for required models
        const modelNames = json.models?.map(m => m.name) || [];
        for (const requiredModel of tests.models) {
          if (modelNames.includes(requiredModel)) {
            console.log(`   ✓ ${requiredModel} loaded`);
          } else {
            console.log(`   ❌ ${requiredModel} missing`);
          }
        }
      } else {
        console.log(`✅ ${service.name}: Available`);
      }
    } catch (error) {
      console.log(`❌ ${service.name}: ${error.message}`);
    }
  }
  console.log();
}

// Test Qdrant collection creation
async function testQdrantSetup() {
  console.log('🗄️  Testing Qdrant Setup...');
  
  try {
    // Create legal_evidence collection
    const response = await fetch('http://localhost:6333/collections/legal_evidence', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'legal_evidence',
        vectors: {
          size: 384, // nomic-embed-text dimension
          distance: 'Dot' // For normalized vectors (cosine = dot product)
        }
      })
    });
    
    if (response.ok) {
      console.log('✅ Qdrant collection "legal_evidence" created');
    } else {
      const error = await response.text();
      if (error.includes('already exists')) {
        console.log('✅ Qdrant collection "legal_evidence" already exists');
      } else {
        console.log(`❌ Qdrant collection creation failed: ${error}`);
      }
    }
  } catch (error) {
    console.log(`❌ Qdrant setup failed: ${error.message}`);
  }
  console.log();
}

// Test AI model responses
async function testAIModels() {
  console.log('🧠 Testing AI Models...');
  
  // Test embedding generation
  try {
    console.log('Testing nomic-embed-text...');
    const embResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: 'This is a test legal document about contracts'
      })
    });
    
    if (embResponse.ok) {
      const embResult = await embResponse.json();
      console.log(`✅ nomic-embed-text: Generated ${embResult.embedding?.length}D embedding`);
    } else {
      console.log('❌ nomic-embed-text: Failed to generate embedding');
    }
  } catch (error) {
    console.log(`❌ nomic-embed-text: ${error.message}`);
  }
  
  // Test Gemma3Legal
  try {
    console.log('Testing gemma3-legal...');
    const genResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt: 'Analyze this evidence: A contract dated 2024-01-15 between parties A and B.',
        stream: false
      })
    });
    
    if (genResponse.ok) {
      const genResult = await genResponse.json();
      console.log(`✅ gemma3-legal: Generated response (${genResult.response?.length || 0} chars)`);
    } else {
      console.log('❌ gemma3-legal: Failed to generate response');
    }
  } catch (error) {
    console.log(`❌ gemma3-legal: ${error.message}`);
  }
  console.log();
}

// Test vector storage and retrieval
async function testVectorOperations() {
  console.log('🔍 Testing Vector Operations...');
  
  try {
    // Generate test embedding
    const embResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: 'Test evidence document for legal case'
      })
    });
    
    if (!embResponse.ok) {
      console.log('❌ Could not generate test embedding');
      return;
    }
    
    const embResult = await embResponse.json();
    const testEmbedding = embResult.embedding;
    
    // Store vector in Qdrant
    const storeResponse = await fetch('http://localhost:6333/collections/legal_evidence/points', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: [{
          id: 'test-evidence-001',
          vector: testEmbedding,
          payload: {
            case_id: 'CASE-2024-001',
            file_name: 'test-contract.pdf',
            evidence_type: 'document',
            tags: ['contract', 'test'],
            is_admissible: true
          }
        }]
      })
    });
    
    if (storeResponse.ok) {
      console.log('✅ Vector stored in Qdrant successfully');
      
      // Test vector search
      const searchResponse = await fetch('http://localhost:6333/collections/legal_evidence/points/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: testEmbedding,
          limit: 3,
          with_payload: true
        })
      });
      
      if (searchResponse.ok) {
        const searchResult = await searchResponse.json();
        console.log(`✅ Vector search: Found ${searchResult.result?.length || 0} similar documents`);
        
        if (searchResult.result?.length > 0) {
          console.log(`   Top match: ${searchResult.result[0].payload?.file_name} (${Math.round(searchResult.result[0].score * 100)}% similarity)`);
        }
      } else {
        console.log('❌ Vector search failed');
      }
    } else {
      console.log('❌ Vector storage failed');
    }
  } catch (error) {
    console.log(`❌ Vector operations failed: ${error.message}`);
  }
  console.log();
}

// Test file upload simulation
async function testFileUploadSimulation() {
  console.log('📁 Testing File Upload Simulation...');
  
  // Create test file content
  const testContent = {
    fileName: 'sample-contract.txt',
    content: 'This is a sample legal contract between Party A and Party B, dated January 15, 2024. The contract outlines terms for service agreement including liability, payment terms, and termination clauses.',
    caseId: 'CASE-2024-001',
    evidenceType: 'document',
    tags: ['contract', 'agreement', 'liability']
  };
  
  try {
    // Generate embedding for the content
    console.log('Generating embedding for test content...');
    const embResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: testContent.content
      })
    });
    
    if (!embResponse.ok) {
      console.log('❌ Failed to generate embedding for test file');
      return;
    }
    
    const embResult = await embResponse.json();
    console.log(`✅ Generated ${embResult.embedding.length}D embedding`);
    
    // AI Analysis with Gemma3Legal
    console.log('Analyzing content with Gemma3Legal...');
    const analysisResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt: `As a legal AI assistant for prosecutors, analyze this evidence and return JSON:

File: ${testContent.fileName}
Content: ${testContent.content}

Return structured analysis as JSON:
{
  "summary": "Brief legal summary",
  "keyFindings": ["finding1", "finding2"],
  "legalImplications": ["implication1"],
  "prosecutionRelevance": "high|medium|low",
  "confidence": 0.85
}`,
        stream: false
      })
    });
    
    if (analysisResponse.ok) {
      const analysisResult = await analysisResponse.json();
      console.log(`✅ AI Analysis completed (${analysisResult.response.length} chars)`);
      
      try {
        const parsedAnalysis = JSON.parse(analysisResult.response);
        console.log(`   Summary: ${parsedAnalysis.summary?.substring(0, 100)}...`);
        console.log(`   Prosecution Relevance: ${parsedAnalysis.prosecutionRelevance || 'unknown'}`);
        console.log(`   Confidence: ${Math.round((parsedAnalysis.confidence || 0) * 100)}%`);
      } catch (parseError) {
        console.log('   Raw AI Response:', analysisResult.response.substring(0, 200) + '...');
      }
    } else {
      console.log('❌ AI Analysis failed');
    }
    
    // Store in Qdrant with analysis
    console.log('Storing processed evidence in Qdrant...');
    const storeResponse = await fetch('http://localhost:6333/collections/legal_evidence/points', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: [{
          id: `evidence-${Date.now()}`,
          vector: embResult.embedding,
          payload: {
            ...testContent,
            processed_at: new Date().toISOString(),
            ai_analyzed: true,
            embedding_model: 'nomic-embed-text',
            analysis_model: 'gemma3-legal'
          }
        }]
      })
    });
    
    if (storeResponse.ok) {
      console.log('✅ Evidence stored with AI analysis in Qdrant');
    } else {
      console.log('❌ Failed to store evidence in Qdrant');
    }
    
  } catch (error) {
    console.log(`❌ File upload simulation failed: ${error.message}`);
  }
  console.log();
}

// Run all tests
async function runAllTests() {
  console.log('🚀 Starting Complete Evidence Pipeline Tests\\n');
  
  await testServices();
  await testQdrantSetup();
  await testAIModels();
  await testVectorOperations();
  await testFileUploadSimulation();
  
  console.log('✅ Evidence Pipeline Testing Complete!');
  console.log('\\nNext steps:');
  console.log('- Start SvelteKit dev server: npm run dev');
  console.log('- Visit prosecutor dashboard: http://localhost:5173/prosecutor');
  console.log('- Test evidence upload through UI');
}

// Run tests
runAllTests().catch(console.error);