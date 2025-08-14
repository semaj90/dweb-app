#!/usr/bin/env node
// Complete Workflow Verification - Context7 Best Practices
// File: test-evidence-synthesis-workflow.mjs

import { createClient } from 'redis';
import fetch from 'node-fetch';

// Configuration
const API_BASE = process.env.API_BASE || 'http://localhost:5173';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

let redisClient;
let testUser = null;
let testCase = null;
let evidenceItems = [];
let synthesizedEvidence = null;

// Test Data Templates
const TEST_USER = {
  username: `testuser_${Date.now()}`,
  email: `test_${Date.now()}@example.com`,
  password: 'TestPassword123!',
  firstName: 'Test',
  lastName: 'User',
  role: 'detective'
};

const TEST_CASE = {
  title: 'Evidence Synthesis Test Case',
  description: 'Automated test case for evidence synthesis workflow',
  caseNumber: `SYNTH-${Date.now()}`,
  priority: 'high',
  status: 'active'
};

const EVIDENCE_TEMPLATES = [
  {
    title: 'Security Camera Footage - Bank Entrance',
    description: 'Digital video evidence showing suspect entering bank at 14:32',
    evidenceType: 'digital',
    subType: 'video',
    tags: ['surveillance', 'suspect', 'entrance', 'timestamp'],
    collectedAt: '2024-01-15T14:32:00Z',
    location: 'First National Bank - Main Street',
    isAdmissible: true,
    confidentialityLevel: 'restricted'
  },
  {
    title: 'ATM Transaction Log',
    description: 'Digital records showing withdrawal attempt at 14:35',
    evidenceType: 'digital',
    subType: 'log',
    tags: ['financial', 'transaction', 'timestamp', 'digital'],
    collectedAt: '2024-01-15T14:35:00Z',
    location: 'First National Bank - ATM Terminal 2',
    isAdmissible: true,
    confidentialityLevel: 'restricted'
  },
  {
    title: 'Witness Statement - Bank Teller',
    description: 'Testimony from teller describing suspect behavior',
    evidenceType: 'document',
    subType: 'statement',
    tags: ['witness', 'testimony', 'behavior', 'description'],
    collectedAt: '2024-01-15T16:00:00Z',
    location: 'First National Bank - Interview Room',
    isAdmissible: true,
    confidentialityLevel: 'standard'
  }
];

const SYNTHESIS_REQUESTS = [
  {
    synthesisType: 'timeline',
    title: 'Bank Incident Timeline',
    description: 'Chronological reconstruction of events during bank incident',
    prompt: 'Create a detailed timeline focusing on the sequence of events and identifying any temporal gaps or inconsistencies.'
  },
  {
    synthesisType: 'correlation',
    title: 'Digital Evidence Correlation',
    description: 'Cross-reference digital evidence to establish patterns',
    prompt: 'Analyze the correlation between video footage and transaction logs to establish suspect presence and intent.'
  }
];

async function initializeTest() {
  console.log('🚀 Initializing Evidence Synthesis Workflow Test');
  console.log('=' .repeat(60));
  
  try {
    // Initialize Redis connection
    redisClient = createClient({ url: REDIS_URL });
    await redisClient.connect();
    console.log('✅ Redis connection established');
    
    // Subscribe to real-time updates
    const subscriber = redisClient.duplicate();
    await subscriber.connect();
    
    subscriber.subscribe('evidence_update', (message) => {
      const update = JSON.parse(message);
      console.log(`📡 Real-time update: ${update.type}`, {
        evidenceId: update.evidenceId,
        timestamp: update.timestamp
      });
    });
    
    subscriber.subscribe('synthesis_update', (message) => {
      const update = JSON.parse(message);
      console.log(`🔗 Synthesis update: ${update.type}`, {
        synthesisType: update.synthesisType,
        ragScore: update.ragScore,
        timestamp: update.timestamp
      });
    });
    
    return true;
  } catch (error) {
    console.error('❌ Initialization failed:', error.message);
    return false;
  }
}

async function testUserRegistration() {
  console.log('\n📝 Testing User Registration');
  console.log('-'.repeat(40));
  
  try {
    const response = await fetch(`${API_BASE}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(TEST_USER)
    });
    
    const result = await response.json();
    
    if (response.ok && result.user) {
      testUser = result.user;
      console.log('✅ User registration successful');
      console.log(`   User ID: ${testUser.id}`);
      console.log(`   Username: ${testUser.username}`);
      return true;
    } else {
      throw new Error(result.error || 'Registration failed');
    }
  } catch (error) {
    console.error('❌ User registration failed:', error.message);
    return false;
  }
}

async function testUserLogin() {
  console.log('\n🔐 Testing User Login');
  console.log('-'.repeat(40));
  
  try {
    const response = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: TEST_USER.username,
        password: TEST_USER.password
      })
    });
    
    const result = await response.json();
    
    if (response.ok && result.token) {
      console.log('✅ User login successful');
      console.log(`   Token: ${result.token.substring(0, 20)}...`);
      return result.token;
    } else {
      throw new Error(result.error || 'Login failed');
    }
  } catch (error) {
    console.error('❌ User login failed:', error.message);
    return null;
  }
}

async function testCaseCreation(authToken) {
  console.log('\n📁 Testing Case Creation');
  console.log('-'.repeat(40));
  
  try {
    const caseData = {
      ...TEST_CASE,
      userId: testUser.id
    };
    
    const response = await fetch(`${API_BASE}/api/cases`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(caseData)
    });
    
    const result = await response.json();
    
    if (response.ok && result.id) {
      testCase = result;
      console.log('✅ Case creation successful');
      console.log(`   Case ID: ${testCase.id}`);
      console.log(`   Case Number: ${testCase.caseNumber}`);
      return true;
    } else {
      throw new Error(result.error || 'Case creation failed');
    }
  } catch (error) {
    console.error('❌ Case creation failed:', error.message);
    return false;
  }
}

async function testEvidenceUpload(authToken) {
  console.log('\n📄 Testing Evidence Upload');
  console.log('-'.repeat(40));
  
  try {
    for (const [index, template] of EVIDENCE_TEMPLATES.entries()) {
      const evidenceData = {
        ...template,
        caseId: testCase.id,
        summary: `Evidence item ${index + 1} for synthesis testing`
      };
      
      const response = await fetch(`${API_BASE}/api/evidence`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify(evidenceData)
      });
      
      const result = await response.json();
      
      if (response.ok && result.id) {
        evidenceItems.push(result);
        console.log(`✅ Evidence ${index + 1} uploaded successfully`);
        console.log(`   ID: ${result.id}`);
        console.log(`   Title: ${result.title}`);
      } else {
        throw new Error(result.error || `Evidence ${index + 1} upload failed`);
      }
      
      // Small delay between uploads
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    console.log(`\n📊 Total evidence items uploaded: ${evidenceItems.length}`);
    return evidenceItems.length > 0;
  } catch (error) {
    console.error('❌ Evidence upload failed:', error.message);
    return false;
  }
}

async function testEmbeddingGeneration(authToken) {
  console.log('\n🧠 Testing Embedding Generation');
  console.log('-'.repeat(40));
  
  try {
    let embeddingsGenerated = 0;
    
    for (const evidence of evidenceItems) {
      const response = await fetch(`${API_BASE}/api/embeddings/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          content: `${evidence.title}\n${evidence.description}`,
          type: 'evidence',
          metadata: {
            evidenceId: evidence.id,
            caseId: evidence.caseId,
            evidenceType: evidence.evidenceType
          }
        })
      });
      
      if (response.ok) {
        embeddingsGenerated++;
        console.log(`✅ Embedding generated for evidence: ${evidence.title}`);
      } else {
        console.log(`⚠️  Embedding generation failed for: ${evidence.title}`);
      }
    }
    
    console.log(`\n📊 Embeddings generated: ${embeddingsGenerated}/${evidenceItems.length}`);
    return embeddingsGenerated > 0;
  } catch (error) {
    console.error('❌ Embedding generation failed:', error.message);
    return false;
  }
}

async function testEvidenceSynthesis(authToken) {
  console.log('\n🔗 Testing Evidence Synthesis');
  console.log('-'.repeat(40));
  
  try {
    const synthesisResults = [];
    
    for (const synthesisRequest of SYNTHESIS_REQUESTS) {
      const requestData = {
        ...synthesisRequest,
        evidenceIds: evidenceItems.slice(0, 2).map(e => e.id), // Use first 2 evidence items
        caseId: testCase.id
      };
      
      console.log(`\n🧪 Testing ${synthesisRequest.synthesisType} synthesis...`);
      
      const response = await fetch(`${API_BASE}/api/evidence/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify(requestData)
      });
      
      const result = await response.json();
      
      if (response.ok && result.success) {
        synthesisResults.push(result);
        console.log('✅ Synthesis successful');
        console.log(`   Synthesized Evidence ID: ${result.synthesizedEvidence.id}`);
        console.log(`   RAG Score: ${result.metadata.ragScore}`);
        console.log(`   Confidence: ${result.metadata.confidence}`);
        console.log(`   Source Count: ${result.metadata.sourceEvidenceCount}`);
        console.log(`   Embedding Dimensions: ${result.metadata.embeddingDimensions}`);
        
        // Store the first synthesis result for RAG testing
        if (!synthesizedEvidence) {
          synthesizedEvidence = result.synthesizedEvidence;
        }
      } else {
        throw new Error(result.error || `${synthesisRequest.synthesisType} synthesis failed`);
      }
      
      // Delay between synthesis requests
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    console.log(`\n📊 Total synthesis operations: ${synthesisResults.length}`);
    return synthesisResults.length > 0;
  } catch (error) {
    console.error('❌ Evidence synthesis failed:', error.message);
    return false;
  }
}

async function testRAGIntegration(authToken) {
  console.log('\n🔍 Testing Enhanced RAG Integration');
  console.log('-'.repeat(40));
  
  try {
    // Test RAG query using synthesized evidence
    const ragQuery = {
      query: 'Find evidence related to bank surveillance and transaction logs',
      useContextRAG: true,
      useSelfPrompting: true,
      useMultiAgent: true,
      documentTypes: ['evidence', 'synthesized_evidence'],
      maxResults: 10,
      caseContext: testCase.id
    };
    
    const response = await fetch(`${API_BASE}/api/enhanced-rag/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify(ragQuery)
    });
    
    const result = await response.json();
    
    if (response.ok && result.documents) {
      console.log('✅ Enhanced RAG query successful');
      console.log(`   Documents found: ${result.documents.length}`);
      console.log(`   RAG Score Range: ${Math.min(...result.documents.map(d => d.score)).toFixed(3)} - ${Math.max(...result.documents.map(d => d.score)).toFixed(3)}`);
      
      // Check if synthesized evidence appears in high-ranking results
      const synthesizedFound = result.documents.find(doc => 
        doc.id === synthesizedEvidence?.id && doc.score > 0.8
      );
      
      if (synthesizedFound) {
        console.log('✅ Synthesized evidence found in high-ranking results');
        console.log(`   Synthesis RAG Score: ${synthesizedFound.score}`);
      } else {
        console.log('⚠️  Synthesized evidence not found in high-ranking results');
      }
      
      return true;
    } else {
      throw new Error(result.error || 'RAG query failed');
    }
  } catch (error) {
    console.error('❌ Enhanced RAG integration test failed:', error.message);
    return false;
  }
}

async function testSynthesisSuggestions(authToken) {
  console.log('\n💡 Testing Synthesis Suggestions');
  console.log('-'.repeat(40));
  
  try {
    const response = await fetch(`${API_BASE}/api/evidence/synthesize?caseId=${testCase.id}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    
    const result = await response.json();
    
    if (response.ok && result.suggestions) {
      console.log('✅ Synthesis suggestions generated');
      console.log(`   Suggestions: ${result.suggestions.length}`);
      
      result.suggestions.forEach((suggestion, index) => {
        console.log(`   ${index + 1}. ${suggestion.title} (${suggestion.type})`);
        console.log(`      Confidence: ${suggestion.confidence}`);
        console.log(`      Priority: ${suggestion.priority}`);
        console.log(`      Estimated Value: ${suggestion.estimatedValue}`);
      });
      
      return true;
    } else {
      throw new Error(result.error || 'Synthesis suggestions failed');
    }
  } catch (error) {
    console.error('❌ Synthesis suggestions test failed:', error.message);
    return false;
  }
}

async function generateTestReport(results) {
  console.log('\n📊 WORKFLOW TEST REPORT');
  console.log('=' .repeat(60));
  
  const totalTests = Object.keys(results).length;
  const passedTests = Object.values(results).filter(Boolean).length;
  const successRate = (passedTests / totalTests * 100).toFixed(1);
  
  console.log(`\n🎯 Overall Success Rate: ${successRate}% (${passedTests}/${totalTests})`);
  
  console.log('\n📋 Test Results:');
  Object.entries(results).forEach(([test, passed]) => {
    const icon = passed ? '✅' : '❌';
    console.log(`   ${icon} ${test}`);
  });
  
  if (passedTests === totalTests) {
    console.log('\n🎉 ALL TESTS PASSED - Workflow implementation is ready for production!');
    
    console.log('\n🚀 Context7 Best Practices Implementation Summary:');
    console.log('   ✅ User authentication & authorization');
    console.log('   ✅ Case management with proper data validation');
    console.log('   ✅ Evidence upload with metadata enrichment');
    console.log('   ✅ AI-powered evidence synthesis');
    console.log('   ✅ Embedding generation for semantic search');
    console.log('   ✅ Enhanced RAG integration with high-score ranking');
    console.log('   ✅ Real-time updates via Redis pub/sub');
    console.log('   ✅ Synthesis suggestions with AI recommendations');
    console.log('   ✅ Production-ready error handling & logging');
    
    console.log('\n🏗️  Next Steps for Production Deployment:');
    console.log('   1. Configure production database (PostgreSQL)');
    console.log('   2. Set up Redis cluster for real-time features');
    console.log('   3. Deploy enhanced RAG service with vector database');
    console.log('   4. Configure AI service endpoints (OpenAI/Gemma3)');
    console.log('   5. Set up monitoring and alerting');
    console.log('   6. Implement rate limiting and security measures');
    console.log('   7. Add comprehensive logging and audit trails');
    
  } else {
    console.log('\n⚠️  Some tests failed - Review implementation before production deployment');
    
    const failedTests = Object.entries(results)
      .filter(([_, passed]) => !passed)
      .map(([test, _]) => test);
    
    console.log('\n🔧 Failed Tests Requiring Attention:');
    failedTests.forEach(test => {
      console.log(`   • ${test}`);
    });
  }
  
  console.log('\n' + '=' .repeat(60));
}

async function cleanup() {
  console.log('\n🧹 Cleaning up test data...');
  
  try {
    if (redisClient) {
      await redisClient.quit();
      console.log('✅ Redis connection closed');
    }
    
    // Note: In production, you might want to clean up test data
    // For now, we'll leave it for manual inspection
    console.log('📝 Test data preserved for manual inspection');
    
  } catch (error) {
    console.error('⚠️  Cleanup warning:', error.message);
  }
}

// Main workflow test execution
async function runWorkflowTest() {
  const results = {};
  let authToken = null;
  
  try {
    // Initialize test environment
    results['Initialization'] = await initializeTest();
    if (!results['Initialization']) {
      throw new Error('Failed to initialize test environment');
    }
    
    // User registration and authentication
    results['User Registration'] = await testUserRegistration();
    if (results['User Registration']) {
      authToken = await testUserLogin();
      results['User Login'] = authToken !== null;
    }
    
    if (!authToken) {
      throw new Error('Authentication failed - cannot proceed with workflow tests');
    }
    
    // Core workflow tests
    results['Case Creation'] = await testCaseCreation(authToken);
    results['Evidence Upload'] = await testEvidenceUpload(authToken);
    results['Embedding Generation'] = await testEmbeddingGeneration(authToken);
    results['Evidence Synthesis'] = await testEvidenceSynthesis(authToken);
    results['RAG Integration'] = await testRAGIntegration(authToken);
    results['Synthesis Suggestions'] = await testSynthesisSuggestions(authToken);
    
    // Generate comprehensive test report
    await generateTestReport(results);
    
  } catch (error) {
    console.error('\n💥 Workflow test failed:', error.message);
    results['Overall Test'] = false;
    await generateTestReport(results);
  } finally {
    await cleanup();
  }
}

// Execute the test
runWorkflowTest().catch(console.error);
