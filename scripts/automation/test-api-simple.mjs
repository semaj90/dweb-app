/**
 * Simple API Test for Enhanced REST API
 * Tests with existing database structure and mock embeddings
 */

import fetch from 'node-fetch';

const API_BASE_URL = 'http://localhost:5173';

// Simple test data
const testData = {
  query: 'contract liability analysis',
  documentIds: ['doc1', 'doc2', 'doc3'],
  sampleEmbedding: Array.from({ length: 384 }, () => Math.random() * 2 - 1)
};

async function testAPI(endpoint, method = 'GET', body = null) {
  console.log(`\\n🔍 Testing ${method} ${endpoint}`);
  
  try {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' }
    };
    
    if (body) {
      options.body = JSON.stringify(body);
    }
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    const data = await response.json();
    
    console.log(`Status: ${response.status}`);
    console.log(`Response:`, JSON.stringify(data, null, 2));
    
    return { ok: response.ok, status: response.status, data };
  } catch (error) {
    console.log(`❌ Error: ${error.message}`);
    return { ok: false, error: error.message };
  }
}

async function runTests() {
  console.log('🚀 Starting Simple API Tests...');
  
  // Test 1: Cluster Status (GET)
  await testAPI('/api/search/semantic');
  
  // Test 2: Semantic Search (POST)
  await testAPI('/api/search/semantic', 'POST', {
    query: testData.query,
    useKnowledge: false, // Disable clustering for initial test
    limit: 5
  });
  
  // Test 3: SOM Training Status (should fail gracefully)
  await testAPI('/api/clustering/som/train?trainingId=test123');
  
  // Test 4: K-Means Prediction (should fail gracefully)
  await testAPI(`/api/clustering/kmeans/cluster?jobId=test&embedding=${JSON.stringify(testData.sampleEmbedding)}`);
  
  console.log('\\n✅ Tests completed!');
}

runTests().catch(console.error);