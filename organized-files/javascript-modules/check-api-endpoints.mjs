/**
 * Quick API Endpoint Checker
 * Verifies all Enhanced REST API endpoints are accessible
 */

import fetch from 'node-fetch';

const API_BASE_URL = 'http://localhost:5173';

const endpoints = [
  { path: '/api/search/semantic', method: 'GET', description: 'Cluster Status' },
  { path: '/api/search/semantic', method: 'POST', description: 'Semantic Search', body: { query: 'test', useKnowledge: false } },
  { path: '/api/clustering/som/train', method: 'GET', description: 'SOM Training Status', query: '?trainingId=test' },
  { path: '/api/clustering/som/train', method: 'POST', description: 'SOM Training', body: { documentIds: ['test'] } },
  { path: '/api/clustering/kmeans/cluster', method: 'GET', description: 'K-Means Prediction', query: '?jobId=test&embedding=[1,2,3]' },
  { path: '/api/clustering/kmeans/cluster', method: 'POST', description: 'K-Means Clustering', body: { documentIds: ['test'] } }
];

async function checkEndpoint(endpoint) {
  const url = `${API_BASE_URL}${endpoint.path}${endpoint.query || ''}`;
  
  try {
    const options = {
      method: endpoint.method,
      headers: { 'Content-Type': 'application/json' },
      timeout: 5000
    };
    
    if (endpoint.body) {
      options.body = JSON.stringify(endpoint.body);
    }
    
    const response = await fetch(url, options);
    const contentType = response.headers.get('content-type');
    
    let data = null;
    if (contentType && contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch (e) {
        data = { error: 'Invalid JSON response' };
      }
    } else {
      data = { error: 'Non-JSON response', contentType };
    }
    
    const status = response.status;
    let statusIcon = '❌';
    
    if (status >= 200 && status < 300) statusIcon = '✅';
    else if (status >= 400 && status < 500) statusIcon = '⚠️'; // Expected for some test endpoints
    else if (status >= 500) statusIcon = '❌';
    
    console.log(`${statusIcon} ${endpoint.method} ${endpoint.path} - ${status} - ${endpoint.description}`);
    
    if (data?.error && !data.error.includes('required') && !data.error.includes('not found')) {
      console.log(`   Error: ${data.error}`);
    }
    
    return { endpoint: endpoint.path, method: endpoint.method, status, ok: response.ok, data };
    
  } catch (error) {
    console.log(`❌ ${endpoint.method} ${endpoint.path} - FAILED - ${endpoint.description}`);
    console.log(`   Error: ${error.message}`);
    return { endpoint: endpoint.path, method: endpoint.method, status: 0, ok: false, error: error.message };
  }
}

async function checkAllEndpoints() {
  console.log('🔍 Checking Enhanced REST API Endpoints...');
  console.log(`Target: ${API_BASE_URL}`);
  console.log('=' * 60);
  
  const results = [];
  
  for (const endpoint of endpoints) {
    const result = await checkEndpoint(endpoint);
    results.push(result);
    
    // Add small delay between requests
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  console.log('\\n' + '=' * 60);
  console.log('📊 ENDPOINT CHECK SUMMARY');
  console.log('=' * 60);
  
  const accessible = results.filter(r => r.ok || (r.status >= 400 && r.status < 500)).length;
  const total = results.length;
  
  console.log(`✅ Accessible: ${accessible}/${total}`);
  console.log(`❌ Failed: ${total - accessible}/${total}`);
  
  if (accessible === total) {
    console.log('\\n🎉 All endpoints are accessible!');
    console.log('You can now run the full test suite with: node test-enhanced-rest-api.mjs');
  } else {
    console.log('\\n⚠️ Some endpoints are not accessible.');
    console.log('Make sure your SvelteKit server is running: cd sveltekit-frontend && npm run dev');
  }
  
  return results;
}

checkAllEndpoints().catch(console.error);