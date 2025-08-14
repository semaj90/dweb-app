/**
 * Enhanced REST API Test Suite
 * Tests all clustering and search endpoints with real data
 */

import { promises as fs } from 'fs';
import fetch from 'node-fetch';

// Test configuration
const API_BASE_URL = 'http://localhost:5173';
const TEST_TIMEOUT = 30000; // 30 seconds

// Test data - sample legal document embeddings (384 dimensions for Gemma3)
const SAMPLE_EMBEDDINGS = [
  Array.from({ length: 384 }, () => Math.random() * 2 - 1), // Contract embedding
  Array.from({ length: 384 }, () => Math.random() * 2 - 1), // Case law embedding
  Array.from({ length: 384 }, () => Math.random() * 2 - 1), // Regulation embedding
  Array.from({ length: 384 }, () => Math.random() * 2 - 1), // Evidence embedding
  Array.from({ length: 384 }, () => Math.random() * 2 - 1)  // Filing embedding
];

const SAMPLE_DOCUMENTS = [
  {
    id: 'doc_contract_001',
    type: 'contract',
    keywords: ['liability', 'indemnification', 'breach', 'termination'],
    title: 'Software License Agreement',
    content: 'This agreement governs the licensing of software with liability limitations...'
  },
  {
    id: 'doc_case_002',
    type: 'case_law',
    keywords: ['precedent', 'negligence', 'damages', 'jurisdiction'],
    title: 'Smith v. Tech Corp - Negligence Case',
    content: 'Plaintiff alleges negligence in software design causing business interruption...'
  },
  {
    id: 'doc_regulation_003',
    type: 'regulation',
    keywords: ['compliance', 'data_protection', 'privacy', 'gdpr'],
    title: 'Data Protection Regulation Compliance',
    content: 'Organizations must implement appropriate technical and organizational measures...'
  },
  {
    id: 'doc_evidence_004',
    type: 'evidence',
    keywords: ['digital_forensics', 'metadata', 'authentication', 'chain_of_custody'],
    title: 'Digital Evidence Analysis Report',
    content: 'Forensic examination of digital devices reveals communication patterns...'
  },
  {
    id: 'doc_filing_005',
    type: 'filing',
    keywords: ['motion', 'discovery', 'admissibility', 'expert_testimony'],
    title: 'Motion for Summary Judgment',
    content: 'Defendant moves for summary judgment on the grounds that no genuine dispute...'
  }
];

// Test results tracking
const testResults = {
  passed: 0,
  failed: 0,
  errors: [],
  timings: {},
  responses: {}
};

// Utility functions
function logTest(testName, status, details = '') {
  const timestamp = new Date().toISOString();
  const statusIcon = status === 'PASS' ? '✅' : status === 'FAIL' ? '❌' : '⚠️';
  console.log(`${statusIcon} [${timestamp}] ${testName}: ${status} ${details}`);
  
  if (status === 'PASS') {
    testResults.passed++;
  } else {
    testResults.failed++;
    testResults.errors.push({ test: testName, details, timestamp });
  }
}

async function makeRequest(method, endpoint, body = null) {
  const startTime = Date.now();
  const url = `${API_BASE_URL}${endpoint}`;
  
  try {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
      timeout: TEST_TIMEOUT
    };
    
    if (body) {
      options.body = JSON.stringify(body);
    }
    
    const response = await fetch(url, options);
    const responseTime = Date.now() - startTime;
    const data = await response.json();
    
    testResults.timings[endpoint] = responseTime;
    testResults.responses[endpoint] = data;
    
    return {
      ok: response.ok,
      status: response.status,
      data,
      responseTime
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    testResults.timings[endpoint] = responseTime;
    
    return {
      ok: false,
      status: 0,
      error: error.message,
      responseTime
    };
  }
}

// Test Suite Functions

async function testSOMTraining() {
  console.log('\\n🧠 Testing SOM Clustering API...');
  
  try {
    // Test SOM training endpoint
    const trainingPayload = {
      documentIds: SAMPLE_DOCUMENTS.map(doc => doc.id),
      config: {
        width: 10,
        height: 10,
        learningRate: 0.3,
        radius: 3,
        iterations: 100 // Reduced for testing
      }
    };
    
    const trainingResponse = await makeRequest('POST', '/api/clustering/som/train', trainingPayload);
    
    if (trainingResponse.ok && trainingResponse.data.success) {
      logTest('SOM Training Start', 'PASS', `Training ID: ${trainingResponse.data.data.trainingId}`);
      
      // Wait briefly and check status
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const statusResponse = await makeRequest('GET', `/api/clustering/som/train?trainingId=${trainingResponse.data.data.trainingId}`);
      
      if (statusResponse.ok && statusResponse.data.success) {
        logTest('SOM Training Status', 'PASS', `Status: ${statusResponse.data.data.status}`);
        return trainingResponse.data.data.trainingId;
      } else {
        logTest('SOM Training Status', 'FAIL', statusResponse.data?.error || 'Status check failed');
      }
    } else {
      logTest('SOM Training Start', 'FAIL', trainingResponse.data?.error || `HTTP ${trainingResponse.status}`);
    }
  } catch (error) {
    logTest('SOM Training', 'FAIL', error.message);
  }
  
  return null;
}

async function testKMeansClustering() {
  console.log('\\n📊 Testing K-Means Clustering API...');
  
  try {
    // Test K-Means clustering endpoint
    const clusteringPayload = {
      documentIds: SAMPLE_DOCUMENTS.map(doc => doc.id),
      k: 3,
      config: {
        maxIterations: 50,
        tolerance: 0.01,
        initMethod: 'kmeans++'
      }
    };
    
    const clusterResponse = await makeRequest('POST', '/api/clustering/kmeans/cluster', clusteringPayload);
    
    if (clusterResponse.ok && clusterResponse.data.success) {
      const clusters = clusterResponse.data.data.clusters;
      const metrics = clusterResponse.data.data.metrics;
      
      logTest('K-Means Clustering', 'PASS', 
        `Clusters: ${clusters.length}, Silhouette: ${metrics.silhouetteScore.toFixed(3)}`);
      
      // Test prediction with sample embedding
      const predictionResponse = await makeRequest('GET', 
        `/api/clustering/kmeans/cluster?jobId=${clusterResponse.data.data.jobId}&embedding=${JSON.stringify(SAMPLE_EMBEDDINGS[0])}`);
      
      if (predictionResponse.ok && predictionResponse.data.success) {
        logTest('K-Means Prediction', 'PASS', 
          `Predicted cluster: ${predictionResponse.data.data.clusterId}`);
        return clusterResponse.data.data.jobId;
      } else {
        logTest('K-Means Prediction', 'FAIL', predictionResponse.data?.error || 'Prediction failed');
      }
    } else {
      logTest('K-Means Clustering', 'FAIL', clusterResponse.data?.error || `HTTP ${clusterResponse.status}`);
    }
  } catch (error) {
    logTest('K-Means Clustering', 'FAIL', error.message);
  }
  
  return null;
}

async function testSemanticSearch() {
  console.log('\\n🔍 Testing Semantic Search API...');
  
  try {
    // Test semantic search with clustering
    const searchPayload = {
      query: 'software liability and negligence in contract law',
      useKnowledge: true,
      limit: 10,
      threshold: 0.5
    };
    
    const searchResponse = await makeRequest('POST', '/api/search/semantic', searchPayload);
    
    if (searchResponse.ok && searchResponse.data.success) {
      const results = searchResponse.data.data.results;
      const insights = searchResponse.data.data.insights;
      
      logTest('Semantic Search', 'PASS', 
        `Results: ${results.length}, Insights: ${insights.length}`);
      
      // Test cluster status endpoint
      const statusResponse = await makeRequest('GET', '/api/search/semantic');
      
      if (statusResponse.ok && statusResponse.data.success) {
        const status = statusResponse.data.data;
        logTest('Cluster Status', 'PASS', 
          `SOM trained: ${status.som.trained}, K-Means clusters: ${status.kmeans.clusters}`);
      } else {
        logTest('Cluster Status', 'FAIL', statusResponse.data?.error || 'Status failed');
      }
    } else {
      logTest('Semantic Search', 'FAIL', searchResponse.data?.error || `HTTP ${searchResponse.status}`);
    }
  } catch (error) {
    logTest('Semantic Search', 'FAIL', error.message);
  }
}

async function testAPIValidation() {
  console.log('\\n🔒 Testing API Validation...');
  
  try {
    // Test invalid SOM training request
    const invalidSOMResponse = await makeRequest('POST', '/api/clustering/som/train', {});
    
    if (!invalidSOMResponse.ok && invalidSOMResponse.status === 400) {
      logTest('SOM Validation', 'PASS', 'Correctly rejected invalid request');
    } else {
      logTest('SOM Validation', 'FAIL', 'Should have rejected invalid request');
    }
    
    // Test invalid K-Means request
    const invalidKMeansResponse = await makeRequest('POST', '/api/clustering/kmeans/cluster', {
      documentIds: [],
      k: -1
    });
    
    if (!invalidKMeansResponse.ok && invalidKMeansResponse.status === 400) {
      logTest('K-Means Validation', 'PASS', 'Correctly rejected invalid request');
    } else {
      logTest('K-Means Validation', 'FAIL', 'Should have rejected invalid request');
    }
    
    // Test invalid search request
    const invalidSearchResponse = await makeRequest('POST', '/api/search/semantic', {});
    
    if (!invalidSearchResponse.ok && invalidSearchResponse.status === 400) {
      logTest('Search Validation', 'PASS', 'Correctly rejected invalid request');
    } else {
      logTest('Search Validation', 'FAIL', 'Should have rejected invalid request');
    }
  } catch (error) {
    logTest('API Validation', 'FAIL', error.message);
  }
}

async function testPerformanceMetrics() {
  console.log('\\n⚡ Testing Performance Metrics...');
  
  try {
    const performanceTests = [
      { endpoint: '/api/clustering/som/train', threshold: 500 },
      { endpoint: '/api/clustering/kmeans/cluster', threshold: 5000 },
      { endpoint: '/api/search/semantic', threshold: 1000 }
    ];
    
    for (const test of performanceTests) {
      const timing = testResults.timings[test.endpoint];
      if (timing !== undefined) {
        if (timing <= test.threshold) {
          logTest(`Performance: ${test.endpoint}`, 'PASS', `${timing}ms (threshold: ${test.threshold}ms)`);
        } else {
          logTest(`Performance: ${test.endpoint}`, 'FAIL', `${timing}ms exceeds ${test.threshold}ms threshold`);
        }
      }
    }
  } catch (error) {
    logTest('Performance Metrics', 'FAIL', error.message);
  }
}

async function testEndToEndWorkflow() {
  console.log('\\n🔄 Testing End-to-End Workflow...');
  
  try {
    // 1. Train SOM
    logTest('E2E Step 1', 'INFO', 'Starting SOM training...');
    const somTrainingId = await testSOMTraining();
    
    // 2. Perform K-Means clustering
    logTest('E2E Step 2', 'INFO', 'Starting K-Means clustering...');
    const kmeansJobId = await testKMeansClustering();
    
    // 3. Run semantic search
    logTest('E2E Step 3', 'INFO', 'Running semantic search...');
    await testSemanticSearch();
    
    // 4. Validate complete workflow
    if (somTrainingId && kmeansJobId) {
      logTest('End-to-End Workflow', 'PASS', 'Complete ML pipeline executed successfully');
    } else {
      logTest('End-to-End Workflow', 'FAIL', 'Some components failed in the pipeline');
    }
  } catch (error) {
    logTest('End-to-End Workflow', 'FAIL', error.message);
  }
}

async function generateTestReport() {
  console.log('\\n📊 Generating Test Report...');
  
  const report = {
    summary: {
      totalTests: testResults.passed + testResults.failed,
      passed: testResults.passed,
      failed: testResults.failed,
      successRate: ((testResults.passed / (testResults.passed + testResults.failed)) * 100).toFixed(2) + '%'
    },
    performance: testResults.timings,
    errors: testResults.errors,
    sampleResponses: testResults.responses,
    timestamp: new Date().toISOString(),
    environment: {
      apiBaseUrl: API_BASE_URL,
      nodeVersion: process.version,
      timeout: TEST_TIMEOUT
    }
  };
  
  // Save report to file
  try {
    await fs.writeFile(
      'enhanced-rest-api-test-report.json',
      JSON.stringify(report, null, 2)
    );
    
    logTest('Test Report', 'PASS', 'Report saved to enhanced-rest-api-test-report.json');
  } catch (error) {
    logTest('Test Report', 'FAIL', `Failed to save report: ${error.message}`);
  }
  
  return report;
}

// Main test execution
async function runAllTests() {
  console.log('🚀 Starting Enhanced REST API Test Suite...');
  console.log(`Target: ${API_BASE_URL}`);
  console.log(`Timeout: ${TEST_TIMEOUT}ms`);
  console.log('=' * 60);
  
  const startTime = Date.now();
  
  try {
    // Update todo status
    testResults.currentTask = 'API Testing';
    
    // Run test suites
    await testAPIValidation();
    await testSOMTraining();
    await testKMeansClustering();
    await testSemanticSearch();
    await testPerformanceMetrics();
    await testEndToEndWorkflow();
    
    // Generate final report
    const report = await generateTestReport();
    
    const totalTime = Date.now() - startTime;
    
    console.log('\\n' + '=' * 60);
    console.log('📈 TEST SUMMARY');
    console.log('=' * 60);
    console.log(`✅ Passed: ${report.summary.passed}`);
    console.log(`❌ Failed: ${report.summary.failed}`);
    console.log(`📊 Success Rate: ${report.summary.successRate}`);
    console.log(`⏱️ Total Time: ${totalTime}ms`);
    
    if (testResults.errors.length > 0) {
      console.log('\\n🔍 ERRORS:');
      testResults.errors.forEach(error => {
        console.log(`  • ${error.test}: ${error.details}`);
      });
    }
    
    console.log('\\n🎯 PERFORMANCE METRICS:');
    Object.entries(testResults.timings).forEach(([endpoint, time]) => {
      console.log(`  • ${endpoint}: ${time}ms`);
    });
    
    // Exit with appropriate code
    process.exit(testResults.failed > 0 ? 1 : 0);
    
  } catch (error) {
    console.error('\\n💥 TEST SUITE CRASHED:', error);
    process.exit(1);
  }
}

// Handle process termination
process.on('SIGINT', async () => {
  console.log('\\n⚠️ Test suite interrupted');
  await generateTestReport();
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('\\n💥 Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Start tests
runAllTests().catch(console.error);