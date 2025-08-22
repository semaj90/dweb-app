/**
 * Playwright API Endpoint Health Check
 * Tests Ollama generation timeout issues and API routing problems
 */

import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:5173';
const API_ENDPOINTS = [
  // Core API endpoints
  '/api/v1/rag',
  '/api/v1/ai',
  '/api/v1/upload',
  '/api/v1/vector/search',
  '/api/v1/graph/query',
  
  // NATS messaging
  '/api/v1/nats/status',
  '/api/v1/nats/publish',
  
  // Cluster management
  '/api/v1/cluster/health',
  '/api/v1/cluster',
  
  // Ollama endpoints
  '/api/ollama',
  '/api/ollama/api/tags',
  '/api/ollama/api/generate',
  
  // Evidence and case management
  '/api/evidence/list',
  '/api/evidence/upload',
  '/api/evidence/synthesize',
  '/api/activities',
  '/api/canvas',
  
  // YoRHa specific endpoints
  '/api/yorha/legal-data',
  '/api/yorha/system/status',
  
  // Context7 autosolve
  '/api/context7-autosolve'
];

const OLLAMA_TEST_REQUESTS = [
  {
    endpoint: '/api/ollama/api/generate',
    payload: {
      model: 'gemma3-legal',
      prompt: 'Test legal analysis prompt',
      stream: false,
      options: {
        temperature: 0.1,
        max_tokens: 100,
        timeout: 30000
      }
    }
  },
  {
    endpoint: '/api/v1/ai',
    payload: {
      prompt: 'Analyze this legal document excerpt',
      model: 'gemma3-legal',
      timeout: 60000
    }
  },
  {
    endpoint: '/api/v1/rag',
    payload: {
      query: 'What are the legal implications?',
      context: ['sample legal text'],
      model: 'gemma3-legal'
    }
  }
];

async function checkAPIEndpoint(page, endpoint, method = 'GET', payload = null, timeout = 10000) {
  const startTime = Date.now();
  
  try {
    console.log(`\n🔍 Testing ${method} ${endpoint}`);
    
    let response;
    if (method === 'GET') {
      response = await page.goto(`${BASE_URL}${endpoint}`, { 
        waitUntil: 'networkidle',
        timeout 
      });
    } else {
      // For POST requests, use page.evaluate to make fetch calls
      response = await page.evaluate(async ({ endpoint, payload, timeout }) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);
          
          return {
            status: response.status,
            statusText: response.statusText,
            headers: Object.fromEntries(response.headers.entries()),
            text: await response.text(),
            ok: response.ok
          };
        } catch (error) {
          clearTimeout(timeoutId);
          throw error;
        }
      }, { endpoint: `${BASE_URL}${endpoint}`, payload, timeout });
    }
    
    const duration = Date.now() - startTime;
    const status = method === 'GET' ? response.status() : response.status;
    
    if (status === 404) {
      console.log(`❌ ${endpoint} - 404 Not Found (${duration}ms)`);
      return { 
        endpoint, 
        status: 404, 
        error: 'Route not found', 
        duration,
        success: false 
      };
    } else if (status >= 500) {
      console.log(`⚠️  ${endpoint} - ${status} Server Error (${duration}ms)`);
      return { 
        endpoint, 
        status, 
        error: 'Server error', 
        duration,
        success: false 
      };
    } else if (status === 200 || status === 201) {
      console.log(`✅ ${endpoint} - ${status} OK (${duration}ms)`);
      return { 
        endpoint, 
        status, 
        duration,
        success: true 
      };
    } else {
      console.log(`⚠️  ${endpoint} - ${status} (${duration}ms)`);
      return { 
        endpoint, 
        status, 
        duration,
        success: status < 400 
      };
    }
    
  } catch (error) {
    const duration = Date.now() - startTime;
    
    if (error.message.includes('timeout') || error.name === 'AbortError') {
      console.log(`⏱️  ${endpoint} - TIMEOUT after ${duration}ms`);
      return { 
        endpoint, 
        error: 'Timeout', 
        duration,
        timeout: true,
        success: false 
      };
    } else {
      console.log(`❌ ${endpoint} - Error: ${error.message} (${duration}ms)`);
      return { 
        endpoint, 
        error: error.message, 
        duration,
        success: false 
      };
    }
  }
}

async function testOllamaGeneration(page) {
  console.log('\n🤖 Testing Ollama Generation Endpoints...');
  
  const results = [];
  
  for (const test of OLLAMA_TEST_REQUESTS) {
    const result = await checkAPIEndpoint(
      page, 
      test.endpoint, 
      'POST', 
      test.payload, 
      test.payload.timeout || 60000
    );
    
    results.push({
      ...result,
      type: 'ollama_generation',
      payload: test.payload
    });
  }
  
  return results;
}

async function testAllAPIEndpoints(page) {
  console.log('\n🌐 Testing All API Endpoints...');
  
  const results = [];
  
  for (const endpoint of API_ENDPOINTS) {
    const result = await checkAPIEndpoint(page, endpoint, 'GET', null, 5000);
    results.push({
      ...result,
      type: 'api_endpoint'
    });
    
    // Brief pause between requests
    await page.waitForTimeout(100);
  }
  
  return results;
}

async function generateReport(endpointResults, ollamaResults) {
  const timestamp = new Date().toISOString();
  const allResults = [...endpointResults, ...ollamaResults];
  
  const summary = {
    total_endpoints: allResults.length,
    successful: allResults.filter(r => r.success).length,
    failed: allResults.filter(r => !r.success).length,
    timeouts: allResults.filter(r => r.timeout).length,
    not_found_404: allResults.filter(r => r.status === 404).length,
    server_errors_5xx: allResults.filter(r => r.status >= 500).length,
    average_response_time: Math.round(
      allResults.filter(r => r.duration).reduce((sum, r) => sum + r.duration, 0) / 
      allResults.filter(r => r.duration).length
    )
  };
  
  const issues = {
    routing_issues: allResults.filter(r => r.status === 404).map(r => ({
      endpoint: r.endpoint,
      issue: 'Route not found - 404',
      recommendation: 'Check route definition in SvelteKit app or proxy configuration'
    })),
    
    timeout_issues: allResults.filter(r => r.timeout || r.duration > 30000).map(r => ({
      endpoint: r.endpoint,
      issue: r.timeout ? 'Request timeout' : `Slow response (${r.duration}ms)`,
      recommendation: r.endpoint.includes('ollama') ? 
        'Check Ollama service status and increase timeout limits' :
        'Investigate service health and optimize response time'
    })),
    
    server_errors: allResults.filter(r => r.status >= 500).map(r => ({
      endpoint: r.endpoint,
      issue: `Server error ${r.status}`,
      recommendation: 'Check server logs and service health'
    }))
  };
  
  const report = {
    timestamp,
    test_configuration: {
      base_url: BASE_URL,
      total_endpoints_tested: API_ENDPOINTS.length,
      ollama_generation_tests: OLLAMA_TEST_REQUESTS.length
    },
    summary,
    issues,
    detailed_results: allResults,
    recommendations: [
      summary.not_found_404 > 0 ? 'Fix API routing configuration' : null,
      summary.timeouts > 0 ? 'Resolve timeout issues, especially for Ollama endpoints' : null,
      summary.server_errors_5xx > 0 ? 'Check service health and server logs' : null,
      summary.average_response_time > 5000 ? 'Optimize API response times' : null
    ].filter(Boolean)
  };
  
  return report;
}

async function runHealthCheck() {
  console.log('🚀 Starting API Health Check with Playwright...\n');
  
  const browser = await chromium.launch({ 
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  try {
    const context = await browser.newContext({
      ignoreHTTPSErrors: true,
      userAgent: 'YoRHa-API-Health-Check/1.0'
    });
    
    const page = await context.newPage();
    
    // Test basic connectivity first
    console.log(`🔗 Testing base connectivity to ${BASE_URL}...`);
    try {
      await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 10000 });
      console.log('✅ Base URL accessible');
    } catch (error) {
      console.log('❌ Base URL not accessible:', error.message);
      throw new Error('Cannot connect to base URL');
    }
    
    // Run API endpoint tests
    const endpointResults = await testAllAPIEndpoints(page);
    
    // Run Ollama generation tests
    const ollamaResults = await testOllamaGeneration(page);
    
    // Generate comprehensive report
    const report = await generateReport(endpointResults, ollamaResults);
    
    // Save report to file
    const reportPath = `./api-health-check-report-${Date.now()}.json`;
    await page.evaluate((reportData, path) => {
      // This would normally save to file, but we'll log instead in browser context
      console.log('HEALTH_CHECK_REPORT:', JSON.stringify(reportData, null, 2));
    }, report, reportPath);
    
    // Print summary
    console.log('\n📊 API Health Check Summary:');
    console.log(`✅ Successful: ${report.summary.successful}/${report.summary.total_endpoints}`);
    console.log(`❌ Failed: ${report.summary.failed}`);
    console.log(`⏱️  Timeouts: ${report.summary.timeouts}`);
    console.log(`🔍 404 Not Found: ${report.summary.not_found_404}`);
    console.log(`⚠️  Server Errors: ${report.summary.server_errors_5xx}`);
    console.log(`⚡ Average Response Time: ${report.summary.average_response_time}ms`);
    
    if (report.recommendations.length > 0) {
      console.log('\n🔧 Recommendations:');
      report.recommendations.forEach(rec => console.log(`   • ${rec}`));
    }
    
    return report;
    
  } finally {
    await browser.close();
  }
}

// Run the health check
runHealthCheck()
  .then(report => {
    console.log('\n✅ Health check completed successfully');
    
    // Write report to file
    const fs = require('fs');
    const reportPath = `./api-health-check-report-${Date.now()}.json`;
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`📝 Report saved to: ${reportPath}`);
    
    process.exit(report.summary.failed > 0 ? 1 : 0);
  })
  .catch(error => {
    console.error('❌ Health check failed:', error);
    process.exit(1);
  });