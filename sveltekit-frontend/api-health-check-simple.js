/**
 * Simple API Health Check for Ollama and Routing Issues
 */

import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:5173';

const API_TESTS = [
  // Test basic API routes that should exist
  { url: '/api/evidence/list', method: 'GET', expected: [200, 500], timeout: 5000 },
  { url: '/api/evidence/upload', method: 'GET', expected: [200, 405, 500], timeout: 5000 },
  { url: '/api/activities', method: 'GET', expected: [200, 500], timeout: 5000 },
  { url: '/api/canvas', method: 'GET', expected: [200, 500], timeout: 5000 },
  { url: '/api/yorha/system/status', method: 'GET', expected: [200, 500], timeout: 5000 },
  { url: '/api/context7-autosolve', method: 'GET', expected: [200, 500], timeout: 5000 },
  
  // Test Ollama endpoints that might have timeout issues
  { url: '/api/ollama/api/tags', method: 'GET', expected: [200, 404, 500], timeout: 10000 },
  { url: '/api/v1/ai', method: 'POST', body: { prompt: 'test' }, expected: [200, 400, 404, 500], timeout: 30000 },
  { url: '/api/v1/rag', method: 'POST', body: { query: 'test' }, expected: [200, 400, 404, 500], timeout: 30000 },
];

async function testAPI() {
  console.log('🚀 Starting API Health Check...\n');
  
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  const results = [];
  
  try {
    // Test basic page load first
    console.log('🔗 Testing base connectivity...');
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded', timeout: 10000 });
    console.log('✅ Base URL accessible\n');
    
    // Test each API endpoint
    for (const test of API_TESTS) {
      const startTime = Date.now();
      
      try {
        console.log(`🔍 Testing ${test.method} ${test.url}`);
        
        const response = await page.evaluate(async ({ url, method, body, timeout }) => {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), timeout);
          
          try {
            const options = {
              method,
              signal: controller.signal,
              headers: { 'Content-Type': 'application/json' }
            };
            
            if (body && method === 'POST') {
              options.body = JSON.stringify(body);
            }
            
            const response = await fetch(url, options);
            clearTimeout(timeoutId);
            
            let responseText = '';
            try {
              responseText = await response.text();
            } catch (e) {
              responseText = 'Could not read response body';
            }
            
            return {
              status: response.status,
              ok: response.ok,
              statusText: response.statusText,
              body: responseText.substring(0, 200) // Limit response size
            };
          } catch (error) {
            clearTimeout(timeoutId);
            throw error;
          }
        }, { 
          url: `${BASE_URL}${test.url}`, 
          method: test.method, 
          body: test.body,
          timeout: test.timeout 
        });
        
        const duration = Date.now() - startTime;
        const isExpected = test.expected.includes(response.status);
        
        const result = {
          endpoint: test.url,
          method: test.method,
          status: response.status,
          duration,
          expected: isExpected,
          timeout: false,
          response_preview: response.body.substring(0, 100)
        };
        
        if (response.status === 404) {
          console.log(`❌ ${test.url} - 404 Not Found (${duration}ms) - ROUTING ISSUE`);
          result.issue = 'Route not found - possible routing configuration problem';
        } else if (response.status >= 500) {
          console.log(`⚠️  ${test.url} - ${response.status} Server Error (${duration}ms)`);
          result.issue = 'Server error - check service health';
        } else if (!isExpected) {
          console.log(`⚠️  ${test.url} - Unexpected ${response.status} (${duration}ms)`);
          result.issue = 'Unexpected response code';
        } else if (duration > 10000) {
          console.log(`⏱️  ${test.url} - ${response.status} OK but SLOW (${duration}ms)`);
          result.issue = 'Slow response - possible timeout risk';
        } else {
          console.log(`✅ ${test.url} - ${response.status} (${duration}ms)`);
        }
        
        results.push(result);
        
      } catch (error) {
        const duration = Date.now() - startTime;
        
        const result = {
          endpoint: test.url,
          method: test.method,
          duration,
          timeout: error.name === 'AbortError' || error.message.includes('timeout'),
          error: error.message,
          expected: false
        };
        
        if (result.timeout) {
          console.log(`⏱️  ${test.url} - TIMEOUT after ${duration}ms - OLLAMA TIMEOUT ISSUE`);
          result.issue = 'Request timeout - likely Ollama generation timeout issue';
        } else {
          console.log(`❌ ${test.url} - Error: ${error.message} (${duration}ms)`);
          result.issue = 'Request failed';
        }
        
        results.push(result);
      }
      
      // Brief pause between requests
      await page.waitForTimeout(200);
    }
    
  } finally {
    await browser.close();
  }
  
  // Generate report
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      total_tests: results.length,
      successful: results.filter(r => r.expected && !r.timeout).length,
      routing_issues_404: results.filter(r => r.status === 404).length,
      server_errors: results.filter(r => r.status >= 500).length,
      timeout_issues: results.filter(r => r.timeout).length,
      slow_responses: results.filter(r => r.duration > 10000).length
    },
    issues: {
      routing_problems: results.filter(r => r.status === 404),
      ollama_timeouts: results.filter(r => r.timeout || r.duration > 20000),
      server_errors: results.filter(r => r.status >= 500)
    },
    all_results: results
  };
  
  // Print summary
  console.log('\n📊 API Health Check Results:');
  console.log(`✅ Successful: ${report.summary.successful}/${report.summary.total_tests}`);
  console.log(`🔍 404 Routing Issues: ${report.summary.routing_issues_404}`);
  console.log(`⏱️  Timeout Issues: ${report.summary.timeout_issues}`);
  console.log(`⚠️  Server Errors: ${report.summary.server_errors}`);
  console.log(`🐌 Slow Responses (>10s): ${report.summary.slow_responses}`);
  
  if (report.issues.routing_problems.length > 0) {
    console.log('\n🔧 Routing Issues Found:');
    report.issues.routing_problems.forEach(issue => {
      console.log(`   • ${issue.endpoint} - Route not found`);
    });
  }
  
  if (report.issues.ollama_timeouts.length > 0) {
    console.log('\n⏱️  Ollama Timeout Issues Found:');
    report.issues.ollama_timeouts.forEach(issue => {
      console.log(`   • ${issue.endpoint} - ${issue.timeout ? 'Timeout' : `Slow (${issue.duration}ms)`}`);
    });
  }
  
  if (report.issues.server_errors.length > 0) {
    console.log('\n⚠️  Server Error Issues Found:');
    report.issues.server_errors.forEach(issue => {
      console.log(`   • ${issue.endpoint} - ${issue.status} ${issue.issue || 'Server error'}`);
    });
  }
  
  // Save report
  const reportPath = `./api-health-report-${Date.now()}.json`;
  const fs = await import('fs');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n📝 Full report saved to: ${reportPath}`);
  
  return report;
}

testAPI()
  .then(report => {
    const hasIssues = report.summary.routing_issues_404 > 0 || 
                     report.summary.timeout_issues > 0 || 
                     report.summary.server_errors > 0;
    
    if (hasIssues) {
      console.log('\n❌ Issues found that need attention');
      process.exit(1);
    } else {
      console.log('\n✅ All tests passed!');
      process.exit(0);
    }
  })
  .catch(error => {
    console.error('\n❌ Health check failed:', error);
    process.exit(1);
  });