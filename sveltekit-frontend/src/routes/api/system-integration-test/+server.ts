import type { RequestHandler } from './$types.js';

// System Integration Test - Complete Health Check and Autosolve
// Tests all system components including Context7, MCP, database orchestrator, and autosolve

import { testContext7Pipeline, testDatabaseOperations, runFullIntegrationTest } from "$lib/services/comprehensive-database-orchestrator";
import { URL } from "url";

// GET /api/system-integration-test - Run comprehensive system test
export const GET: RequestHandler = async ({ url }) => {
  const testType = url.searchParams.get('type') || 'full';

  console.log(`🧪 Starting ${testType} system integration test...`);

  try {
    const testResults: any = {
      test_type: testType,
      start_time: new Date().toISOString(),
      services: {},
      database: {},
      apis: {},
      context7: {},
      autosolve: {},
      overall_status: 'unknown',
      errors: [],
      recommendations: [],
      end_time: undefined,
    };

    // Step 1: Test Core Services
    console.log('🔍 Testing core services...');
    testResults.services = await testCoreServices();

    // Step 2: Test Database Connectivity
    console.log('🗄️ Testing database connectivity...');
    testResults.database = await testDatabaseConnectivity();

    // Step 3: Test API Endpoints
    console.log('🌐 Testing API endpoints...');
    testResults.apis = await testAPIEndpoints();

    // Step 4: Test Context7 Integration (if available)
    if (testType === 'full' || testType === 'context7') {
      console.log('🤖 Testing Context7 MCP integration...');
      testResults.context7 = await testContext7Integration();
    }

    // Step 5: Test Autosolve System (if available)
    if (testType === 'full' || testType === 'autosolve') {
      console.log('🔧 Testing autosolve system...');
      testResults.autosolve = await testAutosolveSystem();
    }

    // Step 6: Determine Overall Status
    testResults.overall_status = determineOverallStatus(testResults);
    testResults.recommendations = generateRecommendations(testResults);
    testResults.end_time = new Date().toISOString();

    console.log(`✅ System integration test completed with status: ${testResults.overall_status}`);

    return json({
      success: true,
      test_results: testResults,
      summary: generateTestSummary(testResults),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('❌ System integration test failed:', error);

    return json(
      {
        success: false,
        error: error.message,
        test_type: testType,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// POST /api/system-integration-test - Trigger specific tests
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, targets } = await request.json();

    switch (action) {
      case 'quick_health_check':
        return await quickHealthCheck();

      case 'autosolve_test':
        return await testAutosolvePipeline();

      case 'context7_test':
        return await testContext7Pipeline();

      case 'database_test':
        return await testDatabaseOperations();

      case 'full_integration':
        return await runFullIntegrationTest();

      default:
        return json(
          {
            success: false,
            error: `Unknown test action: ${action}`,
            available_actions: [
              'quick_health_check',
              'autosolve_test',
              'context7_test',
              'database_test',
              'full_integration',
            ],
          },
          { status: 400 }
        );
    }
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// Test core services
async function testCoreServices(): Promise<any> {
  const services = [
    { name: 'sveltekit_frontend', url: 'http://localhost:5173' },
    { name: 'ollama', url: 'http://localhost:11434/api/tags' },
    { name: 'enhanced_rag', url: 'http://localhost:8097/health' },
    { name: 'aggregate_server', url: 'http://localhost:8123/health' },
    { name: 'recommendation_service', url: 'http://localhost:8096/health' },
  ];

  const results = {};

  for (const service of services) {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 5000);
      const response = await fetch(service.url, { method: 'GET', signal: controller.signal });
      clearTimeout(t);

      results[service.name] = {
        status: response.ok ? 'healthy' : 'unhealthy',
        response_code: response.status,
        response_time: Date.now(),
        url: service.url,
      };
    } catch (error: any) {
      results[service.name] = {
        status: 'error',
        error: error.message,
        url: service.url,
      };
    }
  }

  return results;
}

// Test database connectivity
async function testDatabaseConnectivity(): Promise<any> {
  try {
    // Test basic fetch to database APIs
    const tests = [
      { name: 'cases_api', url: '/api/database-orchestrator?action=query&table=cases&limit=1' },
      { name: 'comprehensive_integration', url: '/api/comprehensive-integration' },
    ];

    const results = {};

    for (const test of tests) {
      try {
        const response = await fetch(`http://localhost:5173${test.url}`);
        results[test.name] = {
          status: response.ok ? 'healthy' : 'unhealthy',
          response_code: response.status,
        };
      } catch (error: any) {
        results[test.name] = {
          status: 'error',
          error: error.message,
        };
      }
    }

    return {
      connectivity_tests: results,
      overall_status: Object.values(results as Record<string, any>).every(
        (r: any) => r && r.status === 'healthy'
      )
        ? 'healthy'
        : 'degraded',
    };
  } catch (error: any) {
    return {
      status: 'error',
      error: error.message,
    };
  }
}

// Test API endpoints
async function testAPIEndpoints(): Promise<any> {
  const endpoints = [
    { name: 'database_orchestrator', path: '/api/database-orchestrator' },
    { name: 'context7_autosolve', path: '/api/context7-autosolve' },
    { name: 'comprehensive_integration', path: '/api/comprehensive-integration' },
    { name: 'context7', path: '/api/context7' },
  ];

  const results = {};

  for (const endpoint of endpoints) {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 3000);
      const response = await fetch(`http://localhost:5173${endpoint.path}`, {
        method: 'GET',
        signal: controller.signal,
      });
      clearTimeout(t);

      results[endpoint.name] = {
        status: response.ok ? 'accessible' : 'error',
        response_code: response.status,
        path: endpoint.path,
      };
    } catch (error: any) {
      results[endpoint.name] = {
        status: 'error',
        error: error.message,
        path: endpoint.path,
      };
    }
  }

  return results;
}

// Test Context7 integration
async function testContext7Integration(): Promise<any> {
  try {
    const context7Tests = [];

    // Test Context7 status
    try {
      const c1 = new AbortController();
      const t1 = setTimeout(() => c1.abort(), 5000);
      const statusResponse = await fetch('http://localhost:5173/api/context7', {
        method: 'GET',
        signal: c1.signal,
      });
      clearTimeout(t1);
      context7Tests.push({
        test: 'context7_status',
        status: statusResponse.ok ? 'pass' : 'fail',
        response_code: statusResponse.status,
      });
    } catch (error: any) {
      context7Tests.push({
        test: 'context7_status',
        status: 'error',
        error: error.message,
      });
    }

    // Test autosolve integration
    try {
      const c2 = new AbortController();
      const t2 = setTimeout(() => c2.abort(), 5000);
      const autosolveResponse = await fetch(
        'http://localhost:5173/api/context7-autosolve?action=status',
        { method: 'GET', signal: c2.signal }
      );
      clearTimeout(t2);
      context7Tests.push({
        test: 'autosolve_integration',
        status: autosolveResponse.ok ? 'pass' : 'fail',
        response_code: autosolveResponse.status,
      });
    } catch (error: any) {
      context7Tests.push({
        test: 'autosolve_integration',
        status: 'error',
        error: error.message,
      });
    }

    return {
      tests: context7Tests,
      overall_status: context7Tests.every((t) => t.status === 'pass') ? 'healthy' : 'degraded',
    };
  } catch (error: any) {
    return {
      status: 'error',
      error: error.message,
    };
  }
}

// Test autosolve system
async function testAutosolveSystem(): Promise<any> {
  try {
    const autosolveTests = [];

    // Test autosolve health
    try {
      const c3 = new AbortController();
      const t3 = setTimeout(() => c3.abort(), 10000);
      const healthResponse = await fetch(
        'http://localhost:5173/api/context7-autosolve?action=health',
        { method: 'GET', signal: c3.signal }
      );
      clearTimeout(t3);
      autosolveTests.push({
        test: 'autosolve_health',
        status: healthResponse.ok ? 'pass' : 'fail',
        response_code: healthResponse.status,
      });
    } catch (error: any) {
      autosolveTests.push({
        test: 'autosolve_health',
        status: 'error',
        error: error.message,
      });
    }

    // Test TypeScript check (if available)
    try {
      const c4 = new AbortController();
      const t4 = setTimeout(() => c4.abort(), 15000);
      const tsResponse = await fetch('http://localhost:5173/api/system/typescript-check', {
        method: 'POST',
        signal: c4.signal,
      });
      clearTimeout(t4);
      autosolveTests.push({
        test: 'typescript_check',
        status: tsResponse.ok ? 'pass' : 'fail',
        response_code: tsResponse.status,
      });
    } catch (error: any) {
      autosolveTests.push({
        test: 'typescript_check',
        status: 'error',
        error: error.message,
      });
    }

    return {
      tests: autosolveTests,
      overall_status: autosolveTests.some((t) => t.status === 'pass') ? 'partial' : 'unavailable',
    };
  } catch (error: any) {
    return {
      status: 'error',
      error: error.message,
    };
  }
}

// Quick health check
async function quickHealthCheck(): Promise<any> {
  const startTime = Date.now();

  const healthChecks = await Promise.allSettled([
    fetch('http://localhost:5173'),
    fetch('http://localhost:11434/api/tags'),
    fetch('http://localhost:5173/api/comprehensive-integration'),
  ]);

  const results = healthChecks.map((result, index) => {
    const services = ['frontend', 'ollama', 'integration_api'];
    return {
      service: services[index],
      status: result.status === 'fulfilled' && result.value?.ok ? 'healthy' : 'unhealthy',
      response: result.status === 'fulfilled' ? result.value?.status : 'error',
    };
  });

  const healthyCount = results.filter((r) => r.status === 'healthy').length;
  const totalCount = results.length;

  return json({
    success: true,
    message: 'Quick health check completed',
    results,
    health_score: Math.round((healthyCount / totalCount) * 100),
    healthy_services: healthyCount,
    total_services: totalCount,
    response_time_ms: Date.now() - startTime,
    timestamp: new Date().toISOString(),
  });
}

// Test autosolve pipeline
async function testAutosolvePipeline(): Promise<any> {
  try {
    const pipelineTest = await fetch('http://localhost:5173/api/context7-autosolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'health_check',
      }),
    });

    return json({
      success: true,
      message: 'Autosolve pipeline test completed',
      pipeline_status: pipelineTest.ok ? 'operational' : 'degraded',
      response_code: pipelineTest.status,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Autosolve pipeline test failed: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Helper functions
function determineOverallStatus(testResults: any): string {
  const statuses = [
    testResults.services,
    testResults.database,
    testResults.apis,
    testResults.context7,
    testResults.autosolve,
  ];

  const healthyCount = statuses.filter(
    (status) =>
      (status as any)?.overall_status === 'healthy' ||
      Object.values(status || {}).some((s: any) => s && (s as any).status === 'healthy')
  ).length;

  if (healthyCount === statuses.length) return 'excellent';
  if (healthyCount >= statuses.length * 0.8) return 'good';
  if (healthyCount >= statuses.length * 0.6) return 'fair';
  return 'poor';
}

function generateRecommendations(testResults: any): string[] {
  const recommendations = [];

  if (testResults.overall_status === 'poor') {
    recommendations.push('🚨 Critical: Multiple system failures detected');
    recommendations.push('Check service startup scripts and logs');
  }

  if (testResults.services?.ollama?.status !== 'healthy') {
    recommendations.push('🤖 Start Ollama service: ollama serve');
  }

  if (testResults.context7?.overall_status !== 'healthy') {
    recommendations.push('🔧 Context7 integration needs attention');
    recommendations.push('Review MCP server configurations');
  }

  if (testResults.autosolve?.overall_status === 'unavailable') {
    recommendations.push('⚡ Autosolve system not available');
    recommendations.push('Check TypeScript and error processing services');
  }

  if (testResults.overall_status === 'excellent') {
    recommendations.push('✅ All systems operational');
    recommendations.push('Ready for production workloads');
  }

  return recommendations;
}

function generateTestSummary(testResults: any): unknown {
  return {
    overall_status: testResults.overall_status,
    services_tested: Object.keys(testResults.services || {}).length,
    healthy_services: Object.values(testResults.services || {}).filter(
      (s: any) => s && s.status === 'healthy'
    ).length,
    api_endpoints_tested: Object.keys(testResults.apis || {}).length,
    context7_available: testResults.context7?.overall_status === 'healthy',
    autosolve_available: testResults.autosolve?.overall_status !== 'error',
    test_duration:
      testResults.end_time && testResults.start_time
        ? new Date(testResults.end_time).getTime() - new Date(testResults.start_time).getTime()
        : 0,
  };
}
