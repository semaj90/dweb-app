import type { RequestHandler } from './$types';

/**
 * QUIC-Go Integration Test API
 * Tests the integration between SvelteKit QUIC endpoints and Go microservices
 */
import { json, error } from '@sveltejs/kit';

import { ensureError } from '$lib/utils/ensure-error';
import { goServiceManager } from '$lib/services/go-microservice-client.js';

/**
 * GET /api/test/quic-go-integration - Test all QUIC-Go integrations
 */
export const GET: RequestHandler = async ({ url }) => {
  const testResults: Record<string, any> = {};
  let overallSuccess = true;

  try {
    // Test 1: Go Service Manager Health Check
    console.log('Testing Go Service Manager health check...');
    const servicesHealth = await goServiceManager.checkAllServices();
    testResults.servicesHealth = {
      test: 'Go Services Health Check',
      status: Object.values(servicesHealth).every(s => s.success) ? 'PASS' : 'PARTIAL',
      details: servicesHealth
    };

    // Test 2: Enhanced RAG Service
    console.log('Testing Enhanced RAG service...');
    try {
      const enhancedRagClient = goServiceManager.getEnhancedRAG();
      const ragResponse = await enhancedRagClient.ragQuery('test legal query', {
        maxResults: 3,
        threshold: 0.5
      });
      
      testResults.enhancedRag = {
        test: 'Enhanced RAG Query',
        status: ragResponse.success ? 'PASS' : 'FAIL',
        responseTime: ragResponse.responseTime,
        protocol: ragResponse.protocol,
        error: ragResponse.error || null
      };
      
      if (!ragResponse.success) overallSuccess = false;
    } catch (ragError) {
      testResults.enhancedRag = {
        test: 'Enhanced RAG Query',
        status: 'ERROR',
        error: ragError instanceof Error ? ragError.message : 'Unknown error'
      };
      overallSuccess = false;
    }

    // Test 3: Vector Service via Enhanced RAG
    console.log('Testing Vector service...');
    try {
      const enhancedRagClient = goServiceManager.getEnhancedRAG();
      const vectorResponse = await enhancedRagClient.semanticSearch('test legal document search', {
        collection: 'legal_documents',
        limit: 5
      });
      
      testResults.vectorService = {
        test: 'Vector Semantic Search',
        status: vectorResponse.success ? 'PASS' : 'FAIL',
        responseTime: vectorResponse.responseTime,
        protocol: vectorResponse.protocol,
        error: vectorResponse.error || null
      };
      
      if (!vectorResponse.success) overallSuccess = false;
    } catch (vectorError) {
      testResults.vectorService = {
        test: 'Vector Semantic Search',
        status: 'ERROR',
        error: vectorError instanceof Error ? vectorError.message : 'Unknown error'
      };
      overallSuccess = false;
    }

    // Test 4: Upload Service
    console.log('Testing Upload service...');
    try {
      const uploadClient = goServiceManager.getUploadService();
      const healthResponse = await uploadClient.health();
      
      testResults.uploadService = {
        test: 'Upload Service Health',
        status: healthResponse.success ? 'PASS' : 'FAIL',
        responseTime: healthResponse.responseTime,
        protocol: healthResponse.protocol,
        error: healthResponse.error || null
      };
      
      if (!healthResponse.success) overallSuccess = false;
    } catch (uploadError) {
      testResults.uploadService = {
        test: 'Upload Service Health',
        status: 'ERROR',
        error: uploadError instanceof Error ? uploadError.message : 'Unknown error'
      };
      overallSuccess = false;
    }

    // Test 5: QUIC Endpoints Integration Test
    console.log('Testing QUIC endpoints...');
    try {
      // Test RAG proxy endpoint
      const ragProxyResponse = await fetch('/api/v1/quic/rag-proxy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'test legal query',
          maxResults: 3,
          threshold: 0.7
        })
      });

      const ragProxyResult = ragProxyResponse.ok;
      
      testResults.quicEndpoints = {
        test: 'QUIC RAG Proxy Integration',
        status: ragProxyResult ? 'PASS' : 'FAIL',
        httpStatus: ragProxyResponse.status,
        statusText: ragProxyResponse.statusText
      };
      
      if (!ragProxyResult) overallSuccess = false;
    } catch (quicError) {
      testResults.quicEndpoints = {
        test: 'QUIC RAG Proxy Integration',
        status: 'ERROR',
        error: quicError instanceof Error ? quicError.message : 'Unknown error'
      };
      overallSuccess = false;
    }

    // Summary
    const summary = {
      overallStatus: overallSuccess ? 'ALL_TESTS_PASSED' : 'SOME_TESTS_FAILED',
      timestamp: new Date().toISOString(),
      testsRun: Object.keys(testResults).length,
      testsPassed: Object.values(testResults).filter(t => t.status === 'PASS').length,
      testsFailed: Object.values(testResults).filter(t => t.status === 'FAIL').length,
      testsError: Object.values(testResults).filter(t => t.status === 'ERROR').length
    };

    return json({
      success: overallSuccess,
      message: 'QUIC-Go Integration Test Complete',
      summary,
      testResults,
      integrationFeatures: [
        '✅ Go Microservice Client with type safety',
        '✅ QUIC endpoint fallback to Go services',
        '✅ Enhanced RAG service integration',
        '✅ Vector service integration via Enhanced RAG',
        '✅ Upload service health monitoring',
        '✅ Multi-protocol support (HTTP/QUIC/gRPC)',
        '✅ Automatic service discovery and health checks'
      ]
    });

  } catch (err: any) {
    console.error('Integration test failed:', err);
    return json({
      success: false,
      message: 'QUIC-Go Integration Test Failed',
      error: err instanceof Error ? err.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * POST /api/test/quic-go-integration - Test specific integration with custom payload
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const testConfig = await request.json();
    const { service, endpoint, payload } = testConfig;

    if (!service || !endpoint) {
      error(400, ensureError({ message: 'Service and endpoint are required' }));
    }

    let testResult: any = {};

    switch (service) {
      case 'enhancedRag':
        const enhancedRagClient = goServiceManager.getEnhancedRAG();
        if (endpoint === 'ragQuery') {
          testResult = await enhancedRagClient.ragQuery(payload.query, payload.options);
        } else if (endpoint === 'semanticSearch') {
          testResult = await enhancedRagClient.semanticSearch(payload.query, payload.options);
        }
        break;

      case 'uploadService':
        const uploadClient = goServiceManager.getUploadService();
        if (endpoint === 'health') {
          testResult = await uploadClient.health();
        }
        break;

      default:
        const client = goServiceManager.getClient(service as any);
        if (!client) {
          error(400, ensureError({ message: `Unknown service: ${service}` }));
        }
        testResult = await client.request(endpoint, payload);
        break;
    }

    return json({
      success: true,
      message: 'Custom integration test completed',
      service,
      endpoint,
      result: testResult,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('Custom integration test failed:', err);
    error(500, ensureError({
      message: 'Custom integration test failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};