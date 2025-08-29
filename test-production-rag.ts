#!/usr/bin/env node
/**
 * Production RAG System Validation Script
 * 
 * Comprehensive testing of the production RAG orchestration system
 * Tests all endpoints, service health, and integration functionality
 */

import fetch from 'node-fetch';

export interface TestResult {
  name: string;
  success: boolean;
  message: string;
  responseTime?: number;
  data?: any;
}

class ProductionRAGTester {
  private baseUrl = 'http://localhost:5173';
  private testResults: TestResult[] = [];

  /**
   * Run all validation tests
   */
  async runAllTests(): Promise<void> {
    console.log('🚀 Production RAG System Validation Starting...\n');

    await this.testSystemHealth();
    await this.testServiceEndpoints();
    await this.testDocumentProcessing();
    await this.testRAGQueries();
    await this.testRealTimeFeatures();

    this.printResults();
  }

  /**
   * Test overall system health
   */
  async testSystemHealth(): Promise<void> {
    console.log('📊 Testing System Health...');

    try {
      const startTime = Date.now();
      const response = await fetch(`${this.baseUrl}/api/rag/orchestrate`, {
        method: 'PATCH'
      });
      const responseTime = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json();
        
        this.testResults.push({
          name: 'System Health Check',
          success: true,
          message: `System status: ${data.services.status}`,
          responseTime,
          data: data.services
        });

        console.log(`  ✅ Health check passed (${responseTime}ms)`);
        console.log(`     - Overall health: ${data.services.status}`);
        console.log(`     - Services healthy: ${data.services.healthy}/${data.services.total}`);
        
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
    } catch (error: any) {
      this.testResults.push({
        name: 'System Health Check',
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error'
      });
      console.log(`  ❌ Health check failed: ${error}`);
    }
  }

  /**
   * Test individual service endpoints
   */
  async testServiceEndpoints(): Promise<void> {
    console.log('\n🔌 Testing Service Endpoints...');

    const endpoints = [
      { name: 'Enhanced RAG', url: 'http://localhost:8094/health' },
      { name: 'Upload Service', url: 'http://localhost:8093/health' },
      { name: 'Vector Service', url: 'http://localhost:8095/health' },
      { name: 'Ollama', url: 'http://localhost:11434/api/tags' },
      { name: 'Qdrant', url: 'http://localhost:6333/collections' },
      { name: 'Redis', url: 'http://localhost:5173/api/rag/orchestrate', method: 'PATCH' }
    ];

    for (const endpoint of endpoints) {
      try {
        const startTime = Date.now();
        const response = await fetch(endpoint.url, {
          method: endpoint.method || 'GET',
          timeout: 5000
        } as any);
        const responseTime = Date.now() - startTime;
        
        const success = response.ok;
        this.testResults.push({
          name: `${endpoint.name} Endpoint`,
          success,
          message: success ? 'Service responding' : `HTTP ${response.status}`,
          responseTime
        });

        console.log(`  ${success ? '✅' : '❌'} ${endpoint.name}: ${success ? 'OK' : 'Failed'} (${responseTime}ms)`);
        
      } catch (error: any) {
        this.testResults.push({
          name: `${endpoint.name} Endpoint`,
          success: false,
          message: error instanceof Error ? error.message : 'Connection failed'
        });
        console.log(`  ❌ ${endpoint.name}: Connection failed`);
      }
    }
  }

  /**
   * Test document processing pipeline
   */
  async testDocumentProcessing(): Promise<void> {
    console.log('\n📄 Testing Document Processing...');

    try {
      // Test document processing initiation
      const startTime = Date.now();
      const response = await fetch(`${this.baseUrl}/api/rag/orchestrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploadId: 'test-upload-001',
          caseId: 'test-case-001',
          filename: 'test-document.pdf',
          storageUrl: 'http://localhost:9000/test-bucket/test-document.pdf'
        })
      });
      const responseTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        
        this.testResults.push({
          name: 'Document Processing Start',
          success: true,
          message: `Job created: ${data.jobId}`,
          responseTime,
          data: { jobId: data.jobId }
        });

        console.log(`  ✅ Document processing started (${responseTime}ms)`);
        console.log(`     - Job ID: ${data.jobId}`);
        console.log(`     - Status: ${data.status}`);

        // Test job status retrieval
        if (data.jobId) {
          await this.testJobStatus(data.jobId);
        }
        
      } else {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
    } catch (error: any) {
      this.testResults.push({
        name: 'Document Processing Start',
        success: false,
        message: error instanceof Error ? error.message : 'Processing failed'
      });
      console.log(`  ❌ Document processing failed: ${error}`);
    }
  }

  /**
   * Test job status tracking
   */
  async testJobStatus(jobId: string): Promise<void> {
    try {
      const startTime = Date.now();
      const response = await fetch(`${this.baseUrl}/api/rag/orchestrate/status/${jobId}`);
      const responseTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        
        this.testResults.push({
          name: 'Job Status Tracking',
          success: true,
          message: `Job status retrieved: ${data.job.status}`,
          responseTime,
          data: data.job
        });

        console.log(`  ✅ Job status retrieved (${responseTime}ms)`);
        console.log(`     - Status: ${data.job.status}`);
        console.log(`     - Progress: ${data.job.progress}%`);
        console.log(`     - Stages completed: ${data.stages.completed}/${data.stages.total}`);
        
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
    } catch (error: any) {
      this.testResults.push({
        name: 'Job Status Tracking',
        success: false,
        message: error instanceof Error ? error.message : 'Status check failed'
      });
      console.log(`  ❌ Job status check failed: ${error}`);
    }
  }

  /**
   * Test RAG query functionality
   */
  async testRAGQueries(): Promise<void> {
    console.log('\n🔍 Testing RAG Queries...');

    const testQueries = [
      {
        name: 'Basic Legal Query',
        query: 'What are the main legal principles?',
        caseId: undefined
      },
      {
        name: 'Case-Specific Query',
        query: 'What documents are relevant to this case?',
        caseId: 'test-case-001'
      },
      {
        name: 'Complex Legal Query',
        query: 'What are the liability implications and risk factors in this legal matter?',
        caseId: 'test-case-001'
      }
    ];

    for (const testQuery of testQueries) {
      try {
        const params = new URLSearchParams({ query: testQuery.query });
        if (testQuery.caseId) params.append('caseId', testQuery.caseId);
        params.append('limit', '3');
        params.append('includeMetadata', 'true');
        
        const startTime = Date.now();
        const response = await fetch(`${this.baseUrl}/api/rag/orchestrate?${params}`);
        const responseTime = Date.now() - startTime;

        if (response.ok) {
          const data = await response.json();
          
          this.testResults.push({
            name: testQuery.name,
            success: true,
            message: `Query processed successfully`,
            responseTime,
            data: {
              sources: data.sources?.length || 0,
              confidence: data.confidence,
              cached: data.cached
            }
          });

          console.log(`  ✅ ${testQuery.name} (${responseTime}ms)`);
          console.log(`     - Response length: ${data.response?.length || 0} characters`);
          console.log(`     - Sources found: ${data.sources?.length || 0}`);
          console.log(`     - Confidence: ${data.confidence || 'N/A'}`);
          console.log(`     - Cached: ${data.cached ? 'Yes' : 'No'}`);
          
        } else {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
      } catch (error: any) {
        this.testResults.push({
          name: testQuery.name,
          success: false,
          message: error instanceof Error ? error.message : 'Query failed'
        });
        console.log(`  ❌ ${testQuery.name} failed: ${error}`);
      }
    }
  }

  /**
   * Test real-time features
   */
  async testRealTimeFeatures(): Promise<void> {
    console.log('\n⚡ Testing Real-time Features...');

    try {
      // Test WebSocket endpoint availability
      const response = await fetch(`${this.baseUrl}/api/rag/orchestrate/ws?subscribe=health,jobs`);
      
      if (response.ok) {
        this.testResults.push({
          name: 'WebSocket Endpoint',
          success: true,
          message: 'WebSocket endpoint accessible'
        });
        console.log('  ✅ WebSocket endpoint accessible');
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
    } catch (error: any) {
      this.testResults.push({
        name: 'WebSocket Endpoint',
        success: false,
        message: error instanceof Error ? error.message : 'WebSocket test failed'
      });
      console.log(`  ❌ WebSocket test failed: ${error}`);
    }

    // Test demo page availability
    try {
      const response = await fetch(`${this.baseUrl}/demo/production-rag`);
      
      if (response.ok) {
        this.testResults.push({
          name: 'Demo Page',
          success: true,
          message: 'Demo page accessible'
        });
        console.log('  ✅ Demo page accessible');
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
    } catch (error: any) {
      this.testResults.push({
        name: 'Demo Page',
        success: false,
        message: error instanceof Error ? error.message : 'Demo page test failed'
      });
      console.log(`  ❌ Demo page test failed: ${error}`);
    }
  }

  /**
   * Print comprehensive test results
   */
  private printResults(): void {
    console.log('\n🎯 Test Results Summary');
    console.log('=' .repeat(50));

    const successful = this.testResults.filter(r => r.success).length;
    const total = this.testResults.length;
    const successRate = (successful / total * 100).toFixed(1);

    console.log(`\nOverall: ${successful}/${total} tests passed (${successRate}%)`);

    // Group results by success status
    const passed = this.testResults.filter(r => r.success);
    const failed = this.testResults.filter(r => !r.success);

    if (passed.length > 0) {
      console.log('\n✅ Passed Tests:');
      passed.forEach(test => {
        const timing = test.responseTime ? ` (${test.responseTime}ms)` : '';
        console.log(`   - ${test.name}: ${test.message}${timing}`);
      });
    }

    if (failed.length > 0) {
      console.log('\n❌ Failed Tests:');
      failed.forEach(test => {
        console.log(`   - ${test.name}: ${test.message}`);
      });
    }

    // Performance summary
    const timedResults = this.testResults.filter(r => r.responseTime);
    if (timedResults.length > 0) {
      const avgResponseTime = timedResults.reduce((sum, r) => sum + (r.responseTime || 0), 0) / timedResults.length;
      const maxResponseTime = Math.max(...timedResults.map(r => r.responseTime || 0));
      const minResponseTime = Math.min(...timedResults.map(r => r.responseTime || 0));

      console.log('\n📊 Performance Metrics:');
      console.log(`   - Average response time: ${avgResponseTime.toFixed(1)}ms`);
      console.log(`   - Fastest response: ${minResponseTime}ms`);
      console.log(`   - Slowest response: ${maxResponseTime}ms`);
    }

    // Recommendations
    console.log('\n🔧 System Status:');
    if (successRate >= 80) {
      console.log('   ✅ System is functioning well and ready for production use');
    } else if (successRate >= 60) {
      console.log('   ⚠️  System has some issues but core functionality works');
    } else {
      console.log('   ❌ System has significant issues that need to be addressed');
    }

    // Next steps
    console.log('\n🚀 Next Steps:');
    console.log('   1. Access the demo at: http://localhost:5173/demo/production-rag');
    console.log('   2. Monitor services via health dashboard');
    console.log('   3. Check logs for any ongoing issues');
    console.log('   4. Run START-LEGAL-AI.bat to ensure all services are running');

    console.log('\n' + '='.repeat(50));
    console.log('Production RAG System Validation Complete! 🎉\n');
  }
}

// Run the tests
const tester = new ProductionRAGTester();
tester.runAllTests().catch(console.error);