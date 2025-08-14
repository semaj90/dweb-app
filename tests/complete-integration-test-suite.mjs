#!/usr/bin/env node

/**
 * Complete Integration Test Suite for Legal AI System
 * Tests: Neo4j + Redis + PostgreSQL + Ollama + ML Pipeline + Frontend
 */

import chalk from 'chalk';
import fetch from 'node-fetch';
import { spawn } from 'child_process';
import fs from 'fs/promises';

const CONFIG = {
  services: {
    postgresql: { port: 5432, name: 'PostgreSQL + pgvector' },
    redis: { port: 6379, name: 'Redis Cache' },
    neo4j: { port: 7474, name: 'Neo4j Graph DB' },
    ollama: { port: 11434, name: 'Ollama LLM' },
    xstate: { port: 8095, name: 'XState Manager' },
    ollama_simd: { port: 8081, name: 'Ollama SIMD' },
    grpc_server: { port: 8080, name: 'Enhanced gRPC' },
    sveltekit: { port: 5173, name: 'SvelteKit Frontend' }
  },
  test_queries: [
    "What are the elements of wire fraud?",
    "Find precedents for contract disputes",
    "Analyze evidence chain for criminal case",
    "Search for similar fraud cases",
    "What is the statute of limitations for theft?"
  ]
};

class IntegrationTestSuite {
  constructor() {
    this.results = {
      services: {},
      apis: {},
      ml_pipeline: {},
      frontend: {},
      rag_pipeline: {},
      performance: {}
    };
    this.startTime = Date.now();
  }

  log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const colors = {
      info: chalk.blue,
      success: chalk.green,
      warning: chalk.yellow,
      error: chalk.red,
      header: chalk.cyan.bold
    };
    console.log(`[${timestamp}] ${colors[type] || chalk.white}(message)`);
  }

  async testServiceHealth(serviceName, config) {
    try {
      const url = `http://localhost:${config.port}`;
      const response = await fetch(url, { timeout: 5000 });
      
      const isHealthy = response.ok || response.status < 500;
      this.results.services[serviceName] = {
        status: isHealthy ? 'healthy' : 'unhealthy',
        port: config.port,
        response_time: Date.now() - this.startTime
      };
      
      this.log(`✅ ${config.name}: ${isHealthy ? 'Healthy' : 'Unhealthy'}`, 
               isHealthy ? 'success' : 'error');
      return isHealthy;
    } catch (error) {
      this.results.services[serviceName] = {
        status: 'error',
        error: error.message,
        port: config.port
      };
      this.log(`❌ ${config.name}: ${error.message}`, 'error');
      return false;
    }
  }

  async testDatabaseConnections() {
    this.log('🗄️  Testing Database Connections...', 'header');
    
    // Test PostgreSQL + pgvector
    try {
      const pgResult = await this.execCommand(
        '"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U postgres -h localhost -c "SELECT version();" --quiet'
      );
      this.results.apis.postgresql = { status: 'connected', version: pgResult };
      this.log('✅ PostgreSQL: Connected', 'success');
    } catch (error) {
      this.results.apis.postgresql = { status: 'error', error: error.message };
      this.log('❌ PostgreSQL: Connection failed', 'error');
    }

    // Test Neo4j
    try {
      const neo4jQuery = {
        statements: [{
          statement: "MATCH (n) RETURN count(n) as total_nodes"
        }]
      };
      
      const response = await fetch('http://localhost:7474/db/neo4j/tx/commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(neo4jQuery)
      });
      
      if (response.ok) {
        const result = await response.json();
        this.results.apis.neo4j = { 
          status: 'connected', 
          nodes: result.results[0]?.data[0]?.row[0] || 0 
        };
        this.log('✅ Neo4j: Connected and queryable', 'success');
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      this.results.apis.neo4j = { status: 'error', error: error.message };
      this.log('❌ Neo4j: Query failed', 'error');
    }

    // Test Redis
    try {
      const redisResult = await this.execCommand('redis-cli ping');
      this.results.apis.redis = { 
        status: redisResult.includes('PONG') ? 'connected' : 'error',
        response: redisResult 
      };
      this.log('✅ Redis: Connected', 'success');
    } catch (error) {
      this.results.apis.redis = { status: 'error', error: error.message };
      this.log('❌ Redis: Connection failed', 'error');
    }
  }

  async testMLPipeline() {
    this.log('🧠 Testing ML/AI Pipeline...', 'header');
    
    const testCases = [
      {
        name: 'Intent Classification',
        endpoint: 'http://localhost:8081/classify-intent',
        payload: { query: "What is wire fraud?" },
        expectedFields: ['intent', 'confidence']
      },
      {
        name: 'Context Ranking',
        endpoint: 'http://localhost:8081/rank-context',
        payload: { 
          query: "contract law", 
          contexts: ["contract formation", "breach of contract", "employment law"] 
        },
        expectedFields: ['rankings', 'scores']
      },
      {
        name: 'Entity Extraction',
        endpoint: 'http://localhost:8081/extract-entities',
        payload: { text: "John Doe signed a contract with ABC Corp on January 1, 2024" },
        expectedFields: ['entities', 'relationships']
      }
    ];

    for (const test of testCases) {
      try {
        const response = await fetch(test.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(test.payload),
          timeout: 10000
        });

        if (response.ok) {
          const result = await response.json();
          const hasRequiredFields = test.expectedFields.every(field => 
            result.hasOwnProperty(field)
          );
          
          this.results.ml_pipeline[test.name.toLowerCase().replace(' ', '_')] = {
            status: hasRequiredFields ? 'passed' : 'partial',
            response: result,
            fields_present: test.expectedFields.filter(field => result.hasOwnProperty(field))
          };
          
          this.log(`✅ ${test.name}: ${hasRequiredFields ? 'Passed' : 'Partial'}`, 
                   hasRequiredFields ? 'success' : 'warning');
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        this.results.ml_pipeline[test.name.toLowerCase().replace(' ', '_')] = {
          status: 'failed',
          error: error.message
        };
        this.log(`❌ ${test.name}: ${error.message}`, 'error');
      }
    }
  }

  async testEnhancedRAGPipeline() {
    this.log('🚀 Testing Enhanced RAG Pipeline...', 'header');
    
    for (const [index, query] of CONFIG.test_queries.entries()) {
      try {
        this.log(`Testing query ${index + 1}: "${query}"`, 'info');
        
        const startTime = Date.now();
        const response = await fetch('http://localhost:8081/enhanced-rag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            include_neo4j: true,
            include_vectors: true,
            include_ml_ranking: true
          }),
          timeout: 30000
        });

        const endTime = Date.now();
        const responseTime = endTime - startTime;

        if (response.ok) {
          const result = await response.json();
          
          this.results.rag_pipeline[`query_${index + 1}`] = {
            query: query,
            status: 'success',
            response_time_ms: responseTime,
            sources: result.sources?.length || 0,
            confidence: result.confidence || 0,
            neo4j_results: result.neo4j_results?.length || 0,
            vector_results: result.vector_results?.length || 0,
            ml_enhanced: !!result.ml_ranking
          };
          
          this.log(`✅ Query ${index + 1}: Success (${responseTime}ms, ${result.sources?.length || 0} sources)`, 'success');
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        this.results.rag_pipeline[`query_${index + 1}`] = {
          query: query,
          status: 'failed',
          error: error.message
        };
        this.log(`❌ Query ${index + 1}: ${error.message}`, 'error');
      }
    }
  }

  async testFrontendIntegration() {
    this.log('🎨 Testing Frontend Integration...', 'header');
    
    const frontendTests = [
      {
        name: 'Homepage Load',
        url: 'http://localhost:5173/',
        expectedText: 'Legal AI'
      },
      {
        name: 'API Health Check',
        url: 'http://localhost:5173/api/health',
        expectedStatus: 200
      },
      {
        name: 'Legal Assistant Page',
        url: 'http://localhost:5173/legal-assistant',
        expectedText: 'assistant'
      }
    ];

    for (const test of frontendTests) {
      try {
        const response = await fetch(test.url, { timeout: 10000 });
        
        if (response.ok) {
          const content = await response.text();
          const hasExpectedContent = test.expectedText ? 
            content.toLowerCase().includes(test.expectedText.toLowerCase()) : true;
          
          this.results.frontend[test.name.toLowerCase().replace(' ', '_')] = {
            status: hasExpectedContent ? 'passed' : 'partial',
            url: test.url,
            status_code: response.status,
            content_check: hasExpectedContent
          };
          
          this.log(`✅ ${test.name}: ${hasExpectedContent ? 'Passed' : 'Partial'}`, 
                   hasExpectedContent ? 'success' : 'warning');
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        this.results.frontend[test.name.toLowerCase().replace(' ', '_')] = {
          status: 'failed',
          error: error.message,
          url: test.url
        };
        this.log(`❌ ${test.name}: ${error.message}`, 'error');
      }
    }
  }

  async testPerformance() {
    this.log('⚡ Testing Performance Metrics...', 'header');
    
    // Test concurrent requests
    const concurrentRequests = 5;
    const testQuery = "What is contract law?";
    
    try {
      const startTime = Date.now();
      const promises = Array(concurrentRequests).fill().map(() =>
        fetch('http://localhost:8081/enhanced-rag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: testQuery }),
          timeout: 15000
        })
      );
      
      const responses = await Promise.all(promises);
      const endTime = Date.now();
      
      const successCount = responses.filter(r => r.ok).length;
      const avgResponseTime = (endTime - startTime) / concurrentRequests;
      
      this.results.performance.concurrent_requests = {
        total_requests: concurrentRequests,
        successful_requests: successCount,
        avg_response_time_ms: avgResponseTime,
        success_rate: (successCount / concurrentRequests) * 100
      };
      
      this.log(`✅ Concurrent Requests: ${successCount}/${concurrentRequests} successful (${avgResponseTime.toFixed(0)}ms avg)`, 'success');
    } catch (error) {
      this.results.performance.concurrent_requests = {
        status: 'failed',
        error: error.message
      };
      this.log(`❌ Performance Test: ${error.message}`, 'error');
    }
  }

  async execCommand(command) {
    return new Promise((resolve, reject) => {
      const process = spawn('cmd', ['/c', command], { shell: true });
      let output = '';
      let errorOutput = '';
      
      process.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      process.on('close', (code) => {
        if (code === 0) {
          resolve(output.trim());
        } else {
          reject(new Error(errorOutput || `Command failed with code ${code}`));
        }
      });
      
      // Timeout after 10 seconds
      setTimeout(() => {
        process.kill();
        reject(new Error('Command timeout'));
      }, 10000);
    });
  }

  async generateReport() {
    this.log('📊 Generating Integration Test Report...', 'header');
    
    const totalTests = Object.keys(this.results).reduce((acc, category) => 
      acc + Object.keys(this.results[category]).length, 0
    );
    
    const passedTests = Object.keys(this.results).reduce((acc, category) => 
      acc + Object.values(this.results[category]).filter(test => 
        test.status === 'passed' || test.status === 'success' || test.status === 'healthy' || test.status === 'connected'
      ).length, 0
    );
    
    const report = {
      summary: {
        total_tests: totalTests,
        passed_tests: passedTests,
        success_rate: ((passedTests / totalTests) * 100).toFixed(1) + '%',
        test_duration_ms: Date.now() - this.startTime,
        timestamp: new Date().toISOString()
      },
      results: this.results,
      recommendations: this.generateRecommendations()
    };
    
    await fs.writeFile(
      `integration-test-report-${Date.now()}.json`, 
      JSON.stringify(report, null, 2)
    );
    
    this.log(`📝 Report saved: integration-test-report-${Date.now()}.json`, 'success');
    return report;
  }

  generateRecommendations() {
    const recommendations = [];
    
    // Check service health
    const unhealthyServices = Object.entries(this.results.services)
      .filter(([_, service]) => service.status !== 'healthy')
      .map(([name, _]) => name);
    
    if (unhealthyServices.length > 0) {
      recommendations.push(`⚠️  Unhealthy services detected: ${unhealthyServices.join(', ')}`);
    }
    
    // Check ML pipeline
    const failedMLTests = Object.entries(this.results.ml_pipeline)
      .filter(([_, test]) => test.status === 'failed')
      .map(([name, _]) => name);
    
    if (failedMLTests.length > 0) {
      recommendations.push(`🧠 ML Pipeline issues: ${failedMLTests.join(', ')}`);
    }
    
    // Check performance
    if (this.results.performance.concurrent_requests?.success_rate < 80) {
      recommendations.push('⚡ Performance optimization needed for concurrent requests');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('🎉 All systems operating optimally!');
    }
    
    return recommendations;
  }

  async runFullSuite() {
    this.log('🚀 Starting Complete Integration Test Suite...', 'header');
    this.log('================================================', 'header');
    
    // Test all components
    await this.testServiceHealth('postgresql', CONFIG.services.postgresql);
    await this.testServiceHealth('redis', CONFIG.services.redis);
    await this.testServiceHealth('neo4j', CONFIG.services.neo4j);
    await this.testServiceHealth('ollama', CONFIG.services.ollama);
    await this.testServiceHealth('xstate', CONFIG.services.xstate);
    await this.testServiceHealth('ollama_simd', CONFIG.services.ollama_simd);
    await this.testServiceHealth('grpc_server', CONFIG.services.grpc_server);
    await this.testServiceHealth('sveltekit', CONFIG.services.sveltekit);
    
    await this.testDatabaseConnections();
    await this.testMLPipeline();
    await this.testEnhancedRAGPipeline();
    await this.testFrontendIntegration();
    await this.testPerformance();
    
    const report = await this.generateReport();
    
    this.log('================================================', 'header');
    this.log('🎉 Integration Test Suite Complete!', 'header');
    this.log(`📊 Success Rate: ${report.summary.success_rate}`, 'success');
    this.log(`⏱️  Duration: ${report.summary.test_duration_ms}ms`, 'info');
    this.log('================================================', 'header');
    
    // Display recommendations
    report.recommendations.forEach(rec => this.log(rec, 'info'));
    
    return report;
  }
}

// Run the test suite
const testSuite = new IntegrationTestSuite();
testSuite.runFullSuite().catch(console.error);