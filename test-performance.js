#!/usr/bin/env node

/**
 * Performance test for optimized Context7 MCP server
 * Tests caching, clustering, and speed improvements
 */

import fetch from 'node-fetch';
import { performance } from 'perf_hooks';

const SERVER_URL = 'http://localhost:4000';

class PerformanceTest {
  constructor() {
    this.results = [];
  }

  async testEndpoint(testName, url, payload = null) {
    const start = performance.now();

    try {
      const options = {
        method: payload ? 'POST' : 'GET',
        headers: { 'Content-Type': 'application/json' },
        ...(payload && { body: JSON.stringify(payload) })
      };

      const response = await fetch(url, options);
      const data = await response.json();
      const end = performance.now();
      const duration = Math.round(end - start);

      const result = {
        test: testName,
        status: response.ok ? 'SUCCESS' : 'FAILED',
        duration: `${duration}ms`,
        response: data,
        timestamp: new Date().toISOString()
      };

      this.results.push(result);
      console.log(`${response.ok ? '✅' : '❌'} ${testName} - ${duration}ms`);

      return result;
    } catch (error) {
      const end = performance.now();
      const duration = Math.round(end - start);

      const result = {
        test: testName,
        status: 'ERROR',
        duration: `${duration}ms`,
        error: error.message,
        timestamp: new Date().toISOString()
      };

      this.results.push(result);
      console.log(`❌ ${testName} - ERROR: ${error.message}`);

      return result;
    }
  }

  async runHealthCheck() {
    console.log('\n🔍 Testing Health Endpoint...');
    return await this.testEndpoint('Health Check', `${SERVER_URL}/health`);
  }

  async runOptimizedRAGTest() {
    console.log('\n⚡ Testing Optimized Enhanced RAG Insight...');

    const payload = {
      tool: 'enhanced-rag-insight-fast',
      args: {
        query: 'contract liability analysis',
        context: 'legal AI system',
        documentType: 'contract'
      }
    };

    return await this.testEndpoint('Enhanced RAG (Optimized)', `${SERVER_URL}/mcp/call`, payload);
  }

  async runCacheTest() {
    console.log('\n🗄️ Testing Cache Performance (Repeated Requests)...');

    const payload = {
      tool: 'enhanced-rag-insight-fast',
      args: {
        query: 'contract liability analysis',
        context: 'legal AI system',
        documentType: 'contract'
      }
    };

    // First request (should populate cache)
    console.log('First request (cache miss):');
    const first = await this.testEndpoint('Cache Test - First', `${SERVER_URL}/mcp/call`, payload);

    // Wait a moment
    await new Promise(resolve => setTimeout(resolve, 100));

    // Second request (should hit cache)
    console.log('Second request (cache hit):');
    const second = await this.testEndpoint('Cache Test - Second', `${SERVER_URL}/mcp/call`, payload);

    // Compare performance
    const firstTime = parseInt(first.duration);
    const secondTime = parseInt(second.duration);
    const improvement = Math.round(((firstTime - secondTime) / firstTime) * 100);

    console.log(`📊 Cache improvement: ${improvement}% faster (${firstTime}ms → ${secondTime}ms)`);

    return { first, second, improvement };
  }

  async runBenchmarkSuite() {
    console.log('\n🚀 Running Performance Benchmark Suite...');

    const queries = [
      { query: 'contract analysis', context: 'legal AI', type: 'contract' },
      { query: 'liability assessment', context: 'risk management', type: 'brief' },
      { query: 'compliance review', context: 'regulatory', type: 'evidence' },
      { query: 'precedent research', context: 'case law', type: 'general' },
      { query: 'damages calculation', context: 'litigation', type: 'contract' }
    ];

    const benchmarks = [];

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      const payload = {
        tool: 'enhanced-rag-insight-fast',
        args: query
      };

      const result = await this.testEndpoint(
        `Benchmark ${i + 1} (${query.query})`,
        `${SERVER_URL}/mcp/call`,
        payload
      );

      benchmarks.push(result);

      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Calculate statistics
    const times = benchmarks
      .filter(b => b.status === 'SUCCESS')
      .map(b => parseInt(b.duration));

    if (times.length > 0) {
      const avg = Math.round(times.reduce((a, b) => a + b, 0) / times.length);
      const min = Math.min(...times);
      const max = Math.max(...times);

      console.log(`\n📊 Benchmark Results:`);
      console.log(`   Average: ${avg}ms`);
      console.log(`   Fastest: ${min}ms`);
      console.log(`   Slowest: ${max}ms`);
      console.log(`   Success Rate: ${times.length}/${benchmarks.length} (${Math.round(times.length/benchmarks.length*100)}%)`);
    }

    return benchmarks;
  }

  async runStressTest() {
    console.log('\n🔥 Running Stress Test (10 concurrent requests)...');

    const payload = {
      tool: 'enhanced-rag-insight-fast',
      args: {
        query: 'stress test concurrent processing',
        context: 'performance testing',
        documentType: 'general'
      }
    };

    const promises = [];
    const startTime = performance.now();

    for (let i = 0; i < 10; i++) {
      promises.push(
        this.testEndpoint(`Stress Test ${i + 1}`, `${SERVER_URL}/mcp/call`, payload)
      );
    }

    const results = await Promise.all(promises);
    const endTime = performance.now();
    const totalTime = Math.round(endTime - startTime);

    const successful = results.filter(r => r.status === 'SUCCESS').length;
    const avgTime = Math.round(
      results
        .filter(r => r.status === 'SUCCESS')
        .map(r => parseInt(r.duration))
        .reduce((a, b) => a + b, 0) / successful
    );

    console.log(`\n🔥 Stress Test Results:`);
    console.log(`   Total Time: ${totalTime}ms`);
    console.log(`   Success Rate: ${successful}/10 (${successful * 10}%)`);
    console.log(`   Average Response: ${avgTime}ms`);
    console.log(`   Throughput: ${Math.round(successful * 1000 / totalTime)} req/sec`);

    return { results, totalTime, successful, avgTime };
  }

  generateReport() {
    console.log('\n📋 Performance Test Report');
    console.log('=' .repeat(50));

    const successfulTests = this.results.filter(r => r.status === 'SUCCESS');
    const failedTests = this.results.filter(r => r.status === 'FAILED' || r.status === 'ERROR');

    console.log(`Total Tests: ${this.results.length}`);
    console.log(`Successful: ${successfulTests.length}`);
    console.log(`Failed: ${failedTests.length}`);
    console.log(`Success Rate: ${Math.round(successfulTests.length / this.results.length * 100)}%`);

    if (successfulTests.length > 0) {
      const times = successfulTests.map(r => parseInt(r.duration));
      const avgTime = Math.round(times.reduce((a, b) => a + b, 0) / times.length);
      const minTime = Math.min(...times);
      const maxTime = Math.max(...times);

      console.log(`\nPerformance Statistics:`);
      console.log(`   Average Response Time: ${avgTime}ms`);
      console.log(`   Fastest Response: ${minTime}ms`);
      console.log(`   Slowest Response: ${maxTime}ms`);

      // Performance target check
      const target = 100; // Target: under 100ms
      const underTarget = times.filter(t => t < target).length;
      const targetPercentage = Math.round(underTarget / times.length * 100);

      console.log(`\n🎯 Performance Target (<${target}ms):`);
      console.log(`   ${underTarget}/${times.length} requests (${targetPercentage}%)`);
      console.log(`   Status: ${targetPercentage >= 80 ? '✅ PASSED' : '❌ NEEDS OPTIMIZATION'}`);
    }

    if (failedTests.length > 0) {
      console.log(`\n❌ Failed Tests:`);
      failedTests.forEach(test => {
        console.log(`   ${test.test}: ${test.error || 'Unknown error'}`);
      });
    }
  }
}

async function main() {
  console.log('🚀 Context7 MCP Server Performance Test');
  console.log('Target: Optimize from 966ms to <100ms\n');

  const tester = new PerformanceTest();

  try {
    // Run all tests
    await tester.runHealthCheck();
    await tester.runOptimizedRAGTest();
    await tester.runCacheTest();
    await tester.runBenchmarkSuite();
    await tester.runStressTest();

    // Generate final report
    tester.generateReport();

    console.log('\n✅ Performance testing complete!');

  } catch (error) {
    console.error('\n❌ Performance test failed:', error);
    process.exit(1);
  }
}

// Run the performance test
main().catch(console.error);
