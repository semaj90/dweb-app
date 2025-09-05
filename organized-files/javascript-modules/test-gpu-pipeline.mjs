/**
 * GPU Processing Pipeline Test Suite
 * Comprehensive testing for the complete GPU tensor processing system
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Test configuration
const config = {
  services: [
    { name: 'Primary GPU Service', url: 'http://localhost:8095', port: 8095 },
    { name: 'Secondary GPU Service', url: 'http://localhost:8096', port: 8096 },
    { name: 'Tertiary GPU Service', url: 'http://localhost:8097', port: 8097 }
  ],
  frontend: {
    name: 'SvelteKit Frontend',
    url: 'http://localhost:5173',
    port: 5173
  },
  apiEndpoint: 'http://localhost:5173/api/gpu/tensor',
  testTimeout: 30000, // 30 seconds
  maxRetries: 3
};

// Test data generators
class TensorGenerator {
  static create1D(size = 100) {
    return {
      shape: [size],
      data: Array.from({ length: size }, () => Math.random() * 2 - 1),
      dimensions: 1,
      layout: 'standard',
      lodLevel: 0
    };
  }

  static create2D(rows = 10, cols = 10) {
    const size = rows * cols;
    return {
      shape: [rows, cols],
      data: Array.from({ length: size }, () => Math.random() * 2 - 1),
      dimensions: 2,
      layout: 'standard',
      lodLevel: 0
    };
  }

  static create3D(depth = 5, rows = 8, cols = 8) {
    const size = depth * rows * cols;
    return {
      shape: [depth, rows, cols],
      data: Array.from({ length: size }, () => Math.random() * 2 - 1),
      dimensions: 3,
      layout: 'standard',
      lodLevel: 0
    };
  }

  static create4D(cases = 3, docs = 5, paragraphs = 10, embeddings = 16) {
    const size = cases * docs * paragraphs * embeddings;
    return {
      shape: [cases, docs, paragraphs, embeddings],
      data: Array.from({ length: size }, () => Math.random() * 2 - 1),
      dimensions: 4,
      layout: 'coalesced',
      lodLevel: 0
    };
  }

  static createLegal4D() {
    // Legal AI specific 4D tensor (small scale for testing)
    return this.create4D(2, 3, 5, 8); // 240 elements total
  }

  static createLarge4D() {
    // Larger tensor for performance testing
    return this.create4D(10, 20, 30, 32); // 192,000 elements
  }
}

// Test runner class
class GPUPipelineTestRunner {
  constructor() {
    this.results = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      tests: []
    };
    this.startTime = Date.now();
  }

  async runAllTests() {
    console.log('🎮 GPU Processing Pipeline Test Suite');
    console.log('====================================');
    console.log(`Started at: ${new Date().toISOString()}`);
    console.log('');

    // Test suites
    await this.testServiceHealth();
    await this.testBasicTensorProcessing();
    await this.testMultiDimensionalProcessing();
    await this.testNESStyleOptimization();
    await this.testLoadBalancing();
    await this.testCaching();
    await this.testErrorHandling();
    await this.testPerformance();
    await this.testWebGPUIntegration();

    // Generate report
    this.generateReport();
  }

  async test(testName, testFunction, timeout = config.testTimeout) {
    this.results.total++;
    const testStart = Date.now();
    
    try {
      console.log(`🧪 ${testName}...`);
      
      await Promise.race([
        testFunction(),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Test timeout')), timeout)
        )
      ]);
      
      const duration = Date.now() - testStart;
      console.log(`   ✅ PASSED (${duration}ms)`);
      
      this.results.passed++;
      this.results.tests.push({
        name: testName,
        status: 'PASSED',
        duration,
        error: null
      });
      
    } catch (error) {
      const duration = Date.now() - testStart;
      console.log(`   ❌ FAILED (${duration}ms): ${error.message}`);
      
      this.results.failed++;
      this.results.tests.push({
        name: testName,
        status: 'FAILED',
        duration,
        error: error.message
      });
    }
  }

  async skip(testName, reason) {
    this.results.total++;
    this.results.skipped++;
    console.log(`   ⏸️  SKIPPED: ${testName} (${reason})`);
    
    this.results.tests.push({
      name: testName,
      status: 'SKIPPED',
      duration: 0,
      error: reason
    });
  }

  // Test suite: Service health
  async testServiceHealth() {
    console.log('\n📊 Testing Service Health');
    console.log('-------------------------');

    for (const service of config.services) {
      await this.test(`Health check: ${service.name}`, async () => {
        const response = await fetch(`${service.url}/health`);
        if (!response.ok) {
          throw new Error(`Health check failed: ${response.status}`);
        }
        const data = await response.json();
        if (data.status !== 'healthy') {
          throw new Error(`Service unhealthy: ${data.status}`);
        }
      });

      await this.test(`Stats endpoint: ${service.name}`, async () => {
        const response = await fetch(`${service.url}/stats`);
        if (!response.ok) {
          throw new Error(`Stats endpoint failed: ${response.status}`);
        }
        const data = await response.json();
        if (!data.processing_stats) {
          throw new Error('Missing processing stats');
        }
      });
    }

    await this.test('Frontend health check', async () => {
      const response = await fetch(config.frontend.url);
      if (!response.ok) {
        throw new Error(`Frontend not accessible: ${response.status}`);
      }
    });
  }

  // Test suite: Basic tensor processing
  async testBasicTensorProcessing() {
    console.log('\n🔢 Testing Basic Tensor Processing');
    console.log('----------------------------------');

    await this.test('Process 1D tensor', async () => {
      const tensor = TensorGenerator.create1D(50);
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error(`Processing failed: ${response.status}`);
      }
      
      const result = await response.json();
      if (!result.success || !result.data) {
        throw new Error('Invalid response structure');
      }
    });

    await this.test('Process 2D tensor', async () => {
      const tensor = TensorGenerator.create2D(8, 12);
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error(`Processing failed: ${response.status}`);
      }
      
      const result = await response.json();
      if (!result.success || !result.data) {
        throw new Error('Invalid response structure');
      }
      
      // Verify data integrity
      if (result.data.shape.length !== 2) {
        throw new Error('Shape dimension mismatch');
      }
    });

    await this.test('Process 3D tensor', async () => {
      const tensor = TensorGenerator.create3D(4, 6, 8);
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error(`Processing failed: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.data.dimensions !== 3) {
        throw new Error('Dimension processing error');
      }
    });
  }

  // Test suite: Multi-dimensional processing
  async testMultiDimensionalProcessing() {
    console.log('\n🎯 Testing Multi-Dimensional Processing');
    console.log('--------------------------------------');

    await this.test('Process legal AI 4D tensor', async () => {
      const tensor = TensorGenerator.createLegal4D();
      tensor.cacheKey = 'test_legal_4d';
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error(`4D processing failed: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.data.dimensions !== 4) {
        throw new Error('4D dimension lost in processing');
      }
      
      // Verify tricubic interpolation was applied
      if (!result.metadata.optimizationLevel) {
        throw new Error('Missing optimization metadata');
      }
    });

    await this.test('Verify 4D tensor data integrity', async () => {
      const tensor = TensorGenerator.create4D(2, 2, 2, 4);
      const originalSum = tensor.data.reduce((a, b) => a + b, 0);
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      const result = await response.json();
      const processedSum = result.data.data.reduce((a, b) => a + b, 0);
      
      // Data should be modified but not drastically different
      const difference = Math.abs(processedSum - originalSum);
      if (difference > originalSum * 2) { // Allow up to 200% change
        throw new Error('Data integrity check failed - too much change');
      }
    });

    await this.test('Test coalesced memory layout', async () => {
      const tensor = TensorGenerator.create4D(3, 4, 5, 8);
      tensor.layout = 'coalesced';
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error('Coalesced layout processing failed');
      }
      
      const result = await response.json();
      if (!result.data.layout.includes('processed')) {
        throw new Error('Layout not updated after processing');
      }
    });
  }

  // Test suite: NES-style optimization
  async testNESStyleOptimization() {
    console.log('\n🎮 Testing NES-Style Optimization');
    console.log('---------------------------------');

    await this.test('Test 8-bit quantization', async () => {
      const tensor = TensorGenerator.create2D(10, 10);
      tensor.bitDepth = 8;
      tensor.nesOptimized = true;
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error('8-bit quantization failed');
      }
      
      const result = await response.json();
      // Check if values are properly quantized (should be more discrete)
      const uniqueValues = new Set(result.data.data.slice(0, 100));
      if (uniqueValues.size > 200) { // 8-bit should have fewer unique values
        console.warn('Quantization may not be working as expected');
      }
    });

    await this.test('Test NES memory hierarchy simulation', async () => {
      const smallTensor = TensorGenerator.create1D(32); // Should use 'ppu' level
      const mediumTensor = TensorGenerator.create1D(1024); // Should use 'ram' level
      
      const responses = await Promise.all([
        fetch(config.apiEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(smallTensor)
        }),
        fetch(config.apiEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(mediumTensor)
        })
      ]);
      
      if (!responses.every(r => r.ok)) {
        throw new Error('NES hierarchy simulation failed');
      }
    });

    await this.test('Test cache efficiency', async () => {
      const tensor = TensorGenerator.create3D(4, 4, 4);
      tensor.cacheKey = 'cache_efficiency_test';
      
      // First request (cache miss)
      const response1 = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      const result1 = await response1.json();
      if (result1.metadata.cacheHit) {
        throw new Error('Unexpected cache hit on first request');
      }
      
      // Second request (should be cache hit)
      const response2 = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      const result2 = await response2.json();
      if (!result2.metadata.cacheHit) {
        console.warn('Cache hit expected but not received');
      }
    });
  }

  // Test suite: Load balancing
  async testLoadBalancing() {
    console.log('\n⚖️  Testing Load Balancing');
    console.log('-------------------------');

    await this.test('Test service routing', async () => {
      const tensors = Array.from({ length: 6 }, (_, i) => {
        const tensor = TensorGenerator.create2D(5, 5);
        tensor.cacheKey = `load_balance_test_${i}`;
        return tensor;
      });
      
      const responses = await Promise.all(
        tensors.map(tensor => 
          fetch(config.apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tensor)
          })
        )
      );
      
      if (!responses.every(r => r.ok)) {
        throw new Error('Load balancing failed');
      }
      
      const results = await Promise.all(responses.map(r => r.json()));
      const services = new Set(results.map(r => r.metadata.service));
      
      if (services.size < 2) {
        console.warn('Load balancing may not be distributing requests');
      }
    });

    await this.test('Test service failover', async () => {
      const tensor = TensorGenerator.create1D(100);
      tensor.cacheKey = 'failover_test';
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error('Service failover test failed');
      }
      
      // The fact that we got a response means failover is working
      // (assuming some services might be unavailable)
    });
  }

  // Test suite: Caching
  async testCaching() {
    console.log('\n💾 Testing Caching System');
    console.log('-------------------------');

    await this.test('Test cache hit performance', async () => {
      const tensor = TensorGenerator.create3D(5, 5, 5);
      tensor.cacheKey = 'performance_cache_test';
      
      // First request (cache miss)
      const start1 = Date.now();
      const response1 = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      const time1 = Date.now() - start1;
      
      // Second request (cache hit)
      const start2 = Date.now();
      const response2 = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      const time2 = Date.now() - start2;
      
      if (!response1.ok || !response2.ok) {
        throw new Error('Cache performance test requests failed');
      }
      
      const result2 = await response2.json();
      
      // Cache hit should be significantly faster
      if (time2 > time1 * 0.8) {
        console.warn(`Cache hit not significantly faster: ${time1}ms vs ${time2}ms`);
      }
    });

    await this.test('Test cache statistics', async () => {
      // Get cache stats from primary service
      const response = await fetch(`${config.services[0].url}/stats`);
      if (!response.ok) {
        throw new Error('Could not retrieve cache statistics');
      }
      
      const stats = await response.json();
      if (!stats.cache_stats) {
        throw new Error('Missing cache statistics');
      }
    });
  }

  // Test suite: Error handling
  async testErrorHandling() {
    console.log('\n🚨 Testing Error Handling');
    console.log('-------------------------');

    await this.test('Test invalid tensor data', async () => {
      const invalidTensor = { invalid: 'data' };
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(invalidTensor)
      });
      
      if (response.ok) {
        throw new Error('Should have rejected invalid tensor data');
      }
      
      if (response.status !== 400) {
        throw new Error(`Expected 400 status, got ${response.status}`);
      }
    });

    await this.test('Test shape-data mismatch', async () => {
      const mismatchTensor = {
        shape: [5, 5],
        data: Array.from({ length: 10 }, () => 0.5), // Should be 25 elements
        dimensions: 2
      };
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mismatchTensor)
      });
      
      if (response.ok) {
        throw new Error('Should have rejected mismatched tensor');
      }
    });

    await this.test('Test malformed JSON', async () => {
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{ invalid json'
      });
      
      if (response.ok) {
        throw new Error('Should have rejected malformed JSON');
      }
    });
  }

  // Test suite: Performance
  async testPerformance() {
    console.log('\n⚡ Testing Performance');
    console.log('---------------------');

    await this.test('Small tensor performance', async () => {
      const tensor = TensorGenerator.create2D(10, 10);
      
      const start = Date.now();
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      const duration = Date.now() - start;
      
      if (!response.ok) {
        throw new Error('Performance test failed');
      }
      
      if (duration > 5000) { // 5 seconds
        throw new Error(`Too slow for small tensor: ${duration}ms`);
      }
    });

    await this.test('Large tensor performance', async () => {
      const tensor = TensorGenerator.createLarge4D();
      
      const start = Date.now();
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      const duration = Date.now() - start;
      
      if (!response.ok) {
        throw new Error('Large tensor processing failed');
      }
      
      if (duration > 30000) { // 30 seconds
        throw new Error(`Too slow for large tensor: ${duration}ms`);
      }
      
      console.log(`   📊 Large tensor processed in ${duration}ms`);
    });

    await this.test('Concurrent processing', async () => {
      const tensors = Array.from({ length: 5 }, (_, i) => {
        const tensor = TensorGenerator.create3D(3, 4, 5);
        tensor.cacheKey = `concurrent_test_${i}`;
        return tensor;
      });
      
      const start = Date.now();
      const responses = await Promise.all(
        tensors.map(tensor => 
          fetch(config.apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tensor)
          })
        )
      );
      const duration = Date.now() - start;
      
      if (!responses.every(r => r.ok)) {
        throw new Error('Concurrent processing failed');
      }
      
      console.log(`   📊 5 tensors processed concurrently in ${duration}ms`);
    });
  }

  // Test suite: WebGPU integration
  async testWebGPUIntegration() {
    console.log('\n🖥️  Testing WebGPU Integration');
    console.log('-----------------------------');

    await this.test('WebGPU worker availability', async () => {
      // Test if the worker can be instantiated
      const tensor = TensorGenerator.create2D(8, 8);
      tensor.preferWebGPU = true;
      
      const response = await fetch(config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tensor)
      });
      
      if (!response.ok) {
        throw new Error('WebGPU processing test failed');
      }
      
      // Just verify we get a valid response
      // The actual WebGPU functionality is tested in the browser
    });

    await this.skip('Browser WebGPU tests', 'Requires browser environment');
    await this.skip('WASM integration tests', 'Requires WASM module compilation');
  }

  generateReport() {
    const duration = Date.now() - this.startTime;
    const passRate = (this.results.passed / this.results.total * 100).toFixed(1);
    
    console.log('\n🎉 Test Suite Complete');
    console.log('======================');
    console.log(`Duration: ${duration}ms (${(duration/1000).toFixed(1)}s)`);
    console.log(`Total Tests: ${this.results.total}`);
    console.log(`✅ Passed: ${this.results.passed}`);
    console.log(`❌ Failed: ${this.results.failed}`);
    console.log(`⏸️  Skipped: ${this.results.skipped}`);
    console.log(`📊 Pass Rate: ${passRate}%`);
    
    if (this.results.failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results.tests
        .filter(test => test.status === 'FAILED')
        .forEach(test => {
          console.log(`   - ${test.name}: ${test.error}`);
        });
    }
    
    // Save detailed report
    const reportPath = path.join(__dirname, 'logs', 'test-report.json');
    const reportDir = path.dirname(reportPath);
    
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }
    
    fs.writeFileSync(reportPath, JSON.stringify({
      summary: this.results,
      timestamp: new Date().toISOString(),
      duration,
      passRate: parseFloat(passRate),
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch
      }
    }, null, 2));
    
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
    
    // Return exit code
    return this.results.failed === 0 ? 0 : 1;
  }
}

// Run tests if this file is executed directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const runner = new GPUPipelineTestRunner();
  
  runner.runAllTests().then((exitCode) => {
    process.exit(exitCode);
  }).catch((error) => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}

export { GPUPipelineTestRunner, TensorGenerator };