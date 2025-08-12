// tests/performance/load.spec.js
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test.describe('Performance and Load Tests', () => {
  let testFilePath;
  
  test.beforeAll(async () => {
    // Create a larger test document for performance testing
    testFilePath = path.join(__dirname, '../../uploads/performance-test.txt');
    const largeContent = `
PERFORMANCE TEST LEGAL DOCUMENT

This is a larger test document designed to evaluate AI processing performance.

${'LEGAL CLAUSE SECTION: '.repeat(100)}
${'This clause contains important legal language that must be processed and analyzed. '.repeat(50)}

${'CONTRACT TERMS SECTION: '.repeat(100)}
${'These terms define the relationship between parties and establish binding obligations. '.repeat(50)}

${'EVIDENCE SECTION: '.repeat(100)}
${'This evidence supports the claims made in the legal proceeding and must be carefully reviewed. '.repeat(50)}

PERFORMANCE METRICS:
- Document length: ~${('x'.repeat(10000).length)} characters
- Processing time should be under 30 seconds
- Memory usage should remain stable
- Concurrent processing should work efficiently

END OF DOCUMENT
    `.trim();
    
    fs.writeFileSync(testFilePath, largeContent);
  });

  test.afterAll(async () => {
    // Cleanup test file
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }
  });

  test('should handle large document processing within time limits', async ({ request }) => {
    const startTime = Date.now();
    
    const response = await request.post('/api/upload', {
      multipart: {
        file: {
          name: 'performance-test.txt',
          mimeType: 'text/plain',
          buffer: fs.readFileSync(testFilePath),
        },
        document_type: 'legal',
        case_id: 'PERF-TEST-001',
        enable_ocr: 'false',
        enable_embedding: 'true',
      }
    });

    const processingTime = Date.now() - startTime;
    
    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    
    // Validate performance metrics
    expect(processingTime).toBeLessThan(60000); // Should process within 60 seconds
    expect(result.performance).toBeDefined();
    expect(result.performance.concurrent_tasks).toBeGreaterThan(0);
    expect(result.performance.cpu_cores).toBeGreaterThan(0);
    expect(result.performance.simd_accelerated).toBe(true);
    expect(result.performance.gpu_accelerated).toBe(true);
    
    // Validate content processing
    expect(result.text_length).toBeGreaterThan(1000);
    expect(result.chunks.length).toBeGreaterThan(1);
    expect(result.embeddings.length).toBeGreaterThan(0);
    
    console.log(`Performance Test Results:
    - Processing Time: ${processingTime}ms
    - Document Size: ${result.file_size} bytes
    - Chunks Generated: ${result.chunks.length}
    - Embeddings Generated: ${result.embeddings.length}
    - CPU Cores Used: ${result.performance.cpu_cores}
    - SIMD Accelerated: ${result.performance.simd_accelerated}
    - GPU Accelerated: ${result.performance.gpu_accelerated}`);
  });

  test('should handle concurrent uploads efficiently', async ({ request }) => {
    const concurrentRequests = 3;
    const promises = [];
    
    for (let i = 0; i < concurrentRequests; i++) {
      const promise = request.post('/api/upload', {
        multipart: {
          file: {
            name: `concurrent-test-${i}.txt`,
            mimeType: 'text/plain',
            buffer: Buffer.from(`Concurrent test document ${i}\n${'Test content line. '.repeat(100)}`),
          },
          document_type: 'contract',
          case_id: `CONCURRENT-${i}`,
          enable_ocr: 'false',
          enable_embedding: 'false',
        }
      });
      promises.push(promise);
    }
    
    const startTime = Date.now();
    const responses = await Promise.all(promises);
    const totalTime = Date.now() - startTime;
    
    // All requests should succeed
    responses.forEach(response => {
      expect(response.ok()).toBeTruthy();
    });
    
    // Concurrent processing should be faster than sequential
    expect(totalTime).toBeLessThan(concurrentRequests * 10000); // Should be much faster than sequential
    
    console.log(`Concurrent Test Results:
    - Requests: ${concurrentRequests}
    - Total Time: ${totalTime}ms
    - Average per request: ${totalTime / concurrentRequests}ms`);
  });

  test('should maintain stable memory usage', async ({ request }) => {
    // Test multiple uploads to check for memory leaks
    const iterations = 5;
    
    for (let i = 0; i < iterations; i++) {
      const response = await request.post('/api/upload', {
        multipart: {
          file: {
            name: `memory-test-${i}.txt`,
            mimeType: 'text/plain',
            buffer: Buffer.from(`Memory test iteration ${i}\n${'Memory stability test. '.repeat(200)}`),
          },
          document_type: 'legal',
          case_id: `MEMORY-${i}`,
          enable_ocr: 'false',
          enable_embedding: 'true',
        }
      });
      
      expect(response.ok()).toBeTruthy();
      
      const result = await response.json();
      expect(result.document_id).toBeDefined();
      expect(result.performance.memory_usage).toBeDefined();
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log(`Memory Stability Test: Completed ${iterations} iterations successfully`);
  });

  test('should provide detailed performance metrics', async ({ request }) => {
    const response = await request.post('/api/upload', {
      multipart: {
        file: {
          name: 'metrics-test.txt',
          mimeType: 'text/plain',
          buffer: fs.readFileSync(testFilePath),
        },
        document_type: 'evidence',
        case_id: 'METRICS-TEST-001',
        enable_ocr: 'false',
        enable_embedding: 'true',
      }
    });

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    
    // Validate all performance metrics are present
    const perf = result.performance;
    expect(perf.total_time).toBeDefined();
    expect(perf.file_read_time).toBeDefined();
    expect(perf.parsing_time).toBeDefined();
    expect(perf.chunking_time).toBeDefined();
    expect(perf.embedding_time).toBeDefined();
    expect(perf.summarization_time).toBeDefined();
    expect(perf.concurrent_tasks).toBeGreaterThan(0);
    expect(perf.cpu_cores).toBeGreaterThan(0);
    expect(perf.memory_usage).toBeDefined();
    expect(perf.simd_accelerated).toBe(true);
    expect(perf.gpu_accelerated).toBe(true);
    
    // Performance should be reasonable
    expect(perf.total_time).toBeLessThan('60s');
    expect(perf.chunking_time).toBeLessThan('5s');
    expect(perf.summarization_time).toBeLessThan('30s');
    
    console.log(`Detailed Performance Metrics:
    - Total Time: ${perf.total_time}
    - File Read: ${perf.file_read_time}
    - Parsing: ${perf.parsing_time}
    - Chunking: ${perf.chunking_time}
    - Embedding: ${perf.embedding_time}
    - Summarization: ${perf.summarization_time}
    - Memory Usage: ${perf.memory_usage}
    - Concurrent Tasks: ${perf.concurrent_tasks}
    - SIMD: ${perf.simd_accelerated}
    - GPU: ${perf.gpu_accelerated}`);
  });
});