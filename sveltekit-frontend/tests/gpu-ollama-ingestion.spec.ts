import { test, expect } from '@playwright/test';

test.describe('GPU-Enabled Ollama Ingestion Pipeline', () => {
  test.beforeEach(async ({ page }) => {
    // Ensure Ollama is running with GPU
    const gpuStatus = await page.request.get('/api/ollama/gpu-status');
    const status = await gpuStatus.json();
    
    if (!status.gpu_enabled) {
      test.skip(); // Skip if GPU is not available
    }
    
    // Login to access ingestion features
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
  });

  test('should verify GPU-enabled Ollama is running', async ({ page }) => {
    // Start Ollama with GPU if not already running
    const startResponse = await page.request.post('/api/ollama/start-gpu');
    expect([200, 409]).toContain(startResponse.status()); // 409 if already running
    
    // Verify GPU status
    const statusResponse = await page.request.get('/api/ollama/gpu-status');
    expect(statusResponse.status()).toBe(200);
    
    const status = await statusResponse.json();
    expect(status).toMatchObject({
      gpu_enabled: true,
      cuda_available: true,
      gpu_layers: expect.any(Number),
      gpu_memory_used: expect.any(Number),
      gpu_memory_total: expect.any(Number)
    });
    
    expect(status.gpu_layers).toBeGreaterThan(0);
    
    // Verify models are loaded with GPU
    const modelsResponse = await page.request.get('/api/ollama/models');
    const models = await modelsResponse.json();
    
    const requiredModels = ['llama3.2', 'nomic-embed-text'];
    requiredModels.forEach(modelName => {
      const model = models.models.find((m: any) => m.name.includes(modelName));
      expect(model).toBeDefined();
      if (model && model.details) {
        expect(model.details.gpu_layers).toBeGreaterThan(0);
      }
    });
  });

  test('should ingest documents with GPU acceleration', async ({ page }) => {
    await page.goto('/dashboard/documents/ingest');
    
    // Upload multiple documents for bulk ingestion
    const documents = [
      { name: 'contract1.pdf', content: 'This is a contract for services between parties A and B. The terms include payment, deliverables, and termination clauses.' },
      { name: 'case_law.pdf', content: 'Supreme Court ruling on intellectual property rights. The court found that the defendant violated patent protections.' },
      { name: 'legal_brief.pdf', content: 'Legal brief arguing for the plaintiff in a negligence case. The argument focuses on duty of care and causation.' }
    ];
    
    // Simulate file uploads
    for (const doc of documents) {
      await page.evaluate((docData) => {
        const file = new File([docData.content], docData.name, { type: 'application/pdf' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        const input = document.querySelector('input[type="file"]') as HTMLInputElement;
        if (input) {
          const currentFiles = Array.from(input.files || []);
          const newFiles = [...currentFiles, file];
          
          const newDataTransfer = new DataTransfer();
          newFiles.forEach(f => newDataTransfer.items.add(f));
          input.files = newDataTransfer.files;
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }, doc);
    }
    
    // Start GPU-accelerated ingestion
    await page.check('[data-testid="gpu-acceleration-enabled"]');
    await page.click('[data-testid="start-ingestion"]');
    
    // Monitor ingestion progress
    await page.waitForSelector('[data-testid="ingestion-status"]');
    
    const ingestionPhases = [
      'initializing',
      'parsing',
      'chunking',
      'embedding',
      'indexing',
      'complete'
    ];
    
    // Wait for each phase
    for (const phase of ingestionPhases) {
      await page.waitForFunction(
        (expectedPhase) => {
          const status = document.querySelector('[data-testid="ingestion-status"]');
          return status?.getAttribute('data-phase') === expectedPhase;
        },
        phase,
        { timeout: 60000 }
      );
      
      // Check GPU utilization during processing
      const gpuMetrics = await page.request.get('/api/gpu/metrics');
      const metrics = await gpuMetrics.json();
      
      if (phase === 'embedding') {
        expect(metrics.gpu_utilization).toBeGreaterThan(50); // Should be high during embedding
        expect(metrics.memory_used).toBeGreaterThan(0);
      }
    }
    
    // Verify ingestion completed successfully
    const completionStatus = await page.locator('[data-testid="ingestion-results"]').textContent();
    expect(completionStatus).toContain('3 documents processed');
    expect(completionStatus).toMatch(/\d+ chunks created/);
    expect(completionStatus).toMatch(/\d+ embeddings generated/);
  });

  test('should compare GPU vs CPU embedding performance', async ({ page }) => {
    const testText = 'This is a comprehensive legal document about contract law, tort liability, and criminal procedure that needs to be processed for semantic search.';
    
    // Test CPU embedding
    console.log('Testing CPU embedding performance...');
    const cpuStartTime = Date.now();
    
    const cpuResponse = await page.request.post('/api/embeddings/generate', {
      data: {
        text: testText,
        use_gpu: false,
        model: 'nomic-embed-text'
      }
    });
    
    const cpuEndTime = Date.now();
    const cpuTime = cpuEndTime - cpuStartTime;
    
    expect(cpuResponse.status()).toBe(200);
    const cpuResult = await cpuResponse.json();
    expect(cpuResult.embedding).toBeDefined();
    expect(cpuResult.embedding.length).toBe(768);
    
    // Test GPU embedding
    console.log('Testing GPU embedding performance...');
    const gpuStartTime = Date.now();
    
    const gpuResponse = await page.request.post('/api/embeddings/generate', {
      data: {
        text: testText,
        use_gpu: true,
        model: 'nomic-embed-text'
      }
    });
    
    const gpuEndTime = Date.now();
    const gpuTime = gpuEndTime - gpuStartTime;
    
    expect(gpuResponse.status()).toBe(200);
    const gpuResult = await gpuResponse.json();
    expect(gpuResult.embedding).toBeDefined();
    expect(gpuResult.embedding.length).toBe(768);
    
    // Compare performance
    console.log(`CPU Embedding Time: ${cpuTime}ms`);
    console.log(`GPU Embedding Time: ${gpuTime}ms`);
    console.log(`GPU Speedup: ${(cpuTime / gpuTime).toFixed(2)}x`);
    
    // GPU should be faster
    expect(gpuTime).toBeLessThan(cpuTime);
    
    // Store benchmark results
    await page.request.post('/api/benchmarks/store', {
      data: {
        test_name: 'embedding_gpu_vs_cpu',
        cpu_time_ms: cpuTime,
        gpu_time_ms: gpuTime,
        speedup_factor: cpuTime / gpuTime,
        model: 'nomic-embed-text',
        text_length: testText.length
      }
    });
  });

  test('should handle batch document processing with GPU', async ({ page }) => {
    await page.goto('/dashboard/ingestion/batch');
    
    // Create a batch of documents
    const batchSize = 20;
    const documents = Array(batchSize).fill(null).map((_, i) => ({
      title: `Batch Document ${i + 1}`,
      content: `This is legal document number ${i + 1} containing information about various legal topics including contracts, torts, and criminal law. Document ${i + 1} has specific clauses and provisions that need to be analyzed.`,
      type: 'legal_document'
    }));
    
    // Submit batch for processing
    const batchResponse = await page.request.post('/api/ingestion/batch', {
      data: {
        documents: documents,
        use_gpu: true,
        batch_size: 5, // Process 5 at a time
        priority: 'high'
      }
    });
    
    expect(batchResponse.status()).toBe(202); // Accepted for processing
    const batch = await batchResponse.json();
    expect(batch).toHaveProperty('batch_id');
    
    const batchId = batch.batch_id;
    
    // Monitor batch progress
    let batchComplete = false;
    let attempts = 0;
    const maxAttempts = 30; // 5 minutes max
    
    while (!batchComplete && attempts < maxAttempts) {
      await page.waitForTimeout(10000); // Wait 10 seconds
      
      const statusResponse = await page.request.get(`/api/ingestion/batch/${batchId}/status`);
      const status = await statusResponse.json();
      
      console.log(`Batch Progress: ${status.processed}/${status.total} documents`);
      
      expect(status).toMatchObject({
        batch_id: batchId,
        status: expect.stringMatching(/processing|completed|failed/),
        processed: expect.any(Number),
        total: batchSize,
        gpu_enabled: true
      });
      
      if (status.status === 'completed') {
        batchComplete = true;
        expect(status.processed).toBe(batchSize);
        expect(status.failed).toBe(0);
      } else if (status.status === 'failed') {
        throw new Error(`Batch processing failed: ${status.error}`);
      }
      
      attempts++;
    }
    
    expect(batchComplete).toBe(true);
    
    // Verify all documents were indexed
    const searchResponse = await page.request.post('/api/search/semantic', {
      data: {
        query: 'legal document contracts',
        limit: 25
      }
    });
    
    const searchResults = await searchResponse.json();
    const batchDocuments = searchResults.results.filter((r: any) => 
      r.metadata.title && r.metadata.title.includes('Batch Document')
    );
    
    expect(batchDocuments.length).toBeGreaterThan(0);
  });

  test('should monitor GPU memory during ingestion', async ({ page }) => {
    // Get baseline GPU memory
    const baselineResponse = await page.request.get('/api/gpu/memory');
    const baseline = await baselineResponse.json();
    
    console.log(`Baseline GPU Memory: ${baseline.used_mb}MB / ${baseline.total_mb}MB`);
    
    // Start memory-intensive ingestion
    const largeDocument = {
      title: 'Large Legal Compendium',
      content: Array(100).fill(null).map((_, i) => 
        `Section ${i + 1}: This section contains detailed legal analysis of various topics including contract formation, tort liability, criminal procedure, and constitutional law. Each section builds upon previous concepts and provides comprehensive coverage of the subject matter.`
      ).join('\n\n')
    };
    
    const ingestionResponse = await page.request.post('/api/ingestion/document', {
      data: {
        document: largeDocument,
        use_gpu: true,
        chunk_size: 1000,
        chunk_overlap: 200
      }
    });
    
    expect(ingestionResponse.status()).toBe(202);
    const { ingestion_id } = await ingestionResponse.json();
    
    // Monitor memory usage during processing
    const memoryReadings = [];
    let processing = true;
    
    while (processing) {
      await page.waitForTimeout(2000);
      
      // Get current memory usage
      const memoryResponse = await page.request.get('/api/gpu/memory');
      const memory = await memoryResponse.json();
      
      memoryReadings.push({
        timestamp: Date.now(),
        used_mb: memory.used_mb,
        utilization: memory.utilization_percent
      });
      
      // Check if processing is complete
      const statusResponse = await page.request.get(`/api/ingestion/${ingestion_id}/status`);
      const status = await statusResponse.json();
      
      processing = status.status === 'processing';
      
      // Safety timeout
      if (memoryReadings.length > 60) { // 2 minutes max
        break;
      }
    }
    
    // Analyze memory usage
    const maxMemoryUsed = Math.max(...memoryReadings.map(r => r.used_mb));
    const avgMemoryUsed = memoryReadings.reduce((sum, r) => sum + r.used_mb, 0) / memoryReadings.length;
    const memoryIncrease = maxMemoryUsed - baseline.used_mb;
    
    console.log(`Max Memory Used: ${maxMemoryUsed}MB`);
    console.log(`Average Memory Used: ${avgMemoryUsed.toFixed(2)}MB`);
    console.log(`Memory Increase: ${memoryIncrease}MB`);
    
    // Memory usage should be reasonable
    expect(memoryIncrease).toBeLessThan(4000); // Less than 4GB increase
    expect(maxMemoryUsed).toBeLessThan(baseline.total_mb * 0.9); // Don't use more than 90%
  });

  test('should handle model switching during ingestion', async ({ page }) => {
    await page.goto('/dashboard/ingestion');
    
    // Check available models
    const modelsResponse = await page.request.get('/api/ollama/models');
    const models = await modelsResponse.json();
    
    const embeddingModels = models.models.filter((m: any) => 
      m.name.includes('embed') || m.capabilities?.includes('embedding')
    );
    
    expect(embeddingModels.length).toBeGreaterThan(0);
    
    // Test with different embedding models
    for (const model of embeddingModels.slice(0, 2)) { // Test first 2 models
      console.log(`Testing ingestion with model: ${model.name}`);
      
      const testDoc = {
        title: `Model Test - ${model.name}`,
        content: `Testing document ingestion with embedding model ${model.name}. This document contains legal content about contract law and tort liability.`
      };
      
      const ingestionResponse = await page.request.post('/api/ingestion/document', {
        data: {
          document: testDoc,
          embedding_model: model.name,
          use_gpu: true
        }
      });
      
      expect(ingestionResponse.status()).toBe(202);
      const { ingestion_id } = await ingestionResponse.json();
      
      // Wait for completion
      let complete = false;
      let attempts = 0;
      
      while (!complete && attempts < 20) {
        await page.waitForTimeout(3000);
        
        const statusResponse = await page.request.get(`/api/ingestion/${ingestion_id}/status`);
        const status = await statusResponse.json();
        
        if (status.status === 'completed') {
          complete = true;
          expect(status.embedding_model).toBe(model.name);
          expect(status.gpu_used).toBe(true);
        } else if (status.status === 'failed') {
          throw new Error(`Ingestion failed with model ${model.name}: ${status.error}`);
        }
        
        attempts++;
      }
      
      expect(complete).toBe(true);
    }
  });

  test('should validate embedding quality with GPU acceleration', async ({ page }) => {
    const testQueries = [
      'contract formation requirements',
      'negligence in tort law',
      'criminal procedure constitutional rights',
      'intellectual property patent law'
    ];
    
    // Generate embeddings for test queries with GPU
    const embeddings = [];
    
    for (const query of testQueries) {
      const response = await page.request.post('/api/embeddings/generate', {
        data: {
          text: query,
          use_gpu: true,
          model: 'nomic-embed-text'
        }
      });
      
      expect(response.status()).toBe(200);
      const result = await response.json();
      
      embeddings.push({
        query,
        embedding: result.embedding,
        generation_time: result.generation_time_ms
      });
    }
    
    // Validate embedding properties
    embeddings.forEach((item, index) => {
      // Check dimensions
      expect(item.embedding.length).toBe(768);
      
      // Check values are in reasonable range
      const values = item.embedding;
      const min = Math.min(...values);
      const max = Math.max(...values);
      expect(min).toBeGreaterThan(-2);
      expect(max).toBeLessThan(2);
      
      // Check generation time is fast with GPU
      expect(item.generation_time).toBeLessThan(1000); // Less than 1 second
      
      console.log(`Query "${item.query}": ${item.generation_time}ms`);
    });
    
    // Test similarity between related queries
    const contractEmbedding = embeddings.find(e => e.query.includes('contract'))?.embedding;
    const tortEmbedding = embeddings.find(e => e.query.includes('tort'))?.embedding;
    
    if (contractEmbedding && tortEmbedding) {
      const similarity = cosineSimilarity(contractEmbedding, tortEmbedding);
      console.log(`Contract-Tort similarity: ${similarity.toFixed(3)}`);
      
      // Should have some similarity but not too high
      expect(similarity).toBeGreaterThan(0.1);
      expect(similarity).toBeLessThan(0.9);
    }
  });

  test('should handle ingestion pipeline errors gracefully', async ({ page }) => {
    // Test with invalid document
    const invalidDoc = {
      title: '', // Empty title should cause validation error
      content: '', // Empty content
      type: 'invalid'
    };
    
    const response = await page.request.post('/api/ingestion/document', {
      data: {
        document: invalidDoc,
        use_gpu: true
      }
    });
    
    expect(response.status()).toBe(400);
    const error = await response.json();
    expect(error.error).toContain('validation');
    
    // Test GPU memory overflow handling
    const hugeDoc = {
      title: 'Memory Overflow Test',
      content: 'x'.repeat(100000000) // 100MB of text
    };
    
    const memoryTestResponse = await page.request.post('/api/ingestion/document', {
      data: {
        document: hugeDoc,
        use_gpu: true
      }
    });
    
    // Should either succeed or fail gracefully
    if (memoryTestResponse.status() === 507) { // Insufficient Storage
      const memoryError = await memoryTestResponse.json();
      expect(memoryError.error).toContain('memory');
      expect(memoryError.fallback).toBe('cpu');
    } else {
      expect(memoryTestResponse.status()).toBe(202);
    }
  });
});

// Helper function for cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}