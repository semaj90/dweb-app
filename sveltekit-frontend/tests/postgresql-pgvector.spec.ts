import { test, expect } from '@playwright/test';

test.describe('PostgreSQL and pgvector Operations', () => {
  test('should verify PostgreSQL connection', async ({ page }) => {
    const response = await page.request.get('/api/db/health');
    expect(response.status()).toBe(200);
    
    const health = await response.json();
    expect(health).toHaveProperty('database');
    expect(health.database.connected).toBe(true);
    expect(health.database.version).toMatch(/PostgreSQL \d+\.\d+/);
  });

  test('should verify pgvector extension is installed', async ({ page }) => {
    const response = await page.request.get('/api/db/extensions');
    expect(response.status()).toBe(200);
    
    const extensions = await response.json();
    const pgvector = extensions.find((ext: any) => ext.name === 'vector');
    
    expect(pgvector).toBeDefined();
    expect(pgvector.installed).toBe(true);
    expect(pgvector.version).toBeDefined();
  });

  test('should create and query vector embeddings', async ({ page }) => {
    // Create a test embedding
    const testData = {
      content: 'Test legal document for vector search',
      embedding: Array(768).fill(0).map(() => Math.random()),
      metadata: {
        type: 'test_document',
        created_at: new Date().toISOString()
      }
    };
    
    // Store the embedding
    const createResponse = await page.request.post('/api/vectors/store', {
      data: testData
    });
    
    expect(createResponse.status()).toBe(201);
    const created = await createResponse.json();
    expect(created).toHaveProperty('id');
    
    // Query similar vectors
    const queryVector = Array(768).fill(0).map(() => Math.random());
    const searchResponse = await page.request.post('/api/vectors/search', {
      data: {
        vector: queryVector,
        limit: 5,
        threshold: 0.5
      }
    });
    
    expect(searchResponse.status()).toBe(200);
    const results = await searchResponse.json();
    
    expect(results).toHaveProperty('matches');
    expect(Array.isArray(results.matches)).toBe(true);
    
    results.matches.forEach((match: any) => {
      expect(match).toHaveProperty('id');
      expect(match).toHaveProperty('distance');
      expect(match).toHaveProperty('content');
      expect(match).toHaveProperty('metadata');
    });
  });

  test('should perform cosine similarity search', async ({ page }) => {
    // Create multiple test embeddings
    const testDocuments = [
      {
        content: 'Contract law and agreements',
        category: 'contract'
      },
      {
        content: 'Criminal law and procedures',
        category: 'criminal'
      },
      {
        content: 'Intellectual property and patents',
        category: 'ip'
      }
    ];
    
    // Generate and store embeddings for each document
    for (const doc of testDocuments) {
      const embeddingResponse = await page.request.post('/api/ai/embeddings', {
        data: { text: doc.content }
      });
      
      const { embedding } = await embeddingResponse.json();
      
      await page.request.post('/api/vectors/store', {
        data: {
          content: doc.content,
          embedding: embedding,
          metadata: { category: doc.category }
        }
      });
    }
    
    // Search for contract-related content
    const queryResponse = await page.request.post('/api/ai/embeddings', {
      data: { text: 'legal agreements and contracts' }
    });
    
    const { embedding: queryEmbedding } = await queryResponse.json();
    
    const searchResponse = await page.request.post('/api/vectors/cosine-search', {
      data: {
        vector: queryEmbedding,
        limit: 3
      }
    });
    
    expect(searchResponse.status()).toBe(200);
    const results = await searchResponse.json();
    
    // The contract document should be the most similar
    expect(results.matches[0].metadata.category).toBe('contract');
    expect(results.matches[0].similarity).toBeGreaterThan(0.8);
  });

  test('should handle vector dimension validation', async ({ page }) => {
    // Try to store vector with wrong dimensions
    const wrongDimensionVector = Array(512).fill(0.5); // Wrong size
    
    const response = await page.request.post('/api/vectors/store', {
      data: {
        content: 'Test document',
        embedding: wrongDimensionVector,
        metadata: {}
      }
    });
    
    expect(response.status()).toBe(400);
    const error = await response.json();
    expect(error.error).toContain('dimension');
  });

  test('should perform efficient batch vector operations', async ({ page }) => {
    // Create batch of vectors
    const batchSize = 100;
    const vectors = Array(batchSize).fill(null).map((_, i) => ({
      content: `Document ${i}`,
      embedding: Array(768).fill(0).map(() => Math.random()),
      metadata: { batch_id: 'test_batch', index: i }
    }));
    
    const startTime = Date.now();
    
    const response = await page.request.post('/api/vectors/batch-store', {
      data: { vectors }
    });
    
    const endTime = Date.now();
    
    expect(response.status()).toBe(201);
    const result = await response.json();
    
    expect(result.stored).toBe(batchSize);
    expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    
    // Verify batch query performance
    const queryStart = Date.now();
    
    const queryResponse = await page.request.post('/api/vectors/search', {
      data: {
        vector: Array(768).fill(0.5),
        limit: 50,
        filters: { 'metadata.batch_id': 'test_batch' }
      }
    });
    
    const queryEnd = Date.now();
    
    expect(queryResponse.status()).toBe(200);
    expect(queryEnd - queryStart).toBeLessThan(1000); // Query should be fast
  });

  test('should update vector embeddings', async ({ page }) => {
    // Create initial vector
    const createResponse = await page.request.post('/api/vectors/store', {
      data: {
        content: 'Original content',
        embedding: Array(768).fill(0.1),
        metadata: { version: 1 }
      }
    });
    
    const { id } = await createResponse.json();
    
    // Update the vector
    const newEmbedding = Array(768).fill(0.9);
    const updateResponse = await page.request.put(`/api/vectors/${id}`, {
      data: {
        embedding: newEmbedding,
        metadata: { version: 2, updated_at: new Date().toISOString() }
      }
    });
    
    expect(updateResponse.status()).toBe(200);
    
    // Verify update
    const getResponse = await page.request.get(`/api/vectors/${id}`);
    const updated = await getResponse.json();
    
    expect(updated.metadata.version).toBe(2);
    expect(updated.metadata.updated_at).toBeDefined();
  });

  test('should delete vector embeddings', async ({ page }) => {
    // Create a vector
    const createResponse = await page.request.post('/api/vectors/store', {
      data: {
        content: 'To be deleted',
        embedding: Array(768).fill(0.5),
        metadata: { temporary: true }
      }
    });
    
    const { id } = await createResponse.json();
    
    // Delete the vector
    const deleteResponse = await page.request.delete(`/api/vectors/${id}`);
    expect(deleteResponse.status()).toBe(204);
    
    // Verify deletion
    const getResponse = await page.request.get(`/api/vectors/${id}`);
    expect(getResponse.status()).toBe(404);
  });

  test('should handle vector index operations', async ({ page }) => {
    // Check index status
    const statusResponse = await page.request.get('/api/vectors/index/status');
    expect(statusResponse.status()).toBe(200);
    
    const status = await statusResponse.json();
    expect(status).toHaveProperty('indexes');
    
    // Create or verify HNSW index exists
    const indexResponse = await page.request.post('/api/vectors/index/create', {
      data: {
        type: 'hnsw',
        parameters: {
          m: 16,
          ef_construction: 200
        }
      }
    });
    
    expect([200, 201]).toContain(indexResponse.status());
    
    // Test index performance
    const perfResponse = await page.request.get('/api/vectors/index/performance');
    expect(perfResponse.status()).toBe(200);
    
    const perf = await perfResponse.json();
    expect(perf).toHaveProperty('avg_query_time_ms');
    expect(perf.avg_query_time_ms).toBeLessThan(100); // Should be fast with index
  });

  test('should handle concurrent vector operations', async ({ page }) => {
    const operations = Array(10).fill(null).map((_, i) => ({
      id: i,
      operation: i % 2 === 0 ? 'create' : 'search'
    }));
    
    const promises = operations.map(async (op) => {
      if (op.operation === 'create') {
        return page.request.post('/api/vectors/store', {
          data: {
            content: `Concurrent doc ${op.id}`,
            embedding: Array(768).fill(Math.random()),
            metadata: { concurrent_test: true, op_id: op.id }
          }
        });
      } else {
        return page.request.post('/api/vectors/search', {
          data: {
            vector: Array(768).fill(Math.random()),
            limit: 5
          }
        });
      }
    });
    
    const results = await Promise.all(promises);
    
    // All operations should succeed
    results.forEach(response => {
      expect([200, 201]).toContain(response.status());
    });
  });

  test('should manage vector storage efficiently', async ({ page }) => {
    // Get storage stats before
    const statsBefore = await page.request.get('/api/vectors/stats');
    const { total_vectors: totalBefore, storage_size: sizeBefore } = await statsBefore.json();
    
    // Add some vectors
    const numVectors = 50;
    for (let i = 0; i < numVectors; i++) {
      await page.request.post('/api/vectors/store', {
        data: {
          content: `Storage test doc ${i}`,
          embedding: Array(768).fill(Math.random()),
          metadata: { test: 'storage' }
        }
      });
    }
    
    // Get storage stats after
    const statsAfter = await page.request.get('/api/vectors/stats');
    const { total_vectors: totalAfter, storage_size: sizeAfter } = await statsAfter.json();
    
    expect(totalAfter - totalBefore).toBe(numVectors);
    expect(sizeAfter).toBeGreaterThan(sizeBefore);
    
    // Clean up test vectors
    await page.request.post('/api/vectors/cleanup', {
      data: {
        filters: { 'metadata.test': 'storage' }
      }
    });
  });
});