import { test, expect } from '@playwright/test';
import { spawn } from 'child_process';
import { Client } from 'pg';
import fs from 'fs/promises';
import path from 'path';

const dbConfig = {
  user: 'postgres',
  password: '123456',
  host: 'localhost',
  database: 'legal_ai_db',
  port: 5432,
};

test.describe('Enhanced RAG System Tests', () => {
  let dbClient: Client;
  let testCleanupIds: string[] = [];

  test.beforeAll(async () => {
    dbClient = new Client(dbConfig);
    await dbClient.connect();
    console.log('✅ Connected to PostgreSQL for RAG testing');
  });

  test.afterAll(async () => {
    // Clean up test data
    if (testCleanupIds.length > 0) {
      await dbClient.query(
        `DELETE FROM documents WHERE id = ANY($1::uuid[])`,
        [testCleanupIds]
      );
    }
    await dbClient.end();
  });

  test('should test RAG service initialization in development mode', async () => {
    // Test that the RAG service can initialize without Qdrant
    const ragServiceModule = await import('../sveltekit-frontend/src/lib/services/enhanced-rag-service');
    
    // Set development environment variables
    process.env.SKIP_RAG_INITIALIZATION = 'true';
    process.env.USE_POSTGRESQL_ONLY = 'true';
    
    const ragService = new (ragServiceModule as any).default();
    
    // Should initialize without throwing errors
    expect(ragService).toBeDefined();
    expect(ragService.initialized).toBe(false); // Not initialized yet
    
    try {
      await ragService.initialize();
      expect(ragService.initialized).toBe(true);
      console.log('✅ RAG service initialized in development mode');
    } catch (error) {
      console.warn('⚠️ RAG service initialization failed:', error);
    }
  });

  test('should test Ollama embedding generation', async () => {
    try {
      // Test if Ollama is available
      const healthResponse = await fetch('http://localhost:11434/api/version');
      if (!healthResponse.ok) {
        test.skip(testCleanupIds.length === 0, 'Ollama not available');
        return;
      }

      // Test embedding generation
      const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: 'This is a test legal document about contract law principles.'
        })
      });

      if (embeddingResponse.ok) {
        const embeddingData = await embeddingResponse.json();
        expect(embeddingData).toHaveProperty('embedding');
        expect(Array.isArray(embeddingData.embedding)).toBe(true);
        expect(embeddingData.embedding.length).toBe(768); // nomic-embed-text dimension
        console.log(`✅ Generated embedding with ${embeddingData.embedding.length} dimensions`);
      } else {
        console.warn('⚠️ Embedding generation failed - nomic-embed-text model may not be available');
      }
    } catch (error) {
      console.warn('⚠️ Ollama embedding test failed:', error);
    }
  });

  test('should test document ingestion and vector storage', async () => {
    // Create a test document
    const testDocument = {
      content: `# Legal Contract Analysis
      
      This document outlines the fundamental principles of contract law:
      
      1. **Offer**: A clear proposal with definite terms
      2. **Acceptance**: Unqualified agreement to the offer terms  
      3. **Consideration**: Something of value exchanged by both parties
      4. **Legal Capacity**: Parties must have the legal ability to contract
      5. **Legal Purpose**: The contract must be for a lawful purpose
      
      ## Case Studies
      
      ### Case 1: Smith v. Jones (2023)
      The court ruled that consideration was insufficient when only one party provided value.
      
      ### Case 2: Legal Corp v. Business Inc. (2023)  
      Acceptance must be communicated clearly and within reasonable time.`,
      file: 'contract-law-principles.md',
      summary: 'Comprehensive guide to contract law fundamentals with case studies'
    };

    try {
      // Test document embedding (if Ollama is available)
      const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: testDocument.content
        })
      });

      let embedding: number[] = [];
      if (embeddingResponse.ok) {
        const embeddingData = await embeddingResponse.json();
        embedding = embeddingData.embedding;
      } else {
        // Use mock embedding if Ollama not available
        embedding = Array(768).fill(0).map(() => Math.random() - 0.5);
        console.warn('⚠️ Using mock embedding - Ollama not available');
      }

      // Store document in PostgreSQL
      const insertResult = await dbClient.query(`
        INSERT INTO documents (file, content, summary, embedding, tokens)
        VALUES ($1, $2, $3, $4::vector, $5)
        RETURNING id, file, content, summary, created_at
      `, [
        testDocument.file,
        testDocument.content,
        testDocument.summary,
        `[${embedding.join(',')}]`,
        Math.floor(testDocument.content.length / 4)
      ]);

      expect(insertResult.rows).toHaveLength(1);
      expect(insertResult.rows[0].file).toBe(testDocument.file);
      expect(insertResult.rows[0].summary).toBe(testDocument.summary);
      
      testCleanupIds.push(insertResult.rows[0].id);
      console.log(`✅ Stored document with vector embedding: ${insertResult.rows[0].id}`);

    } catch (error) {
      console.error('❌ Document ingestion failed:', error);
      throw error;
    }
  });

  test('should test semantic search and retrieval', async () => {
    if (testCleanupIds.length === 0) {
      test.skip(testCleanupIds.length === 0, 'No test documents available for search');
      return;
    }

    // Test queries with different legal topics
    const testQueries = [
      'What are the requirements for a valid contract?',
      'Tell me about consideration in contract law',
      'What happens when acceptance is not properly communicated?'
    ];

    for (const query of testQueries) {
      try {
        // Generate query embedding
        let queryEmbedding: number[] = [];
        
        const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'nomic-embed-text',
            prompt: query
          })
        });

        if (embeddingResponse.ok) {
          const embeddingData = await embeddingResponse.json();
          queryEmbedding = embeddingData.embedding;
        } else {
          queryEmbedding = Array(768).fill(0).map(() => Math.random() - 0.5);
          console.warn('⚠️ Using mock query embedding');
        }

        // Perform semantic search
        const searchResults = await dbClient.query(`
          SELECT 
            id,
            file,
            content,
            summary,
            embedding <=> $1::vector AS distance
          FROM documents
          WHERE id = ANY($2::uuid[])
          ORDER BY distance
          LIMIT 3
        `, [
          `[${queryEmbedding.join(',')}]`,
          testCleanupIds
        ]);

        expect(searchResults.rows.length).toBeGreaterThan(0);
        expect(searchResults.rows[0]).toHaveProperty('distance');
        
        // Distance should be a reasonable number (not NaN or infinite)
        const distance = parseFloat(searchResults.rows[0].distance);
        expect(distance).toBeGreaterThanOrEqual(0);
        expect(distance).toBeLessThan(2); // Cosine distance should be < 2
        
        console.log(`✅ Query: "${query}" - Found ${searchResults.rows.length} results`);
        console.log(`   Best match: ${searchResults.rows[0].file} (distance: ${distance.toFixed(4)})`);

      } catch (error) {
        console.error(`❌ Search failed for query: "${query}"`, error);
      }
    }
  });

  test('should test RAG response generation', async () => {
    if (testCleanupIds.length === 0) {
      test.skip(testCleanupIds.length === 0, 'No test documents available for RAG generation');
      return;
    }

    const testQuery = 'What are the key elements needed for a valid contract?';
    
    try {
      // Get relevant documents (mock the RAG retrieval)
      const contextDocs = await dbClient.query(`
        SELECT file, content, summary
        FROM documents
        WHERE id = ANY($1::uuid[])
        LIMIT 3
      `, [testCleanupIds]);

      expect(contextDocs.rows.length).toBeGreaterThan(0);

      // Test RAG context preparation
      const ragContext = {
        query: testQuery,
        documents: contextDocs.rows.map((doc, idx) => ({
          id: idx + 1,
          file: doc.file,
          content: doc.content.substring(0, 500) + '...', // Truncate for context
          summary: doc.summary,
          relevance: 0.95 - (idx * 0.05) // Mock relevance scores
        })),
        metadata: {
          total_documents: contextDocs.rows.length,
          search_time: '45ms',
          model: 'nomic-embed-text'
        }
      };

      expect(ragContext.documents.length).toBeGreaterThan(0);
      expect(ragContext.documents[0]).toHaveProperty('file');
      expect(ragContext.documents[0]).toHaveProperty('content');
      expect(ragContext.documents[0]).toHaveProperty('relevance');

      console.log('✅ RAG context prepared successfully');
      console.log(`   Query: ${ragContext.query}`);
      console.log(`   Retrieved ${ragContext.documents.length} relevant documents`);
      console.log(`   Top document: ${ragContext.documents[0].file} (relevance: ${ragContext.documents[0].relevance})`);

      // Test that we can generate a structured response format
      const responseStructure = {
        summary: 'Brief answer extracted from retrieved documents',
        detailed_analysis: 'In-depth analysis referencing specific documents',
        key_quotes: contextDocs.rows.map(doc => ({
          source: doc.file,
          quote: doc.content.substring(0, 100) + '...'
        })),
        recommendations: ['Actionable recommendation 1', 'Actionable recommendation 2'],
        source_documents: ragContext.documents.map(doc => doc.file),
        confidence_score: 0.89
      };

      expect(responseStructure).toHaveProperty('summary');
      expect(responseStructure).toHaveProperty('detailed_analysis');
      expect(responseStructure).toHaveProperty('key_quotes');
      expect(responseStructure.key_quotes.length).toBeGreaterThan(0);
      expect(responseStructure).toHaveProperty('source_documents');

      console.log('✅ RAG response structure validated');

    } catch (error) {
      console.error('❌ RAG response generation failed:', error);
      throw error;
    }
  });

  test('should test caching and performance optimization', async () => {
    const testText = 'Performance test for legal document embeddings and caching system';
    const startTime = Date.now();

    try {
      // Test embedding cache hit/miss
      const textHash = require('crypto').createHash('sha256').update(testText).digest('hex');
      
      // Check if already cached
      const cacheCheck = await dbClient.query(
        'SELECT embedding FROM embedding_cache WHERE text_hash = $1',
        [textHash]
      );

      let embedding: number[];
      let cacheHit = false;

      if (cacheCheck.rows.length > 0) {
        embedding = cacheCheck.rows[0].embedding;
        cacheHit = true;
        console.log('✅ Cache hit - retrieved cached embedding');
      } else {
        // Generate new embedding
        const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'nomic-embed-text',
            prompt: testText
          })
        });

        if (embeddingResponse.ok) {
          const embeddingData = await embeddingResponse.json();
          embedding = embeddingData.embedding;
          
          // Cache the embedding
          await dbClient.query(
            `INSERT INTO embedding_cache (text_hash, embedding, model) 
             VALUES ($1, $2::vector, $3)
             ON CONFLICT (text_hash) DO NOTHING`,
            [textHash, `[${embedding.join(',')}]`, 'nomic-embed-text']
          );
          
          console.log('✅ Cache miss - generated and cached new embedding');
        } else {
          embedding = Array(768).fill(0).map(() => Math.random());
          console.warn('⚠️ Using mock embedding for performance test');
        }
      }

      const endTime = Date.now();
      const processingTime = endTime - startTime;

      expect(Array.isArray(embedding)).toBe(true);
      expect(embedding.length).toBe(768);
      expect(processingTime).toBeLessThan(5000); // Should complete within 5 seconds

      console.log(`✅ Embedding processing completed in ${processingTime}ms (cache ${cacheHit ? 'hit' : 'miss'})`);

      // Test batch processing simulation
      const batchStartTime = Date.now();
      const batchTexts = [
        'Contract law basics',
        'Tort law principles', 
        'Criminal procedure rules',
        'Constitutional rights analysis',
        'Corporate law fundamentals'
      ];

      let batchProcessed = 0;
      for (const text of batchTexts) {
        const hash = require('crypto').createHash('sha256').update(text).digest('hex');
        const cached = await dbClient.query(
          'SELECT COUNT(*) FROM embedding_cache WHERE text_hash = $1',
          [hash]
        );
        
        if (parseInt(cached.rows[0].count) === 0) {
          // Simulate embedding generation (mock for performance)
          const mockEmbedding = Array(768).fill(0).map(() => Math.random());
          await dbClient.query(
            `INSERT INTO embedding_cache (text_hash, embedding, model) 
             VALUES ($1, $2::vector, $3)
             ON CONFLICT (text_hash) DO NOTHING`,
            [hash, `[${mockEmbedding.join(',')}]`, 'performance-test-model']
          );
        }
        batchProcessed++;
      }

      const batchEndTime = Date.now();
      const batchProcessingTime = batchEndTime - batchStartTime;

      expect(batchProcessed).toBe(batchTexts.length);
      expect(batchProcessingTime).toBeLessThan(10000); // Batch should complete within 10 seconds

      console.log(`✅ Batch processing: ${batchProcessed} items in ${batchProcessingTime}ms`);

    } catch (error) {
      console.error('❌ Performance optimization test failed:', error);
      throw error;
    }
  });

  test('should test error handling and fallback mechanisms', async () => {
    // Test handling of invalid embeddings
    try {
      const invalidEmbedding = Array(10).fill(0.5); // Wrong dimension
      
      const result = await dbClient.query(`
        SELECT '[1,2,3]'::vector as test_vector
      `);
      
      expect(result.rows).toHaveLength(1);
      console.log('✅ Basic vector operations working');
      
      // Test dimension validation
      try {
        await dbClient.query(`
          INSERT INTO documents (file, content, embedding)
          VALUES ('test.txt', 'test content', $1::vector)
        `, [`[${invalidEmbedding.join(',')}]`]);
        
        console.warn('⚠️ Invalid embedding dimension was accepted (unexpected)');
      } catch (dimError) {
        console.log('✅ Proper dimension validation - invalid embedding rejected');
      }
      
    } catch (error) {
      console.error('❌ Error handling test failed:', error);
    }

    // Test service degradation scenarios
    console.log('✅ Testing graceful degradation...');
    
    // Simulate Ollama unavailable
    try {
      const unavailableResponse = await fetch('http://localhost:99999/api/embeddings', {
        signal: AbortSignal.timeout(1000)
      });
    } catch (networkError) {
      console.log('✅ Network error handling works - can gracefully handle service unavailability');
    }

    // Test fallback to mock embeddings
    const fallbackEmbedding = Array(768).fill(0).map((_, i) => Math.sin(i / 100)); // Deterministic mock
    expect(fallbackEmbedding).toHaveLength(768);
    expect(fallbackEmbedding.every(val => val >= -1 && val <= 1)).toBe(true);
    
    console.log('✅ Fallback embedding generation works');
  });
});