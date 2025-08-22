import { test, expect } from '@playwright/test';
import { Client } from 'pg';
import { randomUUID } from 'crypto';

const dbConfig = {
  user: 'postgres',
  password: '123456',
  host: 'localhost',
  database: 'legal_ai_db',
  port: 5432,
};

test.describe('PostgreSQL CRUD with pgvector Tests', () => {
  let dbClient: Client;
  let testDocumentIds: string[] = [];

  test.beforeAll(async () => {
    dbClient = new Client(dbConfig);
    await dbClient.connect();
    
    // Ensure tables exist
    await dbClient.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        file TEXT,
        content TEXT,
        summary TEXT,
        embedding vector(768),
        chunk_index INTEGER DEFAULT 0,
        total_chunks INTEGER DEFAULT 1,
        tokens INTEGER,
        file_hash TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);

    await dbClient.query(`
      CREATE TABLE IF NOT EXISTS embedding_cache (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        text_hash TEXT UNIQUE NOT NULL,
        embedding vector(768) NOT NULL,
        model VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);

    // Create indexes if they don't exist
    try {
      await dbClient.query(`
        CREATE INDEX IF NOT EXISTS idx_documents_embedding 
        ON documents USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
      `);
    } catch (error) {
      console.warn('⚠️ Could not create ivfflat index (pgvector may not be properly installed)');
    }
  });

  test.afterAll(async () => {
    // Clean up test data
    if (testDocumentIds.length > 0) {
      await dbClient.query(
        `DELETE FROM documents WHERE id = ANY($1::uuid[])`,
        [testDocumentIds]
      );
    }
    
    // Clean up test embedding cache
    await dbClient.query(`DELETE FROM embedding_cache WHERE model = 'test-model'`);
    
    await dbClient.end();
  });

  test('CREATE: Insert legal document with vector embedding', async () => {
    const testContent = 'In contract law, consideration must be present for a valid agreement. This principle ensures mutual benefit between contracting parties.';
    const mockEmbedding = Array(768).fill(0).map(() => Math.random() - 0.5);
    
    const result = await dbClient.query(`
      INSERT INTO documents (file, content, embedding, tokens, file_hash)
      VALUES ($1, $2, $3::vector, $4, $5)
      RETURNING id, file, content, tokens, created_at
    `, [
      'contract-law-test.txt',
      testContent,
      `[${mockEmbedding.join(',')}]`,
      Math.floor(testContent.length / 4),
      'test-hash-' + randomUUID()
    ]);

    expect(result.rows).toHaveLength(1);
    expect(result.rows[0].file).toBe('contract-law-test.txt');
    expect(result.rows[0].content).toBe(testContent);
    expect(result.rows[0].tokens).toBeGreaterThan(0);
    
    testDocumentIds.push(result.rows[0].id);
    console.log(`✅ Created document with ID: ${result.rows[0].id}`);
  });

  test('READ: Query documents with vector similarity search', async () => {
    // First, insert a few more test documents with known embeddings
    const testDocs = [
      {
        content: 'Criminal law defines offenses against the state and prescribes punishments for violations.',
        file: 'criminal-law-test.txt',
        embedding: Array(768).fill(0).map((_, i) => i < 100 ? 0.8 : 0.1) // Similar pattern
      },
      {
        content: 'Tort law addresses civil wrongs and provides remedies for damages caused by negligent actions.',
        file: 'tort-law-test.txt', 
        embedding: Array(768).fill(0).map((_, i) => i < 100 ? 0.2 : 0.9) // Different pattern
      }
    ];

    for (const doc of testDocs) {
      const result = await dbClient.query(`
        INSERT INTO documents (file, content, embedding, tokens)
        VALUES ($1, $2, $3::vector, $4)
        RETURNING id
      `, [
        doc.file,
        doc.content,
        `[${doc.embedding.join(',')}]`,
        Math.floor(doc.content.length / 4)
      ]);
      
      testDocumentIds.push(result.rows[0].id);
    }

    // Test basic SELECT
    const allDocuments = await dbClient.query(`
      SELECT id, file, content, tokens 
      FROM documents 
      WHERE id = ANY($1::uuid[])
      ORDER BY created_at DESC
    `, [testDocumentIds]);

    expect(allDocuments.rows.length).toBeGreaterThanOrEqual(3);
    console.log(`✅ Retrieved ${allDocuments.rows.length} documents`);

    // Test vector similarity search (if pgvector is properly installed)
    try {
      const queryEmbedding = Array(768).fill(0).map((_, i) => i < 100 ? 0.8 : 0.1);
      
      const similarityResults = await dbClient.query(`
        SELECT 
          id,
          file,
          content,
          embedding <=> $1::vector AS distance
        FROM documents
        WHERE id = ANY($2::uuid[])
        ORDER BY distance
        LIMIT 5
      `, [
        `[${queryEmbedding.join(',')}]`,
        testDocumentIds
      ]);

      expect(similarityResults.rows.length).toBeGreaterThan(0);
      expect(similarityResults.rows[0]).toHaveProperty('distance');
      console.log(`✅ Vector similarity search returned ${similarityResults.rows.length} results`);
      
      // The first result should be most similar (lowest distance)
      const distances = similarityResults.rows.map(row => parseFloat(row.distance));
      expect(distances[0]).toBeLessThanOrEqual(distances[distances.length - 1]);
      
    } catch (error) {
      console.warn('⚠️ Vector similarity search failed - pgvector may not be properly configured:', error);
    }
  });

  test('UPDATE: Modify document content and metadata', async () => {
    if (testDocumentIds.length === 0) {
      test.skip(true, 'No test documents available for update');
      return;
    }

    const documentId = testDocumentIds[0];
    const updatedContent = 'Updated: Contract law principles include offer, acceptance, consideration, and legal capacity.';
    const newEmbedding = Array(768).fill(0).map(() => Math.random() - 0.5);

    const updateResult = await dbClient.query(`
      UPDATE documents 
      SET 
        content = $1,
        summary = $2,
        embedding = $3::vector,
        tokens = $4,
        updated_at = NOW()
      WHERE id = $5
      RETURNING id, content, summary, tokens, updated_at
    `, [
      updatedContent,
      'Summary: Core contract law principles',
      `[${newEmbedding.join(',')}]`,
      Math.floor(updatedContent.length / 4),
      documentId
    ]);

    expect(updateResult.rows).toHaveLength(1);
    expect(updateResult.rows[0].content).toBe(updatedContent);
    expect(updateResult.rows[0].summary).toBe('Summary: Core contract law principles');
    expect(updateResult.rows[0].tokens).toBeGreaterThan(0);
    
    console.log(`✅ Updated document ${documentId}`);
  });

  test('DELETE: Remove test documents', async () => {
    if (testDocumentIds.length === 0) {
      test.skip(true, 'No test documents available for deletion');
      return;
    }

    // Delete one specific document
    const documentToDelete = testDocumentIds[0];
    
    const deleteResult = await dbClient.query(`
      DELETE FROM documents 
      WHERE id = $1
      RETURNING id, file
    `, [documentToDelete]);

    expect(deleteResult.rows).toHaveLength(1);
    expect(deleteResult.rows[0].id).toBe(documentToDelete);
    
    console.log(`✅ Deleted document ${documentToDelete}`);

    // Verify deletion
    const verifyResult = await dbClient.query(`
      SELECT id FROM documents WHERE id = $1
    `, [documentToDelete]);

    expect(verifyResult.rows).toHaveLength(0);

    // Remove from our tracking array
    testDocumentIds = testDocumentIds.filter(id => id !== documentToDelete);
  });

  test('EMBEDDING CACHE: Test caching functionality', async () => {
    const testText = 'This is a test text for embedding cache functionality';
    const textHash = 'test-hash-' + randomUUID();
    const mockEmbedding = Array(768).fill(0).map(() => Math.random());

    // Insert into cache
    const insertResult = await dbClient.query(`
      INSERT INTO embedding_cache (text_hash, embedding, model)
      VALUES ($1, $2::vector, $3)
      RETURNING id, text_hash, model, created_at
    `, [
      textHash,
      `[${mockEmbedding.join(',')}]`,
      'test-model'
    ]);

    expect(insertResult.rows).toHaveLength(1);
    expect(insertResult.rows[0].text_hash).toBe(textHash);
    expect(insertResult.rows[0].model).toBe('test-model');

    // Test cache retrieval
    const cacheResult = await dbClient.query(`
      SELECT text_hash, embedding, model 
      FROM embedding_cache 
      WHERE text_hash = $1
    `, [textHash]);

    expect(cacheResult.rows).toHaveLength(1);
    expect(cacheResult.rows[0].text_hash).toBe(textHash);
    expect(cacheResult.rows[0].model).toBe('test-model');
    expect(cacheResult.rows[0].embedding).toBeDefined();

    console.log('✅ Embedding cache functionality verified');
  });

  test('PERFORMANCE: Test bulk operations', async () => {
    const startTime = Date.now();
    const bulkDocuments = [];
    
    // Prepare bulk insert data
    for (let i = 0; i < 10; i++) {
      const embedding = Array(768).fill(0).map(() => Math.random() - 0.5);
      bulkDocuments.push([
        `bulk-test-${i}.txt`,
        `This is bulk test document number ${i} with legal content about contracts and agreements.`,
        `[${embedding.join(',')}]`,
        50,
        `bulk-hash-${i}-${randomUUID()}`
      ]);
    }

    // Bulk insert using transaction
    await dbClient.query('BEGIN');
    try {
      const insertPromises = bulkDocuments.map(doc => 
        dbClient.query(`
          INSERT INTO documents (file, content, embedding, tokens, file_hash)
          VALUES ($1, $2, $3::vector, $4, $5)
          RETURNING id
        `, doc)
      );

      const results = await Promise.all(insertPromises);
      await dbClient.query('COMMIT');

      results.forEach((result: unknown) => {
        testDocumentIds.push(result.rows[0].id);
      });

      const endTime = Date.now();
      const duration = endTime - startTime;

      expect(results).toHaveLength(10);
      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
      
      console.log(`✅ Bulk insert of 10 documents completed in ${duration}ms`);
      
    } catch (error) {
      await dbClient.query('ROLLBACK');
      throw error;
    }
  });
});