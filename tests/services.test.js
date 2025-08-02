// Legal AI System - Service Tests
// Tests for individual service components

import { strict as assert } from 'assert';
import { QdrantClient } from '@qdrant/js-client-rest';
import { createClient } from 'redis';
import pg from 'postgres';

// Test configuration
const config = {
    qdrant: {
        url: 'http://localhost:6333',
        vectorSize: 384
    },
    redis: {
        url: 'redis://localhost:6379'
    },
    postgres: {
        host: 'localhost',
        port: 5432,
        database: 'prosecutor_db',
        username: 'postgres',
        password: 'postgres'
    },
    ollama: {
        url: 'http://localhost:11434'
    }
};

// Test results
const results = {
    passed: 0,
    failed: 0,
    tests: []
};

// Helper function
function logTest(name, status, error = null) {
    const result = {
        name,
        status,
        error: error ? error.message : null,
        timestamp: new Date().toISOString()
    };
    
    results.tests.push(result);
    
    if (status === 'PASS') {
        results.passed++;
        console.log(`✓ ${name}`);
    } else {
        results.failed++;
        console.error(`✗ ${name}${error ? ': ' + error.message : ''}`);
    }
}

// Qdrant Service Tests
async function testQdrantService() {
    console.log('\n=== Qdrant Service Tests ===');
    
    const client = new QdrantClient({ url: config.qdrant.url });
    
    // Test 1: Connection
    try {
        const health = await client.api('GET', '/health');
        assert(health.status === 'ok', 'Qdrant health check failed');
        logTest('Qdrant Connection', 'PASS');
    } catch (error) {
        logTest('Qdrant Connection', 'FAIL', error);
        return; // Skip other tests if connection fails
    }
    
    // Test 2: Collection Operations
    try {
        const collections = await client.getCollections();
        const expectedCollections = ['legal_documents', 'case_embeddings', 'evidence_vectors'];
        
        for (const expected of expectedCollections) {
            const exists = collections.collections.some(c => c.name === expected);
            assert(exists, `Collection ${expected} does not exist`);
        }
        logTest('Qdrant Collections', 'PASS');
    } catch (error) {
        logTest('Qdrant Collections', 'FAIL', error);
    }
    
    // Test 3: Vector Operations
    try {
        const testVector = Array(config.qdrant.vectorSize).fill(0).map(() => Math.random());
        const testPoint = {
            id: Date.now(),
            vector: testVector,
            payload: {
                test: true,
                timestamp: new Date().toISOString()
            }
        };
        
        // Upsert
        await client.upsert('legal_documents', {
            points: [testPoint]
        });
        
        // Search
        const searchResult = await client.search('legal_documents', {
            vector: testVector,
            limit: 1,
            with_payload: true
        });
        
        assert(searchResult.length > 0, 'Vector search returned no results');
        assert(searchResult[0].score > 0.99, 'Vector similarity score too low');
        
        // Cleanup
        await client.delete('legal_documents', {
            points: [testPoint.id]
        });
        
        logTest('Qdrant Vector Operations', 'PASS');
    } catch (error) {
        logTest('Qdrant Vector Operations', 'FAIL', error);
    }
    
    // Test 4: Batch Operations
    try {
        const batchSize = 10;
        const batchPoints = Array(batchSize).fill(0).map((_, i) => ({
            id: `test-batch-${i}`,
            vector: Array(config.qdrant.vectorSize).fill(0).map(() => Math.random()),
            payload: { batch: true, index: i }
        }));
        
        // Batch upsert
        await client.upsert('legal_documents', {
            points: batchPoints
        });
        
        // Batch retrieve
        const retrieved = await client.retrieve('legal_documents', {
            ids: batchPoints.map(p => p.id),
            with_payload: true
        });
        
        assert(retrieved.length === batchSize, 'Batch retrieve count mismatch');
        
        // Batch delete
        await client.delete('legal_documents', {
            points: batchPoints.map(p => p.id)
        });
        
        logTest('Qdrant Batch Operations', 'PASS');
    } catch (error) {
        logTest('Qdrant Batch Operations', 'FAIL', error);
    }
}

// Redis Service Tests
async function testRedisService() {
    console.log('\n=== Redis Service Tests ===');
    
    const redis = createClient({ url: config.redis.url });
    
    try {
        await redis.connect();
        logTest('Redis Connection', 'PASS');
    } catch (error) {
        logTest('Redis Connection', 'FAIL', error);
        return;
    }
    
    // Test 2: Basic Operations
    try {
        const testKey = 'test:service:basic';
        const testValue = { test: true, timestamp: Date.now() };
        
        // Set
        await redis.set(testKey, JSON.stringify(testValue));
        
        // Get
        const retrieved = await redis.get(testKey);
        const parsed = JSON.parse(retrieved);
        
        assert(parsed.test === testValue.test, 'Redis value mismatch');
        
        // Delete
        await redis.del(testKey);
        
        logTest('Redis Basic Operations', 'PASS');
    } catch (error) {
        logTest('Redis Basic Operations', 'FAIL', error);
    }
    
    // Test 3: Cache Operations
    try {
        const cacheKey = 'cache:test:ttl';
        const cacheValue = 'test-value';
        const ttl = 5; // 5 seconds
        
        // Set with TTL
        await redis.setex(cacheKey, ttl, cacheValue);
        
        // Check TTL
        const remainingTTL = await redis.ttl(cacheKey);
        assert(remainingTTL > 0 && remainingTTL <= ttl, 'TTL not set correctly');
        
        // Verify value
        const cached = await redis.get(cacheKey);
        assert(cached === cacheValue, 'Cached value mismatch');
        
        logTest('Redis Cache Operations', 'PASS');
    } catch (error) {
        logTest('Redis Cache Operations', 'FAIL', error);
    }
    
    // Test 4: Hash Operations
    try {
        const hashKey = 'hash:test:service';
        const hashData = {
            field1: 'value1',
            field2: 'value2',
            field3: JSON.stringify({ nested: true })
        };
        
        // Set hash
        await redis.hSet(hashKey, hashData);
        
        // Get all
        const allFields = await redis.hGetAll(hashKey);
        assert(Object.keys(allFields).length === 3, 'Hash field count mismatch');
        
        // Get specific field
        const field1 = await redis.hGet(hashKey, 'field1');
        assert(field1 === hashData.field1, 'Hash field value mismatch');
        
        // Cleanup
        await redis.del(hashKey);
        
        logTest('Redis Hash Operations', 'PASS');
    } catch (error) {
        logTest('Redis Hash Operations', 'FAIL', error);
    }
    
    await redis.disconnect();
}

// PostgreSQL Service Tests
async function testPostgresService() {
    console.log('\n=== PostgreSQL Service Tests ===');
    
    const sql = pg(config.postgres);
    
    // Test 1: Connection
    try {
        const result = await sql`SELECT NOW() as timestamp`;
        assert(result.length > 0, 'PostgreSQL query failed');
        logTest('PostgreSQL Connection', 'PASS');
    } catch (error) {
        logTest('PostgreSQL Connection', 'FAIL', error);
        return;
    }
    
    // Test 2: Extensions
    try {
        const extensions = await sql`
            SELECT extname 
            FROM pg_extension 
            WHERE extname IN ('uuid-ossp', 'vector', 'pg_trgm')
        `;
        
        const required = ['uuid-ossp', 'vector', 'pg_trgm'];
        const installed = extensions.map(e => e.extname);
        
        for (const ext of required) {
            assert(installed.includes(ext), `Extension ${ext} not installed`);
        }
        
        logTest('PostgreSQL Extensions', 'PASS');
    } catch (error) {
        logTest('PostgreSQL Extensions', 'FAIL', error);
    }
    
    // Test 3: Vector Operations
    try {
        // Create test table
        await sql`
            CREATE TEMP TABLE test_vectors (
                id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
                embedding vector(384)
            )
        `;
        
        // Insert test vector
        const testVector = Array(384).fill(0).map(() => Math.random());
        const vectorString = `[${testVector.join(',')}]`;
        
        const inserted = await sql`
            INSERT INTO test_vectors (embedding) 
            VALUES (${vectorString}::vector)
            RETURNING id
        `;
        
        assert(inserted.length > 0, 'Vector insert failed');
        
        // Search similar vectors
        const similar = await sql`
            SELECT id, 1 - (embedding <=> ${vectorString}::vector) as similarity
            FROM test_vectors
            WHERE 1 - (embedding <=> ${vectorString}::vector) > 0.9
            ORDER BY embedding <=> ${vectorString}::vector
            LIMIT 1
        `;
        
        assert(similar.length > 0, 'Vector similarity search failed');
        assert(similar[0].similarity > 0.99, 'Vector similarity too low');
        
        logTest('PostgreSQL Vector Operations', 'PASS');
    } catch (error) {
        logTest('PostgreSQL Vector Operations', 'FAIL', error);
    }
    
    // Test 4: Transactions
    try {
        await sql.begin(async sql => {
            // Create test record
            const testCase = await sql`
                INSERT INTO cases (title, description, status)
                VALUES ('Test Case', 'Service test case', 'open')
                RETURNING id
            `;
            
            // Update
            await sql`
                UPDATE cases 
                SET status = 'closed' 
                WHERE id = ${testCase[0].id}
            `;
            
            // Verify
            const updated = await sql`
                SELECT status 
                FROM cases 
                WHERE id = ${testCase[0].id}
            `;
            
            assert(updated[0].status === 'closed', 'Transaction update failed');
            
            // Rollback
            throw new Error('Intentional rollback');
        }).catch(err => {
            if (err.message === 'Intentional rollback') {
                // Expected
            } else {
                throw err;
            }
        });
        
        logTest('PostgreSQL Transactions', 'PASS');
    } catch (error) {
        logTest('PostgreSQL Transactions', 'FAIL', error);
    }
    
    await sql.end();
}

// Ollama Service Tests
async function testOllamaService() {
    console.log('\n=== Ollama Service Tests ===');
    
    // Test 1: Connection
    try {
        const response = await fetch(`${config.ollama.url}/api/tags`);
        assert(response.ok, 'Ollama API not accessible');
        logTest('Ollama Connection', 'PASS');
    } catch (error) {
        logTest('Ollama Connection', 'FAIL', error);
        return;
    }
    
    // Test 2: Model Availability
    try {
        const response = await fetch(`${config.ollama.url}/api/tags`);
        const data = await response.json();
        
        const requiredModels = ['nomic-embed-text'];
        const availableModels = data.models.map(m => m.name);
        
        for (const model of requiredModels) {
            const exists = availableModels.some(m => m.includes(model));
            assert(exists, `Model ${model} not available`);
        }
        
        logTest('Ollama Models', 'PASS');
    } catch (error) {
        logTest('Ollama Models', 'FAIL', error);
    }
    
    // Test 3: Embedding Generation
    try {
        const response = await fetch(`${config.ollama.url}/api/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'nomic-embed-text',
                prompt: 'Test legal document for embedding generation'
            })
        });
        
        const data = await response.json();
        assert(data.embedding, 'No embedding returned');
        assert(data.embedding.length === config.qdrant.vectorSize, 
               `Embedding dimension mismatch: expected ${config.qdrant.vectorSize}, got ${data.embedding.length}`);
        
        logTest('Ollama Embeddings', 'PASS');
    } catch (error) {
        logTest('Ollama Embeddings', 'FAIL', error);
    }
    
    // Test 4: Generation
    try {
        const response = await fetch(`${config.ollama.url}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'gemma:2b',
                prompt: 'What is a contract?',
                stream: false,
                options: {
                    temperature: 0.7,
                    max_tokens: 100
                }
            })
        });
        
        const data = await response.json();
        assert(data.response, 'No response generated');
        assert(data.response.length > 10, 'Response too short');
        
        logTest('Ollama Generation', 'PASS');
    } catch (error) {
        logTest('Ollama Generation', 'FAIL', error);
    }
}

// Main test runner
async function runTests() {
    console.log('Legal AI System - Service Tests');
    console.log('==============================\n');
    
    const startTime = Date.now();
    
    try {
        await testQdrantService();
        await testRedisService();
        await testPostgresService();
        await testOllamaService();
    } catch (error) {
        console.error('Test suite error:', error);
    }
    
    const duration = Date.now() - startTime;
    
    // Summary
    console.log('\n=== Test Summary ===');
    console.log(`Total: ${results.passed + results.failed}`);
    console.log(`Passed: ${results.passed}`);
    console.log(`Failed: ${results.failed}`);
    console.log(`Duration: ${duration}ms`);
    
    // Save results
    const fs = await import('fs/promises');
    await fs.writeFile(
        'test-results-services.json',
        JSON.stringify(results, null, 2)
    );
    
    // Exit with appropriate code
    process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runTests();
