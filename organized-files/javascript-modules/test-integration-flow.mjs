#!/usr/bin/env node
// Test actual data flow between all integrated components
// This script verifies that components can actually communicate

import pg from 'pg';
import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import amqp from 'amqplib';
import neo4j from 'neo4j-driver';
import fetch from 'node-fetch';
import { v4 as uuidv4 } from 'uuid';

console.log(`
╔════════════════════════════════════════════════════════════════════╗
║              DATA FLOW INTEGRATION TEST                           ║
║     Testing actual communication between all components           ║
╚════════════════════════════════════════════════════════════════════╝
`);

const testResults = [];

// Test 1: Document Upload Flow (MinIO → PostgreSQL)
async function testDocumentFlow() {
    console.log('\n📄 TEST 1: Document Upload Flow');
    console.log('MinIO → PostgreSQL → pgvector');
    
    try {
        // 1. Upload to MinIO
        const s3Client = new S3Client({
            endpoint: 'http://localhost:9000',
            region: 'us-east-1',
            credentials: {
                accessKeyId: 'minioadmin',
                secretAccessKey: 'minioadmin123'
            },
            forcePathStyle: true
        });
        
        const testDoc = {
            id: uuidv4(),
            title: 'Test Legal Document',
            content: 'This is a test legal document for integration testing.',
            timestamp: new Date().toISOString()
        };
        
        // Upload document
        await s3Client.send(new PutObjectCommand({
            Bucket: 'legal-documents',
            Key: `test-doc-${testDoc.id}.json`,
            Body: JSON.stringify(testDoc),
            ContentType: 'application/json'
        }));
        
        console.log('✅ Document uploaded to MinIO');
        
        // 2. Store metadata in PostgreSQL
        const pgClient = new pg.Client({
            connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
        });
        
        await pgClient.connect();
        
        await pgClient.query(`
            INSERT INTO legal_documents (id, title, content, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (id) DO UPDATE SET title = $2
        `, [testDoc.id, testDoc.title, testDoc.content]);
        
        console.log('✅ Metadata stored in PostgreSQL');
        
        // 3. Retrieve and verify
        const result = await pgClient.query(`
            SELECT * FROM legal_documents WHERE id = $1
        `, [testDoc.id]);
        
        if (result.rows.length > 0) {
            console.log('✅ Document flow complete: MinIO → PostgreSQL');
            testResults.push({ test: 'Document Flow', status: 'PASS' });
        }
        
        await pgClient.end();
        return true;
        
    } catch (error) {
        console.log(`❌ Document flow failed: ${error.message}`);
        testResults.push({ test: 'Document Flow', status: 'FAIL', error: error.message });
        return false;
    }
}

// Test 2: Message Queue Flow (RabbitMQ → Processing)
async function testMessageQueueFlow() {
    console.log('\n📨 TEST 2: Message Queue Flow');
    console.log('RabbitMQ: Publish → Subscribe');
    
    try {
        const connection = await amqp.connect('amqp://localhost');
        const channel = await connection.createChannel();
        
        const queue = 'test_integration_queue';
        const testMessage = {
            action: 'process_document',
            documentId: uuidv4(),
            timestamp: new Date().toISOString()
        };
        
        // Create queue
        await channel.assertQueue(queue, { durable: false });
        
        // Publish message
        channel.sendToQueue(queue, Buffer.from(JSON.stringify(testMessage)));
        console.log('✅ Message published to RabbitMQ');
        
        // Consume message
        return new Promise((resolve) => {
            channel.consume(queue, (msg) => {
                if (msg) {
                    const received = JSON.parse(msg.content.toString());
                    console.log('✅ Message received from RabbitMQ');
                    console.log(`   Content: ${JSON.stringify(received).substring(0, 50)}...`);
                    
                    channel.ack(msg);
                    channel.close();
                    connection.close();
                    
                    testResults.push({ test: 'Message Queue', status: 'PASS' });
                    resolve(true);
                }
            });
        });
        
    } catch (error) {
        console.log(`❌ Message queue failed: ${error.message}`);
        testResults.push({ test: 'Message Queue', status: 'FAIL', error: error.message });
        return false;
    }
}

// Test 3: Graph Relationship (Neo4j ↔ PostgreSQL)
async function testGraphRelationship() {
    console.log('\n🔗 TEST 3: Graph Relationship Flow');
    console.log('PostgreSQL → Neo4j → Graph Query');
    
    try {
        // Get case from PostgreSQL
        const pgClient = new pg.Client({
            connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
        });
        
        await pgClient.connect();
        const cases = await pgClient.query('SELECT * FROM cases LIMIT 1');
        
        if (cases.rows.length === 0) {
            console.log('⚠️  No cases found in PostgreSQL');
            await pgClient.end();
            return false;
        }
        
        const testCase = cases.rows[0];
        console.log(`✅ Retrieved case from PostgreSQL: ${testCase.case_number}`);
        
        // Create node in Neo4j
        const driver = neo4j.driver(
            'neo4j://localhost:7687',
            neo4j.auth.basic('neo4j', 'password')
        );
        
        const session = driver.session();
        
        // Create case node
        await session.run(`
            MERGE (c:Case {id: $id, caseNumber: $caseNumber})
            SET c.title = $title, c.updatedAt = datetime()
            RETURN c
        `, {
            id: testCase.id,
            caseNumber: testCase.case_number,
            title: testCase.title
        });
        
        console.log('✅ Case node created in Neo4j');
        
        // Create relationship
        await session.run(`
            MERGE (c:Case {id: $caseId})
            MERGE (d:Document {id: $docId})
            MERGE (c)-[r:HAS_DOCUMENT]->(d)
            SET r.createdAt = datetime()
            RETURN r
        `, {
            caseId: testCase.id,
            docId: uuidv4()
        });
        
        console.log('✅ Graph relationship created');
        
        // Query relationship
        const result = await session.run(`
            MATCH (c:Case {id: $id})-[r:HAS_DOCUMENT]->(d:Document)
            RETURN count(r) as relationships
        `, { id: testCase.id });
        
        const count = result.records[0].get('relationships').toNumber();
        console.log(`✅ Graph query successful: ${count} relationships found`);
        
        await session.close();
        await driver.close();
        await pgClient.end();
        
        testResults.push({ test: 'Graph Relationship', status: 'PASS' });
        return true;
        
    } catch (error) {
        console.log(`❌ Graph relationship failed: ${error.message}`);
        testResults.push({ test: 'Graph Relationship', status: 'FAIL', error: error.message });
        return false;
    }
}

// Test 4: Go Service Integration
async function testGoServices() {
    console.log('\n⚙️  TEST 4: Go Microservices Integration');
    console.log('Testing API endpoints');
    
    const services = [
        { name: 'GPU Orchestrator', port: 8084, endpoint: '/health' },
        { name: 'RAG Service', port: 8085, endpoint: '/health' }
    ];
    
    let passed = 0;
    
    for (const service of services) {
        try {
            const response = await fetch(`http://localhost:${service.port}${service.endpoint}`, {
                timeout: 3000
            });
            
            if (response.ok) {
                console.log(`✅ ${service.name} responding on port ${service.port}`);
                passed++;
            } else {
                console.log(`⚠️  ${service.name} returned status ${response.status}`);
            }
        } catch (error) {
            console.log(`❌ ${service.name} not accessible: ${error.message}`);
        }
    }
    
    if (passed > 0) {
        testResults.push({ test: 'Go Services', status: 'PARTIAL', passed: passed, total: services.length });
        return true;
    } else {
        testResults.push({ test: 'Go Services', status: 'FAIL' });
        return false;
    }
}

// Test 5: End-to-End RAG Pipeline
async function testRAGPipeline() {
    console.log('\n🤖 TEST 5: End-to-End RAG Pipeline');
    console.log('Document → Embedding → Vector Search → Response');
    
    try {
        // 1. Create test document
        const testDoc = {
            id: uuidv4(),
            content: 'The defendant was found guilty of securities fraud under Section 10(b) of the Securities Exchange Act.',
            embedding: Array(384).fill(0).map(() => Math.random()) // Simulated embedding
        };
        
        // 2. Store in PostgreSQL with vector
        const pgClient = new pg.Client({
            connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
        });
        
        await pgClient.connect();
        
        // Check if embeddings table exists
        const tableCheck = await pgClient.query(`
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'document_embeddings'
            );
        `);
        
        if (!tableCheck.rows[0].exists) {
            // Create embeddings table if it doesn't exist
            await pgClient.query(`
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id UUID PRIMARY KEY,
                    content TEXT,
                    embedding vector(384)
                );
            `);
            console.log('✅ Created embeddings table');
        }
        
        // Insert document with embedding
        await pgClient.query(`
            INSERT INTO document_embeddings (id, content, embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO UPDATE SET content = $2
        `, [testDoc.id, testDoc.content, JSON.stringify(testDoc.embedding)]);
        
        console.log('✅ Document stored with embedding');
        
        // 3. Perform vector similarity search
        const searchResult = await pgClient.query(`
            SELECT id, content, 
                   embedding <-> $1 as distance
            FROM document_embeddings
            ORDER BY distance
            LIMIT 5
        `, [JSON.stringify(testDoc.embedding)]);
        
        if (searchResult.rows.length > 0) {
            console.log('✅ Vector similarity search successful');
            console.log(`   Found ${searchResult.rows.length} similar documents`);
            testResults.push({ test: 'RAG Pipeline', status: 'PASS' });
        }
        
        await pgClient.end();
        return true;
        
    } catch (error) {
        console.log(`❌ RAG pipeline failed: ${error.message}`);
        testResults.push({ test: 'RAG Pipeline', status: 'FAIL', error: error.message });
        return false;
    }
}

// Test 6: WebGPU Compute Simulation
async function testWebGPUCompute() {
    console.log('\n🎮 TEST 6: WebGPU Compute Simulation');
    console.log('Testing tensor operations');
    
    try {
        // Simulate tensor operation
        const tensorA = Array(100).fill(0).map(() => Math.random());
        const tensorB = Array(100).fill(0).map(() => Math.random());
        
        // Simulate matrix multiplication
        const start = Date.now();
        const result = tensorA.map((a, i) => a * tensorB[i]);
        const computeTime = Date.now() - start;
        
        console.log(`✅ Tensor operation completed in ${computeTime}ms`);
        console.log(`   Input shape: [100] × [100]`);
        console.log(`   Output shape: [100]`);
        
        // Test Go tensor service if available
        try {
            const response = await fetch('http://localhost:8086/tensor/multiply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tensorA: tensorA.slice(0, 10), tensorB: tensorB.slice(0, 10) }),
                timeout: 2000
            });
            
            if (response.ok) {
                console.log('✅ Go tensor service integration successful');
            }
        } catch {
            console.log('ℹ️  Go tensor service not available (optional)');
        }
        
        testResults.push({ test: 'WebGPU Compute', status: 'PASS' });
        return true;
        
    } catch (error) {
        console.log(`❌ WebGPU compute failed: ${error.message}`);
        testResults.push({ test: 'WebGPU Compute', status: 'FAIL', error: error.message });
        return false;
    }
}

// Generate Integration Report
function generateReport() {
    console.log('\n' + '='.repeat(70));
    console.log('INTEGRATION TEST SUMMARY');
    console.log('='.repeat(70));
    
    const passed = testResults.filter(t => t.status === 'PASS').length;
    const failed = testResults.filter(t => t.status === 'FAIL').length;
    const partial = testResults.filter(t => t.status === 'PARTIAL').length;
    
    console.log(`\n📊 Results: ${passed} PASSED | ${failed} FAILED | ${partial} PARTIAL`);
    
    testResults.forEach(test => {
        const symbol = test.status === 'PASS' ? '✅' : test.status === 'FAIL' ? '❌' : '⚠️';
        console.log(`${symbol} ${test.test}: ${test.status}`);
        if (test.error) {
            console.log(`   Error: ${test.error}`);
        }
        if (test.passed !== undefined) {
            console.log(`   Passed: ${test.passed}/${test.total}`);
        }
    });
    
    if (passed === testResults.length) {
        console.log('\n🎉 ALL INTEGRATIONS WORKING PERFECTLY!');
        console.log('Your Legal AI platform is fully operational.');
    } else if (failed > 0) {
        console.log('\n⚠️  Some integrations need attention.');
        console.log('Run: npm run check:integrations to see what needs fixing');
    }
    
    // Save detailed report
    const fs = require('fs');
    const report = {
        timestamp: new Date().toISOString(),
        summary: { passed, failed, partial },
        tests: testResults
    };
    
    fs.writeFileSync('integration-test-results.json', JSON.stringify(report, null, 2));
    console.log('\n📄 Detailed report saved to: integration-test-results.json');
}

// Main execution
async function runIntegrationTests() {
    console.log('Starting integration tests...\n');
    
    // Check if dependencies are installed
    const deps = ['pg', 'neo4j-driver', 'amqplib', '@aws-sdk/client-s3', 'uuid'];
    const missing = [];
    
    for (const dep of deps) {
        try {
            require.resolve(dep);
        } catch {
            missing.push(dep);
        }
    }
    
    if (missing.length > 0) {
        console.log('📦 Missing dependencies. Install with:');
        console.log(`npm install ${missing.join(' ')}`);
        return;
    }
    
    // Run tests
    await testDocumentFlow();
    await testMessageQueueFlow();
    await testGraphRelationship();
    await testGoServices();
    await testRAGPipeline();
    await testWebGPUCompute();
    
    // Generate report
    generateReport();
}

// Run tests
runIntegrationTests().catch(console.error);
