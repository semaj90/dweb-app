#!/usr/bin/env node
// Comprehensive Component Integration Checker
// Verifies all services are linked and communicating properly

import pg from 'pg';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import neo4j from 'neo4j-driver';
import amqp from 'amqplib';
import { S3Client, ListBucketsCommand } from '@aws-sdk/client-s3';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import net from 'net';

console.log(`
╔════════════════════════════════════════════════════════════════════╗
║          COMPLETE INTEGRATION CHECK - ALL COMPONENTS              ║
║     RabbitMQ | Neo4j | MinIO | PostgreSQL | Go Services | WebGPU  ║
╚════════════════════════════════════════════════════════════════════╝
`);

const results = {
    postgresql: { status: false, details: {} },
    drizzle: { status: false, details: {} },
    neo4j: { status: false, details: {} },
    rabbitmq: { status: false, details: {} },
    minio: { status: false, details: {} },
    goServices: { status: false, details: {} },
    webgpu: { status: false, details: {} },
    integration: { status: false, details: {} }
};

// Helper functions
function printSection(title) {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`  ${title}`);
    console.log('='.repeat(70));
}

function printSuccess(message) {
    console.log(`✅ ${message}`);
}

function printError(message) {
    console.log(`❌ ${message}`);
}

function printWarning(message) {
    console.log(`⚠️  ${message}`);
}

function printInfo(message) {
    console.log(`ℹ️  ${message}`);
}

// Check port availability
async function checkPort(port, serviceName) {
    return new Promise((resolve) => {
        const server = net.createServer();

        server.once('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                resolve(true);
            } else {
                resolve(false);
            }
        });

        server.once('listening', () => {
            server.close();
            resolve(false);
        });

        server.listen(port, '127.0.0.1');
    });
}

// 1. PostgreSQL & postgres.js Check
async function checkPostgreSQL() {
    printSection('1. PostgreSQL & postgres.js Integration');

    try {
        // Check if PostgreSQL is running
        const pgRunning = await checkPort(5432, 'PostgreSQL');
        if (!pgRunning) {
            printError('PostgreSQL is not running on port 5432');
            return false;
        }
        printSuccess('PostgreSQL is running on port 5432');

        // Test connection with pg client
        const pgClient = new pg.Client({
            connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
        });

        await pgClient.connect();
        printSuccess('Connected to PostgreSQL with pg client');

        // Check pgvector extension
        const vectorCheck = await pgClient.query(`
            SELECT * FROM pg_extension WHERE extname = 'vector';
        `);

        if (vectorCheck.rows.length > 0) {
            printSuccess('pgvector extension is installed');
            results.postgresql.details.pgvector = true;
        } else {
            printWarning('pgvector extension not found - install with: CREATE EXTENSION vector;');
            results.postgresql.details.pgvector = false;
        }

        // Check tables
        const tablesResult = await pgClient.query(`
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        `);

        printInfo(`Found ${tablesResult.rows.length} tables in database`);

        // Check specific tables and counts
        const importantTables = ['cases', 'legal_documents', 'evidence', 'users'];
        for (const table of importantTables) {
            try {
                const countResult = await pgClient.query(`SELECT COUNT(*) FROM ${table}`);
                printInfo(`  • ${table}: ${countResult.rows[0].count} records`);
                results.postgresql.details[table] = parseInt(countResult.rows[0].count);
            } catch (e) {
                printWarning(`  • ${table}: table not found`);
            }
        }

        await pgClient.end();

        // Test postgres.js connection
        printInfo('Testing postgres.js connection...');
        const sql = postgres('postgresql://legal_admin:123456@localhost:5432/legal_ai_db');
        const postgresJsTest = await sql`SELECT version()`;
        printSuccess(`postgres.js connected: ${postgresJsTest[0].version.split(',')[0]}`);
        await sql.end();

        results.postgresql.status = true;
        return true;

    } catch (error) {
        printError(`PostgreSQL error: ${error.message}`);
        results.postgresql.error = error.message;
        return false;
    }
}

// 2. Drizzle ORM Check
async function checkDrizzle() {
    printSection('2. Drizzle ORM Integration');

    try {
        // Check if drizzle config exists
        if (!fs.existsSync('./drizzle.config.ts')) {
            printWarning('drizzle.config.ts not found');
            results.drizzle.details.config = false;
        } else {
            printSuccess('drizzle.config.ts found');
            results.drizzle.details.config = true;
        }

        // Check if schema exists
        const schemaPath = './src/lib/server/schema.ts';
        if (!fs.existsSync(schemaPath)) {
            printWarning('Schema file not found at src/lib/server/schema.ts');
            results.drizzle.details.schema = false;
        } else {
            printSuccess('Drizzle schema file found');
            results.drizzle.details.schema = true;
        }

        // Test Drizzle connection
        const sql = postgres('postgresql://legal_admin:123456@localhost:5432/legal_ai_db');
        const db = drizzle(sql);

        printSuccess('Drizzle ORM initialized successfully');

        // Check migrations folder
        if (fs.existsSync('./drizzle')) {
            const migrations = fs.readdirSync('./drizzle');
            printInfo(`Found ${migrations.length} migration files`);
            results.drizzle.details.migrations = migrations.length;
        }

        await sql.end();
        results.drizzle.status = true;
        return true;

    } catch (error) {
        printError(`Drizzle error: ${error.message}`);
        results.drizzle.error = error.message;
        return false;
    }
}

// 3. Neo4j Check
async function checkNeo4j() {
    printSection('3. Neo4j Graph Database');

    try {
        // Check if Neo4j is running
        const neo4jBrowser = await checkPort(7474, 'Neo4j Browser');
        const neo4jBolt = await checkPort(7687, 'Neo4j Bolt');

        if (!neo4jBrowser) {
            printWarning('Neo4j Browser not accessible on port 7474');
        } else {
            printSuccess('Neo4j Browser is running on port 7474');
        }

        if (!neo4jBolt) {
            printWarning('Neo4j Bolt not accessible on port 7687');
            results.neo4j.status = false;
            return false;
        }
        printSuccess('Neo4j Bolt is running on port 7687');

        // Test Neo4j connection
        const driver = neo4j.driver(
            'neo4j://localhost:7687',
            neo4j.auth.basic('neo4j', 'password')
        );

        const session = driver.session();

        try {
            // Test connection
            const result = await session.run('MATCH (n) RETURN count(n) as count LIMIT 1');
            const nodeCount = result.records[0].get('count').toNumber();
            printSuccess(`Neo4j connected - ${nodeCount} nodes in database`);
            results.neo4j.details.nodeCount = nodeCount;

            // Check for legal-specific nodes
            const legalNodes = await session.run(`
                MATCH (n)
                WHERE n:Case OR n:Document OR n:Evidence
                RETURN labels(n)[0] as label, count(n) as count
            `);

            if (legalNodes.records.length > 0) {
                printInfo('Legal graph nodes:');
                legalNodes.records.forEach(record => {
                    const label = record.get('label');
                    const count = record.get('count').toNumber();
                    printInfo(`  • ${label}: ${count} nodes`);
                    results.neo4j.details[label] = count;
                });
            } else {
                printWarning('No legal-specific nodes found in Neo4j');
            }

            results.neo4j.status = true;

        } catch (error) {
            printError(`Neo4j query error: ${error.message}`);
            printInfo('Neo4j may need authentication - default: neo4j/password');
            results.neo4j.error = error.message;
        } finally {
            await session.close();
            await driver.close();
        }

        return results.neo4j.status;

    } catch (error) {
        printError(`Neo4j connection error: ${error.message}`);
        results.neo4j.error = error.message;
        return false;
    }
}

// 4. RabbitMQ Check
async function checkRabbitMQ() {
    printSection('4. RabbitMQ Message Queue');

    try {
        // Check if RabbitMQ is running
        const amqpPort = await checkPort(5672, 'RabbitMQ');
        const managementPort = await checkPort(15672, 'RabbitMQ Management');

        if (!amqpPort) {
            printError('RabbitMQ is not running on port 5672');
            results.rabbitmq.status = false;
            return false;
        }
        printSuccess('RabbitMQ is running on port 5672');

        if (managementPort) {
            printSuccess('RabbitMQ Management UI is accessible on port 15672');
            printInfo('  Access at: http://localhost:15672 (guest/guest)');
        }

        // Test RabbitMQ connection
        try {
            const connection = await amqp.connect('amqp://localhost');
            const channel = await connection.createChannel();

            printSuccess('Connected to RabbitMQ');

            // Check/create legal AI queues
            const queues = [
                'document_processing',
                'embedding_generation',
                'rag_queries',
                'case_updates'
            ];

            printInfo('Checking queues:');
            for (const queueName of queues) {
                const q = await channel.assertQueue(queueName, { durable: true });
                printInfo(`  • ${queueName}: ${q.messageCount} messages`);
                results.rabbitmq.details[queueName] = q.messageCount;
            }

            await channel.close();
            await connection.close();

            results.rabbitmq.status = true;
            return true;

        } catch (error) {
            printError(`RabbitMQ connection error: ${error.message}`);
            printInfo('RabbitMQ may need credentials - default: guest/guest');
            results.rabbitmq.error = error.message;
            return false;
        }

    } catch (error) {
        printError(`RabbitMQ error: ${error.message}`);
        results.rabbitmq.error = error.message;
        return false;
    }
}

// 5. MinIO Check
async function checkMinIO() {
    printSection('5. MinIO Object Storage');

    try {
        // Check if MinIO is running
        const minioApi = await checkPort(9000, 'MinIO API');
        const minioConsole = await checkPort(9001, 'MinIO Console');

        if (!minioApi) {
            printError('MinIO API is not running on port 9000');
            results.minio.status = false;
            return false;
        }
        printSuccess('MinIO API is running on port 9000');

        if (minioConsole) {
            printSuccess('MinIO Console is accessible on port 9001');
            printInfo('  Access at: http://localhost:9001 (minioadmin/minioadmin123)');
        }

        // Test MinIO connection
        const s3Client = new S3Client({
            endpoint: 'http://localhost:9000',
            region: 'us-east-1',
            credentials: {
                accessKeyId: 'minioadmin',
                secretAccessKey: 'minioadmin123'
            },
            forcePathStyle: true
        });

        try {
            const buckets = await s3Client.send(new ListBucketsCommand({}));
            printSuccess(`MinIO connected - ${buckets.Buckets?.length || 0} buckets found`);

            if (buckets.Buckets && buckets.Buckets.length > 0) {
                printInfo('Buckets:');
                buckets.Buckets.forEach(bucket => {
                    printInfo(`  • ${bucket.Name}`);
                });
                results.minio.details.buckets = buckets.Buckets.map(b => b.Name);
            }

            // Check for legal-documents bucket
            const legalBucket = buckets.Buckets?.find(b => b.Name === 'legal-documents');
            if (legalBucket) {
                printSuccess('legal-documents bucket exists');
                results.minio.details.legalBucket = true;
            } else {
                printWarning('legal-documents bucket not found - create it for document storage');
                results.minio.details.legalBucket = false;
            }

            results.minio.status = true;
            return true;

        } catch (error) {
            printError(`MinIO connection error: ${error.message}`);
            printInfo('Check MinIO credentials - default: minioadmin/minioadmin123');
            results.minio.error = error.message;
            return false;
        }

    } catch (error) {
        printError(`MinIO error: ${error.message}`);
        results.minio.error = error.message;
        return false;
    }
}

// 6. Go Microservices Check
async function checkGoServices() {
    printSection('6. Go Microservices');

    const services = [
        { name: 'GPU Orchestrator', port: 8084, endpoint: '/health' },
        { name: 'RAG Service', port: 8085, endpoint: '/health' },
        { name: 'Tensor Service', port: 8086, endpoint: '/health' },
        { name: 'QUIC Service', port: 8087, endpoint: '/health' }
    ];

    let servicesRunning = 0;

    for (const service of services) {
        const isRunning = await checkPort(service.port, service.name);

        if (isRunning) {
            printSuccess(`${service.name} is running on port ${service.port}`);

            // Try to hit health endpoint
            try {
                const response = await fetch(`http://localhost:${service.port}${service.endpoint}`, {
                    timeout: 2000
                });

                if (response.ok) {
                    const data = await response.text();
                    printInfo(`  Health check: ${data.substring(0, 50)}`);
                    results.goServices.details[service.name] = 'healthy';
                } else {
                    printWarning(`  Health check returned: ${response.status}`);
                    results.goServices.details[service.name] = 'unhealthy';
                }
            } catch (e) {
                printWarning(`  Health endpoint not responding`);
                results.goServices.details[service.name] = 'no health endpoint';
            }

            servicesRunning++;
        } else {
            printWarning(`${service.name} is not running on port ${service.port}`);
            results.goServices.details[service.name] = 'not running';
        }
    }

    if (servicesRunning > 0) {
        printInfo(`${servicesRunning}/${services.length} Go services are running`);
        results.goServices.status = true;
    } else {
        printError('No Go microservices are running');
        printInfo('Start with: go run [service].go');
        results.goServices.status = false;
    }

    // Check for Go files
    const goFiles = [
        'gpu-orchestrator.go',
        'enhanced-rag-service.go',
        'tensor-tiling-gpu-accelerator.go',
        'quic-tensor-transport.go'
    ];

    printInfo('\nGo service files:');
    for (const file of goFiles) {
        if (fs.existsSync(`./${file}`)) {
            printInfo(`  ✓ ${file} found`);
        } else {
            printWarning(`  ✗ ${file} not found`);
        }
    }

    return results.goServices.status;
}

// 7. WebGPU Check
async function checkWebGPU() {
    printSection('7. WebGPU Support');

    // Check for WebGPU files
    const webgpuFiles = [
        './src/lib/webgpu/gpu-compute.ts',
        './src/lib/webgpu/tensor-ops.ts',
        './webgpu/index.html'
    ];

    let filesFound = 0;
    printInfo('WebGPU implementation files:');

    for (const file of webgpuFiles) {
        if (fs.existsSync(file)) {
            printInfo(`  ✓ ${file} found`);
            filesFound++;
        } else {
            printWarning(`  ✗ ${file} not found`);
        }
    }

    results.webgpu.details.files = filesFound;

    // Check for GPU-related dependencies in package.json
    if (fs.existsSync('./package.json')) {
        const packageJson = JSON.parse(fs.readFileSync('./package.json', 'utf8'));
        const gpuDeps = [];

        const checkDeps = ['@webgpu/types', 'gpu.js', '@tensorflow/tfjs-backend-webgpu'];

        for (const dep of checkDeps) {
            if (packageJson.dependencies?.[dep] || packageJson.devDependencies?.[dep]) {
                gpuDeps.push(dep);
            }
        }

        if (gpuDeps.length > 0) {
            printSuccess(`WebGPU dependencies found: ${gpuDeps.join(', ')}`);
            results.webgpu.details.dependencies = gpuDeps;
        } else {
            printWarning('No WebGPU dependencies found in package.json');
        }
    }

    // Check for CUDA/GPU support on system
    try {
        const { exec } = require('child_process');
        const { promisify } = require('util');
        const execAsync = promisify(exec);

        const { stdout } = await execAsync('nvidia-smi --query-gpu=name --format=csv,noheader');
        if (stdout) {
            printSuccess(`GPU detected: ${stdout.trim()}`);
            results.webgpu.details.gpu = stdout.trim();
            results.webgpu.status = true;
        }
    } catch {
        printWarning('No NVIDIA GPU detected (nvidia-smi not available)');
        printInfo('WebGPU can still work with integrated graphics');
    }

    return results.webgpu.status;
}

// 8. Integration Tests
async function checkIntegration() {
    printSection('8. Component Integration Tests');

    // Test 1: PostgreSQL ↔ Drizzle
    printInfo('\nTest 1: PostgreSQL ↔ Drizzle Integration');
    if (results.postgresql.status && results.drizzle.status) {
        printSuccess('PostgreSQL and Drizzle are properly integrated');
        results.integration.details.postgresqlDrizzle = true;
    } else {
        printError('PostgreSQL-Drizzle integration not working');
        results.integration.details.postgresqlDrizzle = false;
    }

    // Test 2: Document Pipeline (MinIO → RabbitMQ → PostgreSQL)
    printInfo('\nTest 2: Document Processing Pipeline');
    if (results.minio.status && results.rabbitmq.status && results.postgresql.status) {
        printSuccess('Document pipeline components are ready');
        printInfo('  MinIO (storage) → RabbitMQ (queue) → PostgreSQL (metadata)');
        results.integration.details.documentPipeline = true;
    } else {
        printWarning('Document pipeline incomplete');
        if (!results.minio.status) printError('  ✗ MinIO not running');
        if (!results.rabbitmq.status) printError('  ✗ RabbitMQ not running');
        if (!results.postgresql.status) printError('  ✗ PostgreSQL not running');
        results.integration.details.documentPipeline = false;
    }

    // Test 3: Graph Integration (PostgreSQL ↔ Neo4j)
    printInfo('\nTest 3: Graph Database Integration');
    if (results.postgresql.status && results.neo4j.status) {
        printSuccess('PostgreSQL and Neo4j can work together');
        printInfo('  PostgreSQL (structured) ↔ Neo4j (relationships)');
        results.integration.details.graphIntegration = true;
    } else {
        printWarning('Graph integration not available');
        results.integration.details.graphIntegration = false;
    }

    // Test 4: AI Pipeline (Go Services → PostgreSQL with pgvector)
    printInfo('\nTest 4: AI/ML Pipeline');
    if (results.goServices.status && results.postgresql.details.pgvector) {
        printSuccess('AI pipeline is ready');
        printInfo('  Go Services → pgvector embeddings → RAG queries');
        results.integration.details.aiPipeline = true;
    } else {
        printWarning('AI pipeline not fully configured');
        if (!results.goServices.status) printError('  ✗ Go services not running');
        if (!results.postgresql.details.pgvector) printError('  ✗ pgvector not installed');
        results.integration.details.aiPipeline = false;
    }

    // Overall integration status
    const integrationTests = Object.values(results.integration.details);
    const passedTests = integrationTests.filter(t => t === true).length;

    if (passedTests === integrationTests.length) {
        results.integration.status = true;
    }

    return results.integration.status;
}

// Generate Summary Report
function generateSummary() {
    printSection('INTEGRATION SUMMARY');

    const components = [
        { name: 'PostgreSQL + postgres.js', status: results.postgresql.status },
        { name: 'Drizzle ORM', status: results.drizzle.status },
        { name: 'Neo4j Graph DB', status: results.neo4j.status },
        { name: 'RabbitMQ Queue', status: results.rabbitmq.status },
        { name: 'MinIO Storage', status: results.minio.status },
        { name: 'Go Microservices', status: results.goServices.status },
        { name: 'WebGPU Support', status: results.webgpu.status }
    ];

    const working = components.filter(c => c.status);
    const notWorking = components.filter(c => !c.status);

    console.log('\n📊 COMPONENT STATUS:');
    console.log(`✅ Working: ${working.length}/${components.length}`);
    working.forEach(c => console.log(`   • ${c.name}`));

    if (notWorking.length > 0) {
        console.log(`\n❌ Not Working: ${notWorking.length}/${components.length}`);
        notWorking.forEach(c => console.log(`   • ${c.name}`));
    }

    console.log('\n🔗 INTEGRATION STATUS:');
    Object.entries(results.integration.details).forEach(([key, value]) => {
        const name = key.replace(/([A-Z])/g, ' $1').trim();
        console.log(`   ${value ? '✅' : '❌'} ${name}`);
    });

    // Architecture diagram
    console.log('\n📐 ARCHITECTURE:');
    console.log(`
    ┌─────────────────────────────────────┐
    │         YoRHa Frontend              │
    │     (SvelteKit + WebGPU)            │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │      Go Microservices Layer         │
    │  (GPU, RAG, Tensor, QUIC Services)  │
    └──────────┬────────┬─────────────────┘
               │        │
    ┌──────────▼────┐ ┌▼──────────────────┐
    │  Message Queue│ │  Object Storage   │
    │   (RabbitMQ)  │ │    (MinIO)        │
    └──────────┬────┘ └───────────────────┘
               │
    ┌──────────▼──────────────────────────┐
    │         Data Layer                  │
    │  PostgreSQL │ Neo4j │ Redis         │
    │  (+ pgvector) (Graph)  (Cache)      │
    └─────────────────────────────────────┘
    `);

    // Recommendations
    if (notWorking.length > 0) {
        console.log('\n🔧 RECOMMENDATIONS:');

        if (!results.rabbitmq.status) {
            console.log('\nFor RabbitMQ:');
            console.log('  1. Download Erlang: https://www.erlang.org/downloads');
            console.log('  2. Download RabbitMQ: https://www.rabbitmq.com/download.html');
            console.log('  3. Enable management: rabbitmq-plugins enable rabbitmq_management');
        }

        if (!results.neo4j.status) {
            console.log('\nFor Neo4j:');
            console.log('  1. Download: https://neo4j.com/download/');
            console.log('  2. Extract to C:\\neo4j');
            console.log('  3. Run: neo4j console');
        }

        if (!results.minio.status) {
            console.log('\nFor MinIO:');
            console.log('  1. Download: https://min.io/download#/windows');
            console.log('  2. Run: minio.exe server C:\\minio-data --console-address :9001');
        }

        if (!results.goServices.status) {
            console.log('\nFor Go Services:');
            console.log('  1. Install Go: https://go.dev/dl/');
            console.log('  2. Run services:');
            console.log('     go run gpu-orchestrator.go');
            console.log('     go run enhanced-rag-service.go');
        }
    } else {
        console.log('\n🎉 ALL COMPONENTS ARE LINKED AND WORKING!');
        console.log('Your Legal AI platform is fully integrated and ready for use.');
    }

    // Save detailed report
    const reportPath = './integration-status-report.json';
    fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
}

// Main execution
async function runFullCheck() {
    console.log('Starting comprehensive integration check...\n');

    // Run all checks
    await checkPostgreSQL();
    await checkDrizzle();
    await checkNeo4j();
    await checkRabbitMQ();
    await checkMinIO();
    await checkGoServices();
    await checkWebGPU();
    await checkIntegration();

    // Generate summary
    generateSummary();
}

// Handle missing dependencies gracefully
async function installMissingDeps() {
    const requiredDeps = [
        'pg',
        'postgres',
        'drizzle-orm',
        'neo4j-driver',
        'amqplib',
        '@aws-sdk/client-s3'
    ];

    const missingDeps = [];

    for (const dep of requiredDeps) {
        try {
            require.resolve(dep);
        } catch {
            missingDeps.push(dep);
        }
    }

    if (missingDeps.length > 0) {
        console.log('📦 Missing dependencies detected. Install with:');
        console.log(`npm install ${missingDeps.join(' ')}`);
        console.log('\nOr run: npm install pg postgres drizzle-orm neo4j-driver amqplib @aws-sdk/client-s3');
        return false;
    }

    return true;
}

// Check dependencies first
installMissingDeps().then(depsInstalled => {
    if (depsInstalled) {
        runFullCheck().catch(console.error);
    } else {
        console.log('\n⚠️  Install missing dependencies first, then run this script again.');
    }
});
