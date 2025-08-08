// Test .env configuration
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env file
dotenv.config({ path: join(__dirname, '.env') });

console.log('🔍 Environment Configuration Test\n');
console.log('=' .repeat(50));

// Test critical environment variables
const criticalVars = [
    'DATABASE_URL',
    'PGPASSWORD',
    'GPU_ENABLED',
    'REDIS_URL',
    'API_URL',
    'CUDA_PATH',
    'CGO_ENABLED'
];

console.log('\n📋 Critical Variables:');
criticalVars.forEach(varName => {
    const value = process.env[varName];
    if (value) {
        console.log(`✅ ${varName}: ${value.substring(0, 50)}...`);
    } else {
        console.log(`❌ ${varName}: NOT SET`);
    }
});

// Test database connection with env vars
console.log('\n🔍 Testing Database Connection with .env:');
console.log(`   URL: ${process.env.DATABASE_URL}`);

import postgres from 'postgres';

async function testConnection() {
    try {
        const sql = postgres(process.env.DATABASE_URL, {
            max: 1,
            ssl: false
        });
        
        const result = await sql`SELECT current_user, current_database()`;
        console.log(`✅ Connected as: ${result[0].current_user}`);
        console.log(`✅ Database: ${result[0].current_database}`);
        
        await sql.end();
    } catch (error) {
        console.error(`❌ Connection failed: ${error.message}`);
    }
}

await testConnection();

// Test Redis connection
console.log('\n🔍 Testing Redis Connection:');
import { createClient } from 'redis';

try {
    const client = createClient({
        url: process.env.REDIS_URL
    });
    
    await client.connect();
    await client.ping();
    console.log('✅ Redis connected');
    await client.quit();
} catch (error) {
    console.error(`❌ Redis failed: ${error.message}`);
}

// Check GPU status
console.log('\n🔍 GPU Configuration:');
console.log(`   GPU Enabled: ${process.env.GPU_ENABLED}`);
console.log(`   CUDA Path: ${process.env.CUDA_PATH}`);
console.log(`   CGO Enabled: ${process.env.CGO_ENABLED}`);

// Check service URLs
console.log('\n🔍 Service URLs:');
console.log(`   API: ${process.env.API_URL}`);
console.log(`   Indexer: ${process.env.INDEXER_URL}`);
console.log(`   BullMQ: ${process.env.BULLMQ_URL}`);

console.log('\n' + '=' .repeat(50));
console.log('✅ Environment configuration loaded successfully!');
