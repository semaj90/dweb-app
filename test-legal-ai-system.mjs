// test-legal-ai-system.mjs
// Test script to verify the Legal AI system is working

import axios from 'axios';
import { Queue } from 'bullmq';
import Redis from 'ioredis';

const GO_SERVER_URL = 'http://localhost:8080';
const redis = new Redis({
  host: 'localhost',
  port: 6379,
  maxRetriesPerRequest: null
});

async function testSystem() {
  console.log('🧪 Testing Legal AI System Components\n');
  
  let allTestsPassed = true;
  
  // Test 1: Go Server Health
  console.log('1️⃣ Testing Go Server...');
  try {
    const health = await axios.get(`${GO_SERVER_URL}/health`);
    console.log('   ✅ Go Server is healthy');
    console.log(`   - Version: ${health.data.version}`);
    console.log(`   - Ollama: ${health.data.ollama}`);
    console.log(`   - Database: ${health.data.database}`);
  } catch (error) {
    console.log('   ❌ Go Server is not responding');
    allTestsPassed = false;
  }
  
  // Test 2: Redis Connection
  console.log('\n2️⃣ Testing Redis...');
  try {
    await redis.ping();
    console.log('   ✅ Redis is connected');
  } catch (error) {
    console.log('   ❌ Redis is not available');
    allTestsPassed = false;
  }
  
  // Test 3: Document Processing
  console.log('\n3️⃣ Testing Document Processing...');
  try {
    const testDoc = {
      document_id: `test-${Date.now()}`,
      content: 'This is a test legal document. The agreement between Party A and Party B states that payment of $10,000 is due by December 31, 2024.',
      document_type: 'contract',
      case_id: 'TEST-001',
      options: {
        extract_entities: true,
        generate_summary: true,
        assess_risk: true,
        generate_embedding: false,
        store_in_database: false
      }
    };
    
    const response = await axios.post(`${GO_SERVER_URL}/process-document`, testDoc);
    
    if (response.data.success) {
      console.log('   ✅ Document processing successful');
      console.log(`   - Processing time: ${response.data.processing_time}`);
      if (response.data.summary) {
        console.log(`   - Summary generated: ${response.data.summary.substring(0, 50)}...`);
      }
      if (response.data.entities && response.data.entities.length > 0) {
        console.log(`   - Entities found: ${response.data.entities.length}`);
      }
      if (response.data.risk_assessment) {
        console.log(`   - Risk level: ${response.data.risk_assessment.overall_risk}`);
      }
    } else {
      console.log('   ⚠️ Document processing completed with warnings');
    }
  } catch (error) {
    console.log('   ❌ Document processing failed:', error.message);
    allTestsPassed = false;
  }
  
  // Test 4: BullMQ Queue
  console.log('\n4️⃣ Testing Job Queue...');
  try {
    const queue = new Queue('document-processing', { connection: redis });
    const job = await queue.add('test-job', {
      documentId: `test-${Date.now()}`,
      content: 'Test content'
    });
    console.log(`   ✅ Job queued successfully (ID: ${job.id})`);
    await queue.close();
  } catch (error) {
    console.log('   ❌ Queue system error:', error.message);
    allTestsPassed = false;
  }
  
  // Test 5: Ollama Connection
  console.log('\n5️⃣ Testing Ollama...');
  try {
    const ollamaStatus = await axios.get(`${GO_SERVER_URL}/ollama-status`);
    if (ollamaStatus.data.connected) {
      console.log('   ✅ Ollama is connected');
      if (ollamaStatus.data.models) {
        console.log('   - Available models:', Object.keys(ollamaStatus.data.models).length);
      }
    } else {
      console.log('   ⚠️ Ollama is not connected');
    }
  } catch (error) {
    console.log('   ❌ Could not check Ollama status');
  }
  
  // Test 6: Database Connection
  console.log('\n6️⃣ Testing Database...');
  try {
    const dbStatus = await axios.get(`${GO_SERVER_URL}/database-status`);
    if (dbStatus.data.connected) {
      console.log('   ✅ Database is connected');
      console.log(`   - Pool size: ${dbStatus.data.pool_size}`);
    } else {
      console.log('   ⚠️ Database is not connected');
    }
  } catch (error) {
    console.log('   ❌ Could not check database status');
  }
  
  // Final Report
  console.log('\n' + '='.repeat(50));
  if (allTestsPassed) {
    console.log('✅ All systems operational!');
    console.log('\nYour Legal AI system is ready to use:');
    console.log('🌐 Frontend: http://localhost:5173');
    console.log('🚀 API: http://localhost:8080');
  } else {
    console.log('⚠️ Some components need attention');
    console.log('\nTroubleshooting:');
    console.log('1. Run: .\\Complete-Legal-AI-Startup.ps1');
    console.log('2. Check logs: pm2 logs');
    console.log('3. Verify services are running');
  }
  
  await redis.quit();
}

// Run tests
testSystem().catch(console.error);
