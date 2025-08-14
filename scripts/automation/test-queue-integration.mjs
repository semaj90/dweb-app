import { fetch } from 'node-fetch';

const API_BASE = 'http://localhost:5173';
const GO_SERVER = 'http://localhost:8080';

console.log('🧪 Testing Legal AI Queue Integration...');
console.log('====================================');

// Test 1: Check Go server health
console.log('\n1. Testing Go server health...');
try {
  const response = await fetch(`${GO_SERVER}/health`);
  if (response.ok) {
    const health = await response.json();
    console.log(`✅ Go server healthy: v${health.version}`);
    console.log(`   - Database: ${health.database}`);
    console.log(`   - Ollama: ${health.ollama}`);
  } else {
    console.log(`❌ Go server unhealthy: ${response.status}`);
  }
} catch (error) {
  console.log(`❌ Go server unreachable: ${error.message}`);
}

// Test 2: Check queue status
console.log('\n2. Testing queue status...');
try {
  const response = await fetch(`${API_BASE}/api/queue/status`);
  if (response.ok) {
    const status = await response.json();
    console.log(`✅ Queue accessible`);
    console.log(`   - Waiting: ${status.queue_stats.waiting}`);
    console.log(`   - Active: ${status.queue_stats.active}`);
    console.log(`   - Completed: ${status.queue_stats.completed}`);
  } else {
    console.log(`❌ Queue status failed: ${response.status}`);
  }
} catch (error) {
  console.log(`❌ Queue unreachable: ${error.message}`);
}

// Test 3: Submit a document processing job
console.log('\n3. Testing document processing job submission...');
const testDocument = {
  content: "This is a test legal document for contract analysis. The parties involved are ABC Corp and XYZ Ltd. The contract amount is $50,000.",
  document_type: "contract",
  case_id: "TEST-CASE-001",
  extract_entities: true,
  generate_summary: true,
  assess_risk: true,
  generate_embedding: true,
  store_in_database: false, // Don't store test data
  priority: 5 // Higher priority for test
};

try {
  const response = await fetch(`${API_BASE}/api/legal-ai/process-document`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(testDocument)
  });
  
  if (response.ok) {
    const result = await response.json();
    console.log(`✅ Job submitted successfully`);
    
    if (result.queued) {
      console.log(`   - Job ID: ${result.job_id}`);
      console.log(`   - Estimated: ${result.estimated_seconds}s`);
      console.log(`   - Status URL: ${result.status_url}`);
      
      // Test 4: Check job status
      console.log('\n4. Testing job status check...');
      let completed = false;
      let attempts = 0;
      const maxAttempts = 10;
      
      while (!completed && attempts < maxAttempts) {
        attempts++;
        console.log(`   Attempt ${attempts}: Checking job status...`);
        
        try {
          const statusResponse = await fetch(`${API_BASE}${result.status_url}`);
          if (statusResponse.ok) {
            const jobStatus = await statusResponse.json();
            console.log(`   - Status: ${jobStatus.status}`);
            console.log(`   - Progress: ${jobStatus.progress || 0}%`);
            
            if (jobStatus.status === 'completed') {
              console.log(`✅ Job completed successfully!`);
              if (jobStatus.result) {
                console.log(`   - Processing time: ${jobStatus.result.processingTime}`);
                console.log(`   - Has summary: ${!!jobStatus.result.summary}`);
                console.log(`   - Entities found: ${jobStatus.result.entities?.length || 0}`);
                console.log(`   - Has embedding: ${jobStatus.result.hasEmbedding}`);
              }
              completed = true;
            } else if (jobStatus.status === 'failed') {
              console.log(`❌ Job failed: ${jobStatus.error}`);
              completed = true;
            } else {
              // Still processing, wait a bit
              await new Promise(resolve => setTimeout(resolve, 3000));
            }
          } else {
            console.log(`❌ Status check failed: ${statusResponse.status}`);
          }
        } catch (statusError) {
          console.log(`❌ Status check error: ${statusError.message}`);
        }
      }
      
      if (!completed) {
        console.log(`⚠️  Job did not complete within ${maxAttempts} attempts`);
      }
      
    } else {
      console.log(`✅ Processed directly (queue disabled)`);
      if (result.data) {
        console.log(`   - Processing time: ${result.data.processing_time}`);
        console.log(`   - Has summary: ${!!result.data.summary}`);
        console.log(`   - Entities found: ${result.data.entities?.length || 0}`);
      }
    }
    
  } else {
    const error = await response.text();
    console.log(`❌ Job submission failed: ${response.status}`);
    console.log(`   Error: ${error}`);
  }
} catch (error) {
  console.log(`❌ Job submission error: ${error.message}`);
}

console.log('\n====================================');
console.log('🧪 Queue Integration Test Complete');