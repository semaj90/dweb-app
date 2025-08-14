// Quick Evidence Synthesis API Test
// File: test-synthesis-api.mjs

const API_BASE = 'http://localhost:5173';

async function testSynthesisEndpoint() {
  console.log('🔬 Testing Evidence Synthesis API');
  
  // Mock request data
  const synthesisRequest = {
    evidenceIds: ['mock-id-1', 'mock-id-2'],
    synthesisType: 'correlation',
    caseId: 'mock-case-id',
    title: 'Test Synthesis',
    description: 'API endpoint validation'
  };

  try {
    const response = await fetch(`${API_BASE}/api/evidence/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer mock-token'
      },
      body: JSON.stringify(synthesisRequest)
    });

    console.log(`Status: ${response.status}`);
    
    if (response.status === 401) {
      console.log('✅ API endpoint accessible - authentication required');
      return true;
    }
    
    const result = await response.json();
    console.log('Response:', result);
    
    return response.ok;
  } catch (error) {
    console.log(`❌ API not available: ${error.message}`);
    return false;
  }
}

// Execute test
testSynthesisEndpoint()
  .then(success => {
    console.log(success ? '✅ API Ready' : '❌ API Issues');
    process.exit(success ? 0 : 1);
  })
  .catch(console.error);
