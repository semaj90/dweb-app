// Evidence Synthesis API Test - Local Ollama
// Run: node test-synthesis-local.mjs

const API_BASE = 'http://localhost:5173';

const testSynthesis = async () => {
  console.log('🔬 Testing Evidence Synthesis with Local Ollama');
  
  const testRequest = {
    evidenceIds: ['test-evidence-1', 'test-evidence-2'],
    synthesisType: 'correlation',
    caseId: 'test-case-123',
    title: 'Local Ollama Synthesis Test',
    description: 'Validation of synthesis pipeline with local AI'
  };

  try {
    const response = await fetch(`${API_BASE}/api/evidence/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token'
      },
      body: JSON.stringify(testRequest)
    });

    console.log(`Status: ${response.status}`);
    
    if (response.status === 401) {
      console.log('✅ Endpoint accessible - auth required (expected)');
      return;
    }
    
    const result = await response.json();
    console.log('Response:', JSON.stringify(result, null, 2));
    
    if (result.success) {
      console.log('✅ Synthesis successful');
      console.log(`RAG Score: ${result.metadata.ragScore}`);
      console.log(`Confidence: ${result.metadata.confidence}`);
    }
    
  } catch (error) {
    console.log(`❌ Request failed: ${error.message}`);
    
    if (error.code === 'ECONNREFUSED') {
      console.log('Dev server not running. Start with: npm run dev');
    }
  }
};

// Test suggestions endpoint
const testSuggestions = async () => {
  console.log('\n💡 Testing Synthesis Suggestions');
  
  try {
    const response = await fetch(`${API_BASE}/api/evidence/synthesize?caseId=test-case-123`);
    console.log(`Suggestions Status: ${response.status}`);
    
    if (response.ok) {
      const result = await response.json();
      console.log(`Suggestions: ${result.suggestions.length}`);
    }
  } catch (error) {
    console.log(`Suggestions failed: ${error.message}`);
  }
};

// Execute tests
testSynthesis().then(() => testSuggestions());
