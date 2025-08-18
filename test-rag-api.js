// Test script for RAG API endpoints
const API_BASE = 'http://localhost:5177/api/rag';

async function testStatusEndpoint() {
  try {
    console.log('Testing status endpoint...');
    const response = await fetch(`${API_BASE}/status`);
    const data = await response.json();
    console.log('Status response:', data);
    return true;
  } catch (error) {
    console.error('Status endpoint error:', error);
    return false;
  }
}

async function testFileUpload() {
  try {
    console.log('Testing file upload...');

    // Create a test file
    const testContent = 'This is a test document for RAG processing.';
    const file = new File([testContent], 'test.txt', { type: 'text/plain' });

    // Create form data
    const formData = new FormData();
    formData.append('files', file);
    formData.append('enableOCR', 'true');
    formData.append('enableEmbedding', 'true');
    formData.append('enableRAG', 'true');

    const response = await fetch(`${API_BASE}/process`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    console.log('Upload response:', data);
    return true;
  } catch (error) {
    console.error('Upload test error:', error);
    return false;
  }
}

async function testSearch() {
  try {
    console.log('Testing search endpoint...');

    const response = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'test document',
        searchType: 'semantic',
        limit: 5,
      }),
    });

    const data = await response.json();
    console.log('Search response:', data);
    return true;
  } catch (error) {
    console.error('Search test error:', error);
    return false;
  }
}

// Run all tests
async function runTests() {
  console.log('=== RAG API Testing ===');

  const statusOk = await testStatusEndpoint();
  const uploadOk = await testFileUpload();
  const searchOk = await testSearch();

  console.log('\n=== Test Results ===');
  console.log('Status:', statusOk ? '✅' : '❌');
  console.log('Upload:', uploadOk ? '✅' : '❌');
  console.log('Search:', searchOk ? '✅' : '❌');
}

// Auto-run if in browser console
if (typeof window !== 'undefined') {
  runTests();
}

// Export for Node.js testing
if (typeof module !== 'undefined') {
  module.exports = { testStatusEndpoint, testFileUpload, testSearch, runTests };
}
