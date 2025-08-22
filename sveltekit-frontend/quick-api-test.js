/**
 * Quick API Test for Issue Resolution
 */

const BASE_URL = 'http://localhost:5174';

const tests = [
  // Direct service tests
  { name: 'Ollama Direct', url: 'http://localhost:11434/api/tags' },
  { name: 'Enhanced RAG Direct', url: 'http://localhost:8094/health' },
  { name: 'Upload Service Direct', url: 'http://localhost:8093/health' },
  
  // Proxy tests  
  { name: 'Ollama Proxy', url: `${BASE_URL}/api/ollama/tags` },
  { name: 'Ollama API Proxy', url: `${BASE_URL}/api/ollama/api/tags` },
  { name: 'RAG v1 Proxy', url: `${BASE_URL}/api/v1/rag`, method: 'POST', body: { query: 'test' } },
  { name: 'AI v1 Proxy', url: `${BASE_URL}/api/v1/ai`, method: 'POST', body: { prompt: 'test' } },
  
  // SvelteKit API routes
  { name: 'Evidence List', url: `${BASE_URL}/api/evidence/list` },
  { name: 'YoRHa Status', url: `${BASE_URL}/api/yorha/system/status` },
];

async function testEndpoint(test) {
  try {
    const options = {
      method: test.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      timeout: 5000
    };
    
    if (test.body) {
      options.body = JSON.stringify(test.body);
    }
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    options.signal = controller.signal;
    
    const response = await fetch(test.url, options);
    clearTimeout(timeoutId);
    
    const status = response.status;
    let preview = '';
    
    try {
      const text = await response.text();
      preview = text.substring(0, 50);
    } catch (e) {
      preview = 'Could not read response';
    }
    
    return {
      name: test.name,
      status,
      success: status >= 200 && status < 400,
      preview
    };
    
  } catch (error) {
    return {
      name: test.name,
      status: 'ERROR',
      success: false,
      error: error.message
    };
  }
}

async function runQuickTest() {
  console.log('🔍 Quick API Test - Checking Issues...\n');
  
  const results = [];
  
  for (const test of tests) {
    const result = await testEndpoint(test);
    results.push(result);
    
    const icon = result.success ? '✅' : '❌';
    const detail = result.error || `${result.status} - ${result.preview}`;
    console.log(`${icon} ${result.name}: ${detail}`);
  }
  
  console.log('\n📊 Summary:');
  const successful = results.filter(r => r.success).length;
  console.log(`✅ ${successful}/${results.length} tests passed`);
  
  const failed = results.filter(r => !r.success);
  if (failed.length > 0) {
    console.log('\n❌ Failed tests:');
    failed.forEach(f => {
      console.log(`   • ${f.name}: ${f.error || f.status}`);
    });
  }
  
  return results;
}

runQuickTest().catch(console.error);