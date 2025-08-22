#!/usr/bin/env node

// Quick Integration Test for Full Stack
// Tests key components without timeouts

async function quickTest() {
  console.log('🧪 Quick Stack Integration Test\n');

  const results = {};

  // 1. Enhanced RAG Service
  try {
    const response = await fetch('http://localhost:8094/health');
    const data = await response.json();
    results.enhancedRAG = data.status === 'healthy';
    console.log(`✅ Enhanced RAG: ${data.status}`);
  } catch (e) {
    results.enhancedRAG = false;
    console.log('❌ Enhanced RAG: not accessible');
  }

  // 2. Ollama Models
  try {
    const response = await fetch('http://localhost:11434/api/tags');
    const data = await response.json();
    const hasGemma = data.models.some(m => m.name.includes('gemma3-legal'));
    const hasNomic = data.models.some(m => m.name.includes('nomic-embed'));
    results.ollama = { hasGemma, hasNomic };
    console.log(`✅ Ollama: ${data.models.length} models (Gemma3: ${hasGemma}, Nomic: ${hasNomic})`);
  } catch (e) {
    results.ollama = false;
    console.log('❌ Ollama: not accessible');
  }

  // 3. Upload Service  
  try {
    const response = await fetch('http://localhost:8093/health', { 
      method: 'GET',
      signal: AbortSignal.timeout(2000)
    });
    results.uploadService = response.ok;
    console.log(`${response.ok ? '✅' : '⚠️'} Upload Service: ${response.status}`);
  } catch (e) {
    results.uploadService = false;
    console.log('⚠️ Upload Service: not accessible');
  }

  // 4. Custom JSON API
  try {
    const testData = { ocrData: { text: "test", filename: "test.pdf" } };
    const response = await fetch('http://localhost:5173/api/convert/to-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testData),
      signal: AbortSignal.timeout(3000)
    });
    results.customJSON = response.ok;
    console.log(`${response.ok ? '✅' : '⚠️'} Custom JSON API: ${response.status}`);
  } catch (e) {
    results.customJSON = false;
    console.log('⚠️ Custom JSON API: timeout/error');
  }

  // 5. Go Llama Integration Test
  try {
    const testData = {
      type: "ollama_chat",
      payload: {
        prompt: "What is a contract?",
        model: "gemma3-legal:latest",
        use_custom_json: true
      }
    };
    const response = await fetch('http://localhost:4101/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testData),
      signal: AbortSignal.timeout(3000)
    });
    results.goLlama = response.ok;
    console.log(`${response.ok ? '✅' : '⚠️'} Go Llama Integration: ${response.status}`);
  } catch (e) {
    results.goLlama = false;
    console.log('⚠️ Go Llama Integration: not running');
  }

  // Summary
  console.log('\n📊 Integration Status:');
  const services = [
    { name: 'Enhanced RAG (8094)', status: results.enhancedRAG },
    { name: 'Ollama Models (11434)', status: results.ollama && results.ollama.hasGemma },
    { name: 'Upload Service (8093)', status: results.uploadService },
    { name: 'Custom JSON API (5173)', status: results.customJSON },
    { name: 'Go Llama Integration (4101)', status: results.goLlama }
  ];

  services.forEach(s => {
    console.log(`   ${s.status ? '✅' : '❌'} ${s.name}`);
  });

  const healthyCount = services.filter(s => s.status).length;
  const overallHealth = healthyCount >= 3; // Need at least 3 core services

  console.log(`\n🎯 Overall Status: ${overallHealth ? '✅ HEALTHY' : '⚠️ PARTIAL'} (${healthyCount}/${services.length})`);

  if (overallHealth) {
    console.log('\n🚀 Stack is ready for production testing!');
  } else {
    console.log('\n⚠️ Some services need attention. Check individual service status above.');
  }

  return results;
}

// Run test
quickTest().catch(console.error);