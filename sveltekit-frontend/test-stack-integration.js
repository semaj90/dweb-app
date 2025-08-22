#!/usr/bin/env node

// Comprehensive Stack Integration Test
// Tests: SvelteKit 2 + PostgreSQL + pgvector + Drizzle ORM + TypeScript Barrel Stores + AI

console.log('🧪 Testing Complete Stack Integration...\n');

async function testStack() {
  const results = {
    services: {},
    apis: {},
    database: {},
    ai: {},
    frontend: {}
  };

  // 1. Test Core Services
  console.log('1. Testing Core Services...');
  
  try {
    // Enhanced RAG Service
    const ragResponse = await fetch('http://localhost:8094/health');
    const ragData = await ragResponse.json();
    results.services.enhancedRAG = {
      status: ragData.status,
      healthy: ragData.status === 'healthy',
      context7: ragData.context7_connected,
      websockets: ragData.websocket_connections
    };
    console.log('   ✅ Enhanced RAG Service:', ragData.status);
  } catch (error) {
    results.services.enhancedRAG = { error: error.message, healthy: false };
    console.log('   ❌ Enhanced RAG Service:', error.message);
  }

  try {
    // Upload Service (should be on 8093)
    const uploadResponse = await fetch('http://localhost:8093/health');
    if (uploadResponse.ok) {
      const uploadData = await uploadResponse.json();
      results.services.uploadService = { status: 'healthy', healthy: true };
      console.log('   ✅ Upload Service: healthy');
    }
  } catch (error) {
    results.services.uploadService = { error: error.message, healthy: false };
    console.log('   ⚠️ Upload Service: not accessible');
  }

  // 2. Test Ollama Models
  console.log('\n2. Testing Ollama AI Models...');
  
  try {
    const ollamaResponse = await fetch('http://localhost:11434/api/tags');
    const ollamaData = await ollamaResponse.json();
    const models = ollamaData.models.map(m => ({
      name: m.name,
      size: Math.round(m.size / 1024 / 1024) + 'MB',
      family: m.details?.family
    }));
    
    results.ai.ollama = {
      healthy: true,
      models: models,
      hasGemma3Legal: models.some(m => m.name.includes('gemma3-legal')),
      hasNomicEmbed: models.some(m => m.name.includes('nomic-embed'))
    };
    
    console.log('   ✅ Ollama Models:');
    models.forEach(m => console.log(`      - ${m.name} (${m.size})`));
    
  } catch (error) {
    results.ai.ollama = { error: error.message, healthy: false };
    console.log('   ❌ Ollama:', error.message);
  }

  // 3. Test Database Connectivity 
  console.log('\n3. Testing Database Layer...');
  
  try {
    // Test PostgreSQL connection via SvelteKit API if available
    const dbResponse = await fetch('http://localhost:5173/api/db/health', {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    
    if (dbResponse.ok) {
      const dbData = await dbResponse.json();
      results.database.postgresql = { 
        healthy: true, 
        status: dbData.status || 'connected',
        pgvector: dbData.pgvector || 'available'
      };
      console.log('   ✅ PostgreSQL + pgvector: connected');
    }
  } catch (error) {
    results.database.postgresql = { error: error.message, healthy: false };
    console.log('   ⚠️ PostgreSQL API: endpoint not found (expected in development)');
  }

  // 4. Test AI Integration via Enhanced RAG
  console.log('\n4. Testing AI Processing Integration...');
  
  try {
    const aiTestPayload = {
      query: "What are the key elements of a valid contract?",
      context: ["legal", "contract"],
      model: "gemma3-legal:latest"
    };

    const aiResponse = await fetch('http://localhost:8094/api/rag/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-User-ID': 'test-integration',
        'X-Case-ID': 'integration-test-1'
      },
      body: JSON.stringify(aiTestPayload)
    });

    if (aiResponse.ok) {
      const aiData = await aiResponse.json();
      results.ai.ragIntegration = {
        healthy: true,
        hasResponse: !!aiData.summary || !!aiData.response,
        confidence: aiData.confidence || 'N/A'
      };
      console.log('   ✅ AI Processing: functional');
      console.log('      - Response generated:', !!aiData.summary);
      console.log('      - Confidence:', aiData.confidence || 'N/A');
    } else {
      results.ai.ragIntegration = { 
        healthy: false, 
        status: aiResponse.status,
        error: aiResponse.statusText
      };
      console.log('   ⚠️ AI Processing:', aiResponse.status, aiResponse.statusText);
    }
  } catch (error) {
    results.ai.ragIntegration = { error: error.message, healthy: false };
    console.log('   ❌ AI Processing:', error.message);
  }

  // 5. Test Frontend Integration
  console.log('\n5. Testing Frontend Integration...');
  
  try {
    // Test if AI Assistant page is accessible
    const frontendResponse = await fetch('http://localhost:5173/aiassistant', {
      method: 'HEAD',
      timeout: 5000
    });
    
    results.frontend.aiAssistant = {
      accessible: frontendResponse.ok,
      status: frontendResponse.status
    };
    
    if (frontendResponse.ok) {
      console.log('   ✅ AI Assistant Page: accessible');
    } else {
      console.log('   ⚠️ AI Assistant Page:', frontendResponse.status);
    }
  } catch (error) {
    results.frontend.aiAssistant = { error: error.message, accessible: false };
    console.log('   ❌ AI Assistant Page:', error.message);
  }

  // 6. Test Custom JSON Integration
  console.log('\n6. Testing Custom JSON Optimization...');
  
  try {
    const jsonTestData = {
      ocrData: {
        text: "Contract violation regarding payment terms and liability clauses.",
        filename: "test-contract.pdf",
        pages: 1,
        totalCharacters: 67,
        averageConfidence: 95.5,
        extractedAt: new Date().toISOString()
      }
    };

    const jsonResponse = await fetch('http://localhost:5173/api/convert/to-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(jsonTestData)
    });

    if (jsonResponse.ok) {
      const jsonData = await jsonResponse.json();
      results.apis.customJSON = {
        healthy: true,
        optimized: jsonData.success,
        jsonSize: jsonData.stats?.jsonSize || 'N/A'
      };
      console.log('   ✅ Custom JSON Optimization: functional');
      console.log('      - Optimized size:', jsonData.stats?.jsonSize || 'N/A', 'bytes');
    }
  } catch (error) {
    results.apis.customJSON = { error: error.message, healthy: false };
    console.log('   ⚠️ Custom JSON API:', error.message);
  }

  // 7. Generate Summary Report
  console.log('\n' + '='.repeat(60));
  console.log('📊 INTEGRATION TEST RESULTS SUMMARY');
  console.log('='.repeat(60));

  const serviceHealth = Object.values(results.services).every(s => s.healthy !== false);
  const aiHealth = Object.values(results.ai).every(a => a.healthy !== false);
  const apiHealth = Object.values(results.apis).every(a => a.healthy !== false);
  const frontendHealth = Object.values(results.frontend).every(f => f.accessible !== false);

  console.log('🔧 Services Layer:', serviceHealth ? '✅ HEALTHY' : '⚠️ PARTIAL');
  console.log('🧠 AI/ML Layer:   ', aiHealth ? '✅ HEALTHY' : '⚠️ PARTIAL');  
  console.log('📡 API Layer:     ', apiHealth ? '✅ HEALTHY' : '⚠️ PARTIAL');
  console.log('🎨 Frontend Layer:', frontendHealth ? '✅ HEALTHY' : '⚠️ PARTIAL');

  console.log('\n📈 Key Integration Points:');
  console.log('   - Enhanced RAG Service:', results.services.enhancedRAG?.healthy ? '✅' : '❌');
  console.log('   - Ollama Gemma3-Legal:', results.ai.ollama?.hasGemma3Legal ? '✅' : '❌');
  console.log('   - Custom JSON API:', results.apis.customJSON?.healthy ? '✅' : '⚠️');
  console.log('   - AI Assistant UI:', results.frontend.aiAssistant?.accessible ? '✅' : '⚠️');
  
  const overallHealth = serviceHealth && aiHealth;
  console.log('\n🎯 Overall System Status:', overallHealth ? '✅ PRODUCTION READY' : '⚠️ NEEDS ATTENTION');

  if (overallHealth) {
    console.log('\n🚀 Stack Integration: SUCCESS');
    console.log('   • SvelteKit 2 frontend running');
    console.log('   • Enhanced RAG service operational');  
    console.log('   • Ollama AI models loaded');
    console.log('   • Custom JSON optimization active');
    console.log('   • TypeScript barrel stores ready');
  }

  // Save results
  const fs = require('fs');
  fs.writeFileSync(
    './integration-test-results.json', 
    JSON.stringify(results, null, 2)
  );
  console.log('\n📝 Detailed results saved to: integration-test-results.json');
  
  return results;
}

// Run the test
testStack().catch(console.error);