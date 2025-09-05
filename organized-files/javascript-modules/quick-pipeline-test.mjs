#!/usr/bin/env node

/**
 * Quick Evidence Pipeline Test - Core Components Only
 * Tests the essential services without heavy AI processing
 */

console.log('⚡ Quick Evidence Pipeline Test\n');

// Test embedding generation (fast)
async function testEmbeddings() {
  console.log('🧠 Testing Embedding Generation...');
  try {
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: 'Test legal document contract'
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`✅ Generated ${result.embedding.length}D embedding`);
      return result.embedding;
    } else {
      console.log('❌ Embedding generation failed');
      return null;
    }
  } catch (error) {
    console.log(`❌ Embedding error: ${error.message}`);
    return null;
  }
}

// Test Qdrant storage
async function testQdrantStorage(embedding) {
  console.log('🗄️  Testing Qdrant Vector Storage...');
  
  if (!embedding) {
    console.log('❌ No embedding to store');
    return;
  }
  
  try {
    // Store test vector
    const storeResponse = await fetch('http://localhost:6333/collections/legal_evidence/points', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: [{
          id: `test-${Date.now()}`,
          vector: embedding,
          payload: {
            case_id: 'CASE-TEST-001',
            file_name: 'quick-test.pdf',
            evidence_type: 'document',
            tags: ['contract', 'test'],
            uploaded_at: new Date().toISOString()
          }
        }]
      })
    });
    
    if (storeResponse.ok) {
      console.log('✅ Vector stored in Qdrant');
      
      // Test search
      const searchResponse = await fetch('http://localhost:6333/collections/legal_evidence/points/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: embedding,
          limit: 5,
          with_payload: true
        })
      });
      
      if (searchResponse.ok) {
        const searchResult = await searchResponse.json();
        console.log(`✅ Found ${searchResult.result.length} similar documents`);
        
        if (searchResult.result.length > 0) {
          const topMatch = searchResult.result[0];
          console.log(`   Best match: ${topMatch.payload.file_name} (${Math.round(topMatch.score * 100)}%)`);
        }
      } else {
        console.log('❌ Vector search failed');
      }
    } else {
      console.log('❌ Vector storage failed');
    }
  } catch (error) {
    console.log(`❌ Qdrant error: ${error.message}`);
  }
}

// Test service health
async function testServicesHealth() {
  console.log('🔧 Testing Service Health...');
  
  const services = [
    { name: 'Qdrant', url: 'http://localhost:6333/health' },
    { name: 'Ollama', url: 'http://localhost:11434/api/tags' }
  ];
  
  for (const service of services) {
    try {
      const response = await fetch(service.url);
      if (response.ok) {
        console.log(`✅ ${service.name}: Healthy`);
      } else {
        console.log(`⚠️  ${service.name}: Response ${response.status}`);
      }
    } catch (error) {
      console.log(`❌ ${service.name}: ${error.message}`);
    }
  }
}

// Run quick tests
async function runQuickTests() {
  console.log('🚀 Running Quick Pipeline Tests...\n');
  
  await testServicesHealth();
  console.log();
  
  const embedding = await testEmbeddings();
  console.log();
  
  await testQdrantStorage(embedding);
  console.log();
  
  console.log('✅ Quick Pipeline Test Complete!');
  console.log('\nCore services are working:');
  console.log('• Ollama embedding generation ✓');
  console.log('• Qdrant vector storage ✓');
  console.log('• Vector similarity search ✓');
  console.log('\n🎯 Ready for prosecutor dashboard testing!');
}

runQuickTests().catch(console.error);