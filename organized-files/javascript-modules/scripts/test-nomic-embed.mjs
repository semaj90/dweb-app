#!/usr/bin/env node

// Test script for Nomic Embed integration
import { generateEmbedding } from '../sveltekit-frontend/src/lib/server/ai/embeddings-simple.ts';

console.log('🧪 Testing Nomic Embed Integration...\n');

async function testEmbedding() {
  try {
    console.log('📝 Generating embedding for test text...');
    
    const testText = "This is a test legal document about contract law.";
    const embedding = await generateEmbedding(testText, { 
      model: "local",  // Use Nomic Embed via Ollama
      cache: false 
    });

    if (embedding && Array.isArray(embedding)) {
      console.log('✅ Embedding generated successfully!');
      console.log(`📊 Dimensions: ${embedding.length}`);
      console.log(`🔢 First 5 values: [${embedding.slice(0, 5).map(n => n.toFixed(4)).join(', ')}]`);
      console.log(`🎯 Model: Nomic Embed (via Ollama)`);
    } else {
      console.log('❌ Embedding generation failed');
    }

  } catch (error) {
    console.error('❌ Error testing embedding:', error.message);
  }
}

async function testOllamaConnection() {
  try {
    console.log('🔌 Testing Ollama connection...');
    
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: 'test'
      })
    });

    if (response.ok) {
      const data = await response.json();
      console.log('✅ Ollama connection successful!');
      console.log(`📊 Nomic Embed dimensions: ${data.embedding.length}`);
    } else {
      console.log('❌ Ollama connection failed:', response.status);
    }

  } catch (error) {
    console.error('❌ Ollama connection error:', error.message);
  }
}

// Run tests
await testOllamaConnection();
console.log('');
await testEmbedding();

console.log('\n🎉 Nomic Embed test complete!');
console.log('💡 You can now use local embeddings without OpenAI API key.');