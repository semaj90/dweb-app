/**
 * Test script for AI system integration
 * Tests the gemma3:legal-latest → legal-bert fallback chain
 */

import { ollamaService } from '../src/lib/server/ai/ollama-service.js';

async function testAISystem() {
  console.log('🔍 AI SYSTEM TEST - Legal Models Only');
  console.log('=====================================\n');
  
  // 1. Check Ollama availability
  console.log('1. Checking Ollama service...');
  const isAvailable = await ollamaService.isAvailable();
  if (isAvailable) {
    console.log('   ✅ Ollama service is running');
  } else {
    console.log('   ❌ Ollama service is not available');
    console.log('   💡 Run: ollama serve');
    return;
  }
  
  // 2. List available models
  console.log('\n2. Available models:');
  const models = await ollamaService.listModels();
  if (models.models && models.models.length > 0) {
    models.models.forEach(model => {
      const name = model.name;
      if (name.includes('gemma') && name.includes('legal')) {
        console.log(`   ✅ ${name} (PRIMARY - Legal AI)`);
      } else if (name === 'legal-bert' || (name.includes('legal') && name.includes('bert'))) {
        console.log(`   ✅ ${name} (FALLBACK - Legal BERT)`);
      } else if (name === 'nomic-embed-text') {
        console.log(`   ✓ ${name} (Embeddings)`);
      } else if (name === 'bge-large-en') {
        console.log(`   ✓ ${name} (Embedding Fallback)`);
      } else {
        console.log(`   - ${name} (not in fallback chain)`);
      }
    });
  } else {
    console.log('   ⚠️ No models found');
  }
  
  // 3. Check for primary and fallback models
  console.log('\n3. Checking fallback chain...');
  const hasGemma = await ollamaService.checkLegalModel();
  const hasLegalBert = await ollamaService.checkLegalBert();
  
  console.log('   Fallback Chain Status:');
  console.log(`   1. gemma3:legal-latest: ${hasGemma ? '✅ Available' : '❌ Not found'}`);
  console.log(`   2. legal-bert: ${hasLegalBert ? '✅ Available' : '❌ Not found'}`);
  
  if (!hasGemma && !hasLegalBert) {
    console.log('\n   ⚠️ WARNING: No legal models available!');
    console.log('   The system needs at least one of the legal models to function.');
  }
  
  // 4. Get system status
  console.log('\n4. System Status:');
  const status = await ollamaService.getSystemStatus();
  console.log('   Ollama Available:', status.ollamaAvailable ? '✅' : '❌');
  console.log('   Primary Model:', status.primaryModel);
  console.log('   Legal Fallback:', status.legalFallback);
  console.log('   Cache Size:', status.cacheSize);
  console.log('   Queue Length:', status.queueLength);
  console.log('   Active Requests:', status.activeRequests);
  console.log('\n   Configured Fallback Chains:');
  console.log('   - Legal:', status.fallbackChain.legal.join(' → '));
  console.log('   - General:', status.fallbackChain.general.join(' → '));
  console.log('   - Embedding:', status.fallbackChain.embedding.join(' → '));
  
  // 5. Test text generation with legal prompt
  console.log('\n5. Testing legal text generation...');
  try {
    const response = await ollamaService.generate(
      'What are the key elements of a valid contract under common law?',
      { 
        options: { 
          num_predict: 100,
          temperature: 0.5
        }
      }
    );
    console.log('   ✅ Generation successful');
    console.log('   Model used:', response.model);
    console.log('   Fallback used:', response.fallback_used ? 'Yes' : 'No');
    if (response.models_tried && response.models_tried.length > 1) {
      console.log('   Models tried:', response.models_tried.join(' → '));
    }
    console.log('   Response preview:', response.response?.substring(0, 100) + '...');
  } catch (error) {
    console.log('   ❌ Generation failed:', error.message);
  }
  
  // 6. Test embeddings
  console.log('\n6. Testing embeddings...');
  try {
    const embeddings = await ollamaService.generateEmbeddings(
      'This is a test legal document for embedding generation in a contract dispute case.'
    );
    console.log('   ✅ Embedding generation successful');
    console.log('   Dimensions:', embeddings.length);
    console.log('   Sample values:', embeddings.slice(0, 5).map(v => v.toFixed(4)).join(', '), '...');
  } catch (error) {
    console.log('   ❌ Embedding generation failed:', error.message);
  }
  
  // 7. Test legal document analysis
  console.log('\n7. Testing legal document analysis...');
  try {
    const testDoc = {
      id: 'test-001',
      title: 'Service Agreement',
      type: 'contract',
      content: `This Service Agreement ("Agreement") is entered into as of January 1, 2025, 
                between ABC Corporation, a Delaware corporation ("Company"), and John Doe, 
                an individual contractor ("Contractor"). 
                
                The Contractor agrees to provide software development services for a period 
                of six (6) months at a rate of $150 per hour. Payment terms are Net 30 days 
                from invoice date. 
                
                This Agreement shall be governed by the laws of the State of California. 
                Any disputes arising under this Agreement shall be resolved through binding 
                arbitration in accordance with the rules of the American Arbitration Association.`,
      metadata: {
        dateCreated: new Date(),
        dateModified: new Date()
      },
      chunks: []
    };
    
    const analysis = await ollamaService.analyzeLegalDocument(testDoc);
    console.log('   ✅ Document analysis successful');
    console.log('   Model used:', analysis.metadata?.modelUsed);
    console.log('   Summary:', analysis.summary?.substring(0, 100) + '...');
    console.log('   Key Points:', analysis.keyPoints?.length || 0, 'points identified');
    console.log('   Entities found:');
    console.log('     - People:', analysis.entities.people?.join(', ') || 'none');
    console.log('     - Organizations:', analysis.entities.organizations?.join(', ') || 'none');
    console.log('     - Legal Concepts:', analysis.entities.legalConcepts?.join(', ') || 'none');
  } catch (error) {
    console.log('   ❌ Document analysis failed:', error.message);
  }
  
  // 8. Test fallback behavior
  console.log('\n8. Testing fallback behavior...');
  try {
    // Try to force a model that might not exist
    const response = await ollamaService.generate(
      'Test legal query for fallback testing',
      { model: 'non-existent-model' }
    );
    console.log('   Fallback worked!');
    console.log('   Final model used:', response.model);
  } catch (error) {
    console.log('   Fallback test result:', error.message);
  }
  
  // 9. Cache statistics
  console.log('\n9. Cache statistics:');
  const cacheStats = ollamaService.getCacheStats();
  console.log('   Cache entries:', cacheStats.size);
  console.log('   Cache enabled:', true);
  
  console.log('\n' + '='.repeat(50));
  console.log('✨ AI SYSTEM TEST COMPLETE');
  console.log('='.repeat(50));
  
  console.log('\nFallback Chain Configuration:');
  console.log('  Primary Model: gemma3:legal-latest');
  console.log('  Fallback Model: legal-bert');
  console.log('  Embedding Model: nomic-embed-text');
  console.log('  Embedding Fallback: bge-large-en');
  console.log('  GPU Acceleration: Enabled (35 layers)');
  console.log('  Cache TTL: 3600 seconds');
  
  console.log('\nAPI Endpoints:');
  console.log('  POST /api/ai/generate - Text generation');
  console.log('  POST /api/ai/embeddings - Generate embeddings');
  console.log('  POST /api/ai/analyze - Document analysis');
  console.log('  GET /api/ai/generate - Health check');
  
  console.log('\n📝 Note: llama3.2 has been removed from the fallback chain.');
  console.log('The system now uses a focused legal model chain for better accuracy.');
}

// Run the test
testAISystem().catch(console.error);
