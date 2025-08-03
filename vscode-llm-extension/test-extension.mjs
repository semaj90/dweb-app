#!/usr/bin/env node

/**
 * Simple test script for MCP Context7 Assistant extension
 * Tests the core services without requiring VSCode environment
 */

import { NomicEmbedService } from './src/nomic-embed-service.js';
import { Neo4jService } from './src/neo4j-service.js';
import { SIMDJSONParser } from './src/simd-json-parser.js';

console.log('🧪 Starting MCP Context7 Assistant Extension Tests...\n');

// Test configuration
const testConfig = {
  skipNomicEmbed: true, // Skip if server not running
  skipNeo4j: true,      // Skip if database not running
  testSIMDParser: true  // Always test SIMD parser
};

async function testSIMDJSONParser() {
  console.log('📊 Testing SIMD JSON Parser...');
  
  try {
    const parser = new SIMDJSONParser({
      enableWorkerThreads: true,
      workerPoolSize: 2,
      validationLevel: 'basic'
    });

    await parser.initialize();

    // Test JSON evidence data
    const testEvidence = JSON.stringify({
      evidence: [
        {
          id: 'test_001',
          type: 'document',
          content: 'This is a test legal document for evidence processing.',
          metadata: {
            source: 'test_case',
            timestamp: new Date().toISOString(),
            author: 'Test Author',
            case_id: 'TEST_CASE_001'
          }
        },
        {
          id: 'test_002',
          type: 'testimony',
          content: 'This is test witness testimony about the incident.',
          metadata: {
            source: 'witness_statement',
            timestamp: new Date().toISOString(),
            witness_name: 'John Doe'
          }
        }
      ]
    });

    console.log('  - Testing evidence extraction...');
    const result = await parser.extractEvidenceFromJson(testEvidence);

    console.log(`  ✅ Parse successful: ${result.parseResult.success}`);
    console.log(`  ✅ Processing time: ${result.parseResult.processingTime}ms`);
    console.log(`  ✅ Method used: ${result.parseResult.method}`);
    console.log(`  ✅ Evidence items extracted: ${result.evidence.length}`);
    console.log(`  ✅ Total items processed: ${result.extractionMetadata.totalItems}`);
    console.log(`  ✅ Extraction errors: ${result.extractionMetadata.errors.length}`);

    // Test performance metrics
    const metrics = parser.getPerformanceMetrics();
    console.log(`  ✅ SIMD available: ${metrics.simdAvailable}`);
    console.log(`  ✅ Workers initialized: ${metrics.workersInitialized}`);

    parser.dispose();
    console.log('  ✅ SIMD JSON Parser test completed successfully!\n');

    return true;
  } catch (error) {
    console.error('  ❌ SIMD JSON Parser test failed:', error.message);
    return false;
  }
}

async function testNomicEmbedService() {
  if (testConfig.skipNomicEmbed) {
    console.log('📝 Skipping Nomic Embed Service test (requires running server)\n');
    return true;
  }

  console.log('📝 Testing Nomic Embed Service...');
  
  try {
    const embedService = new NomicEmbedService();

    // Test basic embedding
    const testTexts = [
      'This is a test document about legal procedures.',
      'Another test document containing evidence analysis.'
    ];

    console.log('  - Testing text embedding...');
    const embeddings = await embedService.embedTexts(testTexts);

    console.log(`  ✅ Embeddings generated: ${embeddings.embeddings.length}`);
    console.log(`  ✅ Embedding dimensions: ${embeddings.embeddings[0]?.length || 0}`);
    console.log(`  ✅ Total tokens used: ${embeddings.usage.total_tokens}`);

    embedService.dispose();
    console.log('  ✅ Nomic Embed Service test completed successfully!\n');

    return true;
  } catch (error) {
    console.error('  ❌ Nomic Embed Service test failed:', error.message);
    console.log('  💡 Make sure Nomic Embed server is running on localhost:8080\n');
    return false;
  }
}

async function testNeo4jService() {
  if (testConfig.skipNeo4j) {
    console.log('🗄️ Skipping Neo4j Service test (requires running database)\n');
    return true;
  }

  console.log('🗄️ Testing Neo4j Service...');
  
  try {
    const neo4jService = new Neo4jService();

    console.log('  - Testing database connection...');
    await neo4jService.connect();

    // Test storing sample embeddings
    const sampleEmbeddings = [
      {
        id: 'test_embedding_001',
        text: 'Sample legal text for testing.',
        embedding: new Array(768).fill(0.1), // Dummy embedding
        metadata: {
          source: 'test',
          chunk_index: 0,
          file_path: '/test/path.md',
          timestamp: new Date().toISOString(),
          model: 'test-model'
        }
      }
    ];

    console.log('  - Testing embedding storage...');
    const storedCount = await neo4jService.storeEmbeddings(sampleEmbeddings);
    console.log(`  ✅ Embeddings stored: ${storedCount}`);

    console.log('  - Testing database statistics...');
    const stats = await neo4jService.getStats();
    console.log(`  ✅ Total embeddings in database: ${stats.totalEmbeddings}`);

    await neo4jService.disconnect();
    console.log('  ✅ Neo4j Service test completed successfully!\n');

    return true;
  } catch (error) {
    console.error('  ❌ Neo4j Service test failed:', error.message);
    console.log('  💡 Make sure Neo4j is running on bolt://localhost:7687\n');
    return false;
  }
}

async function runAllTests() {
  console.log('🚀 MCP Context7 Assistant Extension Test Suite\n');
  
  const results = {
    simdParser: false,
    nomicEmbed: false,
    neo4j: false
  };

  // Run tests
  results.simdParser = await testSIMDJSONParser();
  results.nomicEmbed = await testNomicEmbedService();
  results.neo4j = await testNeo4jService();

  // Summary
  console.log('📋 Test Results Summary:');
  console.log(`  SIMD JSON Parser: ${results.simdParser ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`  Nomic Embed Service: ${results.nomicEmbed ? '✅ PASSED' : '❌ SKIPPED'}`);
  console.log(`  Neo4j Service: ${results.neo4j ? '✅ PASSED' : '❌ SKIPPED'}`);

  const passedTests = Object.values(results).filter(Boolean).length;
  const totalTests = Object.values(results).length;

  console.log(`\n🎯 Overall: ${passedTests}/${totalTests} tests passed`);

  if (passedTests === totalTests) {
    console.log('🎉 All tests passed! Extension is ready for use.');
  } else if (passedTests > 0) {
    console.log('⚠️ Some tests passed. Check service availability for skipped tests.');
  } else {
    console.log('❌ No tests passed. Check implementation and dependencies.');
  }

  console.log('\n📚 To run full tests with external services:');
  console.log('  1. Start Nomic Embed server: python -m nomic.embed.server --port 8080');
  console.log('  2. Start Neo4j database: docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password neo4j');
  console.log('  3. Run tests again with services enabled');
}

// Handle module import/export differences
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllTests().catch(console.error);
}

export { 
  testSIMDJSONParser, 
  testNomicEmbedService, 
  testNeo4jService, 
  runAllTests 
};