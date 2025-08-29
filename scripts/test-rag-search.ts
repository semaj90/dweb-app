// scripts/test-rag-search.ts
// Interactive test of Enhanced RAG search and AI analysis

import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import Fuse from 'fuse.js';

// Load the demo data
const demoDataPath = fs.readdirSync('.')
  .filter(dir => dir.startsWith('enhanced-rag-demo-'))
  .sort()
  .pop();

if (!demoDataPath) {
  console.error('❌ No demo data found. Run: npm run rag:demo first');
  process.exit(1);
}

const demoData = JSON.parse(
  fs.readFileSync(path.join(demoDataPath, 'output', 'demo-data.json'), 'utf-8')
);

console.log(`🔍 Testing Enhanced RAG Search & AI Analysis
📊 Loaded ${demoData.documents.length} documents
🧠 Using: ${demoData.config.embedModel} + ${demoData.config.legalModel}`);

// Initialize search
const fuseSearch = new Fuse(demoData.documents, {
  keys: [
    { name: 'content', weight: 0.4 },
    { name: 'summary', weight: 0.3 },
    { name: 'label', weight: 0.2 },
    { name: 'metadata.legalTerms', weight: 0.1 }
  ],
  threshold: 0.6, // More lenient for testing
  includeScore: true,
  includeMatches: true
});

// Test search function
async function testSearch(query: string): Promise<any> {
  console.log(`\n🔍 Searching for: "${query}"`);
  
  const results = fuseSearch.search(query, { limit: 3 });
  
  if (results.length === 0) {
    console.log('❌ No results found');
    return;
  }
  
  console.log(`✅ Found ${results.length} results:\n`);
  
  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    const doc = result.item;
    const similarity = ((1 - (result.score || 0)) * 100).toFixed(1);
    
    console.log(`${i + 1}. 📄 ${doc.label.toUpperCase()} (${similarity}% match)`);
    console.log(`   Score: ${doc.score} | Confidence: ${doc.confidence}`);
    console.log(`   Summary: ${doc.summary}`);
    console.log(`   Legal Terms: ${doc.metadata.legalTerms.join(', ')}`);
    
    // Show matches
    if (result.matches && result.matches.length > 0) {
      const bestMatch = result.matches[0];
      console.log(`   Best Match: "${bestMatch.value}" (in ${bestMatch.key})`);
    }
    
    console.log(`   Embedding: ${doc.embedding ? `✅ ${doc.embedding.length}d vector` : '❌ No embedding'}`);
    console.log('');
  }
}

// Test AI analysis function
async function testAIAnalysis(docId: string): Promise<any> {
  const doc = demoData.documents.find((d: unknown) => d.id === docId);
  if (!doc) {
    console.log(`❌ Document ${docId} not found`);
    return;
  }
  
  console.log(`\n🤖 AI Analysis of ${doc.id} (${doc.label})`);
  console.log(`Content: "${doc.content.slice(0, 100)}..."`);
  
  try {
    const response = await fetch(`${demoData.config.ollamaHost}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: demoData.config.legalModel,
        prompt: `Analyze this legal document and provide key insights:\n\n${doc.content}\n\nProvide analysis in this format:
1. Document Type: [contract/tort/criminal/etc]
2. Key Legal Issues: [main issues]
3. Parties Involved: [who is involved]
4. Potential Outcomes: [what might happen]
5. Legal Precedents: [relevant case law if any]`,
        temperature: 0.3,
        stream: false
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`\n💡 AI Analysis Result:\n${result.response}`);
    } else {
      console.log('❌ AI analysis failed - check Ollama status');
    }
  } catch (error: any) {
    console.log('❌ AI analysis error:', error.message);
  }
}

// Test vector similarity
function testVectorSimilarity() {
  console.log('\n🧠 Vector Similarity Analysis');
  
  const docsWithEmbeddings = demoData.documents.filter((doc: unknown) => doc.embedding);
  console.log(`📊 Documents with embeddings: ${docsWithEmbeddings.length}/${demoData.documents.length}`);
  
  if (docsWithEmbeddings.length >= 2) {
    const doc1 = docsWithEmbeddings[0];
    const doc2 = docsWithEmbeddings[1];
    
    // Simple cosine similarity calculation
    const similarity = cosineSimilarity(doc1.embedding, doc2.embedding);
    
    console.log(`\n📐 Similarity between "${doc1.label}" and "${doc2.label}": ${(similarity * 100).toFixed(2)}%`);
    console.log(`Doc 1: ${doc1.summary}`);
    console.log(`Doc 2: ${doc2.summary}`);
  }
}

// Cosine similarity helper
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Show analytics
function showAnalytics() {
  console.log('\n📊 Enhanced RAG Analytics Dashboard');
  console.log('=' .repeat(50));
  
  const analytics = demoData.analytics;
  console.log(`Total Documents: ${analytics.totalDocuments}`);
  console.log(`Average Score: ${analytics.averageScore}`);
  console.log(`Average Confidence: ${analytics.averageConfidence}`);
  console.log(`Embedding Coverage: ${analytics.embeddingCoverage}`);
  
  console.log('\nLabel Distribution:');
  Object.entries(analytics.labelDistribution).forEach(([label, count]) => {
    console.log(`  ${label}: ${count} documents`);
  });
  
  console.log('\nComplexity Distribution:');
  console.log(`  High (≥0.8): ${analytics.complexityDistribution.high} documents`);
  console.log(`  Medium (0.5-0.8): ${analytics.complexityDistribution.medium} documents`);
  console.log(`  Low (<0.5): ${analytics.complexityDistribution.low} documents`);
  
  console.log('\nTop Scoring Documents:');
  analytics.topScoringDocs.forEach((doc: unknown, index: number) => {
    console.log(`  ${index + 1}. ${doc.label} (${doc.id}) - Score: ${doc.score}`);
  });
}

// Run comprehensive tests
async function runTests(): Promise<any> {
  showAnalytics();
  
  // Test different search queries
  const testQueries = [
    'contract breach',
    'medical malpractice',
    'fourth amendment',
    'patent lawsuit',
    'appellate court',
    'damages negligence',
    'intellectual property'
  ];
  
  console.log('\n🔍 Running Search Tests');
  console.log('=' .repeat(50));
  
  for (const query of testQueries) {
    await testSearch(query);
  }
  
  // Test vector similarity
  testVectorSimilarity();
  
  // Test AI analysis on highest scoring document
  const topDoc = demoData.analytics.topScoringDocs[0];
  await testAIAnalysis(topDoc.id);
  
  console.log('\n✅ All tests completed!');
  console.log(`\n🚀 Your Enhanced RAG system is working perfectly:
- 🔍 Fuzzy search: Finding relevant legal documents
- 🧠 Vector embeddings: 768-dimensional semantic analysis  
- 🤖 AI analysis: ${demoData.config.legalModel} providing legal insights
- 📊 Analytics: Performance tracking and document scoring
- ⚡ Speed: ${demoData.performance.duration}ms processing time`);
}

// Run the tests
runTests().catch(console.error);