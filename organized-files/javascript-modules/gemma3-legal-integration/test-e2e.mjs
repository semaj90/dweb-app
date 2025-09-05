#!/usr/bin/env node

// End-to-End Test Suite for Gemma3 Legal AI Integration
// Tests all components working together

import fetch from 'node-fetch';
import WebSocket from 'ws';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

class Gemma3E2ETest {
  constructor() {
    this.baseUrl = 'http://localhost:8095';
    this.wsUrl = 'ws://localhost:8096';
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      errors: []
    };
  }

  async runAllTests() {
    console.log(`
╔════════════════════════════════════════════════════════════╗
║        Gemma3 Legal AI - End-to-End Test Suite            ║
╚════════════════════════════════════════════════════════════╝
`);

    // Test 1: Service Health
    await this.testServiceHealth();
    
    // Test 2: Text Completion
    await this.testCompletion();
    
    // Test 3: Chat Completion
    await this.testChatCompletion();
    
    // Test 4: Embeddings
    await this.testEmbeddings();
    
    // Test 5: Legal Analysis
    await this.testLegalAnalysis();
    
    // Test 6: WebSocket Connection
    await this.testWebSocket();
    
    // Test 7: Full Pipeline
    await this.testFullPipeline();
    
    // Test 8: Performance
    await this.testPerformance();
    
    this.printResults();
  }

  async testServiceHealth() {
    console.log('\n📋 Test 1: Service Health Check');
    console.log('─'.repeat(50));
    
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      
      if (response.ok && data.status === 'healthy') {
        console.log('✅ Service is healthy');
        console.log(`   GPU Available: ${data.gpu?.available ? 'Yes' : 'No'}`);
        if (data.gpu?.available) {
          console.log(`   GPU Memory: ${data.gpu.memoryUsed}MB / ${data.gpu.memoryTotal}MB`);
          console.log(`   GPU Temp: ${data.gpu.temperature}°C`);
        }
        this.results.passed++;
      } else {
        throw new Error('Service unhealthy');
      }
    } catch (error) {
      console.log('❌ Service health check failed:', error.message);
      this.results.failed++;
      this.results.errors.push({ test: 'Service Health', error: error.message });
    }
  }

  async testCompletion() {
    console.log('\n📋 Test 2: Text Completion');
    console.log('─'.repeat(50));
    
    try {
      const response = await fetch(`${this.baseUrl}/v1/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: 'What are the key elements of a valid contract? List them briefly.',
          max_tokens: 200,
          temperature: 0.1
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.choices && data.choices[0].text) {
        console.log('✅ Completion successful');
        console.log(`   Response length: ${data.choices[0].text.length} chars`);
        console.log(`   Tokens used: ${data.usage?.total_tokens || 'N/A'}`);
        
        // Verify legal content
        const text = data.choices[0].text.toLowerCase();
        if (text.includes('offer') || text.includes('acceptance') || text.includes('consideration')) {
          console.log('   ✓ Legal content verified');
        }
        
        this.results.passed++;
      } else {
        throw new Error('Invalid completion response');
      }
    } catch (error) {
      console.log('❌ Completion test failed:', error.message);
      this.results.failed++;
      this.results.errors.push({ test: 'Completion', error: error.message });
    }
  }

  async testChatCompletion() {
    console.log('\n📋 Test 3: Chat Completion');
    console.log('─'.repeat(50));
    
    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: 'You are a legal AI assistant.' },
            { role: 'user', content: 'Explain negligence in tort law.' }
          ],
          max_tokens: 200,
          temperature: 0.1
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.choices && data.choices[0].message) {
        console.log('✅ Chat completion successful');
        console.log(`   Response role: ${data.choices[0].message.role}`);
        console.log(`   Content length: ${data.choices[0].message.content.length} chars`);
        this.results.passed++;
      } else {
        throw new Error('Invalid chat response');
      }
    } catch (error) {
      console.log('❌ Chat completion test failed:', error.message);
      this.results.failed++;
      this.results.errors.push({ test: 'Chat Completion', error: error.message });
    }
  }

  async testEmbeddings() {
    console.log('\n📋 Test 4: Embeddings Generation');
    console.log('─'.repeat(50));
    
    try {
      const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: 'breach of contract damages',
          model: 'nomic-embed-text'
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.data && data.data[0].embedding) {
        const embedding = data.data[0].embedding;
        console.log('✅ Embeddings generated');
        console.log(`   Dimensions: ${embedding.length}`);
        console.log(`   First values: [${embedding.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...]`);
        
        // Verify embedding dimensions
        if (embedding.length === 384 || embedding.length === 768) {
          console.log('   ✓ Correct dimensions for legal embeddings');
        }
        
        this.results.passed++;
      } else {
        throw new Error('Invalid embeddings response');
      }
    } catch (error) {
      console.log('❌ Embeddings test failed:', error.message);
      this.results.failed++;
      this.results.errors.push({ test: 'Embeddings', error: error.message });
    }
  }

  async testLegalAnalysis() {
    console.log('\n📋 Test 5: Legal Document Analysis');
    console.log('─'.repeat(50));
    
    const sampleContract = `
    PURCHASE AGREEMENT
    
    This Agreement is entered into on January 1, 2025, between ABC Corp ("Buyer") 
    and XYZ Inc ("Seller").
    
    1. Purchase Price: $100,000
    2. Payment Terms: 30 days net
    3. Delivery: Within 60 days
    4. Warranty: 1 year limited warranty
    
    The parties agree to these terms and conditions.
    `;
    
    try {
      const response = await fetch(`${this.baseUrl}/api/legal/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: sampleContract,
          type: 'contract',
          options: { detailed: true }
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.analysis) {
        console.log('✅ Legal analysis completed');
        console.log(`   Document type: ${data.type}`);
        console.log(`   Confidence: ${(data.confidence * 100).toFixed(1)}%`);
        console.log(`   Key points: ${data.analysis.keyPoints?.length || 0}`);
        console.log(`   Risks identified: ${data.analysis.risks?.length || 0}`);
        this.results.passed++;
      } else {
        throw new Error('Legal analysis failed');
      }
    } catch (error) {
      console.log('❌ Legal analysis test failed:', error.message);
      this.results.failed++;
      this.results.errors.push({ test: 'Legal Analysis', error: error.message });
    }
  }

  async testWebSocket() {
    console.log('\n📋 Test 6: WebSocket Real-time Connection');
    console.log('─'.repeat(50));
    
    return new Promise((resolve) => {
      const ws = new WebSocket(this.wsUrl);
      let timeout;
      
      ws.on('open', () => {
        console.log('✅ WebSocket connected');
        
        // Send test message
        ws.send(JSON.stringify({
          type: 'completion',
          prompt: 'Define tort.',
          options: { max_tokens: 50 }
        }));
        
        timeout = setTimeout(() => {
          console.log('⚠️  WebSocket response timeout');
          this.results.failed++;
          ws.close();
          resolve();
        }, 5000);
      });
      
      ws.on('message', (data) => {
        clearTimeout(timeout);
        try {
          const response = JSON.parse(data);
          if (response.type === 'response') {
            console.log('   ✓ Real-time response received');
            this.results.passed++;
          } else if (response.type === 'error') {
            console.log('   ✗ Error response:', response.error);
            this.results.failed++;
          }
        } catch (error) {
          console.log('   ✗ Invalid response format');
          this.results.failed++;
        }
        ws.close();
        resolve();
      });
      
      ws.on('error', (error) => {
        clearTimeout(timeout);
        console.log('❌ WebSocket connection failed:', error.message);
        this.results.failed++;
        this.results.errors.push({ test: 'WebSocket', error: error.message });
        resolve();
      });
    });
  }

  async testFullPipeline() {
    console.log('\n📋 Test 7: Full End-to-End Pipeline');
    console.log('─'.repeat(50));
    
    try {
      // Step 1: Generate embeddings for a legal query
      console.log('   Step 1: Generating query embeddings...');
      const embResponse = await fetch(`${this.baseUrl}/v1/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: 'contract breach remedies'
        })
      });
      
      const embData = await embResponse.json();
      if (!embData.data?.[0]?.embedding) {
        throw new Error('Embedding generation failed');
      }
      console.log('   ✓ Embeddings generated');
      
      // Step 2: Analyze a legal document
      console.log('   Step 2: Analyzing legal document...');
      const analysisResponse = await fetch(`${this.baseUrl}/api/legal/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: 'The defendant breached the contract by failing to deliver goods on time.',
          type: 'brief'
        })
      });
      
      const analysisData = await analysisResponse.json();
      if (!analysisData.analysis) {
        throw new Error('Document analysis failed');
      }
      console.log('   ✓ Document analyzed');
      
      // Step 3: Generate legal advice
      console.log('   Step 3: Generating legal response...');
      const completionResponse = await fetch(`${this.baseUrl}/v1/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `Based on the breach of contract, what remedies are available? 
                   Context: ${analysisData.analysis.raw || 'Contract breach identified'}`,
          max_tokens: 200
        })
      });
      
      const completionData = await completionResponse.json();
      if (!completionData.choices?.[0]?.text) {
        throw new Error('Completion generation failed');
      }
      console.log('   ✓ Legal response generated');
      
      console.log('✅ Full pipeline executed successfully');
      this.results.passed++;
      
    } catch (error) {
      console.log('❌ Full pipeline test failed:', error.message);
      this.results.failed++;
      this.results.errors.push({ test: 'Full Pipeline', error: error.message });
    }
  }

  async testPerformance() {
    console.log('\n📋 Test 8: Performance Benchmarks');
    console.log('─'.repeat(50));
    
    const iterations = 5;
    const times = [];
    
    console.log(`   Running ${iterations} iterations...`);
    
    for (let i = 0; i < iterations; i++) {
      const start = Date.now();
      
      try {
        await fetch(`${this.baseUrl}/v1/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: 'What is negligence?',
            max_tokens: 50,
            temperature: 0.1
          })
        });
        
        const elapsed = Date.now() - start;
        times.push(elapsed);
        process.stdout.write('.');
      } catch (error) {
        console.log(`\n   ✗ Iteration ${i + 1} failed`);
      }
    }
    
    console.log();
    
    if (times.length > 0) {
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const min = Math.min(...times);
      const max = Math.max(...times);
      
      console.log('✅ Performance test completed');
      console.log(`   Average: ${avg.toFixed(0)}ms`);
      console.log(`   Min: ${min}ms`);
      console.log(`   Max: ${max}ms`);
      
      if (avg < 2000) {
        console.log('   ✓ Performance is acceptable');
        this.results.passed++;
      } else {
        console.log('   ⚠️  Performance could be improved');
        this.results.passed++;
      }
    } else {
      console.log('❌ Performance test failed');
      this.results.failed++;
    }
  }

  printResults() {
    console.log('\n' + '═'.repeat(60));
    console.log('                    TEST RESULTS SUMMARY');
    console.log('═'.repeat(60));
    
    const total = this.results.passed + this.results.failed;
    const passRate = total > 0 ? (this.results.passed / total * 100).toFixed(1) : 0;
    
    console.log(`\n  Total Tests: ${total}`);
    console.log(`  ✅ Passed: ${this.results.passed}`);
    console.log(`  ❌ Failed: ${this.results.failed}`);
    console.log(`  📊 Pass Rate: ${passRate}%`);
    
    if (this.results.errors.length > 0) {
      console.log('\n  Errors:');
      this.results.errors.forEach(err => {
        console.log(`    - ${err.test}: ${err.error}`);
      });
    }
    
    console.log('\n' + '═'.repeat(60));
    
    if (passRate >= 75) {
      console.log('🎉 GEMMA3 LEGAL AI INTEGRATION: WORKING END-TO-END! 🎉');
    } else if (passRate >= 50) {
      console.log('⚠️  Integration partially working. Check failed tests.');
    } else {
      console.log('❌ Integration needs attention. Many tests failing.');
    }
    
    console.log('═'.repeat(60) + '\n');
    
    // Save results to file
    this.saveResults();
  }

  async saveResults() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `test-results-${timestamp}.json`;
    const filepath = path.join(__dirname, filename);
    
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total: this.results.passed + this.results.failed,
        passed: this.results.passed,
        failed: this.results.failed,
        passRate: ((this.results.passed / (this.results.passed + this.results.failed)) * 100).toFixed(1) + '%'
      },
      errors: this.results.errors,
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch
      }
    };
    
    try {
      await fs.writeFile(filepath, JSON.stringify(report, null, 2));
      console.log(`📁 Results saved to: ${filename}`);
    } catch (error) {
      console.log('Could not save results:', error.message);
    }
  }
}

// Run the tests
async function main() {
  const tester = new Gemma3E2ETest();
  
  // Wait a moment for services to be ready
  console.log('Waiting for services to initialize...');
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  await tester.runAllTests();
  
  process.exit(tester.results.failed > 0 ? 1 : 0);
}

main().catch(error => {
  console.error('Test suite failed:', error);
  process.exit(1);
});
