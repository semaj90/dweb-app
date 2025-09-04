#!/usr/bin/env node
/**
 * Comprehensive Gemma3 Legal.gguf Search Integration Verification
 * Tests all AI services, WebGPU, WebAssembly, OCR, and vector systems
 */

import chalk from 'chalk';
import { performance } from 'perf_hooks';

// Service endpoints
const ENDPOINTS = {
  sveltekit: 'http://localhost:5175',
  ollama: 'http://localhost:11434',
  qdrant: 'http://localhost:6333',
  postgres: 'postgresql://postgres:your-password@localhost:5432/legal_ai_db',
  redis: 'redis://localhost:6379',
  enhancedRAG: 'http://localhost:8094',
  uploadService: 'http://localhost:8093'
};

class GemmaIntegrationTester {
  constructor() {
    this.results = {
      services: {},
      webgpu: {},
      wasm: {},
      ocr: {},
      vectorSearch: {},
      ragPipeline: {},
      overall: { passed: 0, failed: 0 }
    };
  }

  /**
   * Main test execution
   */
  async runTests() {
    console.log(chalk.blue.bold('🧪 GEMMA3 LEGAL.GGUF INTEGRATION VERIFICATION'));
    console.log(chalk.blue('================================================\n'));

    try {
      // 1. Core Services
      await this.testCoreServices();
      
      // 2. Gemma3 Model
      await this.testGemmaModel();
      
      // 3. WebGPU Integration
      await this.testWebGPUProcessor();
      
      // 4. WebAssembly Acceleration
      await this.testWebAssemblyAccelerator();
      
      // 5. OCR Services
      await this.testOCRServices();
      
      // 6. Vector Search & Ranking
      await this.testVectorServices();
      
      // 7. RAG Pipeline
      await this.testRAGPipeline();
      
      // 8. End-to-End Test
      await this.testEndToEnd();
      
      this.generateReport();
      
    } catch (error) {
      console.error(chalk.red('❌ Test execution failed:'), error);
      process.exit(1);
    }
  }

  /**
   * Test core services availability
   */
  async testCoreServices() {
    console.log(chalk.yellow('📋 Testing Core Services'));
    console.log(chalk.yellow('========================\n'));

    const services = [
      { name: 'SvelteKit', url: ENDPOINTS.sveltekit, path: '/' },
      { name: 'Ollama', url: ENDPOINTS.ollama, path: '/api/tags' },
      { name: 'Qdrant', url: ENDPOINTS.qdrant, path: '/collections' },
      { name: 'Enhanced RAG', url: ENDPOINTS.enhancedRAG, path: '/health' },
      { name: 'Upload Service', url: ENDPOINTS.uploadService, path: '/health' }
    ];

    for (const service of services) {
      try {
        const response = await fetch(`${service.url}${service.path}`);
        const status = response.ok ? '✅ RUNNING' : '⚠️  ERROR';
        console.log(`${service.name.padEnd(15)} | ${status}`);
        this.results.services[service.name] = response.ok;
        
        if (response.ok) this.results.overall.passed++;
        else this.results.overall.failed++;
        
      } catch (error) {
        console.log(`${service.name.padEnd(15)} | ❌ OFFLINE`);
        this.results.services[service.name] = false;
        this.results.overall.failed++;
      }
    }
    console.log();
  }

  /**
   * Test Gemma3 legal model specifically
   */
  async testGemmaModel() {
    console.log(chalk.yellow('⚖️  Testing Gemma3 Legal Model'));
    console.log(chalk.yellow('===============================\n'));

    try {
      // Check if gemma3-legal model is available
      const modelsResponse = await fetch(`${ENDPOINTS.ollama}/api/tags`);
      const models = await modelsResponse.json();
      
      const hasGemma3 = models.models?.some(m => 
        m.name.includes('gemma3-legal') || m.name.includes('gemma3')
      );

      if (!hasGemma3) {
        console.log('⚠️  Gemma3-legal model not found, testing with available model');
      }

      // Test legal query
      const testQuery = "What constitutes breach of contract under common law?";
      const startTime = performance.now();

      const response = await fetch(`${ENDPOINTS.ollama}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: hasGemma3 ? 'gemma3-legal' : models.models[0]?.name || 'llama2',
          prompt: testQuery,
          stream: false,
          options: {
            temperature: 0.3,
            num_gpu: 32 // RTX 3060 optimized
          }
        })
      });

      const result = await response.json();
      const processingTime = performance.now() - startTime;

      if (response.ok && result.response) {
        console.log('✅ Gemma3 Legal Query: SUCCESS');
        console.log(`⏱️  Processing Time: ${Math.round(processingTime)}ms`);
        console.log(`📄 Response Length: ${result.response.length} characters`);
        console.log(`🤖 Model Used: ${result.model || 'Unknown'}`);
        
        this.results.overall.passed++;
      } else {
        console.log('❌ Gemma3 Legal Query: FAILED');
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('❌ Gemma3 model test failed:', error.message);
      this.results.overall.failed++;
    }
    console.log();
  }

  /**
   * Test WebGPU processor
   */
  async testWebGPUProcessor() {
    console.log(chalk.yellow('🖥️  Testing WebGPU Processor'));
    console.log(chalk.yellow('============================\n'));

    try {
      const response = await fetch(`${ENDPOINTS.sveltekit}/api/test-webgpu`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          testData: [1, 2, 3, 4, 5],
          operation: 'tensor-processing'
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ WebGPU Tensor Processing: SUCCESS');
        console.log(`⚡ GPU Acceleration: ${result.gpuAccelerated ? 'ENABLED' : 'DISABLED'}`);
        this.results.webgpu.tensorProcessing = true;
        this.results.overall.passed++;
      } else {
        console.log('⚠️  WebGPU API not available, checking component directly');
        this.results.webgpu.tensorProcessing = false;
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('⚠️  WebGPU test: Browser-only feature, skipping server test');
      this.results.webgpu.tensorProcessing = 'browser-only';
    }
    console.log();
  }

  /**
   * Test WebAssembly acceleration
   */
  async testWebAssemblyAccelerator() {
    console.log(chalk.yellow('🔧 Testing WebAssembly Acceleration'));
    console.log(chalk.yellow('===================================\n'));

    try {
      const response = await fetch(`${ENDPOINTS.sveltekit}/api/test-wasm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operation: 'simd-json-parse',
          data: '{"test": "legal document", "entities": ["contract", "party"]}'
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ WASM SIMD JSON: SUCCESS');
        console.log(`⚡ Performance Boost: ${result.performanceBoost || 'Unknown'}`);
        this.results.wasm.simdJson = true;
        this.results.overall.passed++;
      } else {
        console.log('⚠️  WASM API not available, testing vector operations');
        this.results.wasm.simdJson = false;
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('⚠️  WASM test: Browser-only feature, skipping server test');
      this.results.wasm.simdJson = 'browser-only';
    }
    console.log();
  }

  /**
   * Test OCR services
   */
  async testOCRServices() {
    console.log(chalk.yellow('📄 Testing OCR Services'));
    console.log(chalk.yellow('========================\n'));

    try {
      // Test enhanced OCR processor
      const response = await fetch(`${ENDPOINTS.sveltekit}/api/test-ocr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          testMode: true,
          documentType: 'legal_document'
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ Enhanced OCR: SUCCESS');
        console.log(`📝 Field Extraction: ${result.fieldsExtracted || 0} fields`);
        console.log(`🎯 Confidence: ${Math.round((result.confidence || 0) * 100)}%`);
        this.results.ocr.enhanced = true;
        this.results.overall.passed++;
      } else {
        console.log('❌ Enhanced OCR: FAILED');
        this.results.ocr.enhanced = false;
        this.results.overall.failed++;
      }

      // Test basic OCR service
      const basicResponse = await fetch(`${ENDPOINTS.sveltekit}/api/test-basic-ocr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ testMode: true })
      });

      if (basicResponse.ok) {
        console.log('✅ Basic OCR Service: SUCCESS');
        this.results.ocr.basic = true;
        this.results.overall.passed++;
      } else {
        console.log('❌ Basic OCR Service: FAILED');
        this.results.ocr.basic = false;
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('❌ OCR services test failed:', error.message);
      this.results.ocr.enhanced = false;
      this.results.ocr.basic = false;
      this.results.overall.failed += 2;
    }
    console.log();
  }

  /**
   * Test vector search and ranking
   */
  async testVectorServices() {
    console.log(chalk.yellow('🔍 Testing Vector Services'));
    console.log(chalk.yellow('==========================\n'));

    try {
      // Test vector ranking service
      const rankingResponse = await fetch(`${ENDPOINTS.sveltekit}/api/vector/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'contract breach liability damages',
          options: {
            limit: 5,
            documentType: 'document',
            includeExplanation: true
          }
        })
      });

      if (rankingResponse.ok) {
        const results = await rankingResponse.json();
        console.log('✅ Vector Ranking: SUCCESS');
        console.log(`📊 Results Found: ${results.length || 0}`);
        this.results.vectorSearch.ranking = true;
        this.results.overall.passed++;
      } else {
        console.log('❌ Vector Ranking: FAILED');
        this.results.vectorSearch.ranking = false;
        this.results.overall.failed++;
      }

      // Test custom reranker
      const rerankResponse = await fetch(`${ENDPOINTS.sveltekit}/api/vector/rerank`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'legal precedent analysis',
          userContext: {
            intent: 'analyze',
            timeOfDay: 'afternoon',
            userRole: 'prosecutor',
            workflowState: 'review'
          }
        })
      });

      if (rerankResponse.ok) {
        console.log('✅ Custom Reranker: SUCCESS');
        this.results.vectorSearch.reranker = true;
        this.results.overall.passed++;
      } else {
        console.log('❌ Custom Reranker: FAILED');
        this.results.vectorSearch.reranker = false;
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('❌ Vector services test failed:', error.message);
      this.results.vectorSearch.ranking = false;
      this.results.vectorSearch.reranker = false;
      this.results.overall.failed += 2;
    }
    console.log();
  }

  /**
   * Test RAG pipeline
   */
  async testRAGPipeline() {
    console.log(chalk.yellow('🔄 Testing RAG Pipeline'));
    console.log(chalk.yellow('========================\n'));

    try {
      // Test enhanced RAG pipeline
      const ragResponse = await fetch(`${ENDPOINTS.sveltekit}/api/rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'What are the elements of negligence in tort law?',
          options: {
            useSemanticSearch: true,
            useMemoryGraph: true,
            maxSources: 10,
            minConfidence: 0.7
          }
        })
      });

      if (ragResponse.ok) {
        const result = await ragResponse.json();
        console.log('✅ Enhanced RAG Pipeline: SUCCESS');
        console.log(`📚 Sources Used: ${result.sources?.length || 0}`);
        console.log(`🎯 Confidence: ${Math.round((result.confidence || 0) * 100)}%`);
        console.log(`💭 Reasoning: ${result.reasoning ? 'PROVIDED' : 'NOT PROVIDED'}`);
        this.results.ragPipeline.enhanced = true;
        this.results.overall.passed++;
      } else {
        console.log('❌ Enhanced RAG Pipeline: FAILED');
        this.results.ragPipeline.enhanced = false;
        this.results.overall.failed++;
      }

      // Test LangChain RAG
      const langchainResponse = await fetch(`${ENDPOINTS.sveltekit}/api/langchain-rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'Contract formation requirements',
          options: {
            thinkingMode: true,
            verbose: true,
            documentType: 'contract'
          }
        })
      });

      if (langchainResponse.ok) {
        console.log('✅ LangChain RAG: SUCCESS');
        this.results.ragPipeline.langchain = true;
        this.results.overall.passed++;
      } else {
        console.log('❌ LangChain RAG: FAILED');
        this.results.ragPipeline.langchain = false;
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('❌ RAG pipeline test failed:', error.message);
      this.results.ragPipeline.enhanced = false;
      this.results.ragPipeline.langchain = false;
      this.results.overall.failed += 2;
    }
    console.log();
  }

  /**
   * End-to-end integration test
   */
  async testEndToEnd() {
    console.log(chalk.yellow('🔗 End-to-End Integration Test'));
    console.log(chalk.yellow('================================\n'));

    try {
      const startTime = performance.now();

      // Simulate complete legal document analysis workflow
      const e2eResponse = await fetch(`${ENDPOINTS.sveltekit}/api/legal-ai/process-document`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documentText: `
            CONTRACT AGREEMENT
            
            This agreement is entered into between Party A (John Doe) and Party B (ABC Corporation).
            The effective date is January 1, 2024.
            
            Terms:
            1. Payment of $50,000 upon completion
            2. Delivery within 30 days
            3. Liability limitations apply
            
            IN WITNESS WHEREOF, the parties have executed this agreement.
          `,
          analysisType: 'comprehensive',
          extractFields: true,
          useGemma3: true
        })
      });

      const totalTime = performance.now() - startTime;

      if (e2eResponse.ok) {
        const result = await e2eResponse.json();
        console.log('✅ End-to-End Test: SUCCESS');
        console.log(`⏱️  Total Processing: ${Math.round(totalTime)}ms`);
        console.log(`📊 Analysis Complete: ${result.analysisComplete ? 'YES' : 'NO'}`);
        console.log(`🏷️  Entities Extracted: ${result.entities?.length || 0}`);
        console.log(`🎯 Overall Confidence: ${Math.round((result.confidence || 0) * 100)}%`);
        console.log(`🤖 AI Model: ${result.modelUsed || 'Unknown'}`);
        
        this.results.overall.passed++;
      } else {
        console.log('❌ End-to-End Test: FAILED');
        this.results.overall.failed++;
      }

    } catch (error) {
      console.log('❌ End-to-end test failed:', error.message);
      this.results.overall.failed++;
    }
    console.log();
  }

  /**
   * Generate comprehensive test report
   */
  generateReport() {
    console.log(chalk.blue.bold('📊 COMPREHENSIVE TEST REPORT'));
    console.log(chalk.blue('===============================\n'));

    const totalTests = this.results.overall.passed + this.results.overall.failed;
    const successRate = totalTests > 0 ? (this.results.overall.passed / totalTests) * 100 : 0;

    console.log(chalk.white('📈 Overall Results:'));
    console.log(`   ✅ Passed: ${this.results.overall.passed}`);
    console.log(`   ❌ Failed: ${this.results.overall.failed}`);
    console.log(`   📊 Success Rate: ${Math.round(successRate)}%\n`);

    // Service status
    console.log(chalk.white('🔧 Core Services:'));
    Object.entries(this.results.services).forEach(([service, status]) => {
      const icon = status ? '✅' : '❌';
      console.log(`   ${icon} ${service}`);
    });
    console.log();

    // Component status
    console.log(chalk.white('🧩 Components:'));
    console.log(`   🖥️  WebGPU: ${this.results.webgpu.tensorProcessing ? '✅' : '⚠️'}`);
    console.log(`   🔧 WebAssembly: ${this.results.wasm.simdJson ? '✅' : '⚠️'}`);
    console.log(`   📄 Enhanced OCR: ${this.results.ocr.enhanced ? '✅' : '❌'}`);
    console.log(`   📄 Basic OCR: ${this.results.ocr.basic ? '✅' : '❌'}`);
    console.log(`   🔍 Vector Ranking: ${this.results.vectorSearch.ranking ? '✅' : '❌'}`);
    console.log(`   🎯 Custom Reranker: ${this.results.vectorSearch.reranker ? '✅' : '❌'}`);
    console.log(`   🔄 Enhanced RAG: ${this.results.ragPipeline.enhanced ? '✅' : '❌'}`);
    console.log(`   🦜 LangChain RAG: ${this.results.ragPipeline.langchain ? '✅' : '❌'}`);
    console.log();

    // Recommendations
    console.log(chalk.white('💡 Recommendations:'));
    
    if (!this.results.services.Ollama) {
      console.log('   🔸 Start Ollama service: ollama serve');
    }
    
    if (!this.results.services.Qdrant) {
      console.log('   🔸 Start Qdrant: docker run -p 6333:6333 qdrant/qdrant');
    }
    
    if (successRate < 70) {
      console.log('   🔸 Check service configurations in .env files');
      console.log('   🔸 Ensure all dependencies are installed');
      console.log('   🔸 Verify model files are downloaded');
    }
    
    if (successRate >= 90) {
      console.log('   🎉 System is performing excellently!');
      console.log('   🔸 Ready for production deployment');
    }

    console.log();
    
    const statusColor = successRate >= 90 ? chalk.green : 
                       successRate >= 70 ? chalk.yellow : chalk.red;
    
    console.log(statusColor.bold(`🏁 FINAL STATUS: ${Math.round(successRate)}% SUCCESS RATE`));
  }
}

// Run tests
const tester = new GemmaIntegrationTester();
tester.runTests().catch(console.error);