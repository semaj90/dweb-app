#!/usr/bin/env tsx

/**
 * Comprehensive AI Pipeline End-to-End Test Script
 * Tests all components of the legal AI system integration
 */

import { performance } from 'perf_hooks';
import chalk from 'chalk';
import { ollamaService } from '../sveltekit-frontend/src/lib/services/ollamaService';
import { bullmqService } from '../sveltekit-frontend/src/lib/services/bullmqService';
import { multiLayerCache } from '../sveltekit-frontend/src/lib/services/multiLayerCache';
import { langChainService } from '../sveltekit-frontend/src/lib/ai/langchain-ollama-service';

// Types
interface TestResult {
  name: string;
  success: boolean;
  duration: number;
  error?: string;
  details?: any;
}

interface TestSuite {
  name: string;
  tests: TestResult[];
  totalTime: number;
  passed: number;
  failed: number;
}

// Test data
const SAMPLE_LEGAL_DOCUMENT = `
CASE FILE: People v. Smith
Case Number: CR-2024-001234

SUMMARY OF CHARGES:
Defendant John Smith is charged with theft in the first degree under Penal Code Section 484(a). 
The alleged incident occurred on January 15, 2024, at Best Buy Electronics located at 
123 Main Street, Anytown, CA.

EVIDENCE SUMMARY:
1. Video surveillance showing defendant concealing merchandise
2. Store security report filed by loss prevention officer James Wilson
3. Recovered merchandise valued at $1,247.99
4. Witness statement from store employee Sarah Johnson

PROSECUTION THEORY:
The prosecution alleges that defendant intentionally took merchandise with intent to 
permanently deprive the owner without paying the required amount.

DEFENSE POSITION:
Defense claims this was a misunderstanding and defendant intended to pay for all items.

BACKGROUND INFORMATION:
Defendant has no prior criminal history. Employed as software engineer at TechCorp Inc.
Lives at 456 Oak Avenue, Anytown, CA with spouse and two children.

LEGAL PRECEDENTS:
- People v. Davis (2019) 45 Cal.4th 123: Intent must be proven beyond reasonable doubt
- State v. Johnson (2020) 234 P.3d 567: Value determination for theft classification

NEXT STEPS:
Preliminary hearing scheduled for March 1, 2024 at 9:00 AM, Department 12.
`;

class AITestSuite {
  private results: TestSuite[] = [];
  private startTime: number = 0;

  constructor() {
    this.startTime = performance.now();
  }

  // Test Ollama Service
  async testOllamaService(): Promise<TestSuite> {
    const suite: TestSuite = {
      name: 'Ollama Service',
      tests: [],
      totalTime: 0,
      passed: 0,
      failed: 0
    };

    const startTime = performance.now();

    // Test 1: Health Check
    await this.runTest(suite, 'Health Check', async () => {
      const health = await ollamaService.checkHealth();
      if (health.status !== 'healthy') {
        throw new Error(`Ollama unhealthy: ${JSON.stringify(health)}`);
      }
      return {
        status: health.status,
        models: health.models,
        embedModel: health.embedModel,
        llmModel: health.llmModel
      };
    });

    // Test 2: Embedding Generation
    await this.runTest(suite, 'Single Embedding Generation', async () => {
      const embedding = await ollamaService.generateEmbedding("Legal document test");
      if (!embedding || embedding.length === 0) {
        throw new Error('Empty embedding generated');
      }
      return {
        dimensions: embedding.length,
        sampleValues: embedding.slice(0, 5)
      };
    });

    // Test 3: Batch Embeddings
    await this.runTest(suite, 'Batch Embedding Generation', async () => {
      const texts = ["Contract analysis", "Evidence review", "Case summary"];
      const embeddings = await ollamaService.generateBatchEmbeddings(texts);
      if (embeddings.length !== texts.length) {
        throw new Error(`Expected ${texts.length} embeddings, got ${embeddings.length}`);
      }
      return {
        count: embeddings.length,
        avgDimensions: embeddings.reduce((acc, emb) => acc + emb.length, 0) / embeddings.length
      };
    });

    // Test 4: Document Analysis
    await this.runTest(suite, 'Document Analysis', async () => {
      const analysis = await ollamaService.analyzeDocument(SAMPLE_LEGAL_DOCUMENT, 'summary');
      if (!analysis || analysis.length < 50) {
        throw new Error('Analysis too short or empty');
      }
      return {
        analysisLength: analysis.length,
        preview: analysis.substring(0, 100) + '...'
      };
    });

    // Test 5: Document Embedding with Chunking
    await this.runTest(suite, 'Document Embedding with Chunking', async () => {
      const result = await ollamaService.embedDocument(SAMPLE_LEGAL_DOCUMENT, {
        documentType: 'legal_case',
        caseId: 'test-case-001'
      });
      
      if (!result.chunks || result.chunks.length === 0) {
        throw new Error('No chunks generated');
      }

      return {
        chunksCount: result.chunks.length,
        avgChunkLength: result.chunks.reduce((acc, chunk) => acc + chunk.content.length, 0) / result.chunks.length,
        embeddingDimensions: result.chunks[0].embedding.length
      };
    });

    suite.totalTime = performance.now() - startTime;
    return suite;
  }

  // Test BullMQ Queue System
  async testBullMQService(): Promise<TestSuite> {
    const suite: TestSuite = {
      name: 'BullMQ Queue System',
      tests: [],
      totalTime: 0,
      passed: 0,
      failed: 0
    };

    const startTime = performance.now();

    // Test 1: Queue Stats
    await this.runTest(suite, 'Queue Statistics', async () => {
      const stats = await bullmqService.getAllQueueStats();
      const queueNames = Object.keys(stats);
      
      if (queueNames.length === 0) {
        throw new Error('No queues found');
      }

      return {
        queueCount: queueNames.length,
        queues: queueNames,
        stats
      };
    });

    // Test 2: Document Processing Job
    await this.runTest(suite, 'Document Processing Job', async () => {
      const job = await bullmqService.addDocumentProcessingJob({
        documentId: 'test-doc-001',
        content: SAMPLE_LEGAL_DOCUMENT,
        options: {
          extractText: true,
          generateEmbeddings: true,
          performAnalysis: true
        },
        metadata: {
          userId: 'test-user-001',
          caseId: 'test-case-001',
          filename: 'test-legal-document.txt'
        }
      });

      // Wait for job completion (with timeout)
      const maxWait = 30000; // 30 seconds
      const interval = 1000; // 1 second
      let waited = 0;
      
      while (waited < maxWait) {
        const jobStatus = await bullmqService.getJobStatus('document-processing', job.id!);
        if (jobStatus?.finishedOn) {
          return {
            jobId: job.id,
            status: 'completed',
            processingTime: jobStatus.finishedOn - jobStatus.processedOn!,
            returnValue: jobStatus.returnvalue
          };
        }
        if (jobStatus?.failedReason) {
          throw new Error(`Job failed: ${jobStatus.failedReason}`);
        }
        
        await new Promise(resolve => setTimeout(resolve, interval));
        waited += interval;
      }

      throw new Error('Job timed out');
    });

    // Test 3: Embedding Generation Job
    await this.runTest(suite, 'Embedding Generation Job', async () => {
      const job = await bullmqService.addEmbeddingJob({
        content: "Test content for embedding generation",
        type: 'document',
        entityId: 'test-entity-001',
        metadata: { test: true }
      });

      // Wait for completion
      await this.waitForJobCompletion('embedding-generation', job.id!, 15000);
      
      return {
        jobId: job.id,
        status: 'completed'
      };
    });

    // Test 4: AI Analysis Job
    await this.runTest(suite, 'AI Analysis Job', async () => {
      const job = await bullmqService.addAIAnalysisJob({
        content: SAMPLE_LEGAL_DOCUMENT.substring(0, 500), // Shorter for faster testing
        analysisType: 'summary',
        documentId: 'test-doc-002',
        userId: 'test-user-001'
      });

      await this.waitForJobCompletion('ai-analysis', job.id!, 20000);
      
      return {
        jobId: job.id,
        status: 'completed'
      };
    });

    suite.totalTime = performance.now() - startTime;
    return suite;
  }

  // Test Multi-Layer Cache
  async testMultiLayerCache(): Promise<TestSuite> {
    const suite: TestSuite = {
      name: 'Multi-Layer Cache',
      tests: [],
      totalTime: 0,
      passed: 0,
      failed: 0
    };

    const startTime = performance.now();

    // Test 1: Basic Set/Get Operations
    await this.runTest(suite, 'Basic Set/Get Operations', async () => {
      const testData = { analysis: 'Test legal analysis', confidence: 0.95 };
      const key = 'test-cache-key-001';
      
      await multiLayerCache.set(key, testData, {
        type: 'document',
        userId: 'test-user-001',
        ttl: 300
      });

      const retrieved = await multiLayerCache.get(key, { userId: 'test-user-001' });
      
      if (!retrieved || retrieved.analysis !== testData.analysis) {
        throw new Error('Cache set/get failed');
      }

      return {
        stored: testData,
        retrieved,
        match: JSON.stringify(testData) === JSON.stringify(retrieved)
      };
    });

    // Test 2: Cache Statistics
    await this.runTest(suite, 'Cache Statistics', async () => {
      const stats = multiLayerCache.getStats();
      
      return {
        totalEntries: stats.totalEntries,
        totalSize: stats.totalSize,
        hitRate: stats.hitRate,
        layerStats: stats.layerStats
      };
    });

    // Test 3: Fuzzy Search
    await this.runTest(suite, 'Fuzzy Search', async () => {
      // Add some searchable content
      const documents = [
        { id: '1', title: 'Contract Analysis', content: 'Legal contract review and analysis' },
        { id: '2', title: 'Criminal Case', content: 'Criminal law case documentation' },
        { id: '3', title: 'Evidence Review', content: 'Digital evidence analysis and documentation' }
      ];

      for (const doc of documents) {
        await multiLayerCache.set(`doc-${doc.id}`, doc, {
          type: 'document',
          userId: 'test-user-001'
        });
      }

      const searchResults = await multiLayerCache.fuzzySearch('document', 'contract', {
        keys: ['title', 'content'],
        limit: 5,
        includeScore: true
      });

      return {
        query: 'contract',
        resultsCount: searchResults.length,
        results: searchResults.map(r => ({ 
          title: r.item.title, 
          score: r.score 
        }))
      };
    });

    // Test 4: Cache Invalidation
    await this.runTest(suite, 'Cache Invalidation', async () => {
      // Set some test data
      await multiLayerCache.set('invalidation-test-1', { data: 'test1' }, {
        type: 'document',
        userId: 'test-user-001'
      });
      
      await multiLayerCache.set('invalidation-test-2', { data: 'test2' }, {
        type: 'document',
        userId: 'test-user-001'
      });

      // Invalidate with pattern
      const invalidated = await multiLayerCache.invalidate('invalidation-test', {
        userId: 'test-user-001'
      });

      // Verify they're gone
      const check1 = await multiLayerCache.get('invalidation-test-1');
      const check2 = await multiLayerCache.get('invalidation-test-2');

      return {
        invalidatedCount: invalidated,
        check1: check1 === null,
        check2: check2 === null
      };
    });

    suite.totalTime = performance.now() - startTime;
    return suite;
  }

  // Test LangChain Integration
  async testLangChainService(): Promise<TestSuite> {
    const suite: TestSuite = {
      name: 'LangChain Integration',
      tests: [],
      totalTime: 0,
      passed: 0,
      failed: 0
    };

    const startTime = performance.now();

    // Test 1: Health Check
    await this.runTest(suite, 'LangChain Health Check', async () => {
      const health = await langChainService.healthCheck();
      
      return {
        status: health.status,
        ollama: health.ollama,
        embedding: health.embedding,
        database: health.database,
        cuda: health.cuda,
        models: health.models
      };
    });

    // Test 2: Document Processing
    await this.runTest(suite, 'Document Processing', async () => {
      const testDocId = `test-doc-${Date.now()}`;
      const result = await langChainService.processDocument(
        testDocId,
        SAMPLE_LEGAL_DOCUMENT,
        { documentType: 'legal_case', source: 'test' }
      );

      if (result.chunksCreated === 0) {
        throw new Error('No chunks created');
      }

      return {
        documentId: result.documentId,
        chunksCreated: result.chunksCreated,
        processingTime: result.processingTime,
        totalTokens: result.metadata.totalTokens,
        avgChunkSize: result.metadata.avgChunkSize
      };
    });

    // Test 3: Document Query (RAG)
    await this.runTest(suite, 'Document Query (RAG)', async () => {
      const query = "What are the charges against the defendant?";
      const result = await langChainService.queryDocuments(
        query,
        'test-user-001',
        'test-case-001',
        'test-session-001'
      );

      if (!result.answer || result.answer.length < 10) {
        throw new Error('Query response too short or empty');
      }

      return {
        query,
        answerLength: result.answer.length,
        answerPreview: result.answer.substring(0, 100) + '...',
        sourceDocuments: result.sourceDocuments.length,
        confidence: result.confidence,
        processingTime: result.processingTime,
        tokensUsed: result.tokensUsed
      };
    });

    // Test 4: Document Summarization
    await this.runTest(suite, 'Document Summarization', async () => {
      const result = await langChainService.summarizeDocument(
        'test-summary-doc-001',
        SAMPLE_LEGAL_DOCUMENT,
        {
          extractEntities: true,
          riskAssessment: true,
          generateRecommendations: true
        }
      );

      if (!result.summary || result.summary.length < 50) {
        throw new Error('Summary too short or empty');
      }

      return {
        summaryLength: result.summary.length,
        keyPointsCount: result.keyPoints.length,
        entitiesCount: result.entities.length,
        hasRiskAssessment: !!result.riskAssessment,
        recommendationsCount: result.recommendations.length,
        confidence: result.confidence
      };
    });

    // Test 5: Processing Statistics
    await this.runTest(suite, 'Processing Statistics', async () => {
      const stats = await langChainService.getProcessingStats(1); // Last 1 day
      
      return {
        documentsProcessed: stats.documentsProcessed,
        averageProcessingTime: stats.averageProcessingTime,
        totalQueries: stats.totalQueries,
        averageConfidence: stats.averageConfidence,
        modelUsage: stats.modelUsage
      };
    });

    suite.totalTime = performance.now() - startTime;
    return suite;
  }

  // Test Vector Search & Similarity
  async testVectorOperations(): Promise<TestSuite> {
    const suite: TestSuite = {
      name: 'Vector Operations & Search',
      tests: [],
      totalTime: 0,
      passed: 0,
      failed: 0
    };

    const startTime = performance.now();

    // Test 1: Vector Similarity Calculation
    await this.runTest(suite, 'Vector Similarity Calculation', async () => {
      const text1 = "Contract breach and damages";
      const text2 = "Breach of contract damages";
      const text3 = "Criminal theft case";

      const [emb1, emb2, emb3] = await Promise.all([
        ollamaService.generateEmbedding(text1),
        ollamaService.generateEmbedding(text2),
        ollamaService.generateEmbedding(text3)
      ]);

      // Calculate cosine similarity manually
      const similarity12 = this.cosineSimilarity(emb1, emb2);
      const similarity13 = this.cosineSimilarity(emb1, emb3);

      return {
        similarity_contract_similar: similarity12,
        similarity_contract_different: similarity13,
        expected_higher_similarity: similarity12 > similarity13
      };
    });

    // Test 2: Contextual Embedding Generation
    await this.runTest(suite, 'Contextual Embedding Generation', async () => {
      const text = "Evidence analysis report";
      const result = await ollamaService.generateContextualEmbedding(text, {
        documentType: 'evidence_report',
        caseId: 'case-001',
        userId: 'user-001',
        timestamp: new Date()
      });

      if (!result.embedding || result.embedding.length === 0) {
        throw new Error('No contextual embedding generated');
      }

      return {
        embeddingDimensions: result.embedding.length,
        metadata: result.metadata,
        hasContext: !!result.metadata.documentType
      };
    });

    suite.totalTime = performance.now() - startTime;
    return suite;
  }

  // Helper method to wait for job completion
  private async waitForJobCompletion(queueName: string, jobId: string, timeout: number): Promise<void> {
    const interval = 1000;
    let waited = 0;
    
    while (waited < timeout) {
      const jobStatus = await bullmqService.getJobStatus(queueName, jobId);
      if (jobStatus?.finishedOn) return;
      if (jobStatus?.failedReason) {
        throw new Error(`Job failed: ${jobStatus.failedReason}`);
      }
      
      await new Promise(resolve => setTimeout(resolve, interval));
      waited += interval;
    }
    
    throw new Error('Job timed out');
  }

  // Helper method to calculate cosine similarity
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  // Helper method to run individual tests
  private async runTest(suite: TestSuite, testName: string, testFn: () => Promise<any>): Promise<void> {
    const testStart = performance.now();
    
    try {
      console.log(chalk.blue(`  Running: ${testName}...`));
      const details = await testFn();
      const duration = performance.now() - testStart;
      
      suite.tests.push({
        name: testName,
        success: true,
        duration,
        details
      });
      
      suite.passed++;
      console.log(chalk.green(`  ✓ ${testName} (${Math.round(duration)}ms)`));
    } catch (error) {
      const duration = performance.now() - testStart;
      
      suite.tests.push({
        name: testName,
        success: false,
        duration,
        error: error instanceof Error ? error.message : String(error)
      });
      
      suite.failed++;
      console.log(chalk.red(`  ✗ ${testName} (${Math.round(duration)}ms): ${error}`));
    }
  }

  // Run all test suites
  async runAllTests(): Promise<void> {
    console.log(chalk.bold.cyan('\n🧪 AI Pipeline End-to-End Test Suite\n'));
    console.log(chalk.gray('Testing comprehensive AI system integration...\n'));

    try {
      // Run all test suites
      this.results.push(await this.testOllamaService());
      this.results.push(await this.testBullMQService());
      this.results.push(await this.testMultiLayerCache());
      this.results.push(await this.testLangChainService());
      this.results.push(await this.testVectorOperations());

      // Generate report
      this.generateReport();
    } catch (error) {
      console.error(chalk.red('\n❌ Test suite failed to complete:'), error);
      process.exit(1);
    }
  }

  // Generate comprehensive test report
  private generateReport(): void {
    const totalTime = performance.now() - this.startTime;
    let totalTests = 0;
    let totalPassed = 0;
    let totalFailed = 0;

    console.log(chalk.bold.cyan('\n📊 Test Results Summary\n'));
    console.log(chalk.gray('=' .repeat(80)));

    this.results.forEach(suite => {
      totalTests += suite.tests.length;
      totalPassed += suite.passed;
      totalFailed += suite.failed;

      const status = suite.failed === 0 ? 
        chalk.green('✓ PASSED') : 
        chalk.red(`✗ FAILED (${suite.failed}/${suite.tests.length})`);
      
      console.log(`${status} ${suite.name} - ${suite.passed}/${suite.tests.length} tests (${Math.round(suite.totalTime)}ms)`);
      
      if (suite.failed > 0) {
        suite.tests
          .filter(test => !test.success)
          .forEach(test => {
            console.log(chalk.red(`    ✗ ${test.name}: ${test.error}`));
          });
      }
    });

    console.log(chalk.gray('=' .repeat(80)));
    
    const overallStatus = totalFailed === 0 ? 
      chalk.green.bold('🎉 ALL TESTS PASSED') : 
      chalk.red.bold(`❌ ${totalFailed}/${totalTests} TESTS FAILED`);
    
    console.log(`\n${overallStatus}`);
    console.log(chalk.gray(`Total Time: ${Math.round(totalTime)}ms`));
    console.log(chalk.gray(`Tests: ${totalPassed} passed, ${totalFailed} failed, ${totalTests} total\n`));

    // Performance insights
    console.log(chalk.bold.yellow('🔍 Performance Insights:'));
    
    const slowestTests = this.results
      .flatMap(suite => suite.tests)
      .sort((a, b) => b.duration - a.duration)
      .slice(0, 5);
    
    slowestTests.forEach((test, i) => {
      console.log(`${i + 1}. ${test.name}: ${Math.round(test.duration)}ms`);
    });

    // Detailed results for successful tests
    if (totalFailed === 0) {
      console.log(chalk.bold.green('\n✨ System Health Check Results:'));
      
      const healthTest = this.results
        .find(suite => suite.name === 'Ollama Service')
        ?.tests.find(test => test.name === 'Health Check');
        
      if (healthTest?.details) {
        console.log(`  • Ollama Status: ${healthTest.details.status}`);
        console.log(`  • Available Models: ${healthTest.details.models.join(', ')}`);
        console.log(`  • Embedding Model Ready: ${healthTest.details.embedModel ? '✓' : '✗'}`);
        console.log(`  • LLM Model Ready: ${healthTest.details.llmModel ? '✓' : '✗'}`);
      }

      const cacheStats = this.results
        .find(suite => suite.name === 'Multi-Layer Cache')
        ?.tests.find(test => test.name === 'Cache Statistics');
        
      if (cacheStats?.details) {
        console.log(`  • Cache Entries: ${cacheStats.details.totalEntries}`);
        console.log(`  • Cache Hit Rate: ${Math.round(cacheStats.details.hitRate * 100)}%`);
        console.log(`  • Memory Usage: ${Math.round(cacheStats.details.totalSize / 1024)}KB`);
      }
    }

    // Exit with appropriate code
    process.exit(totalFailed === 0 ? 0 : 1);
  }
}

// Run the test suite
async function main() {
  const testSuite = new AITestSuite();
  await testSuite.runAllTests();
}

// Handle process signals
process.on('SIGINT', () => {
  console.log(chalk.yellow('\n⚠️  Test suite interrupted'));
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection at:', promise, 'reason:', reason));
  process.exit(1);
});

if (require.main === module) {
  main().catch(error => {
    console.error(chalk.red('Test suite error:'), error);
    process.exit(1);
  });
}

export default AITestSuite;