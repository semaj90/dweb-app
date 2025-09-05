/**
 * BVector Store Integration Test Suite
 * Tests embedding worker → BVector store → GPU cache → Go binaries integration
 * Validates contextual prompting and reinforcement learning functionality
 */

import { EnhancedBVectorStore } from '../services/enhanced-bvector-store';
import { EmbeddingWorkerManager } from '../workers/embedding-worker';
import type { BVectorConfig, BVectorEntry, SearchOptions, SearchResult } from '../services/enhanced-bvector-store';

interface TestConfig {
  skipGpuTests?: boolean;
  skipGoBindings?: boolean;
  mockWorkers?: boolean;
  verbose?: boolean;
}

interface TestResult {
  testName: string;
  success: boolean;
  duration: number;
  error?: string;
  details?: Record<string, any>;
}

export class BVectorIntegrationTestSuite {
  private bvectorStore: EnhancedBVectorStore | null = null;
  private embeddingWorker: EmbeddingWorkerManager | null = null;
  private testResults: TestResult[] = [];
  private config: TestConfig;

  constructor(config: TestConfig = {}) {
    this.config = {
      skipGpuTests: false,
      skipGoBindings: false,
      mockWorkers: false,
      verbose: true,
      ...config
    };
  }

  async runFullIntegrationTest(): Promise<{
    success: boolean;
    results: TestResult[];
    summary: {
      totalTests: number;
      passed: number;
      failed: number;
      skipped: number;
      totalDuration: number;
    };
  }> {
    console.log('🧪 Starting BVector Store Integration Test Suite');
    console.log('='.repeat(60));

    const startTime = Date.now();

    try {
      // Initialize components
      await this.initializeComponents();

      // Core integration tests
      await this.testEmbeddingWorkerIntegration();
      await this.testBVectorStoreBasicOperations();
      await this.testMultiLayerCacheSystem();
      await this.testContextualPrompting();
      await this.testReinforcementLearning();

      // Go binaries tests (if not skipped)
      if (!this.config.skipGoBindings) {
        await this.testGoBinaryIntegration();
      }

      // GPU acceleration tests (if not skipped)
      if (!this.config.skipGpuTests) {
        await this.testGPUAcceleration();
      }

      // Performance benchmarks
      await this.runPerformanceBenchmarks();

      // Cleanup
      await this.cleanup();

    } catch (error) {
      this.addTestResult('INITIALIZATION', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }

    const endTime = Date.now();
    const totalDuration = endTime - startTime;

    // Generate summary
    const summary = this.generateTestSummary(totalDuration);
    
    if (this.config.verbose) {
      this.printTestResults(summary);
    }

    return {
      success: summary.failed === 0,
      results: this.testResults,
      summary
    };
  }

  private async initializeComponents(): Promise<void> {
    const startTime = Date.now();

    try {
      // Initialize embedding worker
      this.embeddingWorker = new EmbeddingWorkerManager();
      
      // Initialize BVector store with test configuration
      const testConfig: BVectorConfig = {
        goBinaries: {
          vectorService: 'vector-service.exe',
          cudaService: 'cuda-ai-service.exe',
          enhancedRAG: 'enhanced-rag.exe',
          gpuOrchestrator: 'gpu-orchestrator.exe'
        },
        gpuCache: {
          layers: 4,
          maxMemoryMB: 6144, // RTX 3060 Ti conservative limit
          batchSize: 32,
          enableQuantization: true
        },
        embedding: {
          dimensions: 384, // nomic-embed-text
          model: 'nomic-embed-text',
          workerThreads: 4,
          batchSize: 16
        },
        reinforcementLearning: {
          enabled: true,
          learningRate: 0.01,
          decayFactor: 0.95,
          feedbackThreshold: 0.7
        },
        storage: {
          redis: {
            host: 'localhost',
            port: 6379
          },
          postgresql: {
            enabled: true,
            tableName: 'bvector_store'
          },
          neo4j: {
            enabled: true,
            uri: 'bolt://localhost:7687'
          }
        }
      };

      this.bvectorStore = new EnhancedBVectorStore(testConfig);
      await this.bvectorStore.initialize();

      this.addTestResult('COMPONENT_INITIALIZATION', true, Date.now() - startTime, undefined, {
        embeddingWorkerAvailable: this.embeddingWorker.isAvailable,
        bvectorStoreInitialized: this.bvectorStore.isInitialized,
        gpuCacheEnabled: this.bvectorStore.gpuCacheEnabled
      });

    } catch (error) {
      this.addTestResult('COMPONENT_INITIALIZATION', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }

  private async testEmbeddingWorkerIntegration(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.embeddingWorker || !this.bvectorStore) {
        throw new Error('Components not initialized');
      }

      // Test basic embedding generation
      const testTexts = [
        'This is a legal contract regarding employment terms.',
        'Evidence shows the defendant was present at the scene.',
        'The plaintiff seeks damages for breach of contract.',
        'Chain of custody was properly maintained for digital evidence.'
      ];

      const embeddings = await this.embeddingWorker.processEmbeddings({
        texts: testTexts,
        batchSize: 2,
        model: 'nomic-embed-text',
        dimensions: 384
      });

      // Verify embeddings structure
      if (!embeddings.results || embeddings.results.length !== testTexts.length) {
        throw new Error(`Expected ${testTexts.length} embeddings, got ${embeddings.results?.length || 0}`);
      }

      // Test embedding dimensions
      for (const result of embeddings.results) {
        if (result.embedding.length !== 384) {
          throw new Error(`Expected 384 dimensions, got ${result.embedding.length}`);
        }
      }

      // Test BVector store embedding integration
      const bvectorEntries: BVectorEntry[] = embeddings.results.map((result, index) => ({
        id: `test-${Date.now()}-${index}`,
        content: testTexts[index],
        embedding: result.embedding,
        metadata: {
          userId: 'test-user',
          caseId: 'test-case-001',
          timestamp: Date.now(),
          userRole: 'prosecutor' as const,
          confidence: Math.random() * 0.3 + 0.7, // 0.7-1.0 confidence
          legalDomain: 'contract-law',
          jurisdiction: 'federal'
        }
      }));

      // Store entries in BVector store
      for (const entry of bvectorEntries) {
        await this.bvectorStore.store(entry);
      }

      this.addTestResult('EMBEDDING_WORKER_INTEGRATION', true, Date.now() - startTime, undefined, {
        embeddingsGenerated: embeddings.results.length,
        averageProcessingTime: embeddings.averageTime,
        tokenCount: embeddings.metrics.tokenCount,
        bvectorEntriesStored: bvectorEntries.length
      });

    } catch (error) {
      this.addTestResult('EMBEDDING_WORKER_INTEGRATION', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async testBVectorStoreBasicOperations(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      // Test search functionality
      const searchOptions: SearchOptions = {
        topK: 5,
        threshold: 0.1,
        includeMetadata: true,
        userContext: {
          userId: 'test-user',
          userRole: 'prosecutor',
          currentCase: 'test-case-001'
        },
        filters: {
          legalDomain: 'contract-law'
        }
      };

      const searchResults = await this.bvectorStore.search(
        'What are the employment contract terms?',
        searchOptions
      );

      // Verify search results structure
      if (!Array.isArray(searchResults)) {
        throw new Error('Search results should be an array');
      }

      // Test contextual search with user preferences
      const contextualResults = await this.bvectorStore.contextualSearch(
        'breach of contract evidence',
        {
          ...searchOptions,
          enableRL: true,
          userHistory: ['contract-disputes', 'employment-law']
        }
      );

      // Test similarity calculation
      const similarityMatrix = await this.bvectorStore.calculateSimilarityMatrix([
        'employment contract terms',
        'breach of contract',
        'evidence chain of custody'
      ]);

      this.addTestResult('BVECTOR_BASIC_OPERATIONS', true, Date.now() - startTime, undefined, {
        searchResultsCount: searchResults.length,
        contextualResultsCount: contextualResults.length,
        similarityMatrixSize: similarityMatrix.length,
        averageConfidence: searchResults.reduce((sum, result) => sum + result.metadata.confidence, 0) / searchResults.length
      });

    } catch (error) {
      this.addTestResult('BVECTOR_BASIC_OPERATIONS', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async testMultiLayerCacheSystem(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      // Test cache layer performance
      const query = 'contract liability terms';
      
      // First search (should populate cache)
      const search1Start = Date.now();
      await this.bvectorStore.search(query, { topK: 10 });
      const search1Duration = Date.now() - search1Start;

      // Second search (should use cache)
      const search2Start = Date.now();
      await this.bvectorStore.search(query, { topK: 10 });
      const search2Duration = Date.now() - search2Start;

      // Cache should make second search faster
      const cacheSpeedup = search1Duration / search2Duration;

      // Test cache statistics
      const cacheStats = await this.bvectorStore.getCacheStatistics();

      // Test cache invalidation
      await this.bvectorStore.invalidateCache('user:test-user');
      
      // Test cache layer switching
      await this.bvectorStore.optimizeCacheLayers();

      this.addTestResult('MULTI_LAYER_CACHE_SYSTEM', true, Date.now() - startTime, undefined, {
        search1Duration,
        search2Duration,
        cacheSpeedup: Math.round(cacheSpeedup * 100) / 100,
        cacheHitRate: cacheStats.hitRate,
        cacheLayerCount: cacheStats.activeLayers,
        memoryUsageMB: cacheStats.memoryUsageMB
      });

    } catch (error) {
      this.addTestResult('MULTI_LAYER_CACHE_SYSTEM', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async testContextualPrompting(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      // Test contextual prompt generation
      const userContext = {
        userId: 'test-prosecutor-001',
        userRole: 'prosecutor' as const,
        currentCase: 'criminal-case-2024-001',
        recentQueries: [
          'evidence tampering statutes',
          'chain of custody requirements',
          'digital forensics standards'
        ],
        workflowStage: 'evidence-review'
      };

      // Generate contextual prompts
      const contextualPrompts = await this.bvectorStore.generateContextualPrompts(
        'How do I verify digital evidence integrity?',
        userContext
      );

      // Verify prompt structure
      if (!contextualPrompts.enhancedPrompt || !contextualPrompts.contextualReferences) {
        throw new Error('Contextual prompts missing required fields');
      }

      // Test context-aware search results
      const contextAwareResults = await this.bvectorStore.contextualSearch(
        'digital evidence verification',
        {
          userContext,
          enableRL: true,
          topK: 8
        }
      );

      // Test prompt optimization based on user feedback
      await this.bvectorStore.recordPromptFeedback({
        promptId: contextualPrompts.promptId,
        userId: userContext.userId,
        rating: 4.5,
        actuallyUseful: true,
        improvements: ['more specific to criminal law']
      });

      this.addTestResult('CONTEXTUAL_PROMPTING', true, Date.now() - startTime, undefined, {
        promptLength: contextualPrompts.enhancedPrompt.length,
        contextualReferencesCount: contextualPrompts.contextualReferences.length,
        contextAwareResultsCount: contextAwareResults.length,
        averageContextualBoost: contextAwareResults.reduce((sum, result) => sum + result.contextualBoost, 0) / contextAwareResults.length
      });

    } catch (error) {
      this.addTestResult('CONTEXTUAL_PROMPTING', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async testReinforcementLearning(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      const testUserId = 'rl-test-user-001';
      
      // Simulate user interaction patterns
      const interactions = [
        {
          query: 'contract breach remedies',
          selectedResults: [0, 1], // User clicked first two results
          userSatisfaction: 0.9,
          followUpQuery: 'damages calculation'
        },
        {
          query: 'evidence chain of custody',
          selectedResults: [2], // User only clicked third result
          userSatisfaction: 0.6,
          followUpQuery: 'digital evidence standards'
        },
        {
          query: 'employment law violations',
          selectedResults: [0, 3, 4], // User clicked multiple results
          userSatisfaction: 0.95,
          followUpQuery: null // No follow-up needed
        }
      ];

      // Process interactions for RL training
      for (const interaction of interactions) {
        // Get search results
        const searchResults = await this.bvectorStore.search(interaction.query, {
          topK: 10,
          userContext: { userId: testUserId, userRole: 'prosecutor' }
        });

        // Record user feedback
        await this.bvectorStore.recordUserFeedback({
          userId: testUserId,
          query: interaction.query,
          results: searchResults,
          selectedIndices: interaction.selectedResults,
          userSatisfaction: interaction.userSatisfaction,
          contextualRelevance: Math.random() * 0.4 + 0.6, // 0.6-1.0
          followUpSuccess: interaction.followUpQuery === null
        });

        // Update user preference model
        await this.bvectorStore.updateUserPreferences(testUserId, {
          preferredDomains: ['contract-law', 'employment-law'],
          searchPatterns: ['remedies', 'evidence', 'violations'],
          satisfactionHistory: [interaction.userSatisfaction]
        });
      }

      // Test RL-enhanced search
      const enhancedResults = await this.bvectorStore.searchWithRL(
        'contract dispute resolution',
        {
          userId: testUserId,
          topK: 5,
          usePersonalization: true
        }
      );

      // Verify RL weights are applied
      if (!enhancedResults.every(result => result.rlWeight !== undefined)) {
        throw new Error('RL weights not applied to search results');
      }

      // Test preference learning accuracy
      const userPreferences = await this.bvectorStore.getUserPreferences(testUserId);
      
      this.addTestResult('REINFORCEMENT_LEARNING', true, Date.now() - startTime, undefined, {
        interactionsProcessed: interactions.length,
        enhancedResultsCount: enhancedResults.length,
        averageRLWeight: enhancedResults.reduce((sum, result) => sum + result.rlWeight, 0) / enhancedResults.length,
        learnedPreferences: Object.keys(userPreferences.domains || {}).length,
        avgSatisfactionScore: interactions.reduce((sum, int) => sum + int.userSatisfaction, 0) / interactions.length
      });

    } catch (error) {
      this.addTestResult('REINFORCEMENT_LEARNING', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async testGoBinaryIntegration(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      // Test Go binary service connections
      const goBinaryStatus = await this.bvectorStore.checkGoBinaryStatus();

      // Test vector service integration
      const vectorServiceResult = await this.bvectorStore.callGoBinary('vectorService', {
        operation: 'similarity_search',
        query_embedding: new Array(384).fill(0).map(() => Math.random()),
        top_k: 10,
        threshold: 0.5
      });

      // Test CUDA service integration (if available)
      let cudaServiceResult = null;
      if (goBinaryStatus.cudaService.available) {
        cudaServiceResult = await this.bvectorStore.callGoBinary('cudaService', {
          operation: 'batch_embedding',
          texts: ['test text for CUDA embedding'],
          batch_size: 1
        });
      }

      // Test Enhanced RAG service
      const ragServiceResult = await this.bvectorStore.callGoBinary('enhancedRAG', {
        operation: 'legal_rag_query',
        query: 'contract liability terms',
        case_context: 'employment-dispute',
        include_citations: true
      });

      this.addTestResult('GO_BINARY_INTEGRATION', true, Date.now() - startTime, undefined, {
        vectorServiceAvailable: goBinaryStatus.vectorService.available,
        cudaServiceAvailable: goBinaryStatus.cudaService.available,
        ragServiceAvailable: goBinaryStatus.enhancedRAG.available,
        vectorServiceResponseTime: vectorServiceResult.responseTime,
        ragServiceResponseTime: ragServiceResult.responseTime,
        cudaAcceleration: cudaServiceResult !== null
      });

    } catch (error) {
      this.addTestResult('GO_BINARY_INTEGRATION', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async testGPUAcceleration(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      // Test GPU availability
      const gpuStatus = await this.bvectorStore.getGPUStatus();

      // Large batch processing test
      const largeBatch = Array.from({ length: 100 }, (_, i) => 
        `Test document ${i} with legal content about contract law and employment disputes.`
      );

      // Test GPU-accelerated embedding
      const gpuEmbeddingStart = Date.now();
      const gpuResults = await this.bvectorStore.batchEmbedGPU(largeBatch);
      const gpuDuration = Date.now() - gpuEmbeddingStart;

      // Test CPU fallback (for comparison)
      const cpuEmbeddingStart = Date.now();
      const cpuResults = await this.bvectorStore.batchEmbedCPU(largeBatch.slice(0, 10)); // Smaller batch for CPU
      const cpuDuration = Date.now() - cpuEmbeddingStart;

      // Calculate performance ratio
      const performanceRatio = (cpuDuration / 10) / (gpuDuration / 100); // Normalize for batch size

      // Test GPU memory management
      const memoryStats = await this.bvectorStore.getGPUMemoryStats();

      this.addTestResult('GPU_ACCELERATION', true, Date.now() - startTime, undefined, {
        gpuAvailable: gpuStatus.available,
        gpuModelName: gpuStatus.deviceName,
        gpuMemoryGB: Math.round(gpuStatus.totalMemoryMB / 1024 * 100) / 100,
        largeBatchSize: largeBatch.length,
        gpuProcessingTime: gpuDuration,
        cpuProcessingTime: cpuDuration,
        performanceSpeedup: Math.round(performanceRatio * 100) / 100,
        memoryUsagePercent: memoryStats.usagePercent
      });

    } catch (error) {
      this.addTestResult('GPU_ACCELERATION', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async runPerformanceBenchmarks(): Promise<void> {
    const startTime = Date.now();

    try {
      if (!this.bvectorStore) {
        throw new Error('BVector store not initialized');
      }

      const benchmarks = {
        singleSearch: 0,
        batchSearch: 0,
        contextualSearch: 0,
        rlSearch: 0,
        cacheHitSearch: 0
      };

      // Single search benchmark
      const singleSearchStart = Date.now();
      await this.bvectorStore.search('contract terms legal analysis', { topK: 10 });
      benchmarks.singleSearch = Date.now() - singleSearchStart;

      // Batch search benchmark  
      const batchQueries = [
        'employment law disputes',
        'evidence chain custody',
        'contract breach remedies',
        'digital forensics standards',
        'criminal procedure rules'
      ];

      const batchSearchStart = Date.now();
      await Promise.all(batchQueries.map(query => 
        this.bvectorStore!.search(query, { topK: 5 })
      ));
      benchmarks.batchSearch = Date.now() - batchSearchStart;

      // Contextual search benchmark
      const contextualSearchStart = Date.now();
      await this.bvectorStore.contextualSearch('liability assessment', {
        topK: 8,
        userContext: {
          userId: 'benchmark-user',
          userRole: 'prosecutor',
          currentCase: 'benchmark-case'
        }
      });
      benchmarks.contextualSearch = Date.now() - contextualSearchStart;

      // RL-enhanced search benchmark
      const rlSearchStart = Date.now();
      await this.bvectorStore.searchWithRL('evidence analysis procedures', {
        userId: 'rl-benchmark-user',
        topK: 6,
        usePersonalization: true
      });
      benchmarks.rlSearch = Date.now() - rlSearchStart;

      // Cache hit benchmark (search same query again)
      const cacheHitStart = Date.now();
      await this.bvectorStore.search('contract terms legal analysis', { topK: 10 });
      benchmarks.cacheHitSearch = Date.now() - cacheHitStart;

      this.addTestResult('PERFORMANCE_BENCHMARKS', true, Date.now() - startTime, undefined, benchmarks);

    } catch (error) {
      this.addTestResult('PERFORMANCE_BENCHMARKS', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async cleanup(): Promise<void> {
    const startTime = Date.now();

    try {
      // Cleanup embedding worker
      if (this.embeddingWorker) {
        this.embeddingWorker.terminate();
      }

      // Cleanup BVector store
      if (this.bvectorStore) {
        await this.bvectorStore.cleanup();
      }

      this.addTestResult('CLEANUP', true, Date.now() - startTime);

    } catch (error) {
      this.addTestResult('CLEANUP', false, Date.now() - startTime, error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private addTestResult(testName: string, success: boolean, duration: number, error?: string, details?: Record<string, any>): void {
    this.testResults.push({
      testName,
      success,
      duration,
      error,
      details
    });

    if (this.config.verbose) {
      const status = success ? '✅ PASS' : '❌ FAIL';
      const time = `${duration}ms`;
      console.log(`${status} ${testName} (${time})`);
      
      if (error) {
        console.log(`   Error: ${error}`);
      }
      
      if (details && this.config.verbose) {
        console.log(`   Details:`, details);
      }
    }
  }

  private generateTestSummary(totalDuration: number) {
    const passed = this.testResults.filter(result => result.success).length;
    const failed = this.testResults.filter(result => !result.success).length;
    const skipped = 0; // TODO: Implement skip functionality

    return {
      totalTests: this.testResults.length,
      passed,
      failed,
      skipped,
      totalDuration
    };
  }

  private printTestResults(summary: ReturnType<typeof this.generateTestSummary>): void {
    console.log('\n' + '='.repeat(60));
    console.log('📊 BVector Integration Test Results');
    console.log('='.repeat(60));
    console.log(`Total Tests: ${summary.totalTests}`);
    console.log(`✅ Passed: ${summary.passed}`);
    console.log(`❌ Failed: ${summary.failed}`);
    console.log(`⏭️ Skipped: ${summary.skipped}`);
    console.log(`⏱️ Duration: ${summary.totalDuration}ms`);
    console.log(`🎯 Success Rate: ${Math.round((summary.passed / summary.totalTests) * 100)}%`);
    
    if (summary.failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.testResults.filter(result => !result.success).forEach(result => {
        console.log(`   • ${result.testName}: ${result.error}`);
      });
    }

    console.log('\n📈 Performance Summary:');
    const perfResults = this.testResults.find(result => result.testName === 'PERFORMANCE_BENCHMARKS');
    if (perfResults?.details) {
      Object.entries(perfResults.details).forEach(([key, value]) => {
        console.log(`   • ${key}: ${value}ms`);
      });
    }

    console.log('='.repeat(60));
  }
}

// Export test utilities
export async function runBVectorIntegrationTest(config: TestConfig = {}): Promise<boolean> {
  const testSuite = new BVectorIntegrationTestSuite(config);
  const results = await testSuite.runFullIntegrationTest();
  return results.success;
}

export { TestConfig, TestResult };