import type { RequestHandler } from './$types';

// Production Vector Search Testing API
// Tests vector ranking, reranking, and RAG pipeline integration
import { db } from "$lib/server/db/index";
import { vectorRankingService } from "$lib/services/vectorRankingService";
import { enhancedRAGPipeline } from "$lib/services/enhancedRAGPipeline";

// Logging
const logger = {
  info: (msg: string, data?: unknown) => console.log(`[VECTOR-TEST] ${new Date().toISOString()} - ${msg}`, data || ''),
  error: (msg: string, error?: unknown) => console.error(`[VECTOR-TEST] ${new Date().toISOString()} - ${msg}`, error || '')
};

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  logger.info('Vector search test initiated');

  try {
    const { query, testType = 'all' } = await request.json();

    if (!query || typeof query !== 'string') {
      return json({ success: false, error: 'Query parameter required' }, { status: 400 });
    }

    const results: any = {
      query,
      testType,
      timestamp: new Date().toISOString(),
      tests: {}
    };

    // Test 1: Vector Ranking Service
    if (testType === 'all' || testType === 'ranking') {
      try {
        logger.info('Testing vector ranking service');
        const rankingResults = await vectorRankingService.rankedSearch(query, {
          limit: 5,
          documentType: 'document',
          includeExplanation: true
        });

        results.tests.vectorRanking = {
          success: true,
          resultsCount: rankingResults.length,
          firstResult: rankingResults[0] || null,
          processingTime: Date.now() - startTime
        };
        logger.info(`Vector ranking: ${rankingResults.length} results found`);

      } catch (error: any) {
        logger.error('Vector ranking test failed', error);
        results.tests.vectorRanking = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    }

    // Test 2: Legal Analysis
    if (testType === 'all' || testType === 'analysis') {
      try {
        logger.info('Testing LegalBERT analysis');
        const analysis = await legalBERT.analyzeLegalText(query);

        results.tests.legalAnalysis = {
          success: true,
          entities: analysis.entities.length,
          concepts: analysis.concepts.length,
          sentiment: analysis.sentiment.classification,
          confidence: analysis.sentiment.confidence
        };
        logger.info(`Legal analysis: ${analysis.entities.length} entities, ${analysis.concepts.length} concepts`);

      } catch (error: any) {
        logger.error('Legal analysis test failed', error);
        results.tests.legalAnalysis = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    }

    // Test 3: Enhanced RAG Pipeline
    if (testType === 'all' || testType === 'rag') {
      try {
        logger.info('Testing Enhanced RAG Pipeline');
        const ragResult = await enhancedRAGPipeline.ragQuery(query, {
          useSemanticSearch: true,
          useMemoryGraph: true,
          useMultiAgent: true,
          maxSources: 5,
          minConfidence: 0.7
        });

        results.tests.enhancedRAG = {
          success: true,
          response: (ragResult as any)?.response ? String((ragResult as any).response).substring(0,200) + '...' : '',
          sources: ragResult.sources?.length || 0,
          confidence: ragResult.confidence,
          reasoning: ragResult.reasoning
        };
        logger.info(`Enhanced RAG: ${ragResult.sources?.length || 0} sources, confidence ${ragResult.confidence}`);

      } catch (error: any) {
        logger.error('Enhanced RAG test failed', error);
        results.tests.enhancedRAG = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    }

    // Test 4: LangChain RAG
    if (testType === 'all' || testType === 'langchain') {
      try {
        logger.info('Testing LangChain RAG');
        const langchainResult = await legalRAG.query(query, {
          thinkingMode: true,
          maxRetrievedDocs: 5,
          useCompression: true
        });

        results.tests.langchainRAG = {
          success: true,
          answer: langchainResult.answer?.substring(0, 200) + '...',
          sourceDocuments: langchainResult.sourceDocuments.length,
          confidence: langchainResult.confidence,
          processingTime: langchainResult.metadata.processingTime
        };
        logger.info(`LangChain RAG: ${langchainResult.sourceDocuments.length} sources, confidence ${langchainResult.confidence}`);

      } catch (error: any) {
        logger.error('LangChain RAG test failed', error);
        results.tests.langchainRAG = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    }

    // Test 5: Qdrant Direct Search
    if (testType === 'all' || testType === 'qdrant') {
      try {
        logger.info('Testing Qdrant direct search');
        const embeddingResult = await legalBERT.generateLegalEmbedding(query);
        const qdrantResultsRaw = await qdrantService.searchSimilar(embeddingResult.embedding, {
          topK: 5
        });
        // Normalize qdrant results to expected shape
        const qdrantResults = (qdrantResultsRaw as any[] || []).map(r => ({ id: r.id, score: r.score || 0, payload: r.payload || {} }));

        results.tests.qdrantSearch = {
          success: true,
          resultsCount: qdrantResults.length,
          averageScore: qdrantResults.length > 0 ?
            qdrantResults.reduce((sum, r) => sum + (r.score || 0), 0) / qdrantResults.length : 0,
          embeddingDimensions: embeddingResult.dimensions
        };
        logger.info(`Qdrant search: ${qdrantResults.length} results found`);

      } catch (error: any) {
        logger.error('Qdrant search test failed', error);
        results.tests.qdrantSearch = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    }

    // Calculate overall success rate
    const testResults = Object.values(results.tests);
    const successCount = testResults.filter((test: any) => test.success).length;
    const totalTests = testResults.length;

    results.summary = {
      successRate: totalTests > 0 ? (successCount / totalTests) * 100 : 0,
      passedTests: successCount,
      totalTests,
      overallProcessingTime: Date.now() - startTime,
      status: successCount === totalTests ? 'all_passed' : successCount > 0 ? 'partial_success' : 'all_failed'
    };

    logger.info(`Vector search tests completed: ${successCount}/${totalTests} passed`);
    return json(results);

  } catch (error: any) {
    logger.error('Vector search test failed', error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      processingTime: Date.now() - startTime
    }, { status: 500 });
  }
};

// Health check for vector services
export const GET: RequestHandler = async () => {
  try {
    logger.info('Vector services health check');

    const health = {
      timestamp: new Date().toISOString(),
      services: {
        qdrant: await qdrantService.healthCheck().catch(() => ({ status: 'error' })),
        legalBERT: await legalBERT.healthCheck().catch(() => ({ status: 'error' })),
        enhancedRAG: { status: 'available' }, // Service is always available
        langchainRAG: await legalRAG.healthCheck().catch(() => ({ status: 'error' }))
      }
    };

    return json(health);

  } catch (error: any) {
    logger.error('Health check failed', error);
    return json({
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};