import type { RequestHandler } from './$types.js';

// AI Synthesizer API Route - Full Stack Integration
// Uses Neo4j, PostgreSQL/pgvector, XState, Redis, Ollama with gemma3:legal-latest
// TypeScript-safe with Drizzle ORM and MCP Context7 best practices

import { aiOrchestrator } from "$lib/server/ai/enhanced-ai-synthesis-orchestrator";
import { monitoringService } from "$lib/server/ai/monitoring-service";
import stream from "stream";
import { URL } from "url";

// SSE stream storage for real-time updates
const activeStreams = new Map<string, any>();

// Main synthesis endpoint
export const POST: RequestHandler = async ({ request, url }) => {
  const startTime = Date.now();
  let requestId: string | undefined;

  try {
    // Parse request body
    const body = await request.json();
    const { query, context, options = {} } = body;

    // Validate input
    if (!query || typeof query !== 'string') {
      throw error(400, 'Query is required and must be a string');
    }

    // Generate request ID for tracking
    requestId = `req_${Date.now()}_${Math.random().toString(36).substring(7)}`;

    logger.info(`[API] Processing synthesis request ${requestId}: "${query}"`);

    // Check if streaming is requested
    if (options.stream) {
      // Create stream ID for SSE
      const streamId = `stream_${requestId}`;

      // Initialize stream tracking
      activeStreams.set(streamId, {
        query,
        startTime,
        status: 'initializing',
      });

      // Start async processing
      processStreamingRequest(streamId, query, context, options);

      // Return stream ID immediately
      return json({
        success: true,
        streamId,
        message: 'Streaming synthesis initiated',
        streamUrl: `/api/ai-synthesizer/stream/${streamId}`,
      });
    }

    // Non-streaming request - process synchronously
    const result = await aiOrchestrator.process(query, {
      ...options,
      context,
      requestId,
    });

    // Track metrics
    const processingTime = Date.now() - startTime;
    await monitoringService.recordMetric('api_request_duration', processingTime);
    await monitoringService.recordMetric('api_requests_total', 1);

    // Return successful result
    return json({
      success: true,
      requestId,
      result: {
        synthesis: result.synthesis,
        sources: result.sources,
        confidence: result.confidence,
        metadata: {
          ...result.metadata,
          requestId,
          processingTime,
        },
      },
    });
  } catch (err: any) {
    // Log error
    logger.error('[API] Synthesis error:', err);

    // Track error metrics
    await monitoringService.recordMetric('api_errors_total', 1);

    // Return error response
    return json(
      {
        success: false,
        error: err.message || 'An error occurred during synthesis',
        requestId,
        processingTime: Date.now() - startTime,
      },
      { status: err.status || 500 }
    );
  }
};

// Health check endpoint
export const GET: RequestHandler = async ({ url }) => {
  try {
    // Get orchestrator health
    const health = await aiOrchestrator.health();

    // Get cache stats
    const cacheStats = await cachingLayer.getStats();

    // Get monitoring metrics
    const metrics = await monitoringService.getMetrics();

    // Compile comprehensive health status
    const status = {
      status: health.status,
      timestamp: new Date().toISOString(),
      version: '5.0.0',
      stack: {
        neo4j: health.services.neo4j || 'unknown',
        postgres: health.services.postgres || 'unknown',
        redis: health.services.redis || 'unknown',
        ollama: health.services.ollama || 'unknown',
        enhancedRAG: health.services.enhancedRAG || 'unknown',
        gpuOrchestrator: health.services.gpuOrchestrator || 'unknown',
        context7: health.services.context7 || 'unknown',
      },
      models: {
        primary: 'gemma3:legal-latest',
        embeddings: 'nomic-embed-text',
        fallback: 'gemma2:2b',
      },
      cache: {
        hits: cacheStats.hits,
        misses: cacheStats.misses,
        hitRate: cacheStats.hitRate,
        memoryUsage: cacheStats.memoryUsage,
      },
      monitoring: {
        totalRequests:
          (metrics.find((m: any) => m?.name === 'api_requests_total') as any)?.value ?? 0,
        totalErrors: (metrics.find((m: any) => m?.name === 'api_errors_total') as any)?.value ?? 0,
        avgResponseTime:
          (metrics.find((m: any) => m?.name === 'api_request_duration_avg') as any)?.value ?? 0,
        uptime: process.uptime(),
      },
      features: {
        neo4j: health.services.neo4j === 'healthy',
        pgvector: health.services.postgres === 'healthy',
        redis: health.services.redis === 'healthy',
        ollama: health.services.ollama === 'healthy',
        xstate: true,
        langchain: true,
        legalbert: true,
        drizzle: true,
        autosolve: true,
        streaming: true,
        caching: true,
        monitoring: true,
      },
    };

    // Determine overall health
    const healthyServices = Object.values(health.services).filter((s) => s === 'healthy').length;
    const totalServices = Object.keys(health.services).length;

    if (healthyServices === totalServices) {
      status.status = 'healthy';
    } else if (healthyServices >= totalServices * 0.5) {
      status.status = 'degraded';
    } else {
      status.status = 'unhealthy';
    }

    return json(status);
  } catch (err: any) {
    logger.error('[API] Health check error:', err);

    return json(
      {
        status: 'error',
        error: err.message,
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
};

// Test endpoint for integration testing (consolidated with health check)
export const GET_ALTERNATIVE: RequestHandler = async ({ url }) => {
  if (url.pathname.endsWith('/test')) {
    try {
      logger.info('[API] Running integration test...');

      // Test queries following MCP best practices
      const testQueries = [
        {
          query: 'What are the elements of negligence in tort law?',
          expectedSources: ['neo4j', 'pgvector', 'context7'],
        },
        {
          query: 'Explain the difference between void and voidable contracts',
          expectedSources: ['rag', 'ollama'],
        },
        {
          query: 'What is the statute of limitations for breach of contract?',
          expectedSources: ['neo4j', 'context7', 'ollama'],
        },
      ];

      const results = [];

      for (const test of testQueries) {
        const startTime = Date.now();

        try {
          const result = await aiOrchestrator.process(test.query, {
            test: true,
            timeout: 10000,
          });

          results.push({
            query: test.query,
            success: true,
            processingTime: Date.now() - startTime,
            confidence: result.confidence,
            sourcesUsed: (result.metadata as any)?.sourcesUsed || [],
            expectedSources: test.expectedSources,
          });
        } catch (err: any) {
          results.push({
            query: test.query,
            success: false,
            error: err.message,
            processingTime: Date.now() - startTime,
          });
        }
      }

      // Calculate test metrics
      const successCount = results.filter((r) => r.success).length;
      const avgProcessingTime =
        results.reduce((sum, r) => sum + r.processingTime, 0) / results.length;

      return json({
        success: successCount === results.length,
        testsRun: results.length,
        testsPassed: successCount,
        avgProcessingTime: Math.round(avgProcessingTime),
        results,
        services: await aiOrchestrator.health(),
        timestamp: new Date().toISOString(),
      });
    } catch (err: any) {
      logger.error('[API] Test error:', err);

      return json(
        {
          success: false,
          error: err.message,
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      );
    }
  }

  // Default GET returns health
  // Construct a minimal fake event for GET handler
  return GET({ url, request: new Request(url), params: {} } as any);
};

// Helper function for streaming requests
async function processStreamingRequest(
  streamId: string,
  query: string,
  context: any,
  options: any
): Promise<void> {
  try {
    // Update stream status
    const stream = activeStreams.get(streamId);
    if (stream) {
      stream.status = 'processing';
    }

    // Process with streaming
    const generator = aiOrchestrator.processStream(query, {
      ...options,
      context,
      streamId,
    });

    // Collect stream updates
    const updates = [];

    for await (const update of generator) {
      updates.push(update);

      // Update stream state
      if (stream) {
        stream.lastUpdate = update;
        stream.updates = updates;
      }
    }

    // Mark as complete
    if (stream) {
      stream.status = 'complete';
      stream.result = updates[updates.length - 1]?.result;
    }
  } catch (err: any) {
    logger.error(`[API] Streaming error for ${streamId}:`, err);

    const stream = activeStreams.get(streamId);
    if (stream) {
      stream.status = 'error';
      stream.error = err.message;
    }
  }
}

// Cleanup old streams periodically
setInterval(() => {
  const now = Date.now();
  const maxAge = 5 * 60 * 1000; // 5 minutes

  for (const [streamId, stream] of activeStreams.entries()) {
    if (now - stream.startTime > maxAge) {
      activeStreams.delete(streamId);
      logger.debug(`[API] Cleaned up old stream ${streamId}`);
    }
  }
}, 60000); // Check every minute

// Export for testing
export { activeStreams };
