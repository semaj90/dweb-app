
import { json } from '@sveltejs/kit';
import { db } from '$lib/services/unified-database-service.js';
import { aiService } from '$lib/services/unified-ai-service.js';
import { ragPipeline } from '$lib/services/enhanced-rag-pipeline.js';
import { documentIngestionService } from '$lib/services/document-ingestion-service';
// Lightweight worker spawn (defer full pool integration)
import { Worker } from 'worker_threads';
import path from 'path';
import type { RequestHandler } from './$types';

// Initialize services on startup
let initialized = false;

async function initializeServices() {
  if (initialized) return true;

  try {
    console.log('Initializing Enhanced RAG System...');

    await db.initialize();
    await aiService.initialize();
    await ragPipeline.initialize();

    initialized = true;
    console.log('✅ Enhanced RAG System fully initialized');
    return true;
  } catch (error) {
    console.error('❌ Failed to initialize Enhanced RAG System:', error);
    return false;
  }
}

// Initialize on module load
initializeServices();

/**
 * Enhanced RAG API Endpoint
 * Handles document ingestion, querying, recommendations, and system health
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, data } = await request.json();

    // Ensure services are initialized
    if (!initialized) {
      const initResult = await initializeServices();
      if (!initResult) {
        return json(
          { error: 'System initialization failed', success: false },
          { status: 503 }
        );
      }
    }

    switch (action) {
      case 'ingest':
        return await handleDocumentIngestion(data);

      case 'query':
        return await handleEnhancedQuery(data);

      case 'stream':
        return await handleStreamingQuery(data);

      case 'recommend':
        return await handleRecommendations(data);

      case 'analyze':
        return await handleDocumentAnalysis(data);

      case 'health':
        return await handleHealthCheck();

      case 'search':
        return await handleHybridSearch(data);

      case 'embed':
        return await handleEmbedding(data);

      default:
        return json(
          { error: `Unknown action: ${action}`, success: false },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Enhanced RAG API error:', error);
    return json(
      {
        error: 'Internal server error',
        message: error.message,
        success: false
      },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get('q');
    const caseId = url.searchParams.get('caseId');
    const topK = parseInt(url.searchParams.get('topK') || '5');
    const userId = url.searchParams.get('userId');

    if (!query) {
      return json(
        { error: 'Query parameter "q" is required', success: false },
        { status: 400 }
      );
    }

    // Ensure services are initialized
    if (!initialized) {
      const initResult = await initializeServices();
      if (!initResult) {
        return json(
          { error: 'System initialization failed', success: false },
          { status: 503 }
        );
      }
    }

    // Quick query endpoint
    const result = await ragPipeline.query(query, {
      caseId,
      topK,
      userId,
      hybrid: true,
      diversityFiltering: true
    });

    return json({
      success: true,
      query,
      answer: result.answer,
      sources: result.sources,
      metadata: result.metadata
    });

  } catch (error) {
    console.error('Enhanced RAG GET error:', error);
    return json(
      {
        error: 'Query failed',
        message: error.message,
        success: false
      },
      { status: 500 }
    );
  }
};

// ============ Action Handlers ============

async function handleDocumentIngestion(data: unknown) {
  const { documents, options = {} } = data;
  if (!documents || !Array.isArray(documents)) {
    return json({ error: 'Documents array is required', success: false }, { status: 400 });
  }

  try {
    const enriched: unknown[] = [];

    for (const doc of documents) {
      // If raw path to PDF provided, parse + chunk
      if (doc?.filePath && typeof doc.filePath === 'string' && doc.filePath.endsWith('.pdf')) {
        try {
          const parsed = await documentIngestionService.parsePDF(doc.filePath, { maxPages: options.maxPages });
          enriched.push({
            id: parsed.id,
            title: parsed.title,
            content: parsed.content,
            metadata: { ...parsed.metadata, sourceType: 'pdf', originalPath: doc.filePath },
            chunks: parsed.chunks,
          });
        } catch (e:any) {
          enriched.push({ error: true, filePath: doc.filePath, message: e.message });
        }
        continue;
      }

      // If pre-chunked structure supplied
      if (doc?.content && Array.isArray(doc?.chunks)) {
        enriched.push(doc);
        continue;
      }

      // Fallback: minimal normalization
      if (doc?.content) {
        enriched.push({ ...doc, chunks: [] });
      }
    }

    // Ingest via pipeline (pipeline should internally chunk missing pieces if supported)
    const results = await ragPipeline.ingestDocuments(enriched, options);

    // Optionally spawn background chunking worker for any documents lacking chunks
    const unchunked = enriched.filter(d => d.content && (!d.chunks || d.chunks.length === 0));
    if (unchunked.length && options.backgroundChunking) {
      spawnChunkingWorker(unchunked.map(d => ({ id: d.id, content: d.content, filename: d.title || d.id, metadata: d.metadata, type: 'text' })) );
    }

    return json({
      success: true,
      message: `Processed ${results.processed} documents (ingested: ${results.stored})`,
      results: {
        processed: results.processed,
        failed: results.failed,
        stored: results.stored,
        chunks: results.chunks,
        embeddings: results.embeddings,
        parsed: enriched.length,
      },
    });
  } catch (error:any) {
    return json({ error: 'Document ingestion failed', message: error.message, success: false }, { status: 500 });
  }
}

function spawnChunkingWorker(documents: unknown[]) {
  try {
    const workerPath = path.resolve('src/lib/workers/chunking-worker.js');
    const worker = new Worker(workerPath, { workerData: { workerId: `chunker-${Date.now()}` } });
    worker.postMessage({ taskId: 'chunk-batch', data: { documents } });
    worker.on('message', (msg) => {
      if (process.env.RAG_DEBUG === 'true') console.log('[ChunkWorker]', msg);
    });
    worker.on('error', (err) => console.warn('[ChunkWorker Error]', err.message));
  } catch (e:any) {
    console.warn('Failed to spawn chunking worker:', e.message);
  }
}

async function handleEnhancedQuery(data: unknown) {
  const {
    query,
    caseId,
    topK = 5,
    userId,
    options = {}
  } = data;

  if (!query) {
    return json(
      { error: 'Query is required', success: false },
      { status: 400 }
    );
  }

  try {
    const result = await ragPipeline.query(query, {
      caseId,
      topK,
      userId,
      ...options
    });

    return json({
      success: true,
      query,
      answer: result.answer,
      sources: result.sources,
      metadata: {
        ...result.metadata,
        cached: result.cached || false
      }
    });
  } catch (error) {
    return json(
      {
        error: 'Enhanced query failed',
        message: error.message,
        success: false
      },
      { status: 500 }
    );
  }
}

// (Removed legacy JSON streaming handler; replaced with SSE implementation below)

async function handleStreamingQuery(data: unknown) {
  const { query, caseId, userId, options = {}, enableSSE = true } = data;
  if (!query) return json({ error: 'Query is required', success: false }, { status: 400 });

  if (!enableSSE) {
    // Fallback to legacy JSON stub
    const result = await ragPipeline.query(query, { caseId, userId, stream: true, ...options });
    return json({ success: true, query, answer: result.answer, sources: result.sources, streamFallback: true });
  }

  const stream = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      const send = (obj: unknown) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));
      try {
        send({ type: 'status', message: 'Initializing streaming RAG query', progress: 0 });
        let partial = '';
        // Simulated incremental streaming (ragPipeline.query must be adapted for true token streaming)
        const result = await ragPipeline.query(query, { caseId, userId, stream: true, ...options });
        const text = result.answer || '';
        const chunkSize = 400;
        for (let i = 0; i < text.length; i += chunkSize) {
          const slice = text.slice(i, i + chunkSize);
            partial += slice;
            send({ type: 'chunk', content: slice, progress: Math.min(95, Math.round(((i + chunkSize) / text.length) * 100)) });
            await new Promise(r => setTimeout(r, 10));
        }
        send({ type: 'complete', answer: partial, sources: result.sources, progress: 100 });
        controller.close();
      } catch (e:any) {
        send({ type: 'error', error: e.message });
        controller.close();
      }
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    }
  });
}

async function handleRecommendations(data: unknown) {
  const { userId, context = {}, limit = 10 } = data || {};
  if (!userId) {
    return json(
      { error: 'User ID is required for recommendations', success: false },
      { status: 400 }
    );
  }
  try {
    const recommendations = await ragPipeline.getRecommendations(userId, context) || [];
    return json({
      success: true,
      userId,
      recommendations: recommendations.slice(0, limit),
      count: recommendations.length
    });
  } catch (error: unknown) {
    return json(
      { error: 'Recommendations failed', message: error.message, success: false },
      { status: 500 }
    );
  }
}

async function handleDocumentAnalysis(data: unknown) {
  const { document, operations = ['analyze', 'summarize'] } = data || {};
  if (!document) {
    return json(
      { error: 'Document is required', success: false },
      { status: 400 }
    );
  }
  try {
    const result = await aiService.processDocument(document, { operations });
    return json({
      success: true,
      document: {
        id: result.id,
        title: result.title,
        analysis: result.analysis,
        summary: result.summary,
        chunks: result.chunks?.length || 0,
        embeddings: result.embeddings?.length || 0
      }
    });
  } catch (error: unknown) {
    return json(
      { error: 'Document analysis failed', message: error.message, success: false },
      { status: 500 }
    );
  }
}

async function handleHealthCheck() {
  try {
    const [dbHealth, aiHealth, ragHealth] = await Promise.all([
      db.getHealthStatus(),
      aiService.getHealthStatus(),
      ragPipeline.getHealthStatus()
    ]);

    const overallHealthy = dbHealth.overall === 'healthy' &&
                          aiHealth.status === 'healthy' &&
                          ragHealth.initialized;

    return json({
      success: true,
      status: overallHealthy ? 'healthy' : 'degraded',
      components: {
        database: dbHealth,
        aiService: aiHealth,
        ragPipeline: ragHealth
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return json(
      {
        error: 'Health check failed',
        message: error.message,
        success: false,
        status: 'unhealthy'
      },
      { status: 500 }
    );
  }
}

async function handleHybridSearch(data: unknown) {
  const { query, caseId, limit = 10, filters = {} } = data;

  if (!query) {
    return json(
      { error: 'Query is required', success: false },
      { status: 400 }
    );
  }

  try {
    // Generate embedding for the query
    const queryEmbedding = await aiService.embedSingle(query);

    // Perform hybrid search
    const results = await db.hybridSearch(query, queryEmbedding, caseId);

    return json({
      success: true,
      query,
      results: results.slice(0, limit),
      total: results.length,
      searchType: 'hybrid'
    });
  } catch (error) {
    return json(
      {
        error: 'Hybrid search failed',
        message: error.message,
        success: false
      },
      { status: 500 }
    );
  }
}

async function handleEmbedding(data: unknown) {
  const { texts, model } = data;

  if (!texts || !Array.isArray(texts)) {
    return json(
      { error: 'Texts array is required', success: false },
      { status: 400 }
    );
  }

  try {
    const embeddings = await aiService.embed(texts);

    return json({
      success: true,
      embeddings,
      count: embeddings.length,
      dimensions: embeddings[0]?.length || 0,
      model: model || 'nomic-embed-text'
    });
  } catch (error) {
    return json(
      {
        error: 'Embedding generation failed',
        message: error.message,
        success: false
      },
      { status: 500 }
    );
  }
}