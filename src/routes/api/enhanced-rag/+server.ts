// @ts-nocheck
import { json } from '@sveltejs/kit';
import { db } from '$lib/services/unified-database-service.js';
import { aiService } from '$lib/services/unified-ai-service.js';
import { ragPipeline } from '$lib/services/enhanced-rag-pipeline.js';
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

async function handleDocumentIngestion(data: any) {
  const { documents, options = {} } = data;
  
  if (!documents || !Array.isArray(documents)) {
    return json(
      { error: 'Documents array is required', success: false },
      { status: 400 }
    );
  }
  
  try {
    const results = await ragPipeline.ingestDocuments(documents);
    
    return json({
      success: true,
      message: `Processed ${results.processed} documents`,
      results: {
        processed: results.processed,
        failed: results.failed,
        stored: results.stored,
        chunks: results.chunks,
        embeddings: results.embeddings
      }
    });
  } catch (error) {
    return json(
      { 
        error: 'Document ingestion failed', 
        message: error.message,
        success: false 
      },
      { status: 500 }
    );
  }
}

async function handleEnhancedQuery(data: any) {
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

async function handleStreamingQuery(data: any) {
  const { query, caseId, userId, options = {} } = data;
  
  if (!query) {
    return json(
      { error: 'Query is required', success: false },
      { status: 400 }
    );
  }
  
  try {
    // For streaming, we need to return a ReadableStream
    // This is a simplified version - full implementation would use Server-Sent Events
    const result = await ragPipeline.query(query, {
      caseId,
      userId,
      stream: true,
      ...options
    });
    
    return json({
      success: true,
      query,
      streamId: `stream_${Date.now()}`,
      message: 'Streaming response initiated',
      sources: result.sources
    });
  } catch (error) {
    return json(
      { 
        error: 'Streaming query failed', 
        message: error.message,
        success: false 
      },
      { status: 500 }
    );
  }
}

async function handleRecommendations(data: any) {
  const { userId, context = {}, limit = 10 } = data;
  
  if (!userId) {
    return json(
      { error: 'User ID is required for recommendations', success: false },
      { status: 400 }
    );
  }
  
  try {
    const recommendations = await ragPipeline.getRecommendations(userId, context);
    
    return json({
      success: true,
      userId,
      recommendations: recommendations.slice(0, limit),
      count: recommendations.length
    });
  } catch (error) {
    return json(
      { 
        error: 'Recommendations failed', 
        message: error.message,
        success: false 
      },
      { status: 500 }
    );
  }
}

async function handleDocumentAnalysis(data: any) {
  const { document, operations = ['analyze', 'summarize'] } = data;
  
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
  } catch (error) {
    return json(
      { 
        error: 'Document analysis failed', 
        message: error.message,
        success: false 
      },
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

async function handleHybridSearch(data: any) {
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

async function handleEmbedding(data: any) {
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