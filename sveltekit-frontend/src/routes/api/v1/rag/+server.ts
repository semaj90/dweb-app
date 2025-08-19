/**
 * RAG API Endpoint - Enhanced AI Processing
 * Routes to: enhanced-rag.exe:8094 (HTTP) or rag-quic-proxy.exe:8216 (QUIC)
 */

import { json, error } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler
import {
productionServiceClient, ServiceTier } from "$lib/services/productionServiceClient";

interface RAGRequest {
  query: string;
  context?: string[];
  userId?: string;
  caseId?: string;
  options?: {
    useQUIC?: boolean;
    temperature?: number;
    maxTokens?: number;
  };
}

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const data: RAGRequest = await request.json();
    
    if (!data.query) {
      return error(400, 'Query is required');
    }

    // Route to appropriate service based on options
    const operation = 'rag.query';
    const serviceOptions = {
      timeout: 30000,
      forceTier: data.options?.useQUIC ? ServiceTier.ULTRA_FAST : undefined
    };

    const result = await productionServiceClient.execute(
      operation,
      {
        query: data.query,
        context: data.context || [],
        userId: data.userId,
        caseId: data.caseId,
        temperature: data.options?.temperature || 0.7,
        maxTokens: data.options?.maxTokens || 1000
      },
      serviceOptions
    );

    return json({
      success: true,
      data: result,
      metadata: {
        timestamp: new Date().toISOString(),
        service: data.options?.useQUIC ? 'rag-quic-proxy' : 'enhanced-rag',
        operation: 'query'
      }
    });

  } catch (err) {
    console.error('RAG API Error:', err);
    return error(500, `RAG service unavailable: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  // Health check and service status
  try {
    const health = await productionServiceClient.checkAllServicesHealth();
    
    return json({
      service: 'rag',
      status: 'operational',
      endpoints: {
        query: '/api/v1/rag',
        semantic: '/api/v1/rag/semantic',
        embed: '/api/v1/rag/embed'
      },
      health: {
        'enhanced-rag': health['enhanced-rag'] || false,
        'rag-quic-proxy': health['rag-quic-proxy'] || false
      },
      protocols: ['HTTP', 'QUIC'],
      version: '1.0.0'
    });
  } catch (err) {
    return error(503, { message: 'RAG service health check failed' });
  }
};