import type { RequestHandler } from '@sveltejs/kit';
import { json, type RequestHandler } from '@sveltejs/kit';
import { z } from 'zod';
import { enhancedRAGService } from '$lib/services/enhanced-rag-integration.js';
import { dev } from '$app/environment';

const QuerySchema = z.object({
  query: z.string().min(1).max(2000),
  options: z.object({
    maxResults: z.number().min(1).max(50).optional().default(10),
    includeGraph: z.boolean().optional().default(true),
    streamResponse: z.boolean().optional().default(false),
    confidenceThreshold: z.number().min(0).max(1).optional().default(0.7)
  }).optional().default({})
});

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const { query, options } = QuerySchema.parse(body);

    if (dev) {
      console.log(`🧠 Enhanced RAG Query: "${query.substring(0, 100)}..."`);
    }

    // Process the query through the Enhanced RAG ML Pipeline
    const startTime = Date.now();
    const response = await enhancedRAGService.processLegalQuery(query, options);
    const processingTime = Date.now() - startTime;

    if (dev) {
      console.log(`⚡ Enhanced RAG processed in ${processingTime}ms`);
      console.log(`🎯 Confidence: ${response.confidence}`);
      console.log(`📊 Sources: ${response.sources?.length || 0}`);
    }

    return json({
      success: true,
      response: response.response,
      confidence: response.confidence,
      sources: response.sources,
      mlClassification: response.mlClassification,
      graphRelationships: response.graphRelationships,
      processingTime,
      metadata: {
        timestamp: new Date().toISOString(),
        queryId: response.queryId,
        systemVersion: '2.0.0-enhanced-rag'
      }
    });

  } catch (error: any) {
    console.error('❌ Enhanced RAG Query Error:', error);
    
    return json({
      success: false,
      error: error.message || 'Enhanced RAG processing failed',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};