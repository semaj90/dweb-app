import type { RequestHandler } from './$types.js';

/**
 * AI API Endpoint - Comprehensive AI Services
 * Routes to: ai-enhanced.exe:8096, enhanced-legal-ai.exe:8202, live-agent-enhanced.exe:8200
 */


import { ensureError } from '$lib/utils/ensure-error';
import { productionServiceClient } from "$lib/services/productionServiceClient";
import { URL } from "url";

export interface AIRequest {
  type: 'summary' | 'legal' | 'live' | 'analysis';
  content?: string;
  document?: unknown;
  sessionId?: string;
  userId: string;
  options?: {
    model?: string;
    temperature?: number;
    streaming?: boolean;
  };
}

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const data: AIRequest = await request.json();
    
    if (!data.userId) {
      return error(400, ensureError({ message: 'User ID is required' }));
    }

    let operation: string;
    let serviceData: any;

    switch (data.type) {
      case 'summary':
        if (!data.content) {
          return error(400, ensureError({ message: 'Content is required for summary' }));
        }
        operation = 'ai.summary';
        serviceData = {
          content: data.content,
          userId: data.userId,
          options: data.options
        };
        break;

      case 'legal':
        if (!data.document) {
          return error(400, ensureError({ message: 'Document is required for legal analysis' }));
        }
        operation = 'legal.process';
        serviceData = {
          document: data.document,
          userId: data.userId,
          options: data.options
        };
        break;

      case 'live':
        if (!data.sessionId) {
          return error(400, ensureError({ message: 'Session ID is required for live AI' }));
        }
        operation = 'ai.live';
        serviceData = {
          sessionId: data.sessionId,
          userId: data.userId,
          content: data.content,
          streaming: data.options?.streaming || false
        };
        break;

      case 'analysis':
        operation = 'ai.analysis';
        serviceData = {
          content: data.content || data.document,
          userId: data.userId,
          type: 'general_analysis',
          options: data.options
        };
        break;

      default:
        return error(400, ensureError({ message: 'Invalid AI operation type' }));
    }

    const result = await productionServiceClient.execute(operation, serviceData, {
      timeout: data.options?.streaming ? 60000 : 30000
    });

    return json({
      success: true,
      data: result,
      metadata: {
        timestamp: new Date().toISOString(),
        service: getServiceName(data.type),
        operation: data.type,
        userId: data.userId
      }
    });

  } catch (err: any) {
    console.error('AI API Error:', err);
    return error(500, `AI service unavailable: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const sessionId = url.searchParams.get('sessionId');
  
  if (sessionId) {
    // Get session status for live AI
    try {
      const result = await productionServiceClient.execute('ai.session.status', { sessionId });
      return json({ success: true, data: result });
    } catch (err: any) {
      return error(404, ensureError({ message: 'Session not found' }));
    }
  }

  // Service health check and capabilities
  try {
    const health = await productionServiceClient.checkAllServicesHealth();
    
    return json({
      service: 'ai',
      status: 'operational',
      endpoints: {
        summary: '/api/v1/ai (type: summary)',
        legal: '/api/v1/ai (type: legal)',
        live: '/api/v1/ai (type: live)',
        analysis: '/api/v1/ai (type: analysis)'
      },
      health: {
        'ai-enhanced': health['ai-summary'] || false,
        'legal-ai': health['legal-ai'] || false,
        'live-agent': health['live-agent'] || false
      },
      capabilities: [
        'Document Summarization',
        'Legal Document Analysis',
        'Real-time AI Conversations',
        'Content Analysis',
        'Multi-modal Processing'
      ],
      supportedModels: [
        'gemma3-legal',
        'nomic-embed-text',
        'deeds-web'
      ],
      version: '1.0.0'
    });
  } catch (err: any) {
    return error(503, ensureError({ message: 'AI service health check failed' }));
  }
};

function getServiceName(type: string): string {
  switch (type) {
    case 'summary': return 'ai-enhanced';
    case 'legal': return 'enhanced-legal-ai';
    case 'live': return 'live-agent-enhanced';
    case 'analysis': return 'ai-enhanced';
    default: return 'unknown';
  }
}