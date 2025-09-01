import type { RequestHandler } from './$types.js';

/**
 * QUIC AI Stream API - Real-time AI Streaming Service
 * Provides AI streaming with WebSocket + HTTP/3 support and session management
 * Port: 8447 (QUIC), 8448 (HTTP/2 fallback)
 * Backends: Ollama (11434), Enhanced RAG (8094)
 */
import { json, error } from '@sveltejs/kit';

import { ensureError } from '$lib/utils/ensure-error';
import crypto from "crypto";
import { URL } from "url";

const QUIC_AI_STREAM_CONFIG = {
  primaryPort: 8447,    // QUIC HTTP/3
  fallbackPort: 8448,   // HTTP/2
  baseUrl: 'http://localhost:8447',
  fallbackUrl: 'http://localhost:8448',
  wsUrl: 'ws://localhost:8447',
  timeout: 60000,       // AI operations can take longer
  maxTokens: 4096,
  defaultModel: 'gemma3-legal'
};

export interface AIStreamRequest {
  prompt: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
  sessionId?: string;
  context?: Record<string, any>;
}

export interface AIStreamResponse {
  sessionId: string;
  response?: string;
  streaming?: boolean;
  websocketUrl?: string;
  model: string;
  tokensUsed?: number;
  executionTime?: number;
}

/**
 * GET /api/v1/quic/ai-stream - AI stream service health and session status
 */
export const GET: RequestHandler = async ({ url }) => {
  try {
    const sessionId = url.searchParams.get('sessionId');
    
    // Check AI stream service health
    const healthResponse = await fetch(`${QUIC_AI_STREAM_CONFIG.baseUrl}/health`, {
      signal: AbortSignal.timeout(QUIC_AI_STREAM_CONFIG.timeout)
    });

    let serviceStatus = 'healthy';
    let responseData: any = {};

    if (healthResponse.ok) {
      responseData = await healthResponse.json();
    } else {
      // Try fallback HTTP/2
      const fallbackResponse = await fetch(`${QUIC_AI_STREAM_CONFIG.fallbackUrl}/health`, {
        signal: AbortSignal.timeout(QUIC_AI_STREAM_CONFIG.timeout)
      });

      if (fallbackResponse.ok) {
        responseData = await fallbackResponse.json();
        serviceStatus = 'fallback';
      } else {
        serviceStatus = 'unhealthy';
      }
    }

    // If sessionId provided, get session details
    let sessionInfo = null;
    if (sessionId && serviceStatus !== 'unhealthy') {
      try {
        const sessionUrl = serviceStatus === 'healthy' 
          ? `${QUIC_AI_STREAM_CONFIG.baseUrl}/session/${sessionId}`
          : `${QUIC_AI_STREAM_CONFIG.fallbackUrl}/session/${sessionId}`;
          
        const sessionResponse = await fetch(sessionUrl, {
          signal: AbortSignal.timeout(5000)
        });

        if (sessionResponse.ok) {
          sessionInfo = await sessionResponse.json();
        }
      } catch (sessionError) {
        console.warn('Failed to fetch session info:', sessionError);
      }
    }

    return json({
      service: 'quic-ai-stream',
      status: serviceStatus,
      protocol: serviceStatus === 'healthy' ? 'HTTP/3' : serviceStatus === 'fallback' ? 'HTTP/2' : 'N/A',
      ports: {
        quic: QUIC_AI_STREAM_CONFIG.primaryPort,
        fallback: QUIC_AI_STREAM_CONFIG.fallbackPort
      },
      websocketUrl: QUIC_AI_STREAM_CONFIG.wsUrl,
      backends: {
        ollama: 'http://localhost:11434',
        enhancedRAG: 'http://localhost:8094'
      },
      features: [
        'Real-time AI Streaming',
        'WebSocket Support',
        'Session Management',
        'Multiple AI Models',
        'HTTP/3 Acceleration'
      ],
      models: responseData.models || [
        'gemma3-legal',
        'nomic-embed-text',
        'llama2-legal'
      ],
      session: sessionInfo,
      metrics: responseData.metrics || null,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('QUIC AI Stream health check failed:', err);
    
    return json({
      service: 'quic-ai-stream',
      status: 'error',
      error: err instanceof Error ? err.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * POST /api/v1/quic/ai-stream - Start AI inference with streaming support
 */
export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const aiRequest: AIStreamRequest = await request.json();
    const useHttp3 = url.searchParams.get('http3') !== 'false';
    const enableStreaming = aiRequest.stream !== false;

    // Validate AI request
    if (!aiRequest.prompt || aiRequest.prompt.trim().length === 0) {
      error(400, ensureError({ message: 'Prompt is required and cannot be empty' }));
    }

    if (aiRequest.maxTokens && (aiRequest.maxTokens < 1 || aiRequest.maxTokens > 8192)) {
      error(400, ensureError({ message: 'Max tokens must be between 1 and 8192' }));
    }

    if (aiRequest.temperature && (aiRequest.temperature < 0 || aiRequest.temperature > 2)) {
      error(400, ensureError({ message: 'Temperature must be between 0 and 2' }));
    }

    // Generate session ID if not provided
    const sessionId = aiRequest.sessionId || crypto.randomUUID();

    // Determine target URL
    const targetUrl = useHttp3 
      ? `${QUIC_AI_STREAM_CONFIG.baseUrl}/api/ai/stream`
      : `${QUIC_AI_STREAM_CONFIG.fallbackUrl}/api/ai/stream`;

    // Prepare request payload
    const requestPayload = {
      prompt: aiRequest.prompt,
      model: aiRequest.model || QUIC_AI_STREAM_CONFIG.defaultModel,
      maxTokens: aiRequest.maxTokens || QUIC_AI_STREAM_CONFIG.maxTokens,
      temperature: aiRequest.temperature || 0.7,
      stream: enableStreaming,
      sessionId: sessionId,
      context: aiRequest.context || {},
      meta: {
        requestId: crypto.randomUUID(),
        timestamp: Date.now(),
        protocol: useHttp3 ? 'HTTP/3' : 'HTTP/2'
      }
    };

    let response: Response;
    let protocol: string;

    try {
      response = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId,
          'X-Enable-Streaming': String(enableStreaming),
          'X-QUIC-Request': 'true'
        },
        body: JSON.stringify(requestPayload),
        signal: AbortSignal.timeout(QUIC_AI_STREAM_CONFIG.timeout)
      });
      protocol = useHttp3 ? 'HTTP/3' : 'HTTP/2';

    } catch (quicError) {
      console.error('QUIC AI Stream service failed:', quicError);
      error(503, ensureError({
        message: 'AI streaming service unavailable',
        error: quicError instanceof Error ? quicError.message : 'Unknown error'
      }));
    }

    if (!response.ok) {
      const errorText = await response.text();
      error(response.status, {
        message: `AI stream service error: ${response.statusText}`,
        details: errorText
      });
    }

    const responseData = await response.json();

    const aiResponse: AIStreamResponse = {
      sessionId: sessionId,
      response: responseData.response,
      streaming: enableStreaming,
      websocketUrl: enableStreaming ? `${QUIC_AI_STREAM_CONFIG.wsUrl}/ws/${sessionId}` : undefined,
      model: responseData.model || requestPayload.model,
      tokensUsed: responseData.tokensUsed || 0,
      executionTime: responseData.executionTime || 0
    };

    return json({
      success: true,
      data: aiResponse,
      protocol,
      source: 'quic-ai-stream',
      timestamp: new Date().toISOString(),
      metrics: {
        sessionId: sessionId,
        promptLength: aiRequest.prompt.length,
        responseLength: aiResponse.response?.length || 0,
        executionTimeMs: aiResponse.executionTime || 0,
        streaming: enableStreaming
      }
    });

  } catch (err: any) {
    console.error('QUIC AI Stream error:', err);
    error(500, ensureError({
      message: 'AI streaming failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};

/**
 * DELETE /api/v1/quic/ai-stream - Terminate AI session
 */
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const sessionId = url.searchParams.get('sessionId');
    const useHttp3 = url.searchParams.get('http3') !== 'false';

    if (!sessionId) {
      error(400, ensureError({ message: 'Session ID is required' }));
    }

    const targetUrl = useHttp3 
      ? `${QUIC_AI_STREAM_CONFIG.baseUrl}/session/${sessionId}`
      : `${QUIC_AI_STREAM_CONFIG.fallbackUrl}/session/${sessionId}`;

    const response = await fetch(targetUrl, {
      method: 'DELETE',
      headers: {
        'X-Session-ID': sessionId,
        'X-QUIC-Request': 'true'
      },
      signal: AbortSignal.timeout(10000)
    });

    if (!response.ok) {
      throw new Error(`Session termination failed: ${response.statusText}`);
    }

    const result = await response.json();

    return json({
      success: true,
      message: `AI session '${sessionId}' terminated`,
      result,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('AI session termination error:', err);
    error(500, ensureError({
      message: 'Session termination failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};

/**
 * PUT /api/v1/quic/ai-stream - Update AI streaming configuration
 */
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const config = await request.json();

    // Validate configuration
    if (config.maxTokens && (config.maxTokens < 1 || config.maxTokens > 8192)) {
      error(400, ensureError({ message: 'Max tokens must be between 1 and 8192' }));
    }

    if (config.timeout && (config.timeout < 5000 || config.timeout > 300000)) {
      error(400, ensureError({ message: 'Timeout must be between 5000 and 300000ms' }));
    }

    // Update configuration (in a real implementation, this would be persisted)
    const updatedConfig = {
      ...QUIC_AI_STREAM_CONFIG,
      ...config,
      lastUpdated: new Date().toISOString()
    };

    return json({
      success: true,
      message: 'AI streaming configuration updated',
      config: updatedConfig
    });

  } catch (err: any) {
    console.error('AI stream configuration update failed:', err);
    error(500, ensureError({
      message: 'Configuration update failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    }));
  }
};