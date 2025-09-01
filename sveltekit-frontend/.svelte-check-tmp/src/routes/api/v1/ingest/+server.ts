import { json } from '@sveltejs/kit';
import { INGEST_SERVICE_URL } from '$env/static/private';
import type { RequestHandler } from './$types';


/**
 * SvelteKit API proxy to Go Ingest Service (port 8227)
 * Integrates with your 37-service architecture
 * Follows your established patterns from Enhanced RAG service
 */

const SERVICE_URL = INGEST_SERVICE_URL || 'http://localhost:8227';
const TIMEOUT = 30000; // 30 seconds for document processing

export interface DocumentIngestRequest {
  title: string;
  content: string;
  case_id?: string;
  metadata?: Record<string, any>;
}

export interface BatchIngestRequest {
  documents: DocumentIngestRequest[];
}

export interface IngestResponse {
  id: string;
  status: string;
  document_id: string;
  embedding_id: string;
  process_time_ms: number;
  timestamp: string;
}

// Single document ingestion
export const POST: RequestHandler = async ({ request, fetch }) => {
  try {
    const requestData = await request.json();
    
    // Validate request structure
    if (!requestData.title || !requestData.content) {
      return json(
        { error: 'Missing required fields: title and content are required' },
        { status: 400 }
      );
    }

    // Transform to Go service format
    const ingestRequest: DocumentIngestRequest = {
      title: requestData.title,
      content: requestData.content,
      case_id: requestData.case_id || requestData.caseId, // Support both formats
      metadata: {
        ...requestData.metadata,
        // Add SvelteKit-specific metadata
        source: 'sveltekit-frontend',
        api_version: 'v1',
        timestamp: new Date().toISOString(),
        // Inherit from your established patterns
        user_agent: request.headers.get('user-agent') || 'unknown',
        ip_address: request.headers.get('x-forwarded-for') || 'unknown'
      }
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);

    try {
      // Call Go ingest service using SvelteKit's enhanced fetch
      const response = await fetch(`${SERVICE_URL}/api/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ingestRequest),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        return json(
          { 
            error: `Ingest service error: ${response.status} - ${errorText}`,
            service: 'ingest-service',
            port: '8227'
          },
          { status: response.status }
        );
      }

      const result: IngestResponse = await response.json();

      // Enhanced response with SvelteKit metadata
      return json({
        ...result,
        service_info: {
          go_service: 'ingest-service',
          port: '8227',
          proxy: 'sveltekit-api',
          architecture: 'multi-protocol'
        },
        // Follow your established success pattern
        success: true,
        api_version: 'v1'
      });

    } catch (fetchError) {
      clearTimeout(timeoutId);
      
      if (fetchError.name === 'AbortError') {
        return json(
          { error: 'Request timeout - document processing took too long' },
          { status: 504 }
        );
      }
      
      throw fetchError;
    }

  } catch (error: any) {
    console.error('Ingest API error:', error);
    
    return json(
      {
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'Unknown error',
        service: 'sveltekit-ingest-proxy'
      },
      { status: 500 }
    );
  }
};

// Health check endpoint
export const GET: RequestHandler = async ({ fetch }) => {
  try {
    const response = await fetch(`${SERVICE_URL}/api/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      return json(
        {
          status: 'unhealthy',
          service: 'ingest-service',
          port: '8227',
          error: `Service unreachable: ${response.status}`
        },
        { status: 503 }
      );
    }

    const health = await response.json();
    
    return json({
      status: 'healthy',
      service: 'ingest-service',
      port: '8227',
      proxy: 'sveltekit-api',
      upstream: health,
      // Follow your health check pattern
      timestamp: new Date().toISOString(),
      architecture: 'go-microservice'
    });

  } catch (error: any) {
    return json(
      {
        status: 'error',
        service: 'ingest-service',
        error: error instanceof Error ? error.message : 'Connection failed'
      },
      { status: 503 }
    );
  }
};