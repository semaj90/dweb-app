import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// import { INGEST_SERVICE_URL, MAX_BATCH_SIZE } from '$env/static/private';

/**
 * Batch Document Ingestion API
 * Handles multiple documents with enhanced error handling and progress tracking
 * Integrates with Go ingest service on port 8227
 */

const SERVICE_URL = 'http://localhost:8227'; // INGEST_SERVICE_URL ||
const TIMEOUT = 120000; // 2 minutes for batch processing
const BATCH_SIZE_LIMIT = parseInt('10'); // MAX_BATCH_SIZE ||

export interface BatchIngestRequest {
  documents: Array<{
    title: string;
    content: string;
    case_id?: string;
    metadata?: Record<string, any>;
  }>;
}

export interface BatchIngestResponse {
  results: Array<{
    id: string;
    status: string;
    document_id: string;
    embedding_id: string;
    process_time_ms: number;
    timestamp: string;
  }>;
  processed: number;
  total: number;
  timestamp: string;
  errors?: string[];
}

export const POST: RequestHandler = async ({ request, fetch }) => {
  const startTime = Date.now();

  try {
    const requestData: BatchIngestRequest = await request.json();

    // Validate request structure
    if (!requestData.documents || !Array.isArray(requestData.documents)) {
      return json(
        { error: 'Missing required field: documents array is required' },
        { status: 400 }
      );
    }

    // Validate batch size
    if (requestData.documents.length > BATCH_SIZE_LIMIT) {
      return json(
        {
          error: `Batch size ${requestData.documents.length} exceeds maximum of ${BATCH_SIZE_LIMIT}`,
          max_batch_size: BATCH_SIZE_LIMIT
        },
        { status: 400 }
      );
    }

    // Validate each document
    const validationErrors: string[] = [];
    requestData.documents.forEach((doc, index) => {
      if (!doc.title) {
        validationErrors.push(`Document ${index}: missing title`);
      }
      if (!doc.content) {
        validationErrors.push(`Document ${index}: missing content`);
      }
    });

    if (validationErrors.length > 0) {
      return json(
        {
          error: 'Document validation failed',
          validation_errors: validationErrors
        },
        { status: 400 }
      );
    }

    // Transform to Go service format with enhanced metadata
    const batchRequest = {
      documents: requestData.documents.map((doc, index) => ({
        title: doc.title,
        content: doc.content,
        case_id: doc.case_id,
        metadata: {
          ...doc.metadata,
          // Enhanced batch metadata following your patterns
          source: 'sveltekit-batch-ingest',
          batch_index: index,
          batch_size: requestData.documents.length,
          batch_id: `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          api_version: 'v1',
          timestamp: new Date().toISOString(),
          user_agent: request.headers.get('user-agent') || 'unknown'
        }
      }))
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);

    try {
      // Call Go batch ingest service
      const response = await fetch(`${SERVICE_URL}/api/ingest/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(batchRequest),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        return json(
          {
            error: `Batch ingest service error: ${response.status} - ${errorText}`,
            service: 'ingest-service',
            port: '8227',
            batch_size: requestData.documents.length
          },
          { status: response.status }
        );
      }

      const result: BatchIngestResponse = await response.json();
      const processingTime = Date.now() - startTime;

      // Enhanced response with comprehensive metadata
      return json({
        ...result,
        // Performance metrics following your patterns
        performance: {
          total_processing_time_ms: processingTime,
          average_document_time_ms: result.results.length > 0
            ? result.results.reduce((sum, r) => sum + r.process_time_ms, 0) / result.results.length
            : 0,
          documents_per_second: result.processed > 0
            ? (result.processed / (processingTime / 1000)).toFixed(2)
            : 0
        },
        service_info: {
          go_service: 'ingest-service',
          port: '8227',
          proxy: 'sveltekit-batch-api',
          architecture: 'multi-protocol',
          batch_enabled: true
        },
        // Follow your established success pattern
        success: true,
        api_version: 'v1',
        batch_summary: {
          requested: requestData.documents.length,
          processed: result.processed,
          failed: (result.errors?.length || 0),
          success_rate: result.processed > 0
            ? ((result.processed / requestData.documents.length) * 100).toFixed(1) + '%'
            : '0%'
        }
      });

    } catch (fetchError) {
      clearTimeout(timeoutId);

      if (fetchError.name === 'AbortError') {
        return json(
          {
            error: 'Batch processing timeout',
            message: `Processing ${requestData.documents.length} documents took longer than ${TIMEOUT/1000} seconds`,
            batch_size: requestData.documents.length,
            timeout_ms: TIMEOUT
          },
          { status: 504 }
        );
      }

      throw fetchError;
    }

  } catch (error: any) {
    const processingTime = Date.now() - startTime;

    console.error('Batch ingest API error:', error);

    return json(
      {
        error: 'Internal server error during batch processing',
        message: error instanceof Error ? error.message : 'Unknown error',
        service: 'sveltekit-batch-ingest-proxy',
        processing_time_ms: processingTime
      },
      { status: 500 }
    );
  }
};

// Batch status endpoint
export const GET: RequestHandler = async ({ url, fetch }) => {
  const batchId = url.searchParams.get('batch_id');

  if (!batchId) {
    return json({
      service: 'batch-ingest',
      capabilities: {
        max_batch_size: BATCH_SIZE_LIMIT,
        timeout_ms: TIMEOUT,
        supported_formats: ['json'],
        endpoints: {
          batch_ingest: '/api/v1/ingest/batch',
          single_ingest: '/api/v1/ingest',
          health: '/api/v1/ingest'
        }
      },
      // Follow your API documentation pattern
      architecture: {
        frontend: 'sveltekit-2',
        backend: 'go-gin-microservice',
        database: 'postgresql-pgvector',
        embeddings: 'ollama-nomic-embed-text'
      }
    });
  }

  // Future: implement batch status tracking with your established patterns
  return json({
    batch_id: batchId,
    status: 'not_implemented',
    message: 'Batch status tracking will be implemented with Redis/PostgreSQL integration'
  });
};