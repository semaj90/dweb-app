import type { RequestHandler } from './$types.js';

/**
 * PostgreSQL-First Architecture Test Endpoint
 * 
 * This endpoint demonstrates the complete PostgreSQL-centered workflow:
 * 1. Upload evidence → PostgreSQL
 * 2. Publish Redis event → Worker processing
 * 3. Go ingest service → Embedding generation
 * 4. PostgreSQL notifications → Worker tagging
 * 5. Qdrant sync → Search index mirroring
 */

import { db } from '$lib/server/db/index.js';
import { evidence, documentMetadata, documentEmbeddings } from '$lib/server/db/schema-unified.js';
import { eq, sql } from 'drizzle-orm';
import { v4 as uuidv4 } from 'uuid';
import { createClient } from 'redis';
import { postgresqlQdrantSync } from '$lib/services/postgresql-qdrant-sync.js';
import { URL } from "url";

// Test data for demonstration
const SAMPLE_EVIDENCE = {
  title: 'Test Contract Document',
  description: 'Sample legal contract for testing PostgreSQL-first workflow',
  evidenceType: 'document',
  mimeType: 'application/pdf',
  fileSize: 245760,
  hash: 'sha256_test_hash_' + Date.now(),
  content: `
    LEGAL SERVICES AGREEMENT
    
    This Agreement is entered into between ABC Law Firm and Client Corp.
    
    1. SCOPE OF SERVICES
    The Law Firm shall provide legal counsel and representation in matters relating to corporate law.
    
    2. COMPENSATION
    Client agrees to pay hourly rates as specified in Schedule A.
    
    3. CONFIDENTIALITY
    All information shared between parties shall remain confidential.
    
    4. TERMINATION
    Either party may terminate this agreement with 30 days written notice.
    
    This agreement is governed by the laws of the State of California.
  `
};

export interface TestWorkflowRequest {
  userId?: string;
  caseId?: string;
  enableIngestService?: boolean;
  enableQdrantSync?: boolean;
  testType?: 'full' | 'evidence_only' | 'ingest_only' | 'sync_only';
}

export const POST: RequestHandler = async ({ request, url }) => {
  const testType = url.searchParams.get('type') || 'full';
  const correlationId = uuidv4();
  
  try {
    const body: TestWorkflowRequest = await request.json();
    const {
      userId = 'test-user-' + Date.now(),
      caseId = 'test-case-' + Date.now(),
      enableIngestService = true,
      enableQdrantSync = true
    } = body;

    console.log(`🧪 Starting PostgreSQL-first workflow test (${testType})`);
    console.log(`📋 Correlation ID: ${correlationId}`);

    const results: any = {
      correlationId,
      testType,
      timestamp: new Date().toISOString(),
      steps: [],
      postgresql: { status: 'pending' },
      redis: { status: 'pending' },
      ingestService: { status: 'pending' },
      qdrantSync: { status: 'pending' },
      summary: { success: false, errors: [] }
    };

    // Step 1: Test PostgreSQL Evidence Creation
    if (testType === 'full' || testType === 'evidence_only') {
      try {
        const evidenceId = uuidv4();
        
        // Insert test evidence into PostgreSQL
        const [createdEvidence] = await db.insert(evidence).values({
          id: evidenceId,
          userId: userId,
          caseId: caseId,
          title: SAMPLE_EVIDENCE.title,
          description: SAMPLE_EVIDENCE.description,
          evidenceType: SAMPLE_EVIDENCE.evidenceType,
          mimeType: SAMPLE_EVIDENCE.mimeType,
          fileSize: SAMPLE_EVIDENCE.fileSize,
          hash: SAMPLE_EVIDENCE.hash,
          processingStatus: 'completed',
          ingestStatus: 'pending',
          tags: ['contract', 'legal-agreement'],
          isAdmissible: true,
          confidentialityLevel: 'internal'
        }).returning();

        results.postgresql = {
          status: 'success',
          evidenceId: evidenceId,
          createdAt: createdEvidence.createdAt
        };

        results.steps.push({
          step: 'postgresql_evidence_creation',
          status: 'success',
          timestamp: new Date().toISOString(),
          details: { evidenceId, userId, caseId }
        });

        console.log(`✅ Step 1: Created evidence ${evidenceId} in PostgreSQL`);

        // Step 2: Test Redis Event Publishing
        if (testType === 'full') {
          try {
            const redisClient = createClient({
              url: import.meta.env.REDIS_URL || 'redis://localhost:6379'
            });
            
            await redisClient.connect();

            const eventData = {
              type: 'evidence',
              id: evidenceId,
              action: 'tag',
              caseId: caseId,
              userId: userId,
              correlationId: correlationId,
              priority: 'high',
              timestamp: new Date().toISOString()
            };

            await redisClient.xAdd('autotag:requests', '*', eventData);
            await redisClient.disconnect();

            results.redis = {
              status: 'success',
              streamName: 'autotag:requests',
              eventId: 'published'
            };

            results.steps.push({
              step: 'redis_event_publishing',
              status: 'success',
              timestamp: new Date().toISOString(),
              details: { streamName: 'autotag:requests', eventData }
            });

            console.log(`✅ Step 2: Published Redis event for evidence ${evidenceId}`);
          } catch (error: any) {
            results.redis = {
              status: 'error',
              error: error.message
            };
            results.summary.errors.push(`Redis event publishing failed: ${error.message}`);
            console.error(`❌ Step 2 failed:`, error);
          }
        }

        // Step 3: Test Go Ingest Service Integration
        if ((testType === 'full' || testType === 'ingest_only') && enableIngestService) {
          try {
            const ingestPayload = {
              title: SAMPLE_EVIDENCE.title,
              content: SAMPLE_EVIDENCE.content,
              case_id: caseId,
              metadata: {
                evidence_id: evidenceId,
                source: 'test_workflow',
                correlation_id: correlationId
              }
            };

            const ingestResponse = await fetch('http://localhost:8227/api/ingest', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-Correlation-ID': correlationId
              },
              body: JSON.stringify(ingestPayload)
            });

            if (ingestResponse.ok) {
              const ingestResult = await ingestResponse.json();
              
              results.ingestService = {
                status: 'success',
                documentId: ingestResult.document_id,
                embeddingId: ingestResult.embedding_id,
                processTime: ingestResult.process_time_ms
              };

              // Link document to evidence
              if (ingestResult.document_id) {
                await db.update(documentMetadata)
                  .set({ evidenceId: evidenceId })
                  .where(eq(documentMetadata.id, ingestResult.document_id));
              }

              results.steps.push({
                step: 'go_ingest_service',
                status: 'success',
                timestamp: new Date().toISOString(),
                details: ingestResult
              });

              console.log(`✅ Step 3: Go ingest service processed document ${ingestResult.document_id}`);
            } else {
              throw new Error(`Ingest service responded with ${ingestResponse.status}`);
            }
          } catch (error: any) {
            results.ingestService = {
              status: 'error',
              error: error.message
            };
            results.summary.errors.push(`Ingest service failed: ${error.message}`);
            console.error(`❌ Step 3 failed:`, error);
          }
        }

        // Step 4: Test Qdrant Sync Service
        if ((testType === 'full' || testType === 'sync_only') && enableQdrantSync) {
          try {
            // Wait a moment for ingest service to complete
            await new Promise(resolve => setTimeout(resolve, 2000));

            const syncResult = await postgresqlQdrantSync.syncEvidenceById(evidenceId);
            
            if (syncResult) {
              results.qdrantSync = {
                status: 'success',
                synced: true,
                evidenceId: evidenceId
              };

              results.steps.push({
                step: 'qdrant_sync',
                status: 'success',
                timestamp: new Date().toISOString(),
                details: { evidenceId, synced: true }
              });

              console.log(`✅ Step 4: Evidence ${evidenceId} synced to Qdrant`);
            } else {
              throw new Error('Sync returned false - no embedding available');
            }
          } catch (error: any) {
            results.qdrantSync = {
              status: 'error',
              error: error.message
            };
            results.summary.errors.push(`Qdrant sync failed: ${error.message}`);
            console.error(`❌ Step 4 failed:`, error);
          }
        }

      } catch (error: any) {
        results.postgresql = {
          status: 'error',
          error: error.message
        };
        results.summary.errors.push(`PostgreSQL operation failed: ${error.message}`);
        console.error(`❌ PostgreSQL step failed:`, error);
      }
    }

    // Determine overall success
    const hasErrors = results.summary.errors.length > 0;
    results.summary.success = !hasErrors;

    if (results.summary.success) {
      console.log(`🎉 PostgreSQL-first workflow test completed successfully!`);
    } else {
      console.log(`⚠️  PostgreSQL-first workflow test completed with errors:`, results.summary.errors);
    }

    return json({
      success: results.summary.success,
      message: results.summary.success 
        ? 'PostgreSQL-first workflow test completed successfully'
        : 'PostgreSQL-first workflow test completed with errors',
      data: results
    });

  } catch (error: any) {
    console.error('❌ Test endpoint error:', error);
    return json({
      success: false,
      message: 'Test workflow failed',
      error: error.message,
      correlationId
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get('action') || 'status';

    switch (action) {
      case 'health':
        // Check health of all services
        const health = await checkSystemHealth();
        return json({
          success: true,
          message: 'System health check completed',
          data: health
        });

      case 'stats':
        // Get PostgreSQL statistics
        const stats = await getPostgreSQLStats();
        return json({
          success: true,
          message: 'PostgreSQL statistics retrieved',
          data: stats
        });

      case 'sync-stats':
        // Get Qdrant sync statistics
        const syncStats = postgresqlQdrantSync.getStats();
        return json({
          success: true,
          message: 'Qdrant sync statistics retrieved',
          data: syncStats
        });

      default:
        return json({
          success: true,
          message: 'PostgreSQL-first test endpoint is ready',
          data: {
            endpoints: {
              'POST /': 'Run full workflow test',
              'GET ?action=health': 'Check system health',
              'GET ?action=stats': 'Get PostgreSQL statistics',
              'GET ?action=sync-stats': 'Get Qdrant sync statistics'
            },
            testTypes: [
              'full - Complete workflow test',
              'evidence_only - PostgreSQL evidence creation only',
              'ingest_only - Ingest service test only',
              'sync_only - Qdrant sync test only'
            ]
          }
        });
    }
  } catch (error: any) {
    return json({
      success: false,
      message: 'Failed to process request',
      error: error.message
    }, { status: 500 });
  }
};

async function checkSystemHealth(): Promise<any> {
  const health = {
    postgresql: { status: 'unknown', details: null },
    redis: { status: 'unknown', details: null },
    ingestService: { status: 'unknown', details: null },
    qdrant: { status: 'unknown', details: null },
    overall: 'unknown'
  };

  // Check PostgreSQL
  try {
    await db.execute(sql`SELECT 1`);
    const [evidenceCount] = await db.execute(sql`SELECT COUNT(*) as count FROM evidence`);
    health.postgresql = { 
      status: 'healthy', 
      details: { connected: true, evidenceCount: evidenceCount.count }
    };
  } catch (error: any) {
    health.postgresql = { status: 'unhealthy', details: { error: error.message } };
  }

  // Check Redis
  try {
    const redisClient = createClient({ url: import.meta.env.REDIS_URL || 'redis://localhost:6379' });
    await redisClient.connect();
    await redisClient.ping();
    await redisClient.disconnect();
    health.redis = { status: 'healthy', details: { connected: true } };
  } catch (error: any) {
    health.redis = { status: 'unhealthy', details: { error: error.message } };
  }

  // Check Go Ingest Service
  try {
    const response = await fetch('http://localhost:8227/api/health');
    if (response.ok) {
      const healthData = await response.json();
      health.ingestService = { status: 'healthy', details: healthData };
    } else {
      health.ingestService = { status: 'unhealthy', details: { httpStatus: response.status } };
    }
  } catch (error: any) {
    health.ingestService = { status: 'unhealthy', details: { error: error.message } };
  }

  // Check Qdrant via sync service
  try {
    const qdrantHealth = await postgresqlQdrantSync.healthCheck();
    health.qdrant = { 
      status: qdrantHealth.status === 'healthy' ? 'healthy' : 'unhealthy', 
      details: qdrantHealth 
    };
  } catch (error: any) {
    health.qdrant = { status: 'unhealthy', details: { error: error.message } };
  }

  // Determine overall health
  const healthyServices = Object.values(health).filter(service => 
    typeof service === 'object' && service.status === 'healthy'
  ).length;
  
  const totalServices = 4;
  
  if (healthyServices === totalServices) {
    health.overall = 'healthy';
  } else if (healthyServices >= totalServices / 2) {
    health.overall = 'degraded';
  } else {
    health.overall = 'unhealthy';
  }

  return health;
}

async function getPostgreSQLStats(): Promise<any> {
  try {
    const [evidenceStats] = await db.execute(sql`
      SELECT 
        COUNT(*) as total_evidence,
        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as processed_evidence,
        COUNT(CASE WHEN ingest_status = 'completed' THEN 1 END) as ingested_evidence,
        COUNT(CASE WHEN title_embedding IS NOT NULL THEN 1 END) as evidence_with_title_embeddings,
        COUNT(CASE WHEN content_embedding IS NOT NULL THEN 1 END) as evidence_with_content_embeddings
      FROM evidence
    `);

    const [documentStats] = await db.execute(sql`
      SELECT 
        COUNT(*) as total_documents,
        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents
      FROM document_metadata
    `);

    const [embeddingStats] = await db.execute(sql`
      SELECT 
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT embedding_model) as unique_models,
        AVG(chunk_size) as avg_chunk_size
      FROM document_embeddings
    `);

    return {
      evidence: evidenceStats,
      documents: documentStats,
      embeddings: embeddingStats,
      timestamp: new Date().toISOString()
    };
  } catch (error: any) {
    throw new Error(`Failed to get PostgreSQL stats: ${error.message}`);
  }
}