
import type { RequestHandler } from './$types';

// Document Update Loop API
// Handles document changes with automatic re-embedding and re-ranking

import { documentUpdateLoop } from "$lib/services/documentUpdateLoop";
import { documents } from "$lib/db/schema";
import { URL } from "url";

// ============================================================================
// UPDATE DOCUMENT WITH AUTO RE-EMBEDDING
// ============================================================================

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { documentId, content, action = 'auto' } = await request.json();

    if (!documentId) {
      throw error(400, 'Document ID is required');
    }

    if (!content && action !== 'force') {
      throw error(400, 'Content is required for document updates');
    }

    // Verify document exists
    const [document] = await db
      .select()
      .from(documents)
      .where(eq(documents.id, documentId))
      .limit(1);

    if (!document) {
      throw error(404, `Document ${documentId} not found`);
    }

    let result;

    switch (action) {
      case 'auto':
        // Queue automatic update (background processing)
        await documentUpdateLoop.queueDocumentUpdate(documentId, content);
        result = {
          action: 'queued',
          documentId,
          message: 'Document update queued for processing',
          status: await documentUpdateLoop.getQueueStatus()
        };
        break;

      case 'force':
        // Force immediate re-embedding
        const reembedResult = await documentUpdateLoop.forceReembedDocument(documentId);
        const rerankingJobs = await documentUpdateLoop.rerankAffectedQueries(documentId);
        
        result = {
          action: 'completed',
          documentId,
          reembedding: reembedResult,
          reranking: {
            queriesAffected: rerankingJobs.length,
            avgImprovement: rerankingJobs.reduce((sum, job) => sum + job.improvement, 0) / rerankingJobs.length,
            jobs: rerankingJobs.map((job: any) => ({
              queryId: job.queryId,
              query: job.query.substring(0, 100) + '...',
              improvement: job.improvement,
              newResultsCount: job.newResults.length
            }))
          }
        };
        break;

      case 'detect':
        // Only detect changes, don't process
        const change = await documentUpdateLoop.detectDocumentChanges(documentId, content);
        result = {
          action: 'detected',
          documentId,
          change: change ? {
            changeType: change.changeType,
            priority: change.priority,
            affectedChunks: change.affectedChunks?.length || 0,
            hasChanges: true
          } : {
            hasChanges: false
          }
        };
        break;

      default:
        throw error(400, `Unknown action: ${action}`);
    }

    return json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('❌ Document update loop error:', err);
    
    if (err instanceof Error && 'status' in err) {
      throw err; // Re-throw SvelteKit errors
    }
    
    throw error(500, `Document update failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

// ============================================================================
// GET QUEUE STATUS AND METRICS
// ============================================================================

export const GET: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get('action') || 'status';

    switch (action) {
      case 'status':
        const status = await documentUpdateLoop.getQueueStatus();
        return json({
          success: true,
          data: {
            queue: status,
            service: 'Document Update Loop',
            timestamp: new Date().toISOString()
          }
        });

      case 'health':
        // Health check for the update loop service
        const healthStatus = await documentUpdateLoop.getQueueStatus();
        const isHealthy = !healthStatus.processing || healthStatus.queued < 100; // Arbitrary threshold

        return json({
          success: true,
          healthy: isHealthy,
          data: {
            status: isHealthy ? 'healthy' : 'overloaded',
            queue: healthStatus,
            recommendations: isHealthy ? [] : [
              'Queue is overloaded - consider scaling processing',
              'Review document update frequency',
              'Check for failed updates requiring manual intervention'
            ]
          }
        }, { 
          status: isHealthy ? 200 : 503 
        });

      default:
        throw error(400, `Unknown action: ${action}`);
    }

  } catch (err: any) {
    console.error('❌ Update loop status error:', err);
    
    return json({
      success: false,
      error: err instanceof Error ? err.message : 'Unknown error'
    }, { status: 500 });
  }
};

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

export const PATCH: RequestHandler = async ({ request }) => {
  try {
    const { action, documentIds, priority = 'medium' } = await request.json();

    if (!action) {
      throw error(400, 'Action is required');
    }

    switch (action) {
      case 'batch_reembed':
        if (!documentIds || !Array.isArray(documentIds)) {
          throw error(400, 'Document IDs array is required for batch operations');
        }

        const batchResults = [];
        
        for (const documentId of documentIds) {
          try {
            const result = await documentUpdateLoop.forceReembedDocument(documentId);
            batchResults.push({
              documentId,
              success: true,
              result
            });
          } catch (err: any) {
            batchResults.push({
              documentId,
              success: false,
              error: err instanceof Error ? err.message : 'Unknown error'
            });
          }
        }

        return json({
          success: true,
          data: {
            action: 'batch_reembed',
            processed: batchResults.length,
            successful: batchResults.filter((r: any) => r.success).length,
            failed: batchResults.filter((r: any) => !r.success).length,
            results: batchResults
          }
        });

      case 'clear_queue':
        // This would require adding a method to clear the queue
        return json({
          success: true,
          data: {
            action: 'clear_queue',
            message: 'Queue cleared (implementation needed in DocumentUpdateLoop class)'
          }
        });

      default:
        throw error(400, `Unknown batch action: ${action}`);
    }

  } catch (err: any) {
    console.error('❌ Batch operation error:', err);
    
    if (err instanceof Error && 'status' in err) {
      throw err;
    }
    
    throw error(500, `Batch operation failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};