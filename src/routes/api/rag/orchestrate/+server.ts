/**
 * RAG Orchestration API Endpoint
 * 
 * Provides REST API for document processing orchestration
 * Integrates with ProductionRAGCoordinator for complete pipeline management
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ragCoordinator } from '$lib/orchestration/production-rag-coordinator';
import { serviceHealthMonitor } from '$lib/monitoring/service-health-monitor';

// Initialize systems if not already done
let systemInitialized = false;

async function ensureSystemInitialized() {
  if (!systemInitialized) {
    console.log('[RAG API] Initializing production systems...');
    
    try {
      await ragCoordinator.initialize();
      serviceHealthMonitor.startMonitoring(30000); // 30 second intervals
      systemInitialized = true;
      console.log('[RAG API] ✅ Systems initialized successfully');
    } catch (err) {
      console.error('[RAG API] ❌ System initialization failed:', err);
      throw err;
    }
  }
}

/**
 * Process document through RAG pipeline
 * POST /api/rag/orchestrate
 */
export const POST: RequestHandler = async ({ request }) => {
  await ensureSystemInitialized();

  try {
    const { uploadId, caseId, filename, storageUrl } = await request.json();

    if (!uploadId || !caseId || !filename || !storageUrl) {
      return error(400, {
        message: 'Missing required fields: uploadId, caseId, filename, storageUrl'
      });
    }

    console.log(`[RAG API] 📄 Processing document: ${filename}`);

    // Start document processing
    const job = await ragCoordinator.processDocument(uploadId, caseId, filename, storageUrl);

    return json({
      success: true,
      jobId: job.jobId,
      status: job.status,
      progress: job.progress,
      message: `Document processing started for ${filename}`,
      tracking: {
        jobId: job.jobId,
        statusUrl: `/api/rag/orchestrate/status/${job.jobId}`,
        websocketUrl: `/api/rag/orchestrate/ws?jobId=${job.jobId}`
      }
    });

  } catch (err) {
    console.error('[RAG API] ❌ Document processing failed:', err);
    
    return error(500, {
      message: 'Document processing failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    });
  }
};

/**
 * Query RAG system
 * GET /api/rag/orchestrate?query=...&caseId=...
 */
export const GET: RequestHandler = async ({ url }) => {
  await ensureSystemInitialized();

  try {
    const query = url.searchParams.get('query');
    const caseId = url.searchParams.get('caseId');
    const limit = parseInt(url.searchParams.get('limit') || '5');
    const threshold = parseFloat(url.searchParams.get('threshold') || '0.7');
    const includeMetadata = url.searchParams.get('includeMetadata') === 'true';
    const model = url.searchParams.get('model') || 'gemma3-legal';

    if (!query) {
      return error(400, { message: 'Query parameter is required' });
    }

    console.log(`[RAG API] 🔍 Processing RAG query: ${query.substring(0, 50)}...`);

    // Execute RAG query
    const result = await ragCoordinator.queryRAG(query, caseId || undefined, {
      limit,
      threshold,
      includeMetadata,
      model
    });

    return json({
      success: true,
      ...result
    });

  } catch (err) {
    console.error('[RAG API] ❌ RAG query failed:', err);
    
    return error(500, {
      message: 'RAG query failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    });
  }
};

/**
 * Get system health status
 * PATCH /api/rag/orchestrate (health check)
 */
export const PATCH: RequestHandler = async () => {
  try {
    await ensureSystemInitialized();

    const systemStatus = ragCoordinator.getSystemStatus();
    const healthStatus = serviceHealthMonitor.getOverallHealthStatus();
    const metrics = serviceHealthMonitor.getSystemMetrics();

    return json({
      success: true,
      timestamp: Date.now(),
      coordinator: {
        health: systemStatus.health,
        activeJobs: systemStatus.activeJobs,
        metrics: systemStatus.metrics
      },
      services: {
        total: healthStatus.summary.total,
        healthy: healthStatus.summary.healthy,
        degraded: healthStatus.summary.degraded,
        unhealthy: healthStatus.summary.unhealthy,
        critical: healthStatus.summary.critical,
        status: healthStatus.status
      },
      performance: {
        averageResponseTime: metrics.averageResponseTime,
        uptime: metrics.uptime,
        alertsTriggered: metrics.alertsTriggered
      },
      detailed: {
        services: Array.from(systemStatus.services.entries()).map(([name, status]) => ({
          name,
          ...status
        })),
        serviceHealth: Object.entries(healthStatus.services).map(([name, health]) => ({
          name,
          ...health
        }))
      }
    });

  } catch (err) {
    console.error('[RAG API] ❌ Health check failed:', err);
    
    return error(500, {
      message: 'Health check failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    });
  }
};

/**
 * System management commands
 * DELETE /api/rag/orchestrate (cleanup/reset)
 */
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get('action');
    
    switch (action) {
      case 'cleanup':
        await ragCoordinator.cleanup();
        serviceHealthMonitor.cleanup();
        systemInitialized = false;
        
        return json({
          success: true,
          message: 'System cleanup completed'
        });

      case 'restart-monitoring':
        serviceHealthMonitor.stopMonitoring();
        serviceHealthMonitor.startMonitoring(30000);
        
        return json({
          success: true,
          message: 'Service monitoring restarted'
        });

      default:
        return error(400, { message: 'Invalid action. Supported: cleanup, restart-monitoring' });
    }

  } catch (err) {
    console.error('[RAG API] ❌ System management failed:', err);
    
    return error(500, {
      message: 'System management operation failed',
      error: err instanceof Error ? err.message : 'Unknown error'
    });
  }
};