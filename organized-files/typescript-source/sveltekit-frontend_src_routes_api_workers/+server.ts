// @ts-nocheck
/**
 * Specialized Worker API - Event-Driven Job Processing
 * Endpoints for submitting jobs to the worker "hive"
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { z } from 'zod';
import { createSpecializedWorkerSystem, type SpecializedJob } from '$lib/workers/specialized-worker-system.js';

// Job submission schemas
const summarizationJobSchema = z.object({
  type: z.literal('SUMMARIZE_DOCUMENT'),
  document: z.object({
    id: z.string(),
    content: z.string(),
    metadata: z.object({}).optional()
  }),
  options: z.object({
    maxLength: z.number().optional(),
    focusAreas: z.array(z.string()).optional(),
    style: z.enum(['brief', 'detailed', 'executive']).optional()
  }).optional(),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).default('medium'),
  metadata: z.object({
    caseId: z.string().optional(),
    userId: z.string().optional(),
    confidential: z.boolean().optional()
  }).optional()
});

const caseLawJobSchema = z.object({
  type: z.literal('GET_CASE_LAW'),
  query: z.string(),
  jurisdiction: z.string().optional(),
  dateRange: z.object({
    start: z.string(),
    end: z.string()
  }).optional(),
  maxResults: z.number().min(1).max(50).default(10),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).default('medium'),
  metadata: z.object({
    caseId: z.string().optional(),
    userId: z.string().optional()
  }).optional()
});

const embeddingJobSchema = z.object({
  type: z.literal('GENERATE_EMBEDDING'),
  text: z.string(),
  model: z.string().default('nomic-embed-text'),
  options: z.object({
    dimensions: z.number().optional(),
    normalize: z.string().optional(),
    includeText: z.boolean().default(false)
  }).optional(),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).default('medium'),
  metadata: z.object({
    caseId: z.string().optional(),
    userId: z.string().optional()
  }).optional()
});

const jobSchema = z.union([summarizationJobSchema, caseLawJobSchema, embeddingJobSchema]);

// Initialize worker system (singleton)
let workerSystem: Awaited<ReturnType<typeof createSpecializedWorkerSystem>> | null = null;

async function getWorkerSystem() {
  if (!workerSystem) {
    try {
      workerSystem = await createSpecializedWorkerSystem();
      console.log('🏗️ Worker system initialized for API');
    } catch (error) {
      console.error('Failed to initialize worker system:', error);
      throw new Error('Worker system unavailable');
    }
  }
  return workerSystem;
}

/**
 * POST /api/workers - Submit a new job to the worker system
 */
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    const validation = jobSchema.safeParse(body);

    if (!validation.success) {
      return json(
        { 
          error: 'Invalid job format', 
          details: validation.error.flatten() 
        }, 
        { status: 400 }
      );
    }

    const jobData = validation.data;
    const { orchestrator } = await getWorkerSystem();

    // Create job with timeout and retry settings
    const job: Omit<SpecializedJob, 'id' | 'createdAt'> = {
      type: jobData.type,
      payload: jobData,
      priority: jobData.priority,
      timeout: getTimeoutForJobType(jobData.type),
      retryCount: 3,
      metadata: {
        ...jobData.metadata,
        userId: locals.user?.id,
        source: 'api'
      }
    };

    const jobId = await orchestrator.submitJob(job);

    console.log(`📝 API: Job ${jobId} (${job.type}) submitted by user ${locals.user?.id}`);

    return json({
      success: true,
      jobId,
      type: job.type,
      estimatedTime: getEstimatedTimeForJobType(job.type),
      status: 'queued'
    });

  } catch (error) {
    console.error('Job submission failed:', error);
    return json(
      { 
        error: 'Failed to submit job', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      }, 
      { status: 500 }
    );
  }
};

/**
 * GET /api/workers?jobId=xxx - Get job status and result
 */
export const GET: RequestHandler = async ({ url }) => {
  try {
    const jobId = url.searchParams.get('jobId');
    const statsOnly = url.searchParams.get('stats') === 'true';

    const { orchestrator } = await getWorkerSystem();

    // Return system stats if requested
    if (statsOnly) {
      const stats = orchestrator.getStats();
      return json({
        success: true,
        stats,
        timestamp: new Date().toISOString()
      });
    }

    if (!jobId) {
      return json({ error: 'jobId parameter required' }, { status: 400 });
    }

    const result = await orchestrator.getJobResult(jobId);

    if (!result) {
      return json({
        success: true,
        jobId,
        status: 'processing',
        message: 'Job is still being processed'
      });
    }

    return json({
      success: true,
      jobId,
      status: result.success ? 'completed' : 'failed',
      result: result.success ? result.data : undefined,
      error: result.error,
      processingTime: result.processingTime,
      workerInfo: result.workerInfo
    });

  } catch (error) {
    console.error('Failed to get job status:', error);
    return json(
      { 
        error: 'Failed to get job status', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      }, 
      { status: 500 }
    );
  }
};

/**
 * PUT /api/workers/wait - Wait for job completion (with timeout)
 */
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { jobId, timeout = 30000 } = await request.json();

    if (!jobId) {
      return json({ error: 'jobId required' }, { status: 400 });
    }

    const { orchestrator } = await getWorkerSystem();
    
    try {
      const result = await orchestrator.waitForJobResult(jobId, timeout);
      
      return json({
        success: true,
        jobId,
        status: result.success ? 'completed' : 'failed',
        result: result.success ? result.data : undefined,
        error: result.error,
        processingTime: result.processingTime,
        workerInfo: result.workerInfo
      });

    } catch (timeoutError) {
      return json({
        success: false,
        jobId,
        status: 'timeout',
        error: 'Job did not complete within timeout period',
        timeout
      }, { status: 408 });
    }

  } catch (error) {
    console.error('Wait for job failed:', error);
    return json(
      { 
        error: 'Failed to wait for job', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      }, 
      { status: 500 }
    );
  }
};

// Helper functions
function getTimeoutForJobType(type: SpecializedJob['type']): number {
  const timeouts = {
    'SUMMARIZE_DOCUMENT': 60000,    // 1 minute
    'GET_CASE_LAW': 120000,         // 2 minutes
    'GENERATE_EMBEDDING': 30000,    // 30 seconds
    'ANALYZE_EVIDENCE': 180000,     // 3 minutes
    'LEGAL_RESEARCH': 300000        // 5 minutes
  };
  return timeouts[type] || 60000;
}

function getEstimatedTimeForJobType(type: SpecializedJob['type']): string {
  const estimates = {
    'SUMMARIZE_DOCUMENT': '30-60 seconds',
    'GET_CASE_LAW': '1-2 minutes',
    'GENERATE_EMBEDDING': '10-30 seconds',
    'ANALYZE_EVIDENCE': '2-3 minutes',
    'LEGAL_RESEARCH': '3-5 minutes'
  };
  return estimates[type] || '1 minute';
}