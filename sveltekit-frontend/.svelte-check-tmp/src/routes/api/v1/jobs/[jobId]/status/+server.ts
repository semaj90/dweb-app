import type { RequestHandler } from './$types';

/**
 * Job Status Polling API - Real-time Job Progress Tracking
 * Provides polling endpoint for ingestion job status updates
 */

import { json } from '@sveltejs/kit';
import { redis } from '$lib/server/cache/redis-service';

export interface JobStatus {
  id: string;
  uploadId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: {
    stage: string;
    percentage: number;
    message: string;
  };
  fileName: string;
  bucket: string;
  objectName: string;
  caseId?: string;
  evidenceId?: string;
  error?: string;
  results?: {
    textExtracted?: boolean;
    vectorsGenerated?: boolean;
    metadataExtracted?: boolean;
    thumbnailGenerated?: boolean;
    ocrCompleted?: boolean;
    analysisCompleted?: boolean;
  };
  timing: {
    createdAt: string;
    startedAt?: string;
    completedAt?: string;
    duration?: number;
  };
  metadata?: Record<string, any>;
}

// GET /api/v1/jobs/[jobId]/status - Get job status and progress
export const GET: RequestHandler = async ({ params, url, getClientAddress }) => {
  try {
    const { jobId } = params;

    if (!jobId) {
      return json({
        success: false,
        error: 'jobId parameter is required'
      }, { status: 400 });
    }

    console.log(`🔍 GET /api/v1/jobs/${jobId}/status`);

    // Get job data from Redis
    const jobKey = `ingestion:${jobId}`;
    const jobData = await redis.get(jobKey);

    if (!jobData) {
      return json({
        success: false,
        error: 'Job not found'
      }, { status: 404 });
    }

    const job = JSON.parse(jobData) as JobStatus;

    // Get additional progress data if available
    const progressKey = `progress:${jobId}`;
    const progressData = await redis.get(progressKey);
    
    if (progressData) {
      try {
        const progress = JSON.parse(progressData);
        job.progress = progress;
      } catch (parseError) {
        console.warn('Failed to parse progress data:', parseError);
      }
    }

    // Get error details if job failed
    if (job.status === 'failed') {
      const errorKey = `error:${jobId}`;
      const errorData = await redis.get(errorKey);
      if (errorData) {
        job.error = errorData;
      }
    }

    // Get results if job completed
    if (job.status === 'completed') {
      const resultsKey = `results:${jobId}`;
      const resultsData = await redis.get(resultsKey);
      if (resultsData) {
        try {
          job.results = JSON.parse(resultsData);
        } catch (parseError) {
          console.warn('Failed to parse results data:', parseError);
        }
      }
    }

    // Calculate duration if job has timing info
    if (job.timing) {
      const createdAt = new Date(job.timing.createdAt).getTime();
      const completedAt = job.timing.completedAt ? new Date(job.timing.completedAt).getTime() : null;
      const startedAt = job.timing.startedAt ? new Date(job.timing.startedAt).getTime() : null;
      
      if (completedAt) {
        job.timing.duration = completedAt - createdAt;
      } else if (startedAt) {
        job.timing.duration = Date.now() - createdAt;
      }
    }

    const response = {
      success: true,
      data: {
        job,
        polling: {
          interval: job.status === 'processing' ? 2000 : 10000, // Poll every 2s when processing, 10s otherwise
          shouldContinue: job.status === 'queued' || job.status === 'processing'
        }
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        endpoint: `/api/v1/jobs/${jobId}/status`
      }
    };

    console.log(`✅ Job status retrieved: ${jobId} - ${job.status}`);
    return json(response);

  } catch (error: any) {
    console.error(`❌ GET /api/v1/jobs/${params.jobId}/status error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get job status'
    }, { status: 500 });
  }
};

// POST /api/v1/jobs/[jobId]/status - Update job status (for workers)
export const POST: RequestHandler = async ({ params, request, getClientAddress }) => {
  try {
    const { jobId } = params;

    if (!jobId) {
      return json({
        success: false,
        error: 'jobId parameter is required'
      }, { status: 400 });
    }

    const updateData = await request.json();
    
    console.log(`📝 POST /api/v1/jobs/${jobId}/status - Updating status to ${updateData.status}`);

    // Get existing job data
    const jobKey = `ingestion:${jobId}`;
    const existingData = await redis.get(jobKey);

    if (!existingData) {
      return json({
        success: false,
        error: 'Job not found'
      }, { status: 404 });
    }

    const job = JSON.parse(existingData);

    // Update job status and timing
    if (updateData.status) {
      job.status = updateData.status;
      
      if (updateData.status === 'processing' && !job.timing.startedAt) {
        job.timing.startedAt = new Date().toISOString();
      }
      
      if (['completed', 'failed'].includes(updateData.status) && !job.timing.completedAt) {
        job.timing.completedAt = new Date().toISOString();
      }
    }

    // Update progress if provided
    if (updateData.progress) {
      const progressKey = `progress:${jobId}`;
      await redis.setex(progressKey, 3600, JSON.stringify(updateData.progress));
    }

    // Store error if provided
    if (updateData.error) {
      const errorKey = `error:${jobId}`;
      await redis.setex(errorKey, 86400, updateData.error); // Keep errors for 24h
    }

    // Store results if provided
    if (updateData.results) {
      const resultsKey = `results:${jobId}`;
      await redis.setex(resultsKey, 86400, JSON.stringify(updateData.results)); // Keep results for 24h
    }

    // Update metadata if provided
    if (updateData.metadata) {
      job.metadata = { ...job.metadata, ...updateData.metadata };
    }

    // Save updated job
    await redis.setex(jobKey, 86400, JSON.stringify(job));

    // Remove from queue if status changed to processing or completed
    if (['processing', 'completed', 'failed'].includes(job.status)) {
      const queueData = await redis.lrange('ingestion:queue', 0, -1);
      for (let i = 0; i < queueData.length; i++) {
        try {
          const queuedJob = JSON.parse(queueData[i]);
          if (queuedJob.id === jobId) {
            await redis.lrem('ingestion:queue', 1, queueData[i]);
            break;
          }
        } catch (parseError) {
          // Skip malformed entries
        }
      }
    }

    // Publish status update event
    await redis.publish('job:status_updated', JSON.stringify({
      jobId,
      status: job.status,
      fileName: job.fileName,
      progress: updateData.progress,
      caseId: job.caseId,
      timestamp: Date.now()
    }));

    // If job completed successfully, trigger final processing
    if (job.status === 'completed' && job.caseId && updateData.results) {
      await redis.publish('case:document_processed', JSON.stringify({
        jobId,
        caseId: job.caseId,
        evidenceId: job.metadata?.evidenceId,
        fileName: job.fileName,
        results: updateData.results,
        timestamp: Date.now()
      }));
    }

    const response = {
      success: true,
      data: {
        jobId,
        status: job.status,
        updated: true
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        operation: 'update_status'
      }
    };

    console.log(`✅ Job status updated: ${jobId} - ${job.status}`);
    return json(response);

  } catch (error: any) {
    console.error(`❌ POST /api/v1/jobs/${params.jobId}/status error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to update job status'
    }, { status: 500 });
  }
};

// DELETE /api/v1/jobs/[jobId]/status - Cancel job
export const DELETE: RequestHandler = async ({ params, getClientAddress }) => {
  try {
    const { jobId } = params;

    if (!jobId) {
      return json({
        success: false,
        error: 'jobId parameter is required'
      }, { status: 400 });
    }

    console.log(`🗑️ DELETE /api/v1/jobs/${jobId}/status - Canceling job`);

    // Get existing job data
    const jobKey = `ingestion:${jobId}`;
    const existingData = await redis.get(jobKey);

    if (!existingData) {
      return json({
        success: false,
        error: 'Job not found'
      }, { status: 404 });
    }

    const job = JSON.parse(existingData);

    // Can only cancel queued or processing jobs
    if (!['queued', 'processing'].includes(job.status)) {
      return json({
        success: false,
        error: `Cannot cancel job with status: ${job.status}`
      }, { status: 400 });
    }

    // Update job status to failed with cancellation reason
    job.status = 'failed';
    job.timing.completedAt = new Date().toISOString();
    
    // Store cancellation error
    const errorKey = `error:${jobId}`;
    await redis.setex(errorKey, 86400, 'Job cancelled by user');

    // Save updated job
    await redis.setex(jobKey, 86400, JSON.stringify(job));

    // Remove from processing queue
    const queueData = await redis.lrange('ingestion:queue', 0, -1);
    for (let i = 0; i < queueData.length; i++) {
      try {
        const queuedJob = JSON.parse(queueData[i]);
        if (queuedJob.id === jobId) {
          await redis.lrem('ingestion:queue', 1, queueData[i]);
          break;
        }
      } catch (parseError) {
        // Skip malformed entries
      }
    }

    // Publish cancellation event
    await redis.publish('job:cancelled', JSON.stringify({
      jobId,
      fileName: job.fileName,
      caseId: job.caseId,
      timestamp: Date.now()
    }));

    const response = {
      success: true,
      data: {
        jobId,
        status: 'cancelled',
        message: 'Job cancelled successfully'
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        operation: 'cancel_job'
      }
    };

    console.log(`✅ Job cancelled: ${jobId}`);
    return json(response);

  } catch (error: any) {
    console.error(`❌ DELETE /api/v1/jobs/${params.jobId}/status error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to cancel job'
    }, { status: 500 });
  }
};