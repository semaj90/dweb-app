import { json } from '@sveltejs/kit';
import { createClient } from 'redis';
import { db } from '$lib/db/client';
import { processingJobs, documents } from '$lib/db/schema/rag-integration';
import { eq } from 'drizzle-orm';

const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

const JOB_STATUS_KEY = 'jobs:status:';
const JOB_RESULT_KEY = 'jobs:result:';
const JOB_PROGRESS_KEY = 'jobs:progress:';

export async function GET({ params }) {
  const { jobId } = params;

  try {
    // Connect to Redis if needed
    if (!redis.isOpen) {
      await redis.connect();
    }

    // Get real-time status from Redis
    const redisStatus = await redis.get(`${JOB_STATUS_KEY}${jobId}`);
    const redisProgress = await redis.get(`${JOB_PROGRESS_KEY}${jobId}`);
    const redisResult = await redis.get(`${JOB_RESULT_KEY}${jobId}`);

    // Get persistent record from database
    const [job] = await db
      .select({
        id: processingJobs.id,
        uuid: processingJobs.uuid,
        documentId: processingJobs.documentId,
        jobType: processingJobs.jobType,
        status: processingJobs.status,
        currentStep: processingJobs.currentStep,
        progress: processingJobs.progress,
        result: processingJobs.result,
        error: processingJobs.error,
        startedAt: processingJobs.startedAt,
        completedAt: processingJobs.completedAt,
        createdAt: processingJobs.createdAt
      })
      .from(processingJobs)
      .where(eq(processingJobs.uuid, jobId))
      .limit(1);

    if (!job && !redisStatus) {
      return json({ error: 'Job not found' }, { status: 404 });
    }

    // Merge real-time and persistent data
    let status = job?.status || 'unknown';
    let progress = job?.progress || 0;
    let currentStep = job?.currentStep || 'unknown';
    let result = job?.result || null;
    let error = job?.error || null;

    if (redisStatus) {
      const redisData = JSON.parse(redisStatus);
      status = redisData.status;
      currentStep = redisData.currentStep;
      if (redisData.error) error = redisData.error;
    }

    if (redisProgress) {
      progress = parseInt(redisProgress);
    }

    if (redisResult) {
      try {
        result = JSON.parse(redisResult);
      } catch (parseError) {
        console.warn('Failed to parse Redis result:', parseError);
      }
    }

    // Get related document info
    let document = null;
    if (job?.documentId) {
      const [doc] = await db
        .select({
          id: documents.id,
          uuid: documents.uuid,
          filename: documents.filename,
          originalName: documents.originalName,
          contentType: documents.contentType,
          fileSize: documents.fileSize,
          processingStatus: documents.processingStatus,
          extractedText: documents.extractedText,
          metadata: documents.metadata,
          createdAt: documents.createdAt,
          updatedAt: documents.updatedAt
        })
        .from(documents)
        .where(eq(documents.id, job.documentId))
        .limit(1);

      document = doc || null;
    }

    // Calculate additional metrics
    const isActive = ['queued', 'processing'].includes(status);
    const isCompleted = status === 'completed';
    const isFailed = status === 'failed';
    
    let duration = 0;
    if (job?.startedAt && job?.completedAt) {
      duration = new Date(job.completedAt).getTime() - new Date(job.startedAt).getTime();
    } else if (job?.startedAt && isActive) {
      duration = Date.now() - new Date(job.startedAt).getTime();
    }

    return json({
      jobId,
      status,
      progress,
      currentStep,
      result,
      error,
      isActive,
      isCompleted,
      isFailed,
      duration,
      startedAt: job?.startedAt,
      completedAt: job?.completedAt,
      createdAt: job?.createdAt,
      document,
      metrics: result ? {
        extractedText: result.extractedText?.length || 0,
        chunksCreated: result.chunksCreated || 0,
        embeddingsGenerated: result.embeddingsGenerated || 0,
        entitiesExtracted: result.entitiesExtracted || 0,
        processingSteps: result.steps || []
      } : null
    });

  } catch (error) {
    console.error('Job status retrieval error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Cancel a job
export async function DELETE({ params }) {
  const { jobId } = params;

  try {
    if (!redis.isOpen) {
      await redis.connect();
    }

    // Get current job status
    const redisStatus = await redis.get(`${JOB_STATUS_KEY}${jobId}`);
    
    if (redisStatus) {
      const statusData = JSON.parse(redisStatus);
      
      if (statusData.status === 'processing') {
        // Send cancellation signal to worker
        await redis.setEx(`jobs:cancel:${jobId}`, 300, JSON.stringify({
          cancelled: true,
          timestamp: new Date().toISOString(),
          reason: 'user_requested'
        }));

        // Update status to cancelling
        await redis.setEx(`${JOB_STATUS_KEY}${jobId}`, 3600, JSON.stringify({
          ...statusData,
          status: 'cancelling',
          currentStep: 'cancelling'
        }));
      } else if (statusData.status === 'queued') {
        // Remove from queue if still queued
        const queues = ['jobs:ingest:high', 'jobs:ingest:normal', 'jobs:ingest:low'];
        
        for (const queueName of queues) {
          const queueItems = await redis.lRange(queueName, 0, -1);
          
          for (let i = 0; i < queueItems.length; i++) {
            try {
              const item = JSON.parse(queueItems[i]);
              if (item.jobId === jobId) {
                await redis.lRem(queueName, 1, queueItems[i]);
                break;
              }
            } catch (parseError) {
              console.warn('Failed to parse queue item:', parseError);
            }
          }
        }

        // Update status to cancelled
        await redis.setEx(`${JOB_STATUS_KEY}${jobId}`, 3600, JSON.stringify({
          ...statusData,
          status: 'cancelled',
          currentStep: 'cancelled',
          cancelledAt: new Date().toISOString()
        }));
      }
    }

    // Update database record
    await db
      .update(processingJobs)
      .set({
        status: 'cancelled',
        error: 'Job cancelled by user',
        completedAt: new Date()
      })
      .where(eq(processingJobs.uuid, jobId));

    // Publish cancellation event
    await redis.publish('job_events', JSON.stringify({
      type: 'job_cancelled',
      jobId,
      timestamp: new Date().toISOString()
    }));

    return json({
      success: true,
      message: 'Job cancellation requested'
    });

  } catch (error) {
    console.error('Job cancellation error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}