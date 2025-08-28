import { json } from '@sveltejs/kit';
import { z } from 'zod';
import { createClient } from 'redis';
import { db } from '$lib/db/client';
import { documents, processingJobs } from '$lib/db/schema/rag-integration';
import { eq } from 'drizzle-orm';
import { randomUUID } from 'crypto';

const ingestJobSchema = z.object({
  fileId: z.string().uuid(),
  caseId: z.string().uuid(),
  filename: z.string().min(1),
  contentType: z.string().min(1),
  priority: z.enum(['low', 'normal', 'high']).default('normal')
});

// Redis client for job queue
const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Job queue configuration
const JOB_QUEUES = {
  high: 'jobs:ingest:high',
  normal: 'jobs:ingest:normal', 
  low: 'jobs:ingest:low'
};

const JOB_STATUS_KEY = 'jobs:status:';
const JOB_RESULT_KEY = 'jobs:result:';
const JOB_PROGRESS_KEY = 'jobs:progress:';

export async function POST({ request }) {
  try {
    // Parse and validate request
    const body = await request.json();
    const { fileId, caseId, filename, contentType, priority } = ingestJobSchema.parse(body);

    // Verify document exists
    const [document] = await db
      .select()
      .from(documents)
      .where(eq(documents.uuid, fileId))
      .limit(1);

    if (!document) {
      return json({ error: 'Document not found' }, { status: 404 });
    }

    // Generate job ID
    const jobId = randomUUID();

    // Create job record in database
    const [job] = await db
      .insert(processingJobs)
      .values({
        uuid: jobId,
        documentId: document.id,
        jobType: 'ingest',
        status: 'queued',
        currentStep: 'queued',
        progress: 0,
        result: {
          priority,
          queuedAt: new Date().toISOString()
        }
      })
      .returning();

    // Connect to Redis
    if (!redis.isOpen) {
      await redis.connect();
    }

    // Create job payload for processing workers
    const jobPayload = {
      jobId,
      fileId,
      documentId: document.id,
      caseId,
      filename,
      originalName: document.originalName,
      contentType,
      minioPath: document.minioPath,
      createdAt: new Date().toISOString(),
      priority,
      steps: [
        'download', // Download from MinIO
        'ocr',      // OCR text extraction (Tesseract)
        'nlp',      // NLP entity extraction (Legal-BERT)
        'chunk',    // Text chunking for embeddings
        'embed',    // Generate embeddings (nomic-embed)
        'store',    // Store in PostgreSQL + Qdrant
        'index',    // Update search indexes
        'notify'    // Notify completion
      ]
    };

    // Add job to appropriate Redis queue
    const queueName = JOB_QUEUES[priority];
    await redis.lPush(queueName, JSON.stringify(jobPayload));

    // Set job status in Redis for real-time tracking
    await redis.setEx(`${JOB_STATUS_KEY}${jobId}`, 3600, JSON.stringify({
      status: 'queued',
      progress: 0,
      currentStep: 'queued',
      queuedAt: new Date().toISOString()
    }));

    // Update document status
    await db
      .update(documents)
      .set({ 
        processingStatus: 'queued',
        updatedAt: new Date()
      })
      .where(eq(documents.id, document.id));

    // Publish job queued event for real-time updates
    await redis.publish('job_events', JSON.stringify({
      type: 'job_queued',
      jobId,
      fileId,
      timestamp: new Date().toISOString()
    }));

    return json({
      jobId,
      status: 'queued',
      estimatedTime: getEstimatedProcessingTime(contentType, priority),
      queuePosition: await getQueuePosition(queueName, jobId),
      steps: jobPayload.steps
    });

  } catch (error) {
    console.error('Job creation error:', error);
    
    if (error instanceof z.ZodError) {
      return json({ error: 'Invalid request data', details: error.errors }, { status: 400 });
    }
    
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Get job status
export async function GET({ url }) {
  const jobId = url.searchParams.get('jobId');
  
  if (!jobId) {
    return json({ error: 'Job ID required' }, { status: 400 });
  }

  try {
    // Connect to Redis if needed
    if (!redis.isOpen) {
      await redis.connect();
    }

    // Get job status from Redis (real-time)
    const redisStatus = await redis.get(`${JOB_STATUS_KEY}${jobId}`);
    const redisProgress = await redis.get(`${JOB_PROGRESS_KEY}${jobId}`);
    const redisResult = await redis.get(`${JOB_RESULT_KEY}${jobId}`);

    // Get job record from database (persistent)
    const [job] = await db
      .select()
      .from(processingJobs)
      .where(eq(processingJobs.uuid, jobId))
      .limit(1);

    if (!job && !redisStatus) {
      return json({ error: 'Job not found' }, { status: 404 });
    }

    // Combine Redis and database info
    let status = job?.status || 'unknown';
    let progress = job?.progress || 0;
    let currentStep = job?.currentStep || 'unknown';
    let result = job?.result || null;
    let error = job?.error || null;

    // Redis has more up-to-date info for active jobs
    if (redisStatus) {
      const redisData = JSON.parse(redisStatus);
      status = redisData.status;
      currentStep = redisData.currentStep;
    }

    if (redisProgress) {
      progress = parseInt(redisProgress);
    }

    if (redisResult) {
      result = JSON.parse(redisResult);
    }

    // Get document info
    let document = null;
    if (job?.documentId) {
      const [doc] = await db
        .select()
        .from(documents)
        .where(eq(documents.id, job.documentId))
        .limit(1);
      document = doc;
    }

    return json({
      jobId,
      status,
      progress,
      currentStep,
      result,
      error,
      startedAt: job?.startedAt,
      completedAt: job?.completedAt,
      createdAt: job?.createdAt,
      document: document ? {
        id: document.id,
        uuid: document.uuid,
        filename: document.filename,
        originalName: document.originalName,
        processingStatus: document.processingStatus
      } : null
    });

  } catch (error) {
    console.error('Job status check error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Helper functions
function getEstimatedProcessingTime(contentType: string, priority: string): number {
  // Estimated processing time in seconds
  const baseTime = contentType.includes('pdf') ? 30 : 15;
  const priorityMultiplier = {
    high: 0.5,
    normal: 1.0,
    low: 2.0
  };
  
  return Math.round(baseTime * priorityMultiplier[priority]);
}

async function getQueuePosition(queueName: string, jobId: string): Promise<number> {
  try {
    const queueLength = await redis.lLen(queueName);
    
    // For simplicity, return a rough estimate
    // In production, you'd scan the queue to find exact position
    return Math.min(queueLength, 10);
  } catch (error) {
    console.warn('Queue position check failed:', error);
    return -1;
  }
}

// Cleanup function (called by cleanup worker)
export async function DELETE({ url }) {
  const jobId = url.searchParams.get('jobId');
  
  if (!jobId) {
    return json({ error: 'Job ID required' }, { status: 400 });
  }

  try {
    if (!redis.isOpen) {
      await redis.connect();
    }

    // Clean up Redis keys
    await redis.del(`${JOB_STATUS_KEY}${jobId}`);
    await redis.del(`${JOB_PROGRESS_KEY}${jobId}`);
    await redis.del(`${JOB_RESULT_KEY}${jobId}`);

    // Mark job as cleaned up in database
    await db
      .update(processingJobs)
      .set({ 
        status: 'cleaned',
        updatedAt: new Date()
      })
      .where(eq(processingJobs.uuid, jobId));

    return json({ success: true });

  } catch (error) {
    console.error('Job cleanup error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}