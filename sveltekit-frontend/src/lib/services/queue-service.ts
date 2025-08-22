import { Queue } from "bullmq";
import Redis from "ioredis";


// Redis connection
const redis = new Redis({
  host: import.meta.env.REDIS_HOST || 'localhost',
  port: parseInt(import.meta.env.REDIS_PORT || '6379'),
  maxRetriesPerRequest: null,
  enableReadyCheck: false
});

// Document processing queue
export const documentQueue = new Queue('document-processing', {
  connection: redis,
  defaultJobOptions: {
    removeOnComplete: 100,
    removeOnFail: 50,
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000,
    }
  }
});

// Job types
export interface DocumentProcessingJobData {
  documentId: string;
  content: string;
  documentType: string;
  caseId?: string;
  filePath?: string;
  options: {
    extractEntities?: boolean;
    generateSummary?: boolean;
    assessRisk?: boolean;
    generateEmbedding?: boolean;
    storeInDatabase?: boolean;
    useGemma3Legal?: boolean;
  };
}

export interface DocumentProcessingJobResult {
  success: boolean;
  documentId: string;
  processingTime: string;
  summary?: string;
  entities?: Array<{
    type: string;
    value: string;
    confidence: number;
    start_pos: number;
    end_pos: number;
  }>;
  riskAssessment?: {
    overall_risk: string;
    risk_score: number;
    risk_factors: string[];
    recommendations: string[];
    confidence: number;
  };
  hasEmbedding: boolean;
  error?: string;
}

/**
 * Add document processing job to queue
 */
export async function queueDocumentProcessing(
  data: DocumentProcessingJobData,
  priority = 0
): Promise<{ jobId: string; estimated: number }> {
  const job = await documentQueue.add('process-document', data, {
    priority, // Higher numbers = higher priority
    delay: 0
  });

  // Get queue metrics for estimation
  const waiting = await documentQueue.getWaiting();
  const active = await documentQueue.getActive();
  
  // Rough estimation: 30 seconds per job + queue delay
  const estimatedSeconds = (waiting.length * 30) + (active.length > 0 ? 15 : 0);

  return {
    jobId: job.id as string,
    estimated: estimatedSeconds
  };
}

/**
 * Get job status and result
 */
export async function getJobStatus(jobId: string) {
  const job = await documentQueue.getJob(jobId);
  
  if (!job) {
    return { status: 'not_found', error: 'Job not found' };
  }

  const state = await job.getState();
  const progress = job.progress;

  if (state === 'completed') {
    return {
      status: 'completed',
      progress: 100,
      result: job.returnvalue as DocumentProcessingJobResult
    };
  }

  if (state === 'failed') {
    return {
      status: 'failed',
      progress: progress || 0,
      error: job.failedReason
    };
  }

  return {
    status: state, // 'waiting', 'active', 'delayed'
    progress: progress || 0,
    data: job.data as DocumentProcessingJobData
  };
}

/**
 * Get queue statistics
 */
export async function getQueueStats() {
  const [waiting, active, completed, failed] = await Promise.all([
    documentQueue.getWaiting(),
    documentQueue.getActive(), 
    documentQueue.getCompleted(),
    documentQueue.getFailed()
  ]);

  return {
    waiting: waiting.length,
    active: active.length,
    completed: completed.length,
    failed: failed.length,
    total: waiting.length + active.length + completed.length + failed.length
  };
}

/**
 * Cancel a job
 */
export async function cancelJob(jobId: string): Promise<boolean> {
  try {
    const job = await documentQueue.getJob(jobId);
    if (job) {
      await job.remove();
      return true;
    }
    return false;
  } catch (error) {
    console.error('Error canceling job:', error);
    return false;
  }
}

/**
 * Clear completed jobs (maintenance)
 */
export async function clearCompletedJobs(): Promise<number> {
  return (await documentQueue.clean(24 * 60 * 60 * 1000, 100, 'completed')).length; // 24 hours
}

/**
 * Graceful shutdown
 */
export async function closeQueue(): Promise<void> {
  await documentQueue.close();
  await redis.quit();
}