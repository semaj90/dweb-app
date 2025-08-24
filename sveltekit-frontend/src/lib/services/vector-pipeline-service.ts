// src/lib/services/vector-pipeline-service.ts
// Client-side service for vector processing pipeline coordination
// Integrates with SvelteKit API endpoints and XState machines

// Types for vector processing
export interface VectorOutbox {
  id: string;
  ownerType: string;
  ownerId: string;
  event: string;
  vector: number[] | null;
  payload: any;
  attempts: number;
  processedAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface VectorJob {
  id: string;
  jobId: string;
  ownerType: string;
  ownerId: string;
  event: string;
  status: string;
  progress: number;
  error: string | null;
  result: any;
  startedAt: Date | null;
  completedAt: Date | null;
  createdAt: Date;
}
import { writable } from 'svelte/store';

export interface VectorPipelineJob {
  jobId: string;
  ownerType: 'evidence' | 'report' | 'case' | 'document';
  ownerId: string;
  event: 'upsert' | 'delete' | 'reembed';
  status: 'enqueued' | 'processing' | 'succeeded' | 'failed';
  progress: number;
  error?: string;
  result?: any;
  createdAt: string;
  estimatedTime?: number;
}

export interface PipelineMetrics {
  totalJobs: number;
  enqueuedJobs: number;
  processingJobs: number;
  succeededJobs: number;
  failedJobs: number;
  averageProcessingTime: number;
  throughputPerMinute: number;
}

// Reactive stores for pipeline state
export const pipelineJobs = writable<VectorPipelineJob[]>([]);
export const pipelineMetrics = writable<PipelineMetrics>({
  totalJobs: 0,
  enqueuedJobs: 0,
  processingJobs: 0,
  succeededJobs: 0,
  failedJobs: 0,
  averageProcessingTime: 0,
  throughputPerMinute: 0,
});

export class VectorPipelineService {
  private pollingInterval: number;
  private activePolling: Set<string> = new Set();

  constructor(pollingInterval = 1000) {
    this.pollingInterval = pollingInterval;
  }

  /**
   * Submit a new vector processing job
   */
  async submitJob(params: {
    ownerType: 'evidence' | 'report' | 'case' | 'document';
    ownerId: string;
    event: 'upsert' | 'delete' | 'reembed';
    data?: any;
    jobId?: string;
  }): Promise<VectorPipelineJob> {
    
    console.log(`🚀 Submitting vector job: ${params.event} for ${params.ownerType}:${params.ownerId}`);

    const response = await fetch('/api/compute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Failed to submit job: ${error.message || error.error}`);
    }

    const result = await response.json();
    
    const job: VectorPipelineJob = {
      jobId: result.jobId,
      ownerType: params.ownerType,
      ownerId: params.ownerId,
      event: params.event,
      status: 'enqueued',
      progress: 0,
      createdAt: new Date().toISOString(),
      estimatedTime: result.estimatedTime,
    };

    // Add to reactive store
    pipelineJobs.update(jobs => [...jobs, job]);

    // Start polling for this job
    this.startPolling(job.jobId);

    return job;
  }

  /**
   * Get job status and update reactive store
   */
  async getJobStatus(jobId: string): Promise<VectorPipelineJob | null> {
    try {
      const response = await fetch(`/api/compute?jobId=${encodeURIComponent(jobId)}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          return null; // Job not found
        }
        throw new Error(`Failed to get job status: ${response.statusText}`);
      }

      const result = await response.json();
      
      const job: VectorPipelineJob = {
        jobId: result.job.jobId,
        ownerType: result.job.ownerType || 'unknown',
        ownerId: result.job.ownerId || 'unknown',
        event: result.job.event || 'upsert',
        status: result.job.status,
        progress: result.job.progress,
        error: result.job.error,
        result: result.job.result,
        createdAt: result.job.createdAt,
      };

      // Update reactive store
      pipelineJobs.update(jobs => {
        const index = jobs.findIndex(j => j.jobId === jobId);
        if (index >= 0) {
          jobs[index] = job;
        } else {
          jobs.push(job);
        }
        return jobs;
      });

      return job;

    } catch (error) {
      console.error(`❌ Failed to get job status for ${jobId}:`, error);
      return null;
    }
  }

  /**
   * Start polling for a specific job
   */
  private startPolling(jobId: string) {
    if (this.activePolling.has(jobId)) {
      return; // Already polling this job
    }

    this.activePolling.add(jobId);

    const poll = async () => {
      try {
        const job = await this.getJobStatus(jobId);
        
        if (job && (job.status === 'succeeded' || job.status === 'failed')) {
          // Job completed, stop polling
          this.activePolling.delete(jobId);
          console.log(`✅ Job ${jobId} completed with status: ${job.status}`);
          
          // Update metrics
          this.updateMetrics();
          return;
        }

        // Continue polling if job is still processing
        if (this.activePolling.has(jobId)) {
          setTimeout(poll, this.pollingInterval);
        }

      } catch (error) {
        console.error(`❌ Polling error for job ${jobId}:`, error);
        this.activePolling.delete(jobId);
      }
    };

    // Start initial poll
    setTimeout(poll, this.pollingInterval);
  }

  /**
   * Stop polling for a specific job
   */
  stopPolling(jobId: string) {
    this.activePolling.delete(jobId);
  }

  /**
   * Stop all polling
   */
  stopAllPolling() {
    this.activePolling.clear();
  }

  /**
   * Update pipeline metrics
   */
  private updateMetrics() {
    pipelineJobs.subscribe(jobs => {
      const metrics: PipelineMetrics = {
        totalJobs: jobs.length,
        enqueuedJobs: jobs.filter(j => j.status === 'enqueued').length,
        processingJobs: jobs.filter(j => j.status === 'processing').length,
        succeededJobs: jobs.filter(j => j.status === 'succeeded').length,
        failedJobs: jobs.filter(j => j.status === 'failed').length,
        averageProcessingTime: 0,
        throughputPerMinute: 0,
      };

      // Calculate average processing time for completed jobs
      const completedJobs = jobs.filter(j => j.status === 'succeeded' || j.status === 'failed');
      if (completedJobs.length > 0) {
        const totalTime = completedJobs.reduce((sum, job) => {
          const created = new Date(job.createdAt).getTime();
          const estimated = job.estimatedTime || 1000;
          return sum + estimated;
        }, 0);
        
        metrics.averageProcessingTime = totalTime / completedJobs.length;
      }

      // Calculate throughput (completed jobs in last minute)
      const oneMinuteAgo = Date.now() - 60 * 1000;
      const recentJobs = completedJobs.filter(job => 
        new Date(job.createdAt).getTime() > oneMinuteAgo
      );
      metrics.throughputPerMinute = recentJobs.length;

      pipelineMetrics.set(metrics);
    })();
  }

  /**
   * Convenience methods for common operations
   */
  async upsertEvidence(evidenceId: string, data?: any) {
    return this.submitJob({
      ownerType: 'evidence',
      ownerId: evidenceId,
      event: 'upsert',
      data,
    });
  }

  async upsertReport(reportId: string, data?: any) {
    return this.submitJob({
      ownerType: 'report',
      ownerId: reportId,
      event: 'upsert',
      data,
    });
  }

  async reembedEvidence(evidenceId: string) {
    return this.submitJob({
      ownerType: 'evidence',
      ownerId: evidenceId,
      event: 'reembed',
    });
  }

  async deleteVector(ownerType: 'evidence' | 'report' | 'case' | 'document', ownerId: string) {
    return this.submitJob({
      ownerType,
      ownerId,
      event: 'delete',
    });
  }

  /**
   * Batch operations
   */
  async submitBatch(jobs: Array<{
    ownerType: 'evidence' | 'report' | 'case' | 'document';
    ownerId: string;
    event: 'upsert' | 'delete' | 'reembed';
    data?: any;
  }>) {
    const results = await Promise.allSettled(
      jobs.map(job => this.submitJob(job))
    );

    return results.map((result, index) => ({
      job: jobs[index],
      success: result.status === 'fulfilled',
      result: result.status === 'fulfilled' ? result.value : result.reason,
    }));
  }

  /**
   * Health check for the pipeline services
   */
  async checkHealth() {
    try {
      const response = await fetch('/api/vectors/sync', { method: 'GET' });
      return response.ok ? await response.json() : { success: false };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Health check failed' 
      };
    }
  }
}

// Export singleton instance
export const vectorPipeline = new VectorPipelineService();

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    vectorPipeline.stopAllPolling();
  });
}