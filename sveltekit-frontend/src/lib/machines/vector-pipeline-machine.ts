// src/lib/machines/vector-pipeline-machine.ts
// XState v5 machine for coordinating vector processing pipeline
// Orchestrates PostgreSQL → Redis Streams → Go microservice → CUDA worker → Qdrant

import { setup, assign, createActor, fromPromise } from 'xstate';
import { writable } from 'svelte/store';

// Mock implementations for undefined variables
const webgpuWASM = {
  getStatus: () => ({ capabilities: { webgpuSupported: true } })
};

const vectorPipeline = {
  submitJob: async (job: any) => ({ jobId: `job_${Date.now()}`, status: 'enqueued' })
};

// Add loadModel method to webgpuWASM
const webgpuWASMExtended = {
  ...webgpuWASM,
  loadModel: async () => ({ modelLoaded: true, status: 'ready' })
};
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

export interface VectorPipelineContext {
  // Current job being processed
  currentJob: VectorPipelineJob | null;
  
  // Batch processing
  batch: {
    jobs: VectorPipelineJob[];
    totalJobs: number;
    completedJobs: number;
    failedJobs: number;
    progress: number;
  };

  // Pipeline status
  pipeline: {
    postgresql: boolean;
    redis: boolean;
    goMicroservice: boolean;
    cudaWorker: boolean;
    qdrant: boolean;
    webgpu: boolean;
  };

  // Error tracking
  errors: string[];
  retryAttempts: number;
  maxRetries: number;

  // Performance metrics
  metrics: {
    averageProcessingTime: number;
    totalJobsProcessed: number;
    throughputPerMinute: number;
    lastProcessedAt: Date | null;
  };
}

export type VectorPipelineEvent =
  | { type: 'SUBMIT_JOB'; job: Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'> }
  | { type: 'SUBMIT_BATCH'; jobs: Array<Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'>> }
  | { type: 'JOB_PROGRESS'; jobId: string; progress: number; status: string }
  | { type: 'JOB_COMPLETED'; jobId: string; result: any }
  | { type: 'JOB_FAILED'; jobId: string; error: string }
  | { type: 'RETRY_FAILED_JOBS' }
  | { type: 'HEALTH_CHECK' }
  | { type: 'RESET_PIPELINE' }
  | { type: 'ENABLE_WEBGPU' }
  | { type: 'DISABLE_WEBGPU' };

const initialContext: VectorPipelineContext = {
  currentJob: null,
  batch: {
    jobs: [],
    totalJobs: 0,
    completedJobs: 0,
    failedJobs: 0,
    progress: 0,
  },
  pipeline: {
    postgresql: false,
    redis: false,
    goMicroservice: false,
    cudaWorker: false,
    qdrant: false,
    webgpu: false,
  },
  errors: [],
  retryAttempts: 0,
  maxRetries: 3,
  metrics: {
    averageProcessingTime: 0,
    totalJobsProcessed: 0,
    throughputPerMinute: 0,
    lastProcessedAt: null,
  },
};

export const vectorPipelineMachine = setup({
  types: {} as {
    context: VectorPipelineContext;
    events: VectorPipelineEvent;
  },
  actions: {
    // Job management actions
    setCurrentJob: assign({
      currentJob: ({ event }) => (event as any).job || null,
    }),

    addJobToBatch: assign({
      batch: ({ context, event }) => {
        const jobs = (event as any).jobs || [(event as any).job];
        return {
          ...context.batch,
          jobs: [...context.batch.jobs, ...jobs],
          totalJobs: context.batch.totalJobs + jobs.length,
        };
      },
    }),

    updateJobProgress: assign({
      batch: ({ context, event }) => {
        const { jobId, progress, status } = event as any;
        const jobs = context.batch.jobs.map(job => 
          job.jobId === jobId ? { ...job, progress, status } : job
        );
        
        const completedJobs = jobs.filter(j => j.status === 'succeeded').length;
        const failedJobs = jobs.filter(j => j.status === 'failed').length;
        const overallProgress = jobs.length > 0 
          ? Math.floor((completedJobs + failedJobs) / jobs.length * 100)
          : 0;

        return {
          ...context.batch,
          jobs,
          completedJobs,
          failedJobs,
          progress: overallProgress,
        };
      },
    }),

    completeJob: assign({
      batch: ({ context, event }) => {
        const { jobId, result } = event as any;
        const jobs = context.batch.jobs.map(job => 
          job.jobId === jobId ? { ...job, status: 'succeeded' as const, result } : job
        ) as VectorPipelineJob[];

        return {
          ...context.batch,
          jobs,
          completedJobs: context.batch.completedJobs + 1,
        };
      },
      metrics: ({ context }) => ({
        ...context.metrics,
        totalJobsProcessed: context.metrics.totalJobsProcessed + 1,
        lastProcessedAt: new Date(),
      }),
    }),

    failJob: assign({
      batch: ({ context, event }) => {
        const { jobId, error } = event as any;
        const jobs = context.batch.jobs.map(job => 
          job.jobId === jobId ? { ...job, status: 'failed' as const, error } : job
        ) as VectorPipelineJob[];

        return {
          ...context.batch,
          jobs,
          failedJobs: context.batch.failedJobs + 1,
        };
      },
      errors: ({ context, event }) => [...context.errors, (event as any).error],
    }),

    // Pipeline status actions
    updatePipelineStatus: assign({
      pipeline: ({ event }) => (event as any).status || {},
    }),

    enableWebGPU: assign({
      pipeline: ({ context }) => ({
        ...context.pipeline,
        webgpu: true,
      }),
    }),

    disableWebGPU: assign({
      pipeline: ({ context }) => ({
        ...context.pipeline,
        webgpu: false,
      }),
    }),

    // Error handling actions
    incrementRetry: assign({
      retryAttempts: ({ context }) => context.retryAttempts + 1,
    }),

    resetRetries: assign({
      retryAttempts: () => 0,
    }),

    clearErrors: assign({
      errors: () => [],
    }),

    // Reset actions
    resetBatch: assign({
      batch: () => ({
        jobs: [],
        totalJobs: 0,
        completedJobs: 0,
        failedJobs: 0,
        progress: 0,
      }),
    }),

    resetPipeline: assign(() => initialContext),

    // Metrics actions
    updateMetrics: assign({
      metrics: ({ context }) => {
        const completedJobs = context.batch.jobs.filter(j => j.status === 'succeeded');
        const averageTime = completedJobs.length > 0
          ? completedJobs.reduce((sum, job) => sum + (job.estimatedTime || 1000), 0) / completedJobs.length
          : context.metrics.averageProcessingTime;

        // Calculate throughput (jobs per minute)
        const now = Date.now();
        const oneMinuteAgo = now - 60 * 1000;
        const recentJobs = completedJobs.filter(job => 
          new Date(job.createdAt).getTime() > oneMinuteAgo
        );

        return {
          ...context.metrics,
          averageProcessingTime: averageTime,
          throughputPerMinute: recentJobs.length,
        };
      },
    }),
  },

  guards: {
    hasFailedJobs: ({ context }) => context.batch.failedJobs > 0,
    canRetry: ({ context }) => context.retryAttempts < context.maxRetries,
    allJobsCompleted: ({ context }) => 
      context.batch.totalJobs > 0 && 
      (context.batch.completedJobs + context.batch.failedJobs) >= context.batch.totalJobs,
    webgpuAvailable: () => webgpuWASM.getStatus().capabilities.webgpuSupported,
    pipelineHealthy: ({ context }) => 
      context.pipeline.postgresql && 
      context.pipeline.redis && 
      context.pipeline.goMicroservice,
  },

  actors: {
    submitJob: fromPromise(async ({ input }: { input: any }) => {
      const { job } = input;
      // Mock implementation for now
      return {
        jobId: `job_${Date.now()}`,
        status: 'enqueued',
        progress: 0,
      };
    }),

    submitBatch: fromPromise(async ({ input }: { input: any }) => {
      const { jobs } = input;
      // Mock implementation for now
      return jobs.map((job: any, index: number) => ({
        jobId: `batch_job_${Date.now()}_${index}`,
        status: 'enqueued',
        progress: 0,
      }));
    }),

    healthCheck: fromPromise(async () => {
      // Mock health check - in real implementation would check services
      return {
        postgresql: true,
        redis: true,
        goMicroservice: true,
        cudaWorker: true,
        qdrant: true,
        webgpu: false,
      };
    }),

    retryFailedJobs: fromPromise(async ({ input }: { input: any }) => {
      const { failedJobs } = input;
      
      const retryResults = await Promise.allSettled(
        failedJobs.map(async (job: VectorPipelineJob) => {
          // Reset job for retry
          return await vectorPipeline.submitJob({
            ownerType: job.ownerType,
            ownerId: job.ownerId,
            event: job.event,
          });
        })
      );

      return retryResults.map((result, index) => ({
        job: failedJobs[index],
        success: result.status === 'fulfilled',
        result: result.status === 'fulfilled' ? result.value : result.reason,
      }));
    }),

    loadWebGPUModel: fromPromise(async () => {
      return await webgpuWASMExtended.loadModel();
    }),
  },
}).createMachine({
  id: 'vectorPipeline',
  initial: 'initializing',
  context: initialContext,

  states: {
    initializing: {
      invoke: {
        src: 'healthCheck',
        onDone: {
          target: 'idle',
          actions: ['updatePipelineStatus', 'clearErrors'],
        },
        onError: {
          target: 'error',
          actions: ['failJob'],
        },
      },
    },

    idle: {
      on: {
        SUBMIT_JOB: {
          target: 'processingJob',
          actions: ['setCurrentJob'],
        },
        SUBMIT_BATCH: {
          target: 'processingBatch',
          actions: ['addJobToBatch'],
        },
        HEALTH_CHECK: 'healthCheck',
        ENABLE_WEBGPU: {
          target: 'loadingWebGPU',
          actions: ['enableWebGPU'],
          guard: 'webgpuAvailable',
        },
        DISABLE_WEBGPU: {
          actions: ['disableWebGPU'],
        },
        RESET_PIPELINE: {
          target: 'initializing',
          actions: ['resetPipeline'],
        },
      },
    },

    processingJob: {
      invoke: {
        src: 'submitJob',
        input: ({ context }) => ({ job: context.currentJob }),
        onDone: {
          target: 'idle',
          actions: ['completeJob', 'updateMetrics'],
        },
        onError: [
          {
            target: 'retrying',
            guard: 'canRetry',
            actions: ['failJob', 'incrementRetry'],
          },
          {
            target: 'idle',
            actions: ['failJob', 'resetRetries'],
          },
        ],
      },
      on: {
        JOB_PROGRESS: {
          actions: ['updateJobProgress'],
        },
      },
    },

    processingBatch: {
      invoke: {
        src: 'submitBatch',
        input: ({ context }) => ({ jobs: context.batch.jobs }),
        onDone: [
          {
            target: 'idle',
            guard: 'allJobsCompleted',
            actions: ['updateMetrics', 'resetBatch'],
          },
          {
            target: 'retrying',
            guard: 'hasFailedJobs',
            actions: ['updateMetrics'],
          },
          {
            target: 'idle',
            actions: ['updateMetrics', 'resetBatch'],
          },
        ],
        onError: {
          target: 'error',
          actions: ['failJob'],
        },
      },
      on: {
        JOB_PROGRESS: {
          actions: ['updateJobProgress'],
        },
        JOB_COMPLETED: {
          actions: ['completeJob'],
        },
        JOB_FAILED: {
          actions: ['failJob'],
        },
      },
    },

    retrying: {
      invoke: {
        src: 'retryFailedJobs',
        input: ({ context }) => ({
          failedJobs: context.batch.jobs.filter(j => j.status === 'failed'),
        }),
        onDone: {
          target: 'idle',
          actions: ['resetRetries', 'clearErrors'],
        },
        onError: {
          target: 'error',
          actions: ['incrementRetry'],
        },
      },
      on: {
        RETRY_FAILED_JOBS: {
          target: 'retrying',
          reenter: true,
        },
      },
    },

    loadingWebGPU: {
      invoke: {
        src: 'loadWebGPUModel',
        onDone: {
          target: 'idle',
          actions: ['enableWebGPU'],
        },
        onError: {
          target: 'idle',
          actions: ['disableWebGPU'],
        },
      },
    },

    healthCheck: {
      invoke: {
        src: 'healthCheck',
        onDone: {
          target: 'idle',
          actions: ['updatePipelineStatus'],
        },
        onError: {
          target: 'error',
          actions: ['failJob'],
        },
      },
    },

    error: {
      on: {
        RETRY_FAILED_JOBS: {
          target: 'retrying',
          guard: 'canRetry',
        },
        HEALTH_CHECK: 'healthCheck',
        RESET_PIPELINE: {
          target: 'initializing',
          actions: ['resetPipeline'],
        },
      },
      after: {
        5000: {
          target: 'healthCheck',
          actions: ['clearErrors'],
        },
      },
    },
  },
});

// Create and export the actor
export const vectorPipelineActor = createActor(vectorPipelineMachine);
;
// Create Svelte store for reactive state
export const vectorPipelineState = writable(vectorPipelineActor.getSnapshot());
;
// Update store when state changes
vectorPipelineActor.subscribe((snapshot) => {
  vectorPipelineState.set(snapshot);
});

// Start the actor
vectorPipelineActor.start();

// Convenience functions for common operations
export const vectorPipelineActions = {
  submitJob: (job: Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'>) => {
    vectorPipelineActor.send({ type: 'SUBMIT_JOB', job });
  },

  submitBatch: (jobs: Array<Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'>>) => {
    vectorPipelineActor.send({ type: 'SUBMIT_BATCH', jobs });
  },

  retryFailedJobs: () => {
    vectorPipelineActor.send({ type: 'RETRY_FAILED_JOBS' });
  },

  healthCheck: () => {
    vectorPipelineActor.send({ type: 'HEALTH_CHECK' });
  },

  enableWebGPU: () => {
    vectorPipelineActor.send({ type: 'ENABLE_WEBGPU' });
  },

  disableWebGPU: () => {
    vectorPipelineActor.send({ type: 'DISABLE_WEBGPU' });
  },

  reset: () => {
    vectorPipelineActor.send({ type: 'RESET_PIPELINE' });
  },
};

export default vectorPipelineActor;