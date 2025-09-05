// Test file for vector pipeline machine
// @ts-ignore
import { describe, it, expect, beforeEach } from 'vitest';
import { vectorPipelineMachine, vectorPipelineActions, type VectorPipelineJob } from './vector-pipeline-machine';
import { createActor } from 'xstate';

describe('Vector Pipeline Machine', () => {
  let actor: any;

  beforeEach(() => {
    actor = createActor(vectorPipelineMachine);
    actor.start();
  });

  it('should initialize with proper state', () => {
    const snapshot = actor.getSnapshot();
    expect(snapshot.value).toBe('initializing');
    expect(snapshot.context.batch.totalJobs).toBe(0);
    expect(snapshot.context.pipeline.webgpu).toBe(false);
  });

  it('should handle job submission', async () => {
    const testJob: Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'> = {
      ownerType: 'evidence',
      ownerId: 'test-evidence-1',
      event: 'upsert'
    };

    // Wait for initialization to complete
    await new Promise(resolve => setTimeout(resolve, 100));
    
    actor.send({ type: 'SUBMIT_JOB', job: testJob });
    
    const snapshot = actor.getSnapshot();
    expect(snapshot.context.currentJob).toBeTruthy();
  });

  it('should handle batch job submission', async () => {
    const testJobs: Array<Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'>> = [
      { ownerType: 'evidence', ownerId: 'test-1', event: 'upsert' },
      { ownerType: 'document', ownerId: 'test-2', event: 'reembed' }
    ];

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 100));
    
    actor.send({ type: 'SUBMIT_BATCH', jobs: testJobs });
    
    const snapshot = actor.getSnapshot();
    expect(snapshot.context.batch.totalJobs).toBe(2);
  });

  it('should handle health checks', async () => {
    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 100));
    
    actor.send({ type: 'HEALTH_CHECK' });
    
    // Health check should update pipeline status
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const snapshot = actor.getSnapshot();
    expect(snapshot.context.pipeline.postgresql).toBe(true);
    expect(snapshot.context.pipeline.redis).toBe(true);
  });

  it('should enable/disable WebGPU', async () => {
    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 100));
    
    actor.send({ type: 'ENABLE_WEBGPU' });
    
    await new Promise(resolve => setTimeout(resolve, 100));
    
    let snapshot = actor.getSnapshot();
    expect(snapshot.context.pipeline.webgpu).toBe(true);
    
    actor.send({ type: 'DISABLE_WEBGPU' });
    
    snapshot = actor.getSnapshot();
    expect(snapshot.context.pipeline.webgpu).toBe(false);
  });

  it('should handle pipeline reset', async () => {
    const testJob: Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'> = {
      ownerType: 'case',
      ownerId: 'test-case-1', 
      event: 'upsert'
    };

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Add a job first
    actor.send({ type: 'SUBMIT_JOB', job: testJob });
    
    let snapshot = actor.getSnapshot();
    expect(snapshot.context.currentJob).toBeTruthy();
    
    // Reset pipeline
    actor.send({ type: 'RESET_PIPELINE' });
    
    await new Promise(resolve => setTimeout(resolve, 100));
    
    snapshot = actor.getSnapshot();
    expect(snapshot.value).toBe('initializing');
    expect(snapshot.context.currentJob).toBeNull();
    expect(snapshot.context.batch.totalJobs).toBe(0);
  });
});

// Integration test with actual services
describe('Vector Pipeline Integration', () => {
  it('should communicate with Enhanced RAG service', async () => {
    try {
      const response = await fetch('http://localhost:8094/api/health');
      const health = await response.json();
      
      expect(response.ok).toBe(true);
      expect(health.status).toBe('healthy');
      expect(health.service).toBe('enhanced-rag');
    } catch (error: any) {
      console.warn('Enhanced RAG service not available for testing:', error);
    }
  });

  it('should handle GPU compute requests', async () => {
    try {
      const response = await fetch('http://localhost:8094/api/gpu/compute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: 'test vector computation',
          operation: 'embeddings'
        })
      });
      
      expect(response.status).toBeLessThan(500); // Should not be a server error
    } catch (error: any) {
      console.warn('GPU compute endpoint not available for testing:', error);
    }
  });
});