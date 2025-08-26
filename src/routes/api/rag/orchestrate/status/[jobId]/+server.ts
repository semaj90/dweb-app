/**
 * Job Status API Endpoint
 * 
 * Provides real-time status updates for document processing jobs
 * GET /api/rag/orchestrate/status/[jobId]
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ragCoordinator } from '$lib/orchestration/production-rag-coordinator';

export const GET: RequestHandler = async ({ params }) => {
  try {
    const { jobId } = params;
    
    if (!jobId) {
      return error(400, { message: 'Job ID is required' });
    }

    // Get job status from coordinator
    const job = await ragCoordinator.getJobStatus(jobId);
    
    if (!job) {
      return error(404, { message: `Job ${jobId} not found` });
    }

    // Calculate stage completion percentages
    const stages = Object.entries(job.stages);
    const completedStages = stages.filter(([_, stage]) => stage.status === 'completed').length;
    const failedStages = stages.filter(([_, stage]) => stage.status === 'failed').length;
    const processingStages = stages.filter(([_, stage]) => stage.status === 'processing').length;
    
    const stageDetails = stages.map(([name, stage]) => ({
      name,
      status: stage.status,
      startTime: stage.startTime,
      endTime: stage.endTime,
      processingTime: stage.processingTime,
      error: stage.error
    }));

    // Calculate estimated completion time for processing jobs
    let estimatedCompletion: number | null = null;
    if (job.status === 'processing' && completedStages > 0) {
      const totalStages = stages.length;
      const averageStageTime = Object.values(job.stages)
        .filter(stage => stage.processingTime)
        .reduce((sum, stage) => sum + (stage.processingTime || 0), 0) / completedStages;
      
      const remainingStages = totalStages - completedStages;
      estimatedCompletion = Date.now() + (remainingStages * averageStageTime);
    }

    return json({
      success: true,
      job: {
        jobId: job.jobId,
        uploadId: job.uploadId,
        caseId: job.caseId,
        filename: job.filename,
        status: job.status,
        progress: job.progress,
        startTime: job.startTime,
        endTime: job.endTime,
        processingTime: job.processingTime,
        error: job.error
      },
      stages: {
        total: stages.length,
        completed: completedStages,
        failed: failedStages,
        processing: processingStages,
        pending: stages.length - completedStages - failedStages - processingStages,
        details: stageDetails
      },
      results: job.status === 'completed' ? {
        extractedText: job.extractedText ? `${job.extractedText.substring(0, 500)}...` : null,
        textChunks: job.textChunks?.length || 0,
        embeddings: job.embeddings?.length || 0,
        vectorIds: job.vectorIds?.length || 0,
        summary: job.summary,
        keyTerms: job.keyTerms,
        legalEntities: job.legalEntities?.length || 0,
        metadata: job.metadata
      } : null,
      timing: {
        totalProcessingTime: job.processingTime,
        estimatedCompletion,
        stages: stageDetails.filter(stage => stage.processingTime).map(stage => ({
          name: stage.name,
          processingTime: stage.processingTime
        }))
      }
    });

  } catch (err) {
    console.error(`[RAG API] ❌ Failed to get job status:`, err);
    
    return error(500, {
      message: 'Failed to retrieve job status',
      error: err instanceof Error ? err.message : 'Unknown error'
    });
  }
};