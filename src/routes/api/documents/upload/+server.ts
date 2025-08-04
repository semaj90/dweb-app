import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { aiPipeline } from '$lib/ai/processing-pipeline.js';

/**
 * Document Upload API Endpoint
 * Handles file upload and AI processing pipeline
 */

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return json({ error: 'No file provided' }, { status: 400 });
    }

    // Validate file
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      return json({ error: 'File too large' }, { status: 400 });
    }

    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'application/json'
    ];

    if (!allowedTypes.includes(file.type)) {
      return json({ error: 'Unsupported file type' }, { status: 400 });
    }

    // Get processing options
    const options = {
      includeEmbeddings: formData.get('includeEmbeddings') !== 'false',
      includeSummary: formData.get('includeSummary') !== 'false',
      includeEntities: formData.get('includeEntities') !== 'false',
      includeRiskAnalysis: formData.get('includeRiskAnalysis') !== 'false',
      cacheResults: formData.get('cacheResults') !== 'false',
      priority: (formData.get('priority') as any) || 'medium'
    };

    // Create document upload object
    const upload = {
      file,
      filename: file.name,
      mimeType: file.type,
      metadata: {
        originalName: file.name,
        size: file.size,
        uploadedAt: new Date().toISOString(),
        userAgent: request.headers.get('user-agent') || 'unknown'
      }
    };

    // Process document through AI pipeline
    const result = await aiPipeline.processDocument(upload, options);

    // Return processing result
    return json({
      success: true,
      jobId: result.id,
      status: result.status,
      message: 'Document processing initiated',
      result: result.status === 'completed' ? result.result : undefined,
      metadata: {
        processingTime: result.metadata.processingTime,
        stage: result.metadata.stage,
        filename: file.name,
        size: file.size,
        type: file.type
      }
    });

  } catch (error) {
    console.error('Document upload error:', error);
    
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Document processing failed',
      details: process.env.NODE_ENV === 'development' ? error : undefined
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const jobId = url.searchParams.get('jobId');
    
    if (!jobId) {
      return json({ error: 'Job ID required' }, { status: 400 });
    }

    // Get processing status
    const status = aiPipeline.getProcessingStatus(jobId);
    
    if (!status) {
      return json({ error: 'Job not found' }, { status: 404 });
    }

    return json({
      success: true,
      jobId,
      status: status.status,
      result: status.result,
      error: status.error,
      metadata: status.metadata
    });

  } catch (error) {
    console.error('Status check error:', error);
    
    return json({
      success: false,
      error: 'Failed to get processing status'
    }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const jobId = url.searchParams.get('jobId');
    
    if (!jobId) {
      return json({ error: 'Job ID required' }, { status: 400 });
    }

    // Cancel processing
    const cancelled = aiPipeline.cancelProcessing(jobId);
    
    return json({
      success: true,
      cancelled,
      message: cancelled ? 'Processing cancelled' : 'Job not found or already completed'
    });

  } catch (error) {
    console.error('Cancel processing error:', error);
    
    return json({
      success: false,
      error: 'Failed to cancel processing'
    }, { status: 500 });
  }
};