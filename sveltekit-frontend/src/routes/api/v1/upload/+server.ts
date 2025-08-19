/**
 * Upload API Endpoint - File Processing Service
 * Routes to: upload-service.exe:8093 (Primary) or gin-upload.exe:8207 (Alternative)
 */

import { json, error } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler
import {
productionServiceClient } from "$lib/services/productionServiceClient";
// Orphaned content: import { URL

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const userId = formData.get('userId') as string;
    const caseId = formData.get('caseId') as string;
    const metadata = formData.get('metadata') as string;

    if (!file) {
      return error(400, 'File is required');
    }

    if (!userId) {
      return error(400, 'User ID is required');
    }

    const parsedMetadata = metadata ? JSON.parse(metadata) : {};

    const result = await productionServiceClient.execute('file.upload', {
      file,
      userId,
      caseId,
      metadata: {
        originalName: file.name,
        size: file.size,
        type: file.type,
        uploadedAt: new Date().toISOString(),
        ...parsedMetadata
      }
    });

    return json({
      success: true,
      data: result,
      metadata: {
        timestamp: new Date().toISOString(),
        service: 'upload-service',
        operation: 'upload',
        fileInfo: {
          name: file.name,
          size: file.size,
          type: file.type
        }
      }
    });

  } catch (err) {
    console.error('Upload API Error:', err);
    return error(500, `Upload service unavailable: ${err instanceof Error ? err.message : 'Unknown error'}`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const fileId = url.searchParams.get('fileId');
  
  if (fileId) {
    // Get file metadata
    try {
      const result = await productionServiceClient.execute('file.metadata', { fileId });
      return json({ success: true, data: result });
    } catch (err) {
      return error(404, { message: 'File not found' });
    }
  }

  // Service health check
  try {
    const health = await productionServiceClient.checkAllServicesHealth();
    
    return json({
      service: 'upload',
      status: 'operational',
      endpoints: {
        upload: '/api/v1/upload',
        batch: '/api/v1/upload/batch',
        metadata: '/api/v1/upload/metadata'
      },
      health: {
        'upload-service': health['upload-service'] || false,
        'gin-upload': health['gin-upload'] || false
      },
      supportedTypes: [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'image/jpeg',
        'image/png'
      ],
      maxFileSize: '10MB',
      version: '1.0.0'
    });
  } catch (err) {
    return error(503, { message: 'Upload service health check failed' });
  }
};