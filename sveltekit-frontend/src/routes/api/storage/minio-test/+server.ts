import type { RequestHandler } from '@sveltejs/kit';
/**
 * MinIO Integration Test API Endpoint
 * Tests MinIO connection and file operations
 */

import { json, type RequestHandler } from '@sveltejs/kit'

export const GET: RequestHandler = async () => {
  try {
    // Dynamic import to avoid SSR issues
    const { minioService } = await import('$lib/server/storage/minio-service.js');
    
    console.log('🧪 Testing MinIO service integration...');
    
    // Initialize MinIO service
    const initialized = await minioService.initialize();
    
    if (!initialized) {
      return json({
        success: false,
        error: 'MinIO service failed to initialize',
        timestamp: new Date().toISOString()
      }, { status: 503 });
    }
    
    // Perform health check
    const healthCheck = await minioService.healthCheck();
    
    return json({
      success: true,
      minioStatus: {
        initialized,
        health: healthCheck,
        message: 'MinIO service is working correctly'
      },
      endpoints: {
        api: 'http://localhost:9000',
        console: 'http://localhost:9001',
        credentials: {
          username: 'minioadmin',
          password: 'minioadmin (default - change in production)'
        }
      },
      buckets: {
        available: [
          'legal-documents',
          'evidence-files', 
          'image-assets',
          'thumbnails',
          'temp-uploads',
          'archives',
          'backups'
        ]
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('❌ MinIO test error:', error);
    
    return json({
      success: false,
      error: error.message || 'MinIO test failed',
      details: {
        message: 'MinIO may not be running or accessible',
        suggestion: 'Run: npm run minio:start to start MinIO server',
        endpoints: {
          health: 'http://localhost:9000/minio/health/live',
          console: 'http://localhost:9001'
        }
      },
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { testFile = false } = await request.json();
    
    // Dynamic import to avoid SSR issues
    const { minioService } = await import('$lib/server/storage/minio-service.js');
    
    console.log('🧪 Testing MinIO file upload...');
    
    if (testFile) {
      // Create a test file buffer
      const testContent = `MinIO Test File
Created at: ${new Date().toISOString()}
Test data for Legal AI platform file storage integration.
This file verifies that MinIO can store and retrieve files properly.`;
      
      const testBuffer = Buffer.from(testContent, 'utf-8');
      
      // Upload test file
      const uploadResult = await minioService.uploadFile(
        testBuffer,
        'minio-test.txt',
        {
          bucket: 'legal-documents',
          uploadedBy: 1,
          caseId: 0
        }
      );
      
      return json({
        success: true,
        uploadTest: uploadResult,
        message: 'MinIO file upload test completed successfully',
        timestamp: new Date().toISOString()
      });
    }
    
    return json({
      success: true,
      message: 'MinIO POST endpoint ready for file upload tests',
      usage: 'Send JSON with { "testFile": true } to test file upload',
      timestamp: new Date().toISOString()
    });
    
  } catch (error: any) {
    console.error('❌ MinIO upload test error:', error);
    
    return json({
      success: false,
      error: error.message || 'MinIO upload test failed',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};