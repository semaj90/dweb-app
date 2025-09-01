import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';


// Go Upload Service Configuration
const GO_UPLOAD_SERVICE_URL = 'http://localhost:8093';
const GO_UPLOAD_TIMEOUT = 30000; // 30 seconds

/**
 * Proxy to Go Upload Service
 * Routes file uploads to the Go microservice for processing
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    console.log('📤 Proxying upload request to Go service...');

    // Parse the incoming form data and transform it for Go service
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      throw error(400, 'No file provided');
    }

    // Create new form data for Go service (expects 'files' field)
    const goFormData = new FormData();
    goFormData.append('files', file, file.name);
    
    // Add metadata from form
    const evidenceId = formData.get('evidenceId');
    const caseId = formData.get('caseId');
    const title = formData.get('title');
    const evidenceType = formData.get('evidenceType');
    
    if (caseId) goFormData.append('case_id', caseId.toString());
    if (evidenceId) goFormData.append('user_id', evidenceId.toString()); // Map evidenceId to user_id for now
    if (title) goFormData.append('title', title.toString());
    if (evidenceType) goFormData.append('evidence_type', evidenceType.toString());

    console.log('📋 Sending to Go service:', {
      fileName: file.name,
      fileSize: file.size,
      caseId: caseId?.toString(),
      evidenceId: evidenceId?.toString()
    });

    // Forward to Go service
    const response = await fetch(`${GO_UPLOAD_SERVICE_URL}/upload`, {
      method: 'POST',
      body: goFormData,
      signal: AbortSignal.timeout(GO_UPLOAD_TIMEOUT)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Go service error:', response.status, response.statusText, errorText);
      throw error(response.status, `Upload service error: ${response.statusText}`);
    }

    const result = await response.json();
    console.log('✅ Go service response:', result);

    return json(result);

  } catch (fetchError: any) {
    console.error('❌ Failed to connect to Go upload service:', fetchError);
    
    if (fetchError.name === 'TimeoutError') {
      throw error(504, 'Upload service timeout - please try again');
    }
    
    if (fetchError.name === 'TypeError' && fetchError.message.includes('fetch')) {
      throw error(503, 'Upload service unavailable - please check if Go service is running on port 8093');
    }

    throw error(500, `Upload failed: ${fetchError.message}`);
  }
};

/**
 * Health check endpoint for Go service
 */
export const GET: RequestHandler = async () => {
  try {
    const response = await fetch(`${GO_UPLOAD_SERVICE_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5 second timeout for health check
    });

    const isHealthy = response.ok;
    const status = isHealthy ? 200 : 503;

    return json({
      service: 'go-upload-service',
      url: GO_UPLOAD_SERVICE_URL,
      status: isHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString()
    }, { status });

  } catch (healthError) {
    console.error('❌ Go upload service health check failed:', healthError);
    
    return json({
      service: 'go-upload-service',
      url: GO_UPLOAD_SERVICE_URL,
      status: 'unavailable',
      error: healthError.message,
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
};