import type { RequestHandler } from './$types';

/**
 * Evidence CRUD API for Cases - PostgreSQL + Drizzle ORM Integration
 * Demonstrates nested resource CRUD with file upload support via MinIO
 */

import { json } from '@sveltejs/kit';
import * as casesMCP from '../../../../../../lib/mcp/cases.mcp';
import { minioService } from '../../../../../../lib/server/storage/minio-service';
import type { EvidenceData } from '../../../../../../lib/mcp/cases.mcp';

// POST /api/v1/cases/[caseId]/evidence - Add evidence to case
export const POST: RequestHandler = async ({ params, request, getClientAddress }) => {
  try {
    const { caseId } = params;

    if (!caseId) {
      return json({
        success: false,
        error: 'caseId parameter is required'
      }, { status: 400 });
    }

    // Handle both JSON and FormData (for file uploads)
    const contentType = request.headers.get('content-type') || '';
    let evidenceData: Omit<EvidenceData, 'id' | 'createdAt'>;
    let uploadedFiles: any[] = [];

    if (contentType.includes('multipart/form-data')) {
      // Handle file upload with evidence metadata
      const formData = await request.formData();
      
      // Extract evidence metadata from form
      evidenceData = {
        caseId,
        content: formData.get('content')?.toString() || '',
        evidenceType: (formData.get('evidenceType')?.toString() as any) || 'document',
        source: formData.get('source')?.toString(),
        tags: formData.get('tags')?.toString().split(',').map(t => t.trim()).filter(Boolean) || [],
        confidenceLevel: parseFloat(formData.get('confidenceLevel')?.toString() || '0.8')
      };

      // Handle file uploads to MinIO
      const files = formData.getAll('files') as File[];
      for (const file of files) {
        if (file.size > 0) {
          console.log(`📁 Uploading file: ${file.name} (${file.size} bytes)`);
          
          const uploadResult = await minioService.uploadFile(file, file.name, {
            bucket: 'evidence-files',
            caseId,
            uploadedBy: 1 // TODO: Get from user session
          });

          if (uploadResult.success) {
            uploadedFiles.push({
              fileName: uploadResult.fileName,
              originalName: file.name,
              size: file.size,
              url: uploadResult.url,
              bucket: uploadResult.bucket
            });
          }
        }
      }

      // Add file references to evidence content
      if (uploadedFiles.length > 0) {
        evidenceData.content = `${evidenceData.content}\n\nAttached files: ${uploadedFiles.map(f => f.originalName).join(', ')}`;
        evidenceData.source = evidenceData.source || 'file_upload';
      }

    } else {
      // Handle JSON evidence data
      const requestData = await request.json();
      evidenceData = {
        caseId,
        content: requestData.content || '',
        evidenceType: requestData.evidenceType || 'testimony',
        source: requestData.source,
        tags: requestData.tags || [],
        confidenceLevel: requestData.confidenceLevel || 0.8
      };
    }

    // Validate required fields
    if (!evidenceData.content.trim()) {
      return json({
        success: false,
        error: 'evidence content is required'
      }, { status: 400 });
    }

    console.log(`📋 POST /api/v1/cases/${caseId}/evidence - Adding ${evidenceData.evidenceType} evidence`);

    const result = await casesMCP.addEvidence(caseId, evidenceData);

    if (!result.success) {
      return json({
        success: false,
        error: result.error || 'Failed to add evidence'
      }, { status: 400 });
    }

    const response = {
      success: true,
      data: {
        evidenceId: result.evidenceId,
        uploadedFiles,
        evidence: {
          id: result.evidenceId,
          ...evidenceData,
          createdAt: new Date()
        }
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        operation: 'add_evidence',
        filesUploaded: uploadedFiles.length
      }
    };

    console.log(`✅ Evidence added: ${result.evidenceId}`);
    if (uploadedFiles.length > 0) {
      console.log(`📁 Files uploaded: ${uploadedFiles.length}`);
    }

    return json(response);

  } catch (error: any) {
    console.error(`❌ POST /api/v1/cases/${params.caseId}/evidence error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to add evidence',
      metadata: { timestamp: Date.now() }
    }, { status: 500 });
  }
};

// GET /api/v1/cases/[caseId]/evidence - List evidence for case
export const GET: RequestHandler = async ({ params, url, getClientAddress }) => {
  try {
    const { caseId } = params;

    if (!caseId) {
      return json({
        success: false,
        error: 'caseId parameter is required'
      }, { status: 400 });
    }

    const limit = parseInt(url.searchParams.get('limit') || '50');
    const evidenceType = url.searchParams.get('type') as EvidenceData['evidenceType'] | null;

    console.log(`📋 GET /api/v1/cases/${caseId}/evidence`);

    // Load the case with evidence (this is handled by the loadCase function)
    const caseData = await casesMCP.loadCase(caseId);

    if (!caseData) {
      return json({
        success: false,
        error: 'Case not found'
      }, { status: 404 });
    }

    // Filter evidence by type if specified
    // Note: This is a simplified version. In a full implementation, 
    // you'd want a dedicated MCP tool for evidence queries
    let evidence: any[] = (caseData as any).evidence || [];
    
    if (evidenceType) {
      evidence = evidence.filter(e => e.evidenceType === evidenceType);
    }

    if (evidence.length > limit) {
      evidence = evidence.slice(0, limit);
    }

    const response = {
      success: true,
      data: {
        caseId,
        evidence,
        count: evidence.length,
        filters: {
          type: evidenceType,
          limit
        }
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress()
      }
    };

    console.log(`✅ Evidence retrieved: ${evidence.length} items`);
    return json(response);

  } catch (error: any) {
    console.error(`❌ GET /api/v1/cases/${params.caseId}/evidence error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to retrieve evidence',
      metadata: { timestamp: Date.now() }
    }, { status: 500 });
  }
};