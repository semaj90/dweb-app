/**
 * Individual Evidence API Routes with Lucia v3 Authentication
 * GET /api/v1/evidence/[id] - Get specific evidence
 * PUT /api/v1/evidence/[id] - Update specific evidence
 * DELETE /api/v1/evidence/[id] - Delete specific evidence
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { EvidenceCRUDService, UpdateEvidenceSchema, type UpdateEvidenceData } from '$lib/server/services/user-scoped-crud';
import { z } from 'zod';

// UUID validation schema
const UUIDSchema = z.string().uuid('Invalid evidence ID format');

/**
 * GET /api/v1/evidence/[id]
 * Get a specific evidence by ID
 */
export const GET: RequestHandler = async ({ params, locals }) => {
  try {
    // Check authentication
    if (!locals.session || !locals.user) {
      return error(401, { 
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Validate evidence ID
    const evidenceId = UUIDSchema.parse(params.id);
    
    // Create service instance
    const evidenceService = new EvidenceCRUDService(locals.user.id);
    
    // Get evidence
    const evidenceData = await evidenceService.getById(evidenceId);
    
    return json({
      success: true,
      data: evidenceData,
      meta: {
        userId: locals.user.id,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error fetching evidence:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid evidence ID',
        code: 'INVALID_ID',
        details: err.errors
      });
    }
    
    if (err.message.includes('not found') || err.message.includes('access denied')) {
      return error(404, {
        message: 'Evidence not found',
        code: 'EVIDENCE_NOT_FOUND'
      });
    }
    
    return error(500, {
      message: 'Failed to fetch evidence',
      code: 'FETCH_FAILED',
      details: err.message
    });
  }
};

/**
 * PUT /api/v1/evidence/[id]
 * Update a specific evidence
 */
export const PUT: RequestHandler = async ({ params, request, locals }) => {
  try {
    // Check authentication
    if (!locals.session || !locals.user) {
      return error(401, { 
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Validate evidence ID
    const evidenceId = UUIDSchema.parse(params.id);
    
    // Parse request body
    const body = await request.json();
    const validatedData = UpdateEvidenceSchema.parse({
      id: evidenceId,
      ...body
    }) as UpdateEvidenceData;
    
    // Create service instance
    const evidenceService = new EvidenceCRUDService(locals.user.id);
    
    // Update evidence
    await evidenceService.update(validatedData);
    
    // Get updated evidence details
    const updatedEvidence = await evidenceService.getById(evidenceId);
    
    return json({
      success: true,
      data: updatedEvidence,
      meta: {
        userId: locals.user.id,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error updating evidence:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid evidence data',
        code: 'INVALID_DATA',
        details: err.errors
      });
    }
    
    if (err.message.includes('not found') || err.message.includes('access denied')) {
      return error(404, {
        message: 'Evidence not found',
        code: 'EVIDENCE_NOT_FOUND'
      });
    }
    
    return error(500, {
      message: 'Failed to update evidence',
      code: 'UPDATE_FAILED',
      details: err.message
    });
  }
};

/**
 * DELETE /api/v1/evidence/[id]
 * Delete a specific evidence
 */
export const DELETE: RequestHandler = async ({ params, locals }) => {
  try {
    // Check authentication
    if (!locals.session || !locals.user) {
      return error(401, { 
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Validate evidence ID
    const evidenceId = UUIDSchema.parse(params.id);
    
    // Create service instance
    const evidenceService = new EvidenceCRUDService(locals.user.id);
    
    // Delete evidence
    await evidenceService.delete(evidenceId);
    
    return json({
      success: true,
      message: 'Evidence deleted successfully',
      meta: {
        deletedEvidenceId: evidenceId,
        userId: locals.user.id,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error deleting evidence:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid evidence ID',
        code: 'INVALID_ID',
        details: err.errors
      });
    }
    
    if (err.message.includes('not found') || err.message.includes('access denied')) {
      return error(404, {
        message: 'Evidence not found',
        code: 'EVIDENCE_NOT_FOUND'
      });
    }
    
    return error(500, {
      message: 'Failed to delete evidence',
      code: 'DELETE_FAILED',
      details: err.message
    });
  }
};