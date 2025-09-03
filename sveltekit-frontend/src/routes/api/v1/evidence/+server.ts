/**
 * Evidence API Routes with Lucia v3 Authentication
 * GET /api/v1/evidence - List user's evidence (with pagination)
 * POST /api/v1/evidence - Create new evidence
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { EvidenceCRUDService, CreateEvidenceSchema, type CreateEvidenceData } from '$lib/server/services/user-scoped-crud';
import { z } from 'zod';

// Query parameters schema for GET requests
const EvidenceQuerySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  caseId: z.string().uuid().optional(),
  evidenceType: z.string().optional(),
  isPublic: z.coerce.boolean().optional()
});

/**
 * GET /api/v1/evidence
 * List user's evidence with pagination and filtering
 */
export const GET: RequestHandler = async ({ request, locals }) => {
  try {
    // Check authentication
    if (!locals.session || !locals.user) {
      return error(401, { 
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Parse query parameters
    const url = new URL(request.url);
    const queryParams = Object.fromEntries(url.searchParams.entries());
    const validatedQuery = EvidenceQuerySchema.parse(queryParams);
    
    // Create service instance
    const evidenceService = new EvidenceCRUDService(locals.user.id);
    
    // Get evidence with pagination - filter by case if specified
    const result = validatedQuery.caseId 
      ? await evidenceService.listByCase(validatedQuery.caseId, {
          page: validatedQuery.page,
          limit: validatedQuery.limit
        })
      : await evidenceService.list({
          page: validatedQuery.page,
          limit: validatedQuery.limit
        });
    
    return json({
      success: true,
      data: result.data,
      pagination: {
        page: result.page,
        limit: result.limit,
        total: result.total,
        totalPages: result.totalPages,
        hasNext: result.page < result.totalPages,
        hasPrev: result.page > 1
      },
      meta: {
        userId: locals.user.id,
        caseId: validatedQuery.caseId || null,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error fetching evidence:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid query parameters',
        code: 'INVALID_QUERY',
        details: err.errors
      });
    }
    
    if (err.message.includes('not found') || err.message.includes('access denied')) {
      return error(403, {
        message: err.message,
        code: 'ACCESS_DENIED'
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
 * POST /api/v1/evidence
 * Create new evidence
 */
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Check authentication
    if (!locals.session || !locals.user) {
      return error(401, { 
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    // Parse request body
    const body = await request.json();
    const validatedData = CreateEvidenceSchema.parse(body) as CreateEvidenceData;
    
    // Create service instance
    const evidenceService = new EvidenceCRUDService(locals.user.id);
    
    // Create evidence
    const evidenceId = await evidenceService.create(validatedData);
    
    // Get the created evidence details
    const createdEvidence = await evidenceService.getById(evidenceId);
    
    return json({
      success: true,
      data: createdEvidence,
      meta: {
        evidenceId,
        userId: locals.user.id,
        caseId: validatedData.caseId,
        timestamp: new Date().toISOString()
      }
    }, { status: 201 });

  } catch (err: any) {
    console.error('Error creating evidence:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid evidence data',
        code: 'INVALID_DATA',
        details: err.errors
      });
    }
    
    if (err.message.includes('not found') || err.message.includes('access denied')) {
      return error(403, {
        message: err.message,
        code: 'ACCESS_DENIED'
      });
    }
    
    return error(500, {
      message: 'Failed to create evidence',
      code: 'CREATE_FAILED',
      details: err.message
    });
  }
};