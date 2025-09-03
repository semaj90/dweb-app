/**
 * Cases API Routes with Lucia v3 Authentication
 * GET /api/v1/cases - List user's cases (with pagination)
 * POST /api/v1/cases - Create new case
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { CasesCRUDService, CreateCaseSchema, type CreateCaseData } from '$lib/server/services/user-scoped-crud';
import { z } from 'zod';

// Query parameters schema for GET requests
const CasesQuerySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  sortBy: z.enum(['title', 'created_at', 'updated_at', 'status', 'priority']).default('created_at'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
  status: z.enum(['open', 'closed', 'pending', 'archived']).optional(),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).optional()
});

/**
 * GET /api/v1/cases
 * List user's cases with pagination and filtering
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
    
    const validatedQuery = CasesQuerySchema.parse(queryParams);
    
    // Create service instance
    const casesService = new CasesCRUDService(locals.user.id);
    
    // Get cases with pagination
    const result = await casesService.list({
      page: validatedQuery.page,
      limit: validatedQuery.limit,
      sortBy: validatedQuery.sortBy,
      sortOrder: validatedQuery.sortOrder
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
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error fetching cases:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid query parameters',
        code: 'INVALID_QUERY',
        details: err.errors
      });
    }
    
    return error(500, {
      message: 'Failed to fetch cases',
      code: 'FETCH_FAILED',
      details: err.message
    });
  }
};

/**
 * POST /api/v1/cases
 * Create a new case
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
    const validatedData = CreateCaseSchema.parse(body) as CreateCaseData;
    
    // Create service instance
    const casesService = new CasesCRUDService(locals.user.id);
    
    // Create case
    const caseId = await casesService.create(validatedData);
    
    // Get the created case details
    const createdCase = await casesService.getById(caseId);
    
    return json({
      success: true,
      data: createdCase,
      meta: {
        caseId,
        userId: locals.user.id,
        timestamp: new Date().toISOString()
      }
    }, { status: 201 });

  } catch (err: any) {
    console.error('Error creating case:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid case data',
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
      message: 'Failed to create case',
      code: 'CREATE_FAILED',
      details: err.message
    });
  }
};