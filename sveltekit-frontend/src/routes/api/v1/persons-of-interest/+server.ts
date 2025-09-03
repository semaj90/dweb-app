/**
 * Persons of Interest API Routes with Lucia v3 Authentication
 * GET /api/v1/persons-of-interest - List user's persons of interest (with pagination)
 * POST /api/v1/persons-of-interest - Create new person of interest
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { PersonsOfInterestCRUDService, CreatePersonOfInterestSchema, type CreatePersonOfInterestData } from '$lib/server/services/user-scoped-crud';
import { z } from 'zod';

// Query parameters schema for GET requests
const PersonsOfInterestQuerySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  riskLevel: z.enum(['low', 'medium', 'high', 'critical']).optional(),
  status: z.enum(['active', 'inactive', 'archived']).optional(),
  search: z.string().optional()
});

/**
 * GET /api/v1/persons-of-interest
 * List user's persons of interest with pagination and filtering
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
    const validatedQuery = PersonsOfInterestQuerySchema.parse(queryParams);
    
    // Create service instance
    const personsService = new PersonsOfInterestCRUDService(locals.user.id);
    
    // Get persons of interest with pagination - filter by risk level if specified
    const result = validatedQuery.riskLevel 
      ? await personsService.listByRiskLevel(validatedQuery.riskLevel, {
          page: validatedQuery.page,
          limit: validatedQuery.limit
        })
      : await personsService.list({
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
        riskLevel: validatedQuery.riskLevel || null,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('Error fetching persons of interest:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid query parameters',
        code: 'INVALID_QUERY',
        details: err.errors
      });
    }
    
    return error(500, {
      message: 'Failed to fetch persons of interest',
      code: 'FETCH_FAILED',
      details: err.message
    });
  }
};

/**
 * POST /api/v1/persons-of-interest
 * Create new person of interest
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
    const validatedData = CreatePersonOfInterestSchema.parse(body) as CreatePersonOfInterestData;
    
    // Create service instance
    const personsService = new PersonsOfInterestCRUDService(locals.user.id);
    
    // Create person of interest
    const personId = await personsService.create(validatedData);
    
    // Get the created person details
    const createdPerson = await personsService.getById(personId);
    
    return json({
      success: true,
      data: createdPerson,
      meta: {
        personId,
        userId: locals.user.id,
        caseIds: validatedData.caseIds,
        timestamp: new Date().toISOString()
      }
    }, { status: 201 });

  } catch (err: any) {
    console.error('Error creating person of interest:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid person data',
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
      message: 'Failed to create person of interest',
      code: 'CREATE_FAILED',
      details: err.message
    });
  }
};