/**
 * Reports API Routes with Lucia v3 Authentication
 * GET /api/v1/reports - List user's reports (with pagination)
 * POST /api/v1/reports - Create new report
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { ReportsCRUDService, CreateReportSchema, type CreateReportData } from '$lib/server/services/user-scoped-crud';
import { z } from 'zod';

// Query parameters schema for GET requests
const ReportsQuerySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  caseId: z.string().uuid().optional(),
  status: z.enum(['draft', 'review', 'approved', 'published']).optional(),
  reportType: z.enum(['analysis', 'summary', 'investigation', 'final']).optional()
});

/**
 * GET /api/v1/reports
 * List user's reports with pagination and filtering
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
    const validatedQuery = ReportsQuerySchema.parse(queryParams);
    
    // Create service instance
    const reportsService = new ReportsCRUDService(locals.user.id);
    
    // Get reports with pagination - filter by case if specified
    const result = validatedQuery.caseId 
      ? await reportsService.listByCase(validatedQuery.caseId, {
          page: validatedQuery.page,
          limit: validatedQuery.limit
        })
      : await reportsService.list({
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
    console.error('Error fetching reports:', err);
    
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
      message: 'Failed to fetch reports',
      code: 'FETCH_FAILED',
      details: err.message
    });
  }
};

/**
 * POST /api/v1/reports
 * Create new report
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
    const validatedData = CreateReportSchema.parse(body) as CreateReportData;
    
    // Create service instance
    const reportsService = new ReportsCRUDService(locals.user.id);
    
    // Create report
    const reportId = await reportsService.create(validatedData);
    
    // Get the created report details
    const createdReport = await reportsService.getById(reportId);
    
    return json({
      success: true,
      data: createdReport,
      meta: {
        reportId,
        userId: locals.user.id,
        caseId: validatedData.caseId,
        timestamp: new Date().toISOString()
      }
    }, { status: 201 });

  } catch (err: any) {
    console.error('Error creating report:', err);
    
    if (err instanceof z.ZodError) {
      return error(400, {
        message: 'Invalid report data',
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
      message: 'Failed to create report',
      code: 'CREATE_FAILED',
      details: err.message
    });
  }
};