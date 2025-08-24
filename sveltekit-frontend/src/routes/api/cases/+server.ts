import type { RequestHandler } from '@sveltejs/kit';
import { z } from "zod";
import { withApiHandler, parseRequestBody, apiSuccess, validationError, createPagination, CommonErrors } from '$lib/server/api/response';
import { CaseOperations } from '$lib/server/db/enhanced-operations';
import type { Case } from '$lib/server/db/schema-postgres';

// Enhanced case schemas with comprehensive validation
const createCaseSchema = z.object({
  title: z.string().min(1, "Case title is required").max(500, "Case title too long"),
  description: z.string().optional(),
  priority: z.enum(["low", "medium", "high", "critical"]).default("medium"),
  status: z.enum(["open", "investigating", "pending", "closed", "archived"]).default("open"),
  incidentDate: z.string().datetime().optional().transform(str => str ? new Date(str) : undefined),
  location: z.string().optional(),
  jurisdiction: z.string().optional()
});

const searchCasesSchema = z.object({
  query: z.string().optional(),
  status: z.array(z.string()).optional(),
  priority: z.array(z.string()).optional(),
  assignedTo: z.string().optional(),
  dateRange: z.object({
    start: z.string().datetime().transform(str => new Date(str)),
    end: z.string().datetime().transform(str => new Date(str))
  }).optional(),
  page: z.number().min(1).default(1),
  limit: z.number().min(1).max(100).default(50),
  useVectorSearch: z.boolean().default(true)
});

// GET - List cases with advanced search and filtering
export const GET: RequestHandler = async (event) => {
  return withApiHandler(async ({ url, locals }) => {
    // Get user from session
    const user = locals.user;
    if (!user) {
      throw CommonErrors.Unauthorized('User authentication required');
    }

    // Parse and validate query parameters
    const searchParams = {
      query: url.searchParams.get('query') || undefined,
      status: url.searchParams.get('status')?.split(',').filter(Boolean) || undefined,
      priority: url.searchParams.get('priority')?.split(',').filter(Boolean) || undefined,
      assignedTo: url.searchParams.get('assignedTo') || undefined,
      dateRange: url.searchParams.get('dateStart') && url.searchParams.get('dateEnd') ? {
        start: new Date(url.searchParams.get('dateStart')!),
        end: new Date(url.searchParams.get('dateEnd')!)
      } : undefined,
      page: parseInt(url.searchParams.get('page') || '1'),
      limit: Math.min(parseInt(url.searchParams.get('limit') || '50'), 100),
      useVectorSearch: url.searchParams.get('useVectorSearch') !== 'false'
    };

    // Validate search parameters
    try {
      const validatedParams = searchCasesSchema.parse(searchParams);
      
      // Calculate offset from page
      const offset = (validatedParams.page - 1) * validatedParams.limit;
      
      // Perform case search
      const { cases: caseResults, total } = await CaseOperations.search({
        ...validatedParams,
        offset
      });

      // Create pagination info
      const pagination = createPagination(validatedParams.page, validatedParams.limit, total);

      return {
        cases: caseResults,
        pagination,
        search: validatedParams.query ? {
          term: validatedParams.query,
          resultsCount: caseResults.length,
          vectorSearchUsed: validatedParams.useVectorSearch
        } : null
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw CommonErrors.ValidationFailed('search parameters', error.errors[0]?.message || 'Invalid parameters');
      }
      throw error;
    }
  }, event);
};

// POST - Create new case
export const POST: RequestHandler = async (event) => {
  return withApiHandler(async ({ request, locals }) => {
    // Get authenticated user
    const user = locals.user;
    if (!user) {
      throw CommonErrors.Unauthorized('User authentication required');
    }

    // Parse and validate request body
    const caseData = await parseRequestBody(request, createCaseSchema);
    
    try {
      // Create case using enhanced operations
      const newCase = await CaseOperations.create({
        ...caseData,
        createdBy: user.id
      });

      console.log(`✅ Case created successfully: ${newCase.caseNumber} by user ${user.id}`);

      return {
        case: newCase,
        message: `Case ${newCase.caseNumber} created successfully`
      };
    } catch (error) {
      if (error instanceof Error && error.message.includes('duplicate')) {
        throw CommonErrors.BadRequest('Case with similar details already exists');
      }
      throw error;
    }
  }, event);
};

// Additional endpoints

// PUT - Update existing case
export const PUT: RequestHandler = async (event) => {
  return withApiHandler(async ({ request, url, locals }) => {
    const user = locals.user;
    if (!user) {
      throw CommonErrors.Unauthorized('User authentication required');
    }

    const caseId = url.searchParams.get('id');
    if (!caseId) {
      throw CommonErrors.BadRequest('Case ID is required');
    }

    // Parse and validate update data
    const updateSchema = createCaseSchema.partial().omit({ status: true });
    const updates = await parseRequestBody(request, updateSchema);

    try {
      const updatedCase = await CaseOperations.update(caseId, updates, user.id);
      
      return {
        case: updatedCase,
        message: 'Case updated successfully'
      };
    } catch (error) {
      if (error instanceof Error && error.message.includes('not found')) {
        throw CommonErrors.NotFound('Case');
      }
      throw error;
    }
  }, event);
};

// OPTIONS - CORS preflight
export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
};