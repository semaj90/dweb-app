import { z } from "zod";
import { withApiHandler, parseRequestBody, apiSuccess, validationError, createPagination, CommonErrors } from '$lib/server/api/response';
import { CaseOperations } from '$lib/server/db/enhanced-operations';
import type { Case } from '$lib/server/db/schema-postgres';
import { createClient } from 'redis';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// Redis client for worker communication
let redisClient: ReturnType<typeof createClient> | null = null;
let redisUnavailable = false;

async function getRedisClient(): Promise<any> {
  if (redisUnavailable) return null;
  if (!redisClient) {
    try {
      redisClient = createClient({
        url: import.meta.env.REDIS_URL || 'redis://localhost:6379',
        socket: { connectTimeout: 5000 }
      });
      await redisClient.connect();
    } catch (e) {
      console.warn('⚠️ Redis not available, continuing without stream worker integration');
      redisUnavailable = true;
      return null;
    }
  }
  return redisClient;
}

// Worker trigger function
async function triggerWorkerProcessing(caseId: string, options: {
  priority: string;
  caseType: string;
  userId: string;
  trigger: string;
  metadata?: any;
}): Promise<any> {
  const redis = await getRedisClient();
  if (!redis) return; // silently skip if unavailable in dev
  const correlationId = `case-${caseId}-${Date.now()}`;

  // Create Redis stream event for worker
  const eventData = {
    id: correlationId,
    type: 'case_created',
    action: 'process',
    caseId: caseId,
    evidenceId: '',
    documentId: '',
    metadata: JSON.stringify({
      priority: options.priority,
      caseType: options.caseType,
      userId: options.userId,
      trigger: options.trigger,
      timestamp: new Date().toISOString(),
      ...options.metadata
    }),
    retry: '0',
    timestamp: Date.now().toString()
  };

  // Add to Redis stream for worker consumption
  const streamName = 'autotag:requests';
  await redis.xAdd(streamName, '*', eventData);

  console.log(`📡 Worker event sent: ${streamName} -> ${correlationId}`);
}

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
export const GET: RequestHandler = async (event: any) => {
  return withApiHandler(async ({ url, locals }) => {
    // Get user from session
    const user = locals.user;
    if (!user) {
      // In development we may want a clearer hint rather than raw 401
      if ((process.env.DEV_BYPASS_AUTH === 'true') || (import.meta as any).env?.DEV_BYPASS_AUTH === 'true') {
        console.warn('DEV_BYPASS_AUTH enabled but locals.user missing; returning empty case list');
        return { cases: [], pagination: createPagination(1, 1, 0), search: null, devBypass: true };
      }
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
    } catch (error: any) {
      if (error instanceof z.ZodError) {
        throw CommonErrors.ValidationFailed('search parameters', error.errors[0]?.message || 'Invalid parameters');
      }
      throw error;
    }
  }, event);
};

// POST - Create new case
export const POST: RequestHandler = async (event: any) => {
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

      // Trigger PostgreSQL-first worker for auto-tagging and processing
      try {
        await triggerWorkerProcessing(newCase.id, {
          priority: caseData.priority,
          caseType: 'civil', // Default case type, could be enhanced
          userId: user.id,
          trigger: 'api-case-creation',
          metadata: {
            caseNumber: newCase.caseNumber,
            title: caseData.title,
            status: caseData.status,
            location: caseData.location,
            jurisdiction: caseData.jurisdiction
          }
        });
        console.log(`🚀 Worker processing triggered for case: ${newCase.id}`);
      } catch (workerError) {
        console.warn(`⚠️ Worker trigger failed for case ${newCase.id}:`, workerError);
        // Don't fail the case creation if worker trigger fails
      }

      return {
        case: newCase,
        message: `Case ${newCase.caseNumber} created successfully`,
        metadata: {
          workerTriggered: true,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error: any) {
      if (error instanceof Error && error.message.includes('duplicate')) {
        throw CommonErrors.BadRequest('Case with similar details already exists');
      }
      throw error;
    }
  }, event);
};

// Additional endpoints

// PUT - Update existing case
export const PUT: RequestHandler = async (event: any) => {
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
    } catch (error: any) {
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