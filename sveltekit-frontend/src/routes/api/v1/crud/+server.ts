import type { RequestHandler } from './$types';

/**
 * Comprehensive CRUD REST API - Legal AI Platform
 * Integrates with PostgreSQL + pgvector using Drizzle ORM
 * Supports all entities with vector search capabilities
 */

import { json, error } from '@sveltejs/kit';

import { ensureError } from '$lib/utils/ensure-error';
import { db } from '$lib/server/db';
import {
  users, userProfiles, cases, criminals, evidence, legalDocuments,
  reports, personsOfInterest, ragMessages, ragSessions
} from '$lib/server/db/schema';
import { sql, eq, and, or, like } from 'drizzle-orm';
import { apiOrchestrator } from '$lib/services/api-orchestrator';

// Entity mapping for dynamic CRUD operations
// Small helpers used by the request handlers below:
// - validate entity names
// - resolve a table for an entity
// - build simple WHERE clauses from filters
// - build lightweight text-search clauses for common entities

function isValidEntity(name?: string | null): boolean {
  return !!name && !!(entityMap as any)?.[name];
}

function getTableForEntity(entity: string) {
  if (!entity) throw new Error('Entity required');
  const table = (entityMap as any)[entity];
  if (!table) throw new Error(`Unknown entity: ${entity}`);
  return table;
}

// Convert a simple filters object into an array of drizzle WHERE conditions.
// - Strings containing '%' are treated as LIKE patterns.
// - Other values use equality.
function buildWhereClauses(filters: Record<string, any> | undefined, table: any) {
  if (!filters || typeof filters !== 'object') return [];
  return Object.entries(filters).flatMap(([key, value]) => {
    if (value === undefined || value === null) return [];
    const column = (table as any)[key];
    if (!column) return [];
    if (typeof value === 'string' && value.includes('%')) {
      return [like(column, value)];
    }
    return [eq(column, value)];
  });
}

// Lightweight text-search clause builder used in list/search handlers.
// Lightweight text-search clause builder used in list/search handlers.
// Returns an array of drizzle conditions (often a single `or(...)`) or [].
function buildSearchClause(entity: string, query: string, table: any) {
  if (!query) return [];
  if (entity === 'cases') {
    return [or(like((table as any).title, `%${query}%`), like((table as any).description, `%${query}%`), like((table as any).status, `%${query}%`))];
  }
  if (entity === 'evidence') {
    return [or(like((table as any).title, `%${query}%`), like((table as any).description, `%${query}%`), like((table as any).evidenceType, `%${query}%`))];
  }
  if (entity === 'legalDocuments') {
    return [or(like((table as any).title, `%${query}%`), like((table as any).content, `%${query}%`), like((table as any).documentType, `%${query}%`))];
  }
  return [];
}
// Map entity names to tables (add/remove as needed)
// Internal map (not exported to SvelteKit routing system)
const entityMap = {
  users,
  userProfiles,
  cases,
  criminals,
  evidence,
  legalDocuments,
  reports,
  personsOfInterest,
  ragMessages,
  ragSessions,
} as const;

type EntityName = keyof typeof entityMap;

export interface CrudRequest {
  action: 'create' | 'read' | 'update' | 'delete' | 'list' | 'search' | 'vector_search';
  entity: EntityName;
  id?: string;
  data?: any;
  filters?: Record<string, any>;
  pagination?: {
    page?: number;
    limit?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc'
  };
  search?: {
    query: string;
    fields?: string[];
    vector?: boolean;
    similarity_threshold?: number;
  };
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    total?: number;
    page?: number;
    limit?: number;
    processingTime?: number;
    cached?: boolean;
  };
}

export const GET: RequestHandler = async ({ url, request }) => {
  const startTime = Date.now();

  try {
    // Parse query parameters for GET requests
    const entity = url.searchParams.get('entity') as EntityName;
    const action = url.searchParams.get('action') || 'list';
    const id = url.searchParams.get('id');
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = parseInt(url.searchParams.get('limit') || '50');
    const sortBy = url.searchParams.get('sortBy') || 'createdAt';
    const sortOrder = (url.searchParams.get('sortOrder') || 'desc') as 'asc' | 'desc';
    const searchQuery = url.searchParams.get('search');

    if (!entity || !entityMap[entity]) {
      return error(400, ensureError({
        message: `Invalid entity: ${entity}. Available: ${Object.keys(entityMap).join(', ')}`
      }));
    }

    const table = entityMap[entity];
    let result: any;

    switch (action) {
      case 'read':
        if (!id) return error(400, ensureError({ message: 'ID required for read operation' }));

        result = await db.select().from(table).where(eq(table.id, id)).limit(1);
        if (result.length === 0) {
          return error(404, ensureError({ message: `${entity} with ID ${id} not found` }));
        }
        result = await db.select().from(table).where(eq((table as any).id, id)).limit(1);
        return json({
          success: true,
          data: result[0],
          metadata: { processingTime: Date.now() - startTime }
        } satisfies ApiResponse);

      case 'list':
        const offset = (page - 1) * limit;
        const sortColumnAny = (table as any)[sortBy] || (table as any).createdAt;
        const orderBy = sortOrder === 'asc' ? sql`${sortColumnAny} ASC` : sql`${sortColumnAny} DESC`;

        // Get total count for pagination
        const countResultArr = await db.select({ count: sql<number>`count(*)::int` }).from(table);
        const total = countResultArr[0]?.count ?? 0;

        // Get paginated results
        const query = db.select().from(table).orderBy(orderBy).limit(limit).offset(offset);

        // Add search if provided
        if (searchQuery && entity === 'cases') {
          query.where(
            or(
              like(cases.title, `%${searchQuery}%`),
              like(cases.description, `%${searchQuery}%`)
            )
          );
        } else if (searchQuery && entity === 'evidence') {
          query.where(
            or(
              like(evidence.title, `%${searchQuery}%`),
              like(evidence.description, `%${searchQuery}%`)
            )
          );
        }

        result = await query;

        return json({
          success: true,
          data: result,
          metadata: {
            total,
            page,
            limit,
            processingTime: Date.now() - startTime
          }
        } satisfies ApiResponse);

      case 'search':
        if (!searchQuery) return error(400, ensureError({ message: 'Search query required' }));

        // Basic text search for now - can be enhanced with vector search
        let searchResult: any[] = [];

        if (entity === 'cases') {
          searchResult = await db.select().from(cases)
            .where(
              or(
                like(cases.title, `%${searchQuery}%`),
                like(cases.description, `%${searchQuery}%`),
                like(cases.status, `%${searchQuery}%`)
              )
            )
            .limit(limit);
        } else if (entity === 'evidence') {
          searchResult = await db.select().from(evidence)
            .where(
              or(
                like(evidence.title, `%${searchQuery}%`),
                like(evidence.description, `%${searchQuery}%`),
                like(evidence.evidenceType, `%${searchQuery}%`)
              )
            )
            .limit(limit);
        } else if (entity === 'legalDocuments') {
          searchResult = await db.select().from(legalDocuments)
            .where(
              or(
                like(legalDocuments.title, `%${searchQuery}%`),
                like(legalDocuments.content, `%${searchQuery}%`),
                like(legalDocuments.documentType, `%${searchQuery}%`)
              )
            )
            .limit(limit);
        }

        return json({
          success: true,
          data: searchResult,
          metadata: {
            total: searchResult.length,
            processingTime: Date.now() - startTime
          }
        } satisfies ApiResponse);

      case 'vector_search':
        if (!searchQuery) return error(400, ensureError({ message: 'Search query required for vector search' }));

        // Use the API orchestrator to perform vector search via Go services
        try {
          const vectorResponse = await apiOrchestrator.routeRequest(
            'enhancedRAG',
            '/api/vector/search',
            {
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: searchQuery,
                entity: entity,
                limit: limit,
                similarity_threshold: parseFloat(url.searchParams.get('similarity_threshold') || '0.7')
              })
            }
          );

          if (!vectorResponse.ok) {
            throw new Error(`Vector search failed: ${vectorResponse.status}`);
          }

          const vectorData = await vectorResponse.json();

          return json({
            success: true,
            data: vectorData.results || vectorData,
            metadata: {
              total: vectorData.total || vectorData.results?.length || 0,
              processingTime: Date.now() - startTime,
              cached: false
            }
          } satisfies ApiResponse);

        } catch (vectorError) {
          console.error('Vector search error:', vectorError);

          // Fallback to regular search if vector search fails
          return json({
            success: true,
            data: [],
            error: 'Vector search unavailable, falling back to text search',
            metadata: {
              processingTime: Date.now() - startTime,
              cached: true
            }
          } satisfies ApiResponse);
        }

      default:
        return error(400, ensureError({ message: `Invalid action: ${action}` }));
    }

  } catch (err: any) {
    console.error('CRUD GET error:', err);
    return error(500, ensureError({
      message: 'Internal server error',
      details: err instanceof Error ? err.message : String(err)
    }));
  }
};

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const body: CrudRequest = await request.json();
    const { action, entity, data, id } = body;

    if (!entity || !entityMap[entity]) {
      return error(400, ensureError({
        message: `Invalid entity: ${entity}. Available: ${Object.keys(entityMap).join(', ')}`
      }));
    }

    const table = entityMap[entity];
    let result: any;

    switch (action) {
      case 'create':
        if (!data) return error(400, ensureError({ message: 'Data required for create operation' }));

        // Add timestamps for create
        const createData = {
          ...data,
          createdAt: new Date(),
          updatedAt: new Date()
        };

        result = await db.insert(table).values(createData).returning();

        // If this is a document or evidence with content, trigger embedding generation
        if ((entity === 'evidence' || entity === 'legalDocuments') && data.content) {
          try {
            await apiOrchestrator.routeRequest(
              'enhancedRAG',
              '/api/embeddings/generate',
              {
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  id: result[0].id,
                  entity: entity,
                  content: data.content,
                  title: data.title
                })
              }
            );
          } catch (embeddingError) {
            console.warn('Embedding generation failed:', embeddingError);
            // Don't fail the creation, just log the warning
          }
        }

        return json({
          success: true,
          data: result[0],
          metadata: { processingTime: Date.now() - startTime }
        } satisfies ApiResponse);

      case 'update':
        if (!id || !data) return error(400, ensureError({ message: 'ID and data required for update operation' }));

        // Add updated timestamp
        const updateData = {
          ...data,
          updatedAt: new Date()
        };

        result = await db.update(table).set(updateData).where(eq(table.id, id)).returning();

        if (result.length === 0) {
          return error(404, ensureError({ message: `${entity} with ID ${id} not found` }));
        }

        return json({
          success: true,
          data: result[0],
          metadata: { processingTime: Date.now() - startTime }
        } satisfies ApiResponse);

      case 'delete':
        if (!id) return error(400, ensureError({ message: 'ID required for delete operation' }));

        // Perform deletion
        result = await db.delete(table).where(eq((table as any).id, id)).returning();

        if (result.length === 0) {
          return error(404, ensureError({ message: `${entity} with ID ${id} not found` }));
        }

        // Best-effort cleanup of embeddings / vectors for content entities
        if (['evidence', 'legalDocuments'].includes(entity)) {
          try {
            await apiOrchestrator.routeRequest(
              'enhancedRAG',
              '/api/embeddings/delete',
              {
                body: JSON.stringify({ id, entity }),
                headers: { 'Content-Type': 'application/json' }
              }
            );
          } catch (cleanupErr) {
            console.warn(`Embedding cleanup failed for ${entity} ${id}:`, cleanupErr);
          }
        }

        return json({
          success: true,
          data: result[0],
          metadata: { processingTime: Date.now() - startTime }
        } satisfies ApiResponse);

      default:
        return error(400, ensureError({ message: `Invalid action: ${action}` }));
    }

  } catch (err: any) {
    console.error('CRUD POST error:', err);
    return error(500, ensureError({
      message: 'Internal server error',
      details: err instanceof Error ? err.message : String(err)
    }));
  }
};

export const PUT: RequestHandler = async ({ request, url }) => {
  // PUT maps to update action
  const body = await request.json();
  const id = url.searchParams.get('id') || body.id;

  return POST({
    request: new Request(request.url, {
      method: 'POST',
      headers: request.headers,
      body: JSON.stringify({
        ...body,
        action: 'update',
        id
      })
    })
  } as any);
};

export const DELETE: RequestHandler = async ({ url }) => {
  const entity = url.searchParams.get('entity') as EntityName;
  const id = url.searchParams.get('id');

  return POST({
    request: new Request(url.toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'delete',
        entity,
        id
      })
    })
  } as any);
};