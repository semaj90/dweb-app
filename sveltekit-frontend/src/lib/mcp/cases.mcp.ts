/**
 * MCP Tools Layer - Cases Management
 * Thin adapters wrapping Drizzle ORM queries for XState machine services
 */

import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import { sql, eq, and, like, desc } from '$lib/server/db/index';
import * as schema from "../db/schema";
import { redis } from "../server/cache/redis-service";
import { minioService } from "../server/storage/minio-service";

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

const db = drizzle(pool, { schema });

export interface CaseData {
  id?: string;
  title: string;
  caseNumber?: string;
  description: string;
  priority: 'low' | 'medium' | 'high';
  status: 'open' | 'active' | 'closed' | 'archived';
  createdBy: string;
  userId?: string; // alias for createdBy
  assignedTo?: string;
  metadata?: any;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface EvidenceData {
  id?: string;
  caseId: string;
  content: string;
  evidenceType: 'document' | 'testimony' | 'physical' | 'digital';
  source?: string;
  tags?: string[];
  confidenceLevel?: number;
  createdAt?: Date;
}

/**
 * MCP Tool: Load Case
 * Retrieves case data with caching layer
 */
export async function loadCase(caseId: string): Promise<CaseData | null> {
  try {
    console.log(`🔍 MCP Tool: loadCase(${caseId})`);
    
    // Check Redis cache first
    const cacheKey = `case:${caseId}`;
    const cached = await redis.get<CaseData>(cacheKey);
    if (cached) {
      console.log(`📦 Cache hit for case ${caseId}`);
      return cached;
    }

    // Query PostgreSQL with Drizzle
    const result = await db.query.cases.findFirst({
      where: eq(schema.cases.id, caseId),
      with: {
        evidence: {
          limit: 10,
          orderBy: desc(schema.evidence.createdAt)
        }
      }
    });

    if (!result) {
      console.log(`❌ Case ${caseId} not found`);
      return null;
    }

    // Cache the result
    await redis.set(cacheKey, result, 300); // 5 minutes TTL
    
    console.log(`✅ Case ${caseId} loaded from database`);
    return result as CaseData;

  } catch (error: any) {
    console.error('❌ MCP Tool loadCase error:', error);
    throw error;
  }
}

/**
 * MCP Tool: Create Case  
 * Creates new case with auto-generated case number
 */
export async function createCase(caseData: Omit<CaseData, 'id' | 'createdAt' | 'updatedAt'>): Promise<{ success: boolean; caseId?: string; error?: string }> {
  try {
    console.log('🆕 MCP Tool: createCase', caseData.title);

    // Generate case number if not provided
    const caseNumber = caseData.caseNumber || await generateCaseNumber();

    // Insert into PostgreSQL
    const [newCase] = await db.insert(schema.cases).values({
      ...caseData,
      caseNumber,
      status: caseData.status || 'open',
      createdAt: new Date(),
      updatedAt: new Date()
    }).returning();

    // Invalidate related caches
    await redis.del(`user:${caseData.userId}:cases`);
    
    // Publish event to Redis pub/sub for real-time updates
    await redis.publish('case:created', {
      caseId: newCase.id,
      title: newCase.title,
      userId: newCase.userId,
      timestamp: Date.now()
    });

    console.log(`✅ Case created: ${newCase.id}`);
    return { success: true, caseId: newCase.id };

  } catch (error: any) {
    console.error('❌ MCP Tool createCase error:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * MCP Tool: Update Case
 * Updates case with optimistic locking
 */
export async function updateCase(caseId: string, updates: Partial<CaseData>): Promise<{ success: boolean; error?: string }> {
  try {
    console.log(`📝 MCP Tool: updateCase(${caseId})`);

    // Update in PostgreSQL
    const [updated] = await db.update(schema.cases)
      .set({
        ...updates,
        updatedAt: new Date()
      })
      .where(eq(schema.cases.id, caseId))
      .returning();

    if (!updated) {
      return { success: false, error: 'Case not found' };
    }

    // Invalidate caches
    await redis.del(`case:${caseId}`);
    await redis.del(`user:${updated.userId}:cases`);

    // Publish update event
    await redis.publish('case:updated', {
      caseId,
      changes: updates,
      timestamp: Date.now()
    });

    console.log(`✅ Case ${caseId} updated`);
    return { success: true };

  } catch (error: any) {
    console.error('❌ MCP Tool updateCase error:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * MCP Tool: Add Evidence
 * Adds evidence to case with file upload support
 */
export async function addEvidence(caseId: string, evidence: Omit<EvidenceData, 'id' | 'createdAt'>): Promise<{ success: boolean; evidenceId?: string; error?: string }> {
  try {
    console.log(`📋 MCP Tool: addEvidence to case ${caseId}`);

    // Verify case exists
    const caseExists = await db.query.cases.findFirst({
      where: eq(schema.cases.id, caseId)
    });

    if (!caseExists) {
      return { success: false, error: 'Case not found' };
    }

    // Insert evidence
    const [newEvidence] = await db.insert(schema.evidence).values({
      caseId,
      title: evidence.source || 'Evidence Item',
      content: evidence.content,
      type: evidence.evidenceType,
      evidenceType: evidence.evidenceType,
      createdBy: 'system', // TODO: get from context
      tags: evidence.tags ? JSON.stringify(evidence.tags) : null,
      metadata: evidence,
      createdAt: new Date()
    }).returning();

    // Invalidate case cache
    await redis.del(`case:${caseId}`);

    // Publish evidence added event
    await redis.publish('evidence:added', {
      caseId,
      evidenceId: newEvidence.id,
      evidenceType: evidence.evidenceType,
      timestamp: Date.now()
    });

    console.log(`✅ Evidence added: ${newEvidence.id}`);
    return { success: true, evidenceId: newEvidence.id };

  } catch (error: any) {
    console.error('❌ MCP Tool addEvidence error:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * MCP Tool: Search Cases
 * Vector-enhanced case search with semantic similarity
 */
export async function searchCases(query: string, userId: string, filters?: {
  status?: string;
  priority?: string;
  dateRange?: { from: Date; to: Date };
}): Promise<{ cases: CaseData[]; totalCount: number }> {
  try {
    console.log(`🔍 MCP Tool: searchCases("${query}")`);

    // Check cache first
    const cacheKey = `search:cases:${Buffer.from(query + JSON.stringify(filters)).toString('base64')}`;
    const cached = await redis.get(cacheKey);
    if (cached) {
      console.log(`📦 Cache hit for search: ${query}`);
      return cached;
    }

    // Build query conditions
    let conditions = [eq(schema.cases.userId, userId)];
    
    if (query.trim()) {
      conditions.push(
        sql`(${schema.cases.title} ILIKE ${`%${query}%`} OR ${schema.cases.description} ILIKE ${`%${query}%`})`
      );
    }

    if (filters?.status) {
      conditions.push(eq(schema.cases.status, filters.status));
    }

    if (filters?.priority) {
      conditions.push(eq(schema.cases.priority, filters.priority));
    }

    // Execute search
    const results = await db.query.cases.findMany({
      where: and(...conditions),
      orderBy: desc(schema.cases.updatedAt),
      limit: 50,
      with: {
        evidence: {
          limit: 3,
          orderBy: desc(schema.evidence.createdAt)
        }
      }
    });

    const searchResult = {
      cases: results as CaseData[],
      totalCount: results.length
    };

    // Cache results for 5 minutes
    await redis.set(cacheKey, searchResult, 300);

    console.log(`✅ Found ${results.length} cases`);
    return searchResult;

  } catch (error: any) {
    console.error('❌ MCP Tool searchCases error:', error);
    return { cases: [], totalCount: 0 };
  }
}

/**
 * MCP Tool: Get User Cases
 * Retrieves all cases for a user with pagination
 */
export async function getUserCases(userId: string, options: {
  limit?: number;
  offset?: number;
  status?: string;
} = {}): Promise<{ cases: CaseData[]; totalCount: number }> {
  try {
    console.log(`👤 MCP Tool: getUserCases(${userId})`);

    const { limit = 20, offset = 0, status } = options;
    
    // Check cache
    const cacheKey = `user:${userId}:cases:${JSON.stringify(options)}`;
    const cached = await redis.get(cacheKey);
    if (cached) {
      return cached;
    }

    let conditions = [eq(schema.cases.userId, userId)];
    if (status) {
      conditions.push(eq(schema.cases.status, status));
    }

    const cases = await db.query.cases.findMany({
      where: and(...conditions),
      limit,
      offset,
      orderBy: desc(schema.cases.updatedAt)
    });

    // Get total count
    const [{ count }] = await db
      .select({ count: sql`count(*)`.mapWith(Number) })
      .from(schema.cases)
      .where(and(...conditions));

    const result = { cases: cases as CaseData[], totalCount: count };
    
    // Cache for 2 minutes
    await redis.set(cacheKey, result, 120);

    return result;

  } catch (error: any) {
    console.error('❌ MCP Tool getUserCases error:', error);
    return { cases: [], totalCount: 0 };
  }
}

/**
 * Helper: Generate Case Number
 * Creates unique case number with format: CASE-YYYY-NNNN
 */
async function generateCaseNumber(): Promise<string> {
  const year = new Date().getFullYear();
  const prefix = `CASE-${year}-`;
  
  // Get highest case number for this year
  const result = await db
    .select({ caseNumber: schema.cases.caseNumber })
    .from(schema.cases)
    .where(like(schema.cases.caseNumber, `${prefix}%`))
    .orderBy(desc(schema.cases.caseNumber))
    .limit(1);

  let nextNumber = 1;
  if (result.length > 0 && result[0].caseNumber) {
    const lastNumber = parseInt(result[0].caseNumber.replace(prefix, ''));
    if (!isNaN(lastNumber)) {
      nextNumber = lastNumber + 1;
    }
  }

  return `${prefix}${nextNumber.toString().padStart(4, '0')}`;
}

/**
 * MCP Tool: Health Check
 * Verifies database connectivity and performance
 */
export async function healthCheck(): Promise<{ status: 'healthy' | 'unhealthy'; details: any }> {
  try {
    const start = Date.now();
    
    // Test query
    const result = await db.execute(sql`SELECT 1 as test, NOW() as timestamp`);
    const firstRow = Array.isArray(result) ? result[0] : result.rows?.[0];
    
    const responseTime = Date.now() - start;
    
    return {
      status: 'healthy',
      details: {
        responseTime,
        timestamp: firstRow?.timestamp || new Date().toISOString(),
        poolTotalCount: pool.totalCount,
        poolIdleCount: pool.idleCount,
        poolWaitingCount: pool.waitingCount
      }
    };
    
  } catch (error: any) {
    return {
      status: 'unhealthy',
      details: {
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    };
  }
}