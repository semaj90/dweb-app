/**
 * MCP Tool: Cases Management
 * Clean abstraction layer for case operations using Drizzle ORM + pgvector
 * Following the suggested architecture pattern
 */

import { drizzle } from 'drizzle-orm/node-postgres';
import postgres from 'postgres';
import { eq, desc, and, sql } from 'drizzle-orm';
import { l2Distance } from 'drizzle-orm';
import { cases, evidence, users } from '$lib/server/db/schema';
import type { Case, Evidence } from '$lib/types';

// Database connection (based on MCP pgvector docs)
const connectionString = process.env.DATABASE_URL!;
const client = postgres(connectionString);
const db = drizzle(client);

export interface MCPToolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: Record<string, any>;
}

export interface CaseCreateParams {
  title: string;
  description?: string;
  userId: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

export interface CaseUpdateParams {
  caseId: string;
  updates: Partial<CaseCreateParams>;
}

export interface CaseSearchParams {
  query?: string;
  userId?: string;
  status?: 'active' | 'closed' | 'archived';
  limit?: number;
  offset?: number;
}

export interface EvidenceAddParams {
  caseId: string;
  title: string;
  description?: string;
  evidenceType: string;
  tags?: string[];
  embedding?: number[]; // For vector similarity search
}

/**
 * MCP Tool: Legal Cases Management
 * Thin adapter wrapping Drizzle ORM operations
 */
export class CasesMCPTool {
  
  /**
   * Create a new case
   */
  async createCase(params: CaseCreateParams): Promise<MCPToolResponse<Case>> {
    try {
      const newCase = await db.insert(cases).values({
        id: crypto.randomUUID(),
        title: params.title,
        description: params.description,
        userId: params.userId,
        priority: params.priority || 'medium',
        status: 'active',
        createdAt: new Date(),
        updatedAt: new Date()
      }).returning();

      return {
        success: true,
        data: newCase[0],
        metadata: {
          tool: 'cases.createCase',
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'cases.createCase',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Load cases for a user with optional filtering
   */
  async loadCases(params: CaseSearchParams): Promise<MCPToolResponse<Case[]>> {
    try {
      let query = db.select().from(cases);
      
      // Add filters based on params
      const conditions = [];
      if (params.userId) {
        conditions.push(eq(cases.userId, params.userId));
      }
      if (params.status) {
        conditions.push(eq(cases.status, params.status));
      }
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }
      
      query = query.orderBy(desc(cases.updatedAt));
      
      if (params.limit) {
        query = query.limit(params.limit);
      }
      
      if (params.offset) {
        query = query.offset(params.offset);
      }

      const results = await query;

      return {
        success: true,
        data: results,
        metadata: {
          tool: 'cases.loadCases',
          count: results.length,
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'cases.loadCases',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Update an existing case
   */
  async updateCase(params: CaseUpdateParams): Promise<MCPToolResponse<Case>> {
    try {
      const updatedCase = await db.update(cases)
        .set({
          ...params.updates,
          updatedAt: new Date()
        })
        .where(eq(cases.id, params.caseId))
        .returning();

      if (updatedCase.length === 0) {
        return {
          success: false,
          error: 'Case not found',
          metadata: {
            tool: 'cases.updateCase',
            caseId: params.caseId,
            timestamp: Date.now()
          }
        };
      }

      return {
        success: true,
        data: updatedCase[0],
        metadata: {
          tool: 'cases.updateCase',
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'cases.updateCase',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Add evidence to a case with vector embedding support
   */
  async addEvidence(params: EvidenceAddParams): Promise<MCPToolResponse<Evidence>> {
    try {
      const newEvidence = await db.insert(evidence).values({
        id: crypto.randomUUID(),
        caseId: params.caseId,
        title: params.title,
        description: params.description,
        evidenceType: params.evidenceType,
        tags: params.tags,
        embedding: params.embedding, // pgvector field
        createdAt: new Date()
      }).returning();

      return {
        success: true,
        data: newEvidence[0],
        metadata: {
          tool: 'cases.addEvidence',
          hasEmbedding: !!params.embedding,
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'cases.addEvidence',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Find similar cases using vector similarity (pgvector)
   */
  async findSimilarCases(embedding: number[], limit: number = 5): Promise<MCPToolResponse<Case[]>> {
    try {
      // Using pgvector L2 distance for similarity search
      const similarCases = await db.select()
        .from(cases)
        .leftJoin(evidence, eq(evidence.caseId, cases.id))
        .where(sql`${evidence.embedding} IS NOT NULL`)
        .orderBy(l2Distance(evidence.embedding, embedding))
        .limit(limit);

      return {
        success: true,
        data: similarCases.map(row => row.cases),
        metadata: {
          tool: 'cases.findSimilarCases',
          vectorDimensions: embedding.length,
          similarityMethod: 'l2Distance',
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'cases.findSimilarCases',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Get case analytics and statistics
   */
  async getCaseAnalytics(userId?: string): Promise<MCPToolResponse<any>> {
    try {
      const conditions = userId ? [eq(cases.userId, userId)] : [];
      
      const totalCases = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(cases)
        .where(conditions.length > 0 ? and(...conditions) : undefined);
        
      const activeCases = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(cases)
        .where(and(
          eq(cases.status, 'active'),
          ...(conditions.length > 0 ? conditions : [])
        ));

      const evidenceCount = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(evidence)
        .leftJoin(cases, eq(evidence.caseId, cases.id))
        .where(conditions.length > 0 ? and(...conditions) : undefined);

      return {
        success: true,
        data: {
          totalCases: totalCases[0].count,
          activeCases: activeCases[0].count,
          evidenceCount: evidenceCount[0].count,
          lastUpdated: new Date()
        },
        metadata: {
          tool: 'cases.getCaseAnalytics',
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'cases.getCaseAnalytics',
          timestamp: Date.now()
        }
      };
    }
  }
}

// Export singleton instance
export const casesMCPTool = new CasesMCPTool();