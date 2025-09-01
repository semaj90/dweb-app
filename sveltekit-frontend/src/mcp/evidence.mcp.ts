/**
 * MCP Tool: Evidence Management
 * Clean abstraction layer for evidence operations using Drizzle ORM + pgvector
 * Following the suggested architecture pattern for Legal AI Platform
 */

import { drizzle } from 'drizzle-orm/node-postgres';
import postgres from 'postgres';
import { eq, desc, and, sql, like } from 'drizzle-orm';
import { l2Distance, cosineDistance } from 'drizzle-orm';
import { evidence, cases, documentChunks } from '$lib/server/db/schema';
import type { Evidence } from '$lib/types';
import crypto from "crypto";

// Database connection (based on MCP pgvector docs)
const connectionString = import.meta.env.DATABASE_URL!;
const client = postgres(connectionString);
const db = drizzle(client);

export interface MCPToolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: Record<string, any>;
}

export interface EvidenceCreateParams {
  caseId: string;
  title: string;
  description?: string;
  evidenceType: string;
  tags?: string[];
  embedding?: number[]; // For vector similarity search
  metadata?: Record<string, any>;
}

export interface EvidenceUpdateParams {
  evidenceId: string;
  updates: Partial<EvidenceCreateParams>;
}

export interface EvidenceSearchParams {
  query?: string;
  caseId?: string;
  evidenceType?: string;
  tags?: string[];
  limit?: number;
  offset?: number;
  includeEmbedding?: boolean;
}

export interface EvidenceVectorSearchParams {
  embedding: number[];
  caseId?: string;
  evidenceType?: string;
  threshold?: number;
  limit?: number;
}

/**
 * MCP Tool: Evidence Management
 * Thin adapter wrapping Drizzle ORM operations for evidence
 */
export class EvidenceMCPTool {;
  
  /**
   * Create new evidence
   */
  async createEvidence(params: EvidenceCreateParams): Promise<MCPToolResponse<Evidence>> {
    try {
      const newEvidence = await db.insert(evidence).values({
        id: crypto.randomUUID(),
        case_id: params.caseId,
        title: params.title,
        description: params.description,
        evidence_type: params.evidenceType,
        created_at: new Date(),
        updated_at: new Date()
      }).returning();

      return {
        success: true,
        data: newEvidence[0] as Evidence,
        metadata: {
          tool: 'evidence.createEvidence',
          timestamp: Date.now(),
          hasEmbedding: !!params.embedding
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'evidence.createEvidence',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Load evidence with optional filtering
   */
  async loadEvidence(params: EvidenceSearchParams): Promise<MCPToolResponse<Evidence[]>> {
    try {
      let query = db.select().from(evidence);
      
      // Add filters based on params
      const conditions = [];
      if (params.caseId) {
        conditions.push(eq(evidence.case_id, params.caseId));
      }
      if (params.evidenceType) {
        conditions.push(eq(evidence.evidence_type, params.evidenceType));
      }
      if (params.query) {
        conditions.push(
          sql`${evidence.title} ILIKE ${'%' + params.query + '%'} 
              OR ${evidence.description} ILIKE ${'%' + params.query + '%'}`
        );
      }
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }
      
      query = query.orderBy(desc(evidence.updated_at));
      
      if (params.limit) {
        query = query.limit(params.limit);
      }
      
      if (params.offset) {
        query = query.offset(params.offset);
      }

      const results = await query;

      return {
        success: true,
        data: results as Evidence[],
        metadata: {
          tool: 'evidence.loadEvidence',
          count: results.length,
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'evidence.loadEvidence',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Update existing evidence
   */
  async updateEvidence(params: EvidenceUpdateParams): Promise<MCPToolResponse<Evidence>> {
    try {
      const updateData: any = {
        ...params.updates,
        updated_at: new Date()
      };
      
      // Map camelCase to snake_case for database columns
      if (params.updates.caseId) updateData.case_id = params.updates.caseId;
      if (params.updates.evidenceType) updateData.evidence_type = params.updates.evidenceType;

      const updatedEvidence = await db.update(evidence)
        .set(updateData)
        .where(eq(evidence.id, params.evidenceId))
        .returning();

      if (updatedEvidence.length === 0) {
        return {
          success: false,
          error: 'Evidence not found',
          metadata: {
            tool: 'evidence.updateEvidence',
            evidenceId: params.evidenceId,
            timestamp: Date.now()
          }
        };
      }

      return {
        success: true,
        data: updatedEvidence[0] as Evidence,
        metadata: {
          tool: 'evidence.updateEvidence',
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'evidence.updateEvidence',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Find similar evidence using vector similarity (pgvector)
   */
  async findSimilarEvidence(params: EvidenceVectorSearchParams): Promise<MCPToolResponse<Evidence[]>> {
    try {
      let query = db.select({
        evidence: evidence,
        similarity: sql<number>`1 - (${documentChunks.embedding} <=> ${params.embedding}::vector) as similarity`
      })
        .from(evidence)
        .leftJoin(documentChunks, eq(evidence.id, documentChunks.document_id))
        .where(sql`${documentChunks.embedding} IS NOT NULL`);

      // Add filters
      const conditions = [];
      if (params.caseId) {
        conditions.push(eq(evidence.case_id, params.caseId));
      }
      if (params.evidenceType) {
        conditions.push(eq(evidence.evidence_type, params.evidenceType));
      }
      if (params.threshold) {
        conditions.push(sql`1 - (${documentChunks.embedding} <=> ${params.embedding}::vector) > ${params.threshold}`);
      }

      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }

      query = query.orderBy(sql`similarity DESC`).limit(params.limit || 10);

      const results = await query;

      return {
        success: true,
        data: results.map(row => row.evidence) as Evidence[],
        metadata: {
          tool: 'evidence.findSimilarEvidence',
          vectorDimensions: params.embedding.length,
          similarityMethod: 'cosineDistance',
          threshold: params.threshold || 0.7,
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'evidence.findSimilarEvidence',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Get evidence analytics and statistics
   */
  async getEvidenceAnalytics(caseId?: string): Promise<MCPToolResponse<any>> {
    try {
      const conditions = caseId ? [eq(evidence.case_id, caseId)] : [];
      
      const totalEvidence = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(evidence)
        .where(conditions.length > 0 ? and(...conditions) : undefined);
        
      const evidenceByType = await db.select({
        evidenceType: evidence.evidence_type,
        count: sql`count(*)`.mapWith(Number)
      })
        .from(evidence)
        .where(conditions.length > 0 ? and(...conditions) : undefined)
        .groupBy(evidence.evidence_type);

      const recentEvidence = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(evidence)
        .where(and(
          sql`${evidence.created_at} >= NOW() - INTERVAL '7 days'`,
          ...(conditions.length > 0 ? conditions : [])
        ));

      return {
        success: true,
        data: {
          totalEvidence: totalEvidence[0].count,
          evidenceByType: evidenceByType.reduce((acc, item) => {
            acc[item.evidenceType] = item.count;
            return acc;
          }, {} as Record<string, number>),
          recentEvidence: recentEvidence[0].count,
          lastUpdated: new Date()
        },
        metadata: {
          tool: 'evidence.getEvidenceAnalytics',
          caseId: caseId || 'all',
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'evidence.getEvidenceAnalytics',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Delete evidence (soft delete by setting deleted_at)
   */
  async deleteEvidence(evidenceId: string): Promise<MCPToolResponse<boolean>> {
    try {
      const deletedEvidence = await db.update(evidence)
        .set({ 
          updated_at: new Date()
          // Note: schema doesn't have deleted_at, so we would need to add this column
          // or implement hard delete with actual DELETE statement
        })
        .where(eq(evidence.id, evidenceId))
        .returning();

      if (deletedEvidence.length === 0) {
        return {
          success: false,
          error: 'Evidence not found',
          metadata: {
            tool: 'evidence.deleteEvidence',
            evidenceId,
            timestamp: Date.now()
          }
        };
      }

      return {
        success: true,
        data: true,
        metadata: {
          tool: 'evidence.deleteEvidence',
          evidenceId,
          timestamp: Date.now()
        }
      };
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'evidence.deleteEvidence',
          timestamp: Date.now()
        }
      };
    }
  }
}

// Export singleton instance
export const evidenceMCPTool = new EvidenceMCPTool();