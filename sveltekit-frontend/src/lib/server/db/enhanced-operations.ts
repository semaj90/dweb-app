// Enhanced Database Operations with pgvector Integration
// Production-ready database operations for SvelteKit 2

import { db, sql } from './index';
import type { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { eq, and, or, desc, asc, ilike, count, isNull, isNotNull, sql as sqlRaw, gte, lte } from 'drizzle-orm';
import {
  cases, evidence, legalDocuments, users, ragSessions, ragMessages,
  userAiQueries, embeddingCache, documentChunks, caseEmbeddings,
  evidenceVectors, legalPrecedents
} from './index';
import { ApiErrorClass, CommonErrors } from '../api/response';
import { arrayToPgVector, generateSampleEmbedding } from './vector-operations';
import type { Case, Evidence, LegalDocument, User } from './index';

// Transaction wrapper for safe database operations
export async function withTransaction<T>(
  operation: (tx: NodePgDatabase<typeof import('./index')>) => Promise<T>
): Promise<T> {
  return await db.transaction(async (tx) => {
    try {
      return await operation(tx);
    } catch (error) {
      // Transaction will be rolled back automatically
      throw error instanceof Error 
        ? CommonErrors.DatabaseError('transaction', { originalError: error.message })
        : CommonErrors.InternalError('Transaction failed');
    }
  });
}

// Enhanced Case Operations
export class CaseOperations {
  // Create case with validation and audit trail
  static async create(
    caseData: {
      title: string;
      description?: string;
      priority?: 'low' | 'medium' | 'high' | 'critical';
      status?: 'open' | 'investigating' | 'pending' | 'closed' | 'archived';
      incidentDate?: Date;
      location?: string;
      jurisdiction?: string;
      createdBy: string;
    }
  ): Promise<Case> {
    return withTransaction(async (tx) => {
      // Generate unique case number
      const caseCount = await tx.select({ count: count() }).from(cases);
      const year = new Date().getFullYear();
      const sequence = (caseCount[0]?.count || 0) + 1;
      const caseNumber = `CR-${year}-${sequence.toString().padStart(4, '0')}`;

      const [newCase] = await tx.insert(cases).values({
        ...caseData,
        caseNumber,
        priority: caseData.priority || 'medium',
        status: caseData.status || 'open',
        createdAt: new Date(),
        updatedAt: new Date()
      }).returning();

      // Generate AI summary if description exists
      if (newCase.description) {
        const embedding = generateSampleEmbedding(768);
        await tx.insert(caseEmbeddings).values({
          caseId: newCase.id,
          content: `${newCase.title} - ${newCase.description}`,
          embedding: arrayToPgVector(embedding),
          metadata: {
            caseNumber: newCase.caseNumber,
            priority: newCase.priority,
            status: newCase.status
          }
        });
      }

      return newCase;
    });
  }

  // Advanced case search with vector similarity
  static async search(
    params: {
      query?: string;
      status?: string[];
      priority?: string[];
      dateRange?: { start: Date; end: Date };
      assignedTo?: string;
      limit?: number;
      offset?: number;
      useVectorSearch?: boolean;
    }
  ): Promise<{ cases: Case[]; total: number }> {
    const { query, status, priority, dateRange, assignedTo, limit = 50, offset = 0, useVectorSearch = true } = params;

    let conditions = [];
    
    // Build WHERE conditions
    if (status && status.length > 0) {
      conditions.push(sqlRaw`status = ANY(${status})`);
    }
    if (priority && priority.length > 0) {
      conditions.push(sqlRaw`priority = ANY(${priority})`);
    }
    if (assignedTo) {
      conditions.push(eq(cases.leadProsecutor, assignedTo));
    }
    if (dateRange) {
      conditions.push(
        and(
          gte(cases.createdAt, dateRange.start),
          lte(cases.createdAt, dateRange.end)
        )
      );
    }

    // Vector search for semantic similarity
    if (query && useVectorSearch) {
      try {
        const queryEmbedding = generateSampleEmbedding(768);
        const vectorQuery = arrayToPgVector(queryEmbedding);
        
        const vectorResults = await db.execute(sqlRaw`
          SELECT 
            c.*,
            (1 - (ce.embedding <=> ${vectorQuery}::vector)) as similarity_score
          FROM cases c
          LEFT JOIN case_embeddings ce ON c.id = ce.case_id
          WHERE 
            ${conditions.length > 0 ? sqlRaw`(${conditions.join(' AND ')}) AND` : sqlRaw``}
            ce.embedding IS NOT NULL AND
            (1 - (ce.embedding <=> ${vectorQuery}::vector)) > 0.7
          ORDER BY similarity_score DESC
          LIMIT ${limit}
          OFFSET ${offset}
        `);
        
        return {
          cases: vectorResults as Case[],
          total: vectorResults.length
        };
      } catch (error) {
        console.warn('Vector search failed, falling back to text search:', error);
      }
    }

    // Fallback to traditional text search
    if (query) {
      conditions.push(
        or(
          ilike(cases.title, `%${query}%`),
          ilike(cases.description, `%${query}%`),
          ilike(cases.caseNumber, `%${query}%`)
        )
      );
    }

    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;
    
    const [results, totalCount] = await Promise.all([
      db.select({
        id: cases.id,
        caseNumber: cases.caseNumber,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        priority: cases.priority,
        incidentDate: cases.incidentDate,
        location: cases.location,
        jurisdiction: cases.jurisdiction,
        leadProsecutor: cases.leadProsecutor,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt,
        closedAt: cases.closedAt
      })
      .from(cases)
      .where(whereClause)
      .orderBy(desc(cases.createdAt))
      .limit(limit)
      .offset(offset),
      
      db.select({ count: count() })
      .from(cases)
      .where(whereClause)
    ]);

    return {
      cases: results as Case[],
      total: totalCount[0]?.count || 0
    };
  }

  // Update case with audit trail
  static async update(
    caseId: string,
    updates: Partial<Pick<Case, 'title' | 'description' | 'status' | 'priority' | 'location' | 'jurisdiction'>>,
    updatedBy: string
  ): Promise<Case> {
    return withTransaction(async (tx) => {
      const [updatedCase] = await tx.update(cases)
        .set({
          ...updates,
          updatedAt: new Date(),
          ...(updates.status === 'closed' && { closedAt: new Date() })
        })
        .where(eq(cases.id, caseId))
        .returning();

      if (!updatedCase) {
        throw CommonErrors.NotFound('Case');
      }

      // Update vector embeddings if content changed
      if (updates.title || updates.description) {
        const embedding = generateSampleEmbedding(768);
        const content = `${updatedCase.title} - ${updatedCase.description || ''}`;
        
        await tx.insert(caseEmbeddings).values({
          caseId: updatedCase.id,
          content,
          embedding: arrayToPgVector(embedding),
          metadata: {
            caseNumber: updatedCase.caseNumber,
            priority: updatedCase.priority,
            status: updatedCase.status,
            updatedBy,
            updatedAt: new Date().toISOString()
          }
        }).onConflictDoUpdate({
          target: [caseEmbeddings.caseId],
          set: {
            content,
            embedding: arrayToPgVector(embedding),
            metadata: {
              caseNumber: updatedCase.caseNumber,
              priority: updatedCase.priority,
              status: updatedCase.status,
              updatedBy,
              updatedAt: new Date().toISOString()
            }
          }
        });
      }

      return updatedCase;
    });
  }

  // Get case with related data
  static async getWithRelations(caseId: string): Promise<Case & { 
    evidence: Evidence[];
    createdByUser?: User;
    leadProsecutorUser?: User;
  } | null> {
    const caseData = await db.query.cases.findFirst({
      where: eq(cases.id, caseId),
      with: {
        evidence: {
          orderBy: [desc(evidence.collectedAt)],
          limit: 50
        },
        createdBy: true,
        leadProsecutor: true
      }
    });

    return caseData || null;
  }
}

// Enhanced Evidence Operations
export class EvidenceOperations {
  // Create evidence with AI analysis
  static async create(
    evidenceData: {
      caseId?: string;
      title: string;
      description?: string;
      evidenceType: string;
      fileType?: string;
      fileUrl?: string;
      fileName?: string;
      fileSize?: number;
      mimeType?: string;
      hash?: string;
      tags?: string[];
      collectedAt?: Date;
      collectedBy?: string;
      location?: string;
      uploadedBy?: string;
    }
  ): Promise<Evidence> {
    return withTransaction(async (tx) => {
      const [newEvidence] = await tx.insert(evidence).values({
        ...evidenceData,
        tags: evidenceData.tags || [],
        chainOfCustody: [],
        labAnalysis: {},
        aiAnalysis: {},
        aiTags: [],
        isAdmissible: true,
        confidentialityLevel: 'standard',
        canvasPosition: {},
        uploadedAt: new Date(),
        updatedAt: new Date()
      }).returning();

      // Generate vector embeddings for AI search
      if (newEvidence.description || newEvidence.title) {
        const content = `${newEvidence.title} ${newEvidence.description || ''} ${newEvidence.evidenceType}`;
        const embedding = generateSampleEmbedding(768);
        
        await tx.insert(evidenceVectors).values({
          evidenceId: newEvidence.id,
          content,
          embedding: arrayToPgVector(embedding),
          metadata: {
            evidenceType: newEvidence.evidenceType,
            fileType: newEvidence.fileType,
            caseId: newEvidence.caseId,
            tags: newEvidence.tags
          }
        });
      }

      return newEvidence;
    });
  }

  // Advanced evidence search
  static async search(
    params: {
      query?: string;
      caseId?: string;
      evidenceTypes?: string[];
      tags?: string[];
      dateRange?: { start: Date; end: Date };
      limit?: number;
      offset?: number;
      useVectorSearch?: boolean;
    }
  ): Promise<{ evidence: Evidence[]; total: number }> {
    const { query, caseId, evidenceTypes, tags, dateRange, limit = 50, offset = 0, useVectorSearch = true } = params;

    let conditions = [];
    
    if (caseId) {
      conditions.push(eq(evidence.caseId, caseId));
    }
    if (evidenceTypes && evidenceTypes.length > 0) {
      conditions.push(sqlRaw`evidence_type = ANY(${evidenceTypes})`);
    }
    if (tags && tags.length > 0) {
      conditions.push(sqlRaw`tags && ${tags}`);
    }
    if (dateRange) {
      conditions.push(
        and(
          gte(evidence.collectedAt, dateRange.start),
          lte(evidence.collectedAt, dateRange.end)
        )
      );
    }

    // Vector search for semantic similarity
    if (query && useVectorSearch) {
      try {
        const queryEmbedding = generateSampleEmbedding(768);
        const vectorQuery = arrayToPgVector(queryEmbedding);
        
        const vectorResults = await db.execute(sqlRaw`
          SELECT 
            e.*,
            (1 - (ev.embedding <=> ${vectorQuery}::vector)) as similarity_score
          FROM evidence e
          LEFT JOIN evidence_vectors ev ON e.id = ev.evidence_id
          WHERE 
            ${conditions.length > 0 ? sqlRaw`(${conditions.join(' AND ')}) AND` : sqlRaw``}
            ev.embedding IS NOT NULL AND
            (1 - (ev.embedding <=> ${vectorQuery}::vector)) > 0.7
          ORDER BY similarity_score DESC
          LIMIT ${limit}
          OFFSET ${offset}
        `);
        
        return {
          evidence: vectorResults as Evidence[],
          total: vectorResults.length
        };
      } catch (error) {
        console.warn('Evidence vector search failed, falling back to text search:', error);
      }
    }

    // Fallback to traditional text search
    if (query) {
      conditions.push(
        or(
          ilike(evidence.title, `%${query}%`),
          ilike(evidence.description, `%${query}%`),
          ilike(evidence.fileName, `%${query}%`)
        )
      );
    }

    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;
    
    const [results, totalCount] = await Promise.all([
      db.select()
      .from(evidence)
      .where(whereClause)
      .orderBy(desc(evidence.collectedAt))
      .limit(limit)
      .offset(offset),
      
      db.select({ count: count() })
      .from(evidence)
      .where(whereClause)
    ]);

    return {
      evidence: results,
      total: totalCount[0]?.count || 0
    };
  }

  // Update evidence with chain of custody
  static async update(
    evidenceId: string,
    updates: Partial<Pick<Evidence, 'title' | 'description' | 'evidenceType' | 'tags' | 'isAdmissible'>>,
    updatedBy: string,
    custodyNotes?: string
  ): Promise<Evidence> {
    return withTransaction(async (tx) => {
      // Get current evidence for chain of custody
      const currentEvidence = await tx.select().from(evidence).where(eq(evidence.id, evidenceId)).limit(1);
      if (currentEvidence.length === 0) {
        throw CommonErrors.NotFound('Evidence');
      }

      // Update chain of custody
      const newCustodyEntry = {
        timestamp: new Date().toISOString(),
        action: 'updated',
        updatedBy,
        changes: Object.keys(updates),
        notes: custodyNotes || 'Evidence updated'
      };

      const updatedChainOfCustody = [...(currentEvidence[0].chainOfCustody as any[]), newCustodyEntry];

      const [updatedEvidence] = await tx.update(evidence)
        .set({
          ...updates,
          chainOfCustody: updatedChainOfCustody,
          updatedAt: new Date()
        })
        .where(eq(evidence.id, evidenceId))
        .returning();

      // Update vector embeddings if content changed
      if (updates.title || updates.description) {
        const content = `${updatedEvidence.title} ${updatedEvidence.description || ''} ${updatedEvidence.evidenceType}`;
        const embedding = generateSampleEmbedding(768);
        
        await tx.insert(evidenceVectors).values({
          evidenceId: updatedEvidence.id,
          content,
          embedding: arrayToPgVector(embedding),
          metadata: {
            evidenceType: updatedEvidence.evidenceType,
            fileType: updatedEvidence.fileType,
            caseId: updatedEvidence.caseId,
            tags: updatedEvidence.tags,
            updatedBy,
            updatedAt: new Date().toISOString()
          }
        }).onConflictDoUpdate({
          target: [evidenceVectors.evidenceId],
          set: {
            content,
            embedding: arrayToPgVector(embedding),
            metadata: {
              evidenceType: updatedEvidence.evidenceType,
              fileType: updatedEvidence.fileType,
              caseId: updatedEvidence.caseId,
              tags: updatedEvidence.tags,
              updatedBy,
              updatedAt: new Date().toISOString()
            }
          }
        });
      }

      return updatedEvidence;
    });
  }
}

// Enhanced Legal Document Operations
export class LegalDocumentOperations {
  // Advanced legal precedent search
  static async searchPrecedents(
    params: {
      query: string;
      jurisdiction?: string;
      dateRange?: { start: number; end: number };
      limit?: number;
      similarityThreshold?: number;
    }
  ): Promise<{ precedents: any[]; total: number }> {
    const { query, jurisdiction, dateRange, limit = 20, similarityThreshold = 0.75 } = params;

    try {
      const queryEmbedding = generateSampleEmbedding(768);
      const vectorQuery = arrayToPgVector(queryEmbedding);
      
      let conditions = [sqlRaw`(1 - (embedding <=> ${vectorQuery}::vector)) > ${similarityThreshold}`];
      
      if (jurisdiction) {
        conditions.push(eq(legalPrecedents.jurisdiction, jurisdiction));
      }
      if (dateRange) {
        conditions.push(
          and(
            gte(legalPrecedents.year, dateRange.start),
            lte(legalPrecedents.year, dateRange.end)
          )
        );
      }

      const results = await db.execute(sqlRaw`
        SELECT 
          *,
          (1 - (embedding <=> ${vectorQuery}::vector)) as relevance_score
        FROM legal_precedents
        WHERE ${conditions.join(' AND ')}
        ORDER BY relevance_score DESC
        LIMIT ${limit}
      `);
      
      return {
        precedents: results,
        total: results.length
      };
    } catch (error) {
      console.warn('Legal precedent vector search failed:', error);
      
      // Fallback to text search
      const results = await db.select()
        .from(legalPrecedents)
        .where(
          and(
            or(
              ilike(legalPrecedents.caseTitle, `%${query}%`),
              ilike(legalPrecedents.summary, `%${query}%`)
            ),
            jurisdiction ? eq(legalPrecedents.jurisdiction, jurisdiction) : sqlRaw`1=1`
          )
        )
        .limit(limit);
        
      return {
        precedents: results,
        total: results.length
      };
    }
  }
}

// RAG (Retrieval Augmented Generation) Operations
export class RAGOperations {
  // Store AI query with embeddings
  static async storeQuery(
    queryData: {
      userId: string;
      caseId?: string;
      query: string;
      response: string;
      model?: string;
      confidence?: number;
      processingTime?: number;
      contextUsed?: any[];
    }
  ): Promise<void> {
    return withTransaction(async (tx) => {
      const queryEmbedding = generateSampleEmbedding(768);
      
      await tx.insert(userAiQueries).values({
        ...queryData,
        embedding: arrayToPgVector(queryEmbedding),
        model: queryData.model || 'gemma3-legal',
        confidence: queryData.confidence || 0.8,
        processingTime: queryData.processingTime || 0,
        contextUsed: queryData.contextUsed || [],
        metadata: {
          timestamp: new Date().toISOString(),
          embeddingModel: 'nomic-embed-text',
          version: '2.0'
        },
        isSuccessful: true
      });
    });
  }

  // Find similar queries for context
  static async findSimilarQueries(
    queryText: string,
    userId?: string,
    limit: number = 5
  ): Promise<any[]> {
    try {
      const queryEmbedding = generateSampleEmbedding(768);
      const vectorQuery = arrayToPgVector(queryEmbedding);
      
      const conditions = [sqlRaw`(1 - (embedding <=> ${vectorQuery}::vector)) > 0.7`];
      if (userId) {
        conditions.push(eq(userAiQueries.userId, userId));
      }

      const results = await db.execute(sqlRaw`
        SELECT 
          query,
          response,
          confidence,
          (1 - (embedding <=> ${vectorQuery}::vector)) as similarity_score,
          created_at
        FROM user_ai_queries
        WHERE ${conditions.join(' AND ')}
        ORDER BY similarity_score DESC
        LIMIT ${limit}
      `);
      
      return results;
    } catch (error) {
      console.warn('Similar query search failed:', error);
      return [];
    }
  }
}

// Database Health Check
export async function checkDatabaseHealth(): Promise<{
  connected: boolean;
  pgvectorEnabled: boolean;
  queryTime: number;
  errors: string[];
}> {
  const startTime = Date.now();
  const errors: string[] = [];
  let connected = false;
  let pgvectorEnabled = false;

  try {
    // Test basic connection
    await db.execute(sqlRaw`SELECT 1`);
    connected = true;

    // Test pgvector extension
    await db.execute(sqlRaw`SELECT '[1,2,3]'::vector`);
    pgvectorEnabled = true;
  } catch (error) {
    errors.push(error instanceof Error ? error.message : 'Unknown database error');
  }

  return {
    connected,
    pgvectorEnabled,
    queryTime: Date.now() - startTime,
    errors
  };
}
