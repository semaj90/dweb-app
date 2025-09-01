/**
 * JSONB Legal Service
 * 
 * High-performance service for managing legal metadata using PostgreSQL JSONB.
 * Integrates with the unified dimensional storage system and provides
 * optimized query patterns for legal document operations.
 * 
 * Features:
 * - Legal-specific JSONB schema operations
 * - Optimized query patterns with GIN indexes
 * - Integration with vector embeddings and graph data
 * - Real-time analytics and reporting
 * - Chain of custody verification for evidence
 * - Citation network analysis
 */

import { drizzle } from 'drizzle-orm/node-postgres';
import { eq, sql, and, or, gt, lt, desc, asc, inArray, like, ilike } from 'drizzle-orm';
import postgres from 'postgres';
import { 
  legalDocumentsJsonb, 
  casesJsonb, 
  evidenceJsonb, 
  documentRelationshipsJsonb,
  LegalJsonbOperations,
  type LegalDocument,
  type NewLegalDocument,
  type Case,
  type NewCase,
  type Evidence,
  type NewEvidence,
  type LegalMetadata,
  type CaseMetadata,
  type EvidenceMetadata
} from '$lib/server/db/jsonb-legal-schema.js';
import { logger } from '$lib/logging/structured-logger.js';
import type { VectorEmbedding } from '$lib/storage/vector-quantization.js';

// PostgreSQL connection with JSONB optimizations
const connectionString = import.meta.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';
const sql_client = postgres(connectionString, {
  max: 20,
  idle_timeout: 20,
  connect_timeout: 60,
  prepare: false, // Disable prepared statements for JSONB flexibility
});

const db = drizzle(sql_client);

// ============================================================================
// CORE JSONB LEGAL SERVICE
// ============================================================================

export class JsonbLegalService {
  private static instance: JsonbLegalService;

  static getInstance(): JsonbLegalService {
    if (!JsonbLegalService.instance) {
      JsonbLegalService.instance = new JsonbLegalService();
    }
    return JsonbLegalService.instance;
  }

  // ========================================================================
  // LEGAL DOCUMENT OPERATIONS
  // ========================================================================

  /**
   * Create legal document with optimized JSONB metadata
   */
  async createLegalDocument(
    data: Omit<NewLegalDocument, 'id' | 'createdAt' | 'updatedAt'>,
    embeddings?: { title: VectorEmbedding; content: VectorEmbedding }
  ): Promise<LegalDocument> {
    const startTime = performance.now();

    try {
      const documentData: NewLegalDocument = {
        ...data,
        titleEmbedding: embeddings?.title ? `[${embeddings.title.join(',')}]` : null,
        contentEmbedding: embeddings?.content ? `[${embeddings.content.join(',')}]` : null,
      };

      const [document] = await db
        .insert(legalDocumentsJsonb)
        .values(documentData)
        .returning();

      const duration = performance.now() - startTime;

      await logger.logDocumentProcessing({
        documentId: document.id,
        operation: 'create',
        documentType: data.metadata.documentType,
        processingTime: duration,
        hasEmbeddings: !!embeddings,
        metadataSize: JSON.stringify(data.metadata).length,
        success: true
      });

      return document;
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logDocumentProcessing({
        operation: 'create',
        documentType: data.metadata.documentType,
        processingTime: duration,
        hasEmbeddings: !!embeddings,
        metadataSize: JSON.stringify(data.metadata).length,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  /**
   * Advanced JSONB metadata queries with legal context
   */
  async findDocumentsByLegalCriteria(criteria: {
    documentTypes?: string[];
    practiceAreas?: string[];
    jurisdictions?: string[];
    confidentialityLevels?: string[];
    minConfidence?: number;
    hasHumanVerification?: boolean;
    dateRange?: { start: Date; end: Date };
    keyTerms?: string[];
    parties?: { name: string; role?: string }[];
    limit?: number;
    offset?: number;
  }): Promise<{ documents: LegalDocument[]; totalCount: number }> {
    const startTime = performance.now();

    try {
      let query = db.select().from(legalDocumentsJsonb);
      let countQuery = db.select({ count: sql<number>`count(*)` }).from(legalDocumentsJsonb);

      const conditions = [];

      // Document type filter
      if (criteria.documentTypes?.length) {
        conditions.push(inArray(legalDocumentsJsonb.documentType, criteria.documentTypes));
      }

      // Practice area filter
      if (criteria.practiceAreas?.length) {
        conditions.push(inArray(legalDocumentsJsonb.practiceArea, criteria.practiceAreas));
      }

      // Jurisdiction filter
      if (criteria.jurisdictions?.length) {
        conditions.push(inArray(legalDocumentsJsonb.jurisdiction, criteria.jurisdictions));
      }

      // Confidentiality filter
      if (criteria.confidentialityLevels?.length) {
        conditions.push(inArray(legalDocumentsJsonb.confidentialityLevel, criteria.confidentialityLevels));
      }

      // AI confidence filter
      if (criteria.minConfidence !== undefined) {
        conditions.push(
          sql`(metadata->'aiMetadata'->>'confidence')::real >= ${criteria.minConfidence}`
        );
      }

      // Human verification filter
      if (criteria.hasHumanVerification !== undefined) {
        conditions.push(
          sql`(metadata->'aiMetadata'->>'humanVerified')::boolean = ${criteria.hasHumanVerification}`
        );
      }

      // Date range filter
      if (criteria.dateRange) {
        conditions.push(
          and(
            gt(legalDocumentsJsonb.createdAt, criteria.dateRange.start),
            lt(legalDocumentsJsonb.createdAt, criteria.dateRange.end)
          )
        );
      }

      // Key terms filter (JSONB array contains)
      if (criteria.keyTerms?.length) {
        for (const term of criteria.keyTerms) {
          conditions.push(
            sql`metadata->'semantics'->'keyTerms' ? ${term}`
          );
        }
      }

      // Party filter (JSONB array element matching)
      if (criteria.parties?.length) {
        for (const party of criteria.parties) {
          if (party.role) {
            conditions.push(
              sql`EXISTS (
                SELECT 1 FROM jsonb_array_elements(metadata->'parties') AS p
                WHERE p->>'name' ILIKE ${`%${party.name}%`}
                  AND p->>'role' = ${party.role}
              )`
            );
          } else {
            conditions.push(
              sql`EXISTS (
                SELECT 1 FROM jsonb_array_elements(metadata->'parties') AS p
                WHERE p->>'name' ILIKE ${`%${party.name}%`}
              )`
            );
          }
        }
      }

      // Apply conditions
      if (conditions.length > 0) {
        const whereCondition = and(...conditions);
        query = query.where(whereCondition);
        countQuery = countQuery.where(whereCondition);
      }

      // Apply pagination and ordering
      query = query
        .orderBy(desc(legalDocumentsJsonb.updatedAt))
        .limit(criteria.limit || 50)
        .offset(criteria.offset || 0);

      // Execute queries
      const [documents, [{ count: totalCount }]] = await Promise.all([
        query,
        countQuery
      ]);

      const duration = performance.now() - startTime;

      await logger.logSearch({
        searchType: 'jsonb_legal_criteria',
        query: JSON.stringify(criteria),
        resultsCount: documents.length,
        totalResults: totalCount,
        processingTime: duration,
        indexesUsed: ['metadata_gin', 'practice_area', 'document_type'],
        success: true
      });

      return { documents, totalCount };
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logSearch({
        searchType: 'jsonb_legal_criteria',
        query: JSON.stringify(criteria),
        resultsCount: 0,
        totalResults: 0,
        processingTime: duration,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  /**
   * Legal concept extraction and analysis
   */
  async analyzeLegalConcepts(documentIds: string[]): Promise<{
    concepts: Array<{
      concept: string;
      frequency: number;
      documentIds: string[];
      documentTitles: string[];
      importanceLevel: 'CORE' | 'IMPORTANT' | 'RELEVANT' | 'PERIPHERAL';
    }>;
    totalDocuments: number;
    conceptNetwork: Record<string, string[]>;
  }> {
    const startTime = performance.now();

    try {
      // Execute the complex JSONB concept analysis query
      const conceptResults = await sql_client.unsafe(
        LegalJsonbOperations.extractLegalConcepts(documentIds).strings[0],
        ...LegalJsonbOperations.extractLegalConcepts(documentIds).values
      );

      // Build concept network (concepts that appear together)
      const conceptNetwork: Record<string, string[]> = {};
      const documentConcepts = await db
        .select({
          documentId: legalDocumentsJsonb.id,
          concepts: sql<string[]>`
            ARRAY(
              SELECT jsonb_array_elements_text(metadata->'semantics'->'legalConcepts')
            )
          `
        })
        .from(legalDocumentsJsonb)
        .where(inArray(legalDocumentsJsonb.id, documentIds));

      // Calculate co-occurrence network
      for (const doc of documentConcepts) {
        if (doc.concepts && doc.concepts.length > 1) {
          for (let i = 0; i < doc.concepts.length; i++) {
            for (let j = i + 1; j < doc.concepts.length; j++) {
              const concept1 = doc.concepts[i];
              const concept2 = doc.concepts[j];
              
              if (!conceptNetwork[concept1]) conceptNetwork[concept1] = [];
              if (!conceptNetwork[concept2]) conceptNetwork[concept2] = [];
              
              if (!conceptNetwork[concept1].includes(concept2)) {
                conceptNetwork[concept1].push(concept2);
              }
              if (!conceptNetwork[concept2].includes(concept1)) {
                conceptNetwork[concept2].push(concept1);
              }
            }
          }
        }
      }

      const duration = performance.now() - startTime;

      await logger.logAIInteraction({
        operation: 'legal_concept_analysis',
        inputSize: documentIds.length,
        outputSize: conceptResults.length,
        processingTime: duration,
        model: 'jsonb_analysis',
        success: true
      });

      return {
        concepts: conceptResults as any[],
        totalDocuments: documentIds.length,
        conceptNetwork
      };
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logAIInteraction({
        operation: 'legal_concept_analysis',
        inputSize: documentIds.length,
        outputSize: 0,
        processingTime: duration,
        model: 'jsonb_analysis',
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  // ========================================================================
  // CASE MANAGEMENT OPERATIONS
  // ========================================================================

  /**
   * Create case with JSONB metadata
   */
  async createCase(data: Omit<NewCase, 'id' | 'createdAt' | 'updatedAt'>): Promise<Case> {
    const startTime = performance.now();

    try {
      const [caseRecord] = await db
        .insert(casesJsonb)
        .values(data)
        .returning();

      const duration = performance.now() - startTime;

      await logger.logDocumentProcessing({
        documentId: caseRecord.id,
        operation: 'create_case',
        documentType: 'case',
        processingTime: duration,
        metadataSize: JSON.stringify(data.metadata).length,
        success: true
      });

      return caseRecord;
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logDocumentProcessing({
        operation: 'create_case',
        documentType: 'case',
        processingTime: duration,
        metadataSize: data.metadata ? JSON.stringify(data.metadata).length : 0,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  /**
   * Find similar cases using JSONB metadata analysis
   */
  async findSimilarCases(caseId: string, threshold = 0.8): Promise<Array<{
    id: string;
    title: string;
    metadata: CaseMetadata;
    similarityScore: number;
    similarityLevel: 'VERY_HIGH' | 'HIGH' | 'MEDIUM' | 'LOW';
  }>> {
    const startTime = performance.now();

    try {
      const similarCases = await sql_client.unsafe(
        LegalJsonbOperations.findSimilarCases(caseId, threshold).strings[0],
        ...LegalJsonbOperations.findSimilarCases(caseId, threshold).values
      );

      const duration = performance.now() - startTime;

      await logger.logAIInteraction({
        operation: 'case_similarity_analysis',
        inputSize: 1,
        outputSize: similarCases.length,
        processingTime: duration,
        model: 'jsonb_similarity',
        contextMetadata: { caseId, threshold },
        success: true
      });

      return similarCases as any[];
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logAIInteraction({
        operation: 'case_similarity_analysis',
        inputSize: 1,
        outputSize: 0,
        processingTime: duration,
        model: 'jsonb_similarity',
        contextMetadata: { caseId, threshold },
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  /**
   * Update case timeline with JSONB array operations
   */
  async addCaseTimelineEvent(
    caseId: string, 
    event: {
      date: string;
      event: string;
      significance: 'low' | 'medium' | 'high' | 'critical';
    }
  ): Promise<Case> {
    const startTime = performance.now();

    try {
      const [updatedCase] = await db
        .update(casesJsonb)
        .set({
          metadata: sql`
            jsonb_set(
              metadata,
              '{timeline}',
              COALESCE(metadata->'timeline', '[]'::jsonb) || ${JSON.stringify(event)}::jsonb
            )
          `,
          updatedAt: new Date()
        })
        .where(eq(casesJsonb.id, caseId))
        .returning();

      const duration = performance.now() - startTime;

      await logger.logDocumentProcessing({
        documentId: caseId,
        operation: 'update_timeline',
        documentType: 'case',
        processingTime: duration,
        metadataSize: JSON.stringify(event).length,
        success: true
      });

      return updatedCase;
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logDocumentProcessing({
        documentId: caseId,
        operation: 'update_timeline',
        documentType: 'case',
        processingTime: duration,
        metadataSize: JSON.stringify(event).length,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  // ========================================================================
  // EVIDENCE OPERATIONS WITH CHAIN OF CUSTODY
  // ========================================================================

  /**
   * Create evidence with chain of custody
   */
  async createEvidence(
    data: Omit<NewEvidence, 'id' | 'createdAt' | 'updatedAt'>,
    embedding?: VectorEmbedding
  ): Promise<Evidence> {
    const startTime = performance.now();

    try {
      const evidenceData: NewEvidence = {
        ...data,
        embedding: embedding ? `[${embedding.join(',')}]` : null,
      };

      const [evidence] = await db
        .insert(evidenceJsonb)
        .values(evidenceData)
        .returning();

      const duration = performance.now() - startTime;

      await logger.logDocumentProcessing({
        documentId: evidence.id,
        operation: 'create_evidence',
        documentType: 'evidence',
        processingTime: duration,
        hasEmbeddings: !!embedding,
        metadataSize: JSON.stringify(data.metadata).length,
        success: true
      });

      return evidence;
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logDocumentProcessing({
        operation: 'create_evidence',
        documentType: 'evidence',
        processingTime: duration,
        hasEmbeddings: !!embedding,
        metadataSize: JSON.stringify(data.metadata).length,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  /**
   * Add custody transfer to evidence chain
   */
  async addCustodyTransfer(
    evidenceId: string,
    transfer: {
      timestamp: string;
      custodian: string;
      action: 'collected' | 'transferred' | 'analyzed' | 'stored' | 'retrieved';
      location?: string;
      condition?: string;
    }
  ): Promise<Evidence> {
    const startTime = performance.now();

    try {
      const [updatedEvidence] = await db
        .update(evidenceJsonb)
        .set({
          metadata: sql`
            jsonb_set(
              metadata,
              '{chainOfCustody}',
              COALESCE(metadata->'chainOfCustody', '[]'::jsonb) || ${JSON.stringify(transfer)}::jsonb
            )
          `,
          updatedAt: new Date()
        })
        .where(eq(evidenceJsonb.id, evidenceId))
        .returning();

      const duration = performance.now() - startTime;

      await logger.logSecurityEvent({
        eventType: 'custody_transfer',
        resourceId: evidenceId,
        resourceType: 'evidence',
        action: 'update_custody_chain',
        actor: transfer.custodian,
        metadata: { action: transfer.action, location: transfer.location },
        success: true,
        processingTime: duration
      });

      return updatedEvidence;
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logSecurityEvent({
        eventType: 'custody_transfer',
        resourceId: evidenceId,
        resourceType: 'evidence',
        action: 'update_custody_chain',
        actor: transfer.custodian,
        metadata: { action: transfer.action, location: transfer.location },
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime: duration
      });

      throw error;
    }
  }

  /**
   * Verify evidence chain of custody integrity
   */
  async verifyEvidenceChain(evidenceId: string): Promise<{
    isValid: boolean;
    evidence: Evidence;
    chainValidation: {
      chronologicallyValid: boolean;
      custodyCount: number;
      currentCustodian: string;
      gaps: Array<{ from: string; to: string; duration: number }>;
      transfers: Array<{
        timestamp: string;
        custodian: string;
        action: string;
        location?: string;
        condition?: string;
      }>;
    };
  }> {
    const startTime = performance.now();

    try {
      const verificationResult = await sql_client.unsafe(
        LegalJsonbOperations.verifyEvidenceChain(evidenceId).strings[0],
        ...LegalJsonbOperations.verifyEvidenceChain(evidenceId).values
      );

      if (!verificationResult.length) {
        throw new Error(`Evidence not found: ${evidenceId}`);
      }

      const result = verificationResult[0] as any;
      const evidence = result as Evidence;

      // Analyze chain for gaps and issues
      const chain = (evidence.metadata as EvidenceMetadata).chainOfCustody || [];
      const gaps = [];
      
      for (let i = 1; i < chain.length; i++) {
        const prevTime = new Date(chain[i-1].timestamp);
        const currTime = new Date(chain[i].timestamp);
        const duration = currTime.getTime() - prevTime.getTime();
        
        // Flag gaps longer than 24 hours
        if (duration > 24 * 60 * 60 * 1000) {
          gaps.push({
            from: chain[i-1].custodian,
            to: chain[i].custodian,
            duration: Math.round(duration / (60 * 60 * 1000)) // hours
          });
        }
      }

      const chainValidation = {
        chronologicallyValid: result.chain_validity === 'VALID',
        custodyCount: result.custody_count,
        currentCustodian: result.current_custodian,
        gaps,
        transfers: chain
      };

      const isValid = chainValidation.chronologicallyValid && gaps.length === 0;
      const duration = performance.now() - startTime;

      await logger.logSecurityEvent({
        eventType: 'custody_verification',
        resourceId: evidenceId,
        resourceType: 'evidence',
        action: 'verify_chain',
        metadata: { 
          isValid, 
          custodyCount: chainValidation.custodyCount,
          gapCount: gaps.length 
        },
        success: true,
        processingTime: duration
      });

      return { isValid, evidence, chainValidation };
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logSecurityEvent({
        eventType: 'custody_verification',
        resourceId: evidenceId,
        resourceType: 'evidence',
        action: 'verify_chain',
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime: duration
      });

      throw error;
    }
  }

  // ========================================================================
  // CITATION NETWORK ANALYSIS
  // ========================================================================

  /**
   * Build citation network using JSONB recursive queries
   */
  async buildCitationNetwork(documentId: string, depth = 2): Promise<{
    nodes: Array<{
      id: string;
      title: string;
      documentType: string;
      depth: number;
      citationCount: number;
    }>;
    edges: Array<{
      from: string;
      to: string;
      type: string;
      relevance?: number;
    }>;
  }> {
    const startTime = performance.now();

    try {
      const networkResult = await sql_client.unsafe(
        LegalJsonbOperations.findCitationNetwork(documentId, depth).strings[0],
        ...LegalJsonbOperations.findCitationNetwork(documentId, depth).values
      );

      // Process results into nodes and edges
      const nodes = networkResult.map((row: any) => ({
        id: row.id,
        title: row.title,
        documentType: row.document_type || 'unknown',
        depth: row.depth,
        citationCount: Array.isArray(row.citations) ? row.citations.length : 0
      }));

      // Build edges from citation relationships
      const edges = [];
      for (const row of networkResult) {
        if (row.citations && Array.isArray(row.citations)) {
          for (const citation of row.citations) {
            // Try to find the cited document in our network
            const targetNode = nodes.find(n => 
              n.title.toLowerCase().includes(citation.citation?.toLowerCase()) ||
              citation.citation?.toLowerCase().includes(n.title.toLowerCase())
            );

            if (targetNode && targetNode.id !== row.id) {
              edges.push({
                from: row.id,
                to: targetNode.id,
                type: citation.type || 'citation',
                relevance: citation.relevance
              });
            }
          }
        }
      }

      const duration = performance.now() - startTime;

      await logger.logAIInteraction({
        operation: 'citation_network_analysis',
        inputSize: 1,
        outputSize: nodes.length,
        processingTime: duration,
        model: 'jsonb_network',
        contextMetadata: { documentId, depth, edgeCount: edges.length },
        success: true
      });

      return { nodes, edges };
    } catch (error: any) {
      const duration = performance.now() - startTime;
      
      await logger.logAIInteraction({
        operation: 'citation_network_analysis',
        inputSize: 1,
        outputSize: 0,
        processingTime: duration,
        model: 'jsonb_network',
        contextMetadata: { documentId, depth },
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  // ========================================================================
  // PERFORMANCE ANALYTICS
  // ========================================================================

  /**
   * Get JSONB performance metrics
   */
  async getPerformanceMetrics(): Promise<{
    indexUsage: Array<{
      tableName: string;
      indexName: string;
      indexSize: string;
      indexScans: number;
      tuplesRead: number;
      tuplesFetched: number;
    }>;
    storageEfficiency: Array<{
      tableName: string;
      totalSize: string;
      jsonbSizeEstimate: string;
      rowCount: number;
      avgJsonbSizeBytes: number;
    }>;
    queryPerformance: {
      averageQueryTime: number;
      slowQueries: number;
      indexHitRatio: number;
    };
  }> {
    try {
      const [indexUsage, storageEfficiency] = await Promise.all([
        sql_client.unsafe('SELECT * FROM analyze_jsonb_performance()'),
        sql_client.unsafe('SELECT * FROM analyze_jsonb_storage()')
      ]);

      // Mock query performance data (would be from pg_stat_statements in production)
      const queryPerformance = {
        averageQueryTime: 45.2, // ms
        slowQueries: 3,
        indexHitRatio: 0.95
      };

      return {
        indexUsage: indexUsage as any[],
        storageEfficiency: storageEfficiency as any[],
        queryPerformance
      };
    } catch (error: any) {
      await logger.logError({
        error: error instanceof Error ? error.message : 'Unknown error',
        context: 'jsonb_performance_metrics',
        severity: 'medium',
        category: 'database'
      });

      throw error;
    }
  }

  /**
   * Get legal analytics dashboard data
   */
  async getLegalAnalytics(): Promise<{
    documentAnalytics: Array<{
      documentType: string;
      practiceArea: string;
      jurisdiction: string;
      confidentialityLevel: string;
      documentCount: number;
      avgAiConfidence: number;
      humanVerifiedCount: number;
      latestDocument: Date;
      earliestDocument: Date;
    }>;
    caseAnalytics: Array<{
      status: string;
      caseCount: number;
      avgDocumentsPerCase: number;
      avgEvidencePerCase: number;
      avgCaseAgeDays: number;
    }>;
    evidenceIntegrity: Array<{
      id: string;
      title: string;
      evidenceType: string;
      authenticated: boolean;
      admissibilityStatus: string;
      custodySteps: number;
      chronologicallyValid: boolean;
    }>;
  }> {
    try {
      const [documentAnalytics, caseAnalytics, evidenceIntegrity] = await Promise.all([
        sql_client.unsafe('SELECT * FROM legal_document_analytics'),
        sql_client.unsafe('SELECT * FROM case_status_analytics'),
        sql_client.unsafe('SELECT * FROM evidence_chain_integrity')
      ]);

      return {
        documentAnalytics: documentAnalytics as any[],
        caseAnalytics: caseAnalytics as any[],
        evidenceIntegrity: evidenceIntegrity as any[]
      };
    } catch (error: any) {
      await logger.logError({
        error: error instanceof Error ? error.message : 'Unknown error',
        context: 'legal_analytics',
        severity: 'medium',
        category: 'analytics'
      });

      throw error;
    }
  }

  // ========================================================================
  // CLEANUP AND MAINTENANCE
  // ========================================================================

  /**
   * Update case document counters
   */
  async updateCaseCounters(): Promise<void> {
    try {
      await sql_client.unsafe('SELECT update_case_document_counters()');
      
      await logger.logSystemEvent({
        eventType: 'maintenance',
        operation: 'update_case_counters',
        success: true
      });
    } catch (error: any) {
      await logger.logSystemEvent({
        eventType: 'maintenance',
        operation: 'update_case_counters',
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      throw error;
    }
  }

  /**
   * Close database connections
   */
  async close(): Promise<void> {
    await sql_client.end();
  }
}

// Singleton export
export const jsonbLegalService = JsonbLegalService.getInstance();