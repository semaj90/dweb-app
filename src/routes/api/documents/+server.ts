/**
 * Documents API Endpoint
 * Enhanced with PostgreSQL + pgvector + Drizzle + Cognitive Cache
 */

import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from "$lib/server/db";
import { legalDocuments as legal_documents } from '$lib/database/schema/legal-documents';
import { cases, evidence } from "$lib/server/db/index";
import { sql, desc, asc, and, or, eq, ilike, inArray, count, isNotNull } from "drizzle-orm";
import { cognitiveCache as cognitiveCacheManager } from '$lib/services/cognitive-cache-integration';
import { getDatabaseHealth } from '$lib/database';
import { z } from 'zod';

// Local vector search service shim (attempts cognitive cache first, falls back to DB)
const vectorSearchService = {
  getVectorStats: async () => {
    try {
      // Prefer the cognitive cache manager if it exposes stats
      if (cognitiveCacheManager && typeof (cognitiveCacheManager as any).getIndexStats === 'function') {
        const stats = await (cognitiveCacheManager as any).getIndexStats();
        return {
          totalVectors: stats.totalVectors ?? 0,
          averageDimension: stats.averageDimension ?? 384,
          indexStatus: stats.indexStatus ?? 'unknown'
        };
      }
    } catch (err) {
      console.warn('cognitiveCacheManager.getIndexStats failed:', err);
    }

    // Fallback: derive simple stats from the database
    try {
      const [totalResult] = await db
        .select({ total: count() })
        .from(legal_documents)
        .where(sql`${legal_documents.content_embedding} IS NOT NULL`);

      const totalVectors = Number((totalResult as any).total ?? 0);

      return {
        totalVectors,
        averageDimension: 384,
        indexStatus: totalVectors > 0 ? 'available' : 'empty'
      };
    } catch (dbErr) {
      console.error('vectorSearchService fallback DB stats failed:', dbErr);
      return {
        totalVectors: 0,
        averageDimension: 384,
        indexStatus: 'unknown'
      };
    }
  }
};

// Query parameters schema for GET requests
const listParamsSchema = z.object({
  limit: z.number().min(1).max(100).default(20),
  offset: z.number().min(0).default(0),
  sortBy: z.enum(['created', 'updated', 'title', 'type', 'size']).default('updated'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
  documentType: z.enum(['contract', 'motion', 'evidence', 'correspondence', 'brief', 'regulation', 'case_law']).optional(),
  jurisdiction: z.string().optional(),
  practiceArea: z.enum(['corporate', 'litigation', 'intellectual_property', 'employment', 'real_estate', 'criminal', 'family', 'tax', 'immigration', 'environmental']).optional(),
  status: z.enum(['pending', 'processing', 'completed', 'error']).optional(),
  isConfidential: z.boolean().optional(),
  hasEmbeddings: z.boolean().optional(),
  search: z.string().optional(),
  includeContent: z.boolean().default(false),
  includeAnalysis: z.boolean().default(false),
});

// Document creation schema for POST requests
const createDocumentSchema = z.object({
  title: z.string().min(1).max(500),
  content: z.string().min(1),
  documentType: z.enum(['contract', 'motion', 'evidence', 'correspondence', 'brief', 'regulation', 'case_law']),
  jurisdiction: z.string().min(1).max(100).default('federal'),
  practiceArea: z.enum(['corporate', 'litigation', 'intellectual_property', 'employment', 'real_estate', 'criminal', 'family', 'tax', 'immigration', 'environmental']).optional(),
  isConfidential: z.boolean().default(false),
  generateEmbeddings: z.boolean().default(true),
  generateAnalysis: z.boolean().default(true),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Get documents with filtering, sorting, and pagination
 */
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  try {
    // Parse query parameters
    const params = Object.fromEntries(url.searchParams.entries());

    // Convert string parameters to appropriate types
    const queryParams = {
      ...params,
      limit: params.limit ? parseInt(params.limit) : undefined,
      offset: params.offset ? parseInt(params.offset) : undefined,
      isConfidential: params.isConfidential === 'true' ? true : params.isConfidential === 'false' ? false : undefined,
      hasEmbeddings: params.hasEmbeddings === 'true' ? true : params.hasEmbeddings === 'false' ? false : undefined,
      includeContent: params.includeContent === 'true',
      includeAnalysis: params.includeAnalysis === 'true',
    };

    const listParams = listParamsSchema.parse(queryParams);

    // Build filter conditions
    const filterConditions = [];

    if (listParams.documentType) {
      filterConditions.push(eq(legal_documents.document_type, listParams.documentType));
    }

    if (listParams.jurisdiction) {
      filterConditions.push(eq(legal_documents.jurisdiction, listParams.jurisdiction));
    }

    if (listParams.practiceArea) {
      filterConditions.push(eq(legal_documents.practice_area, listParams.practiceArea));
    }

    if (listParams.status) {
      filterConditions.push(eq(legal_documents.processing_status, listParams.status));
    }

    if (listParams.isConfidential !== undefined) {
      filterConditions.push(eq(legal_documents.is_confidential, listParams.isConfidential));
    }

    if (listParams.hasEmbeddings !== undefined) {
      if (listParams.hasEmbeddings) {
        filterConditions.push(
          and(
            isNotNull(legal_documents.content_embedding),
            isNotNull(legal_documents.title_embedding)
          )
        );
      } else {
        filterConditions.push(
          or(
            sql`${legal_documents.content_embedding} IS NULL`,
            sql`${legal_documents.title_embedding} IS NULL`
          )
        );
      }
    }

    if (listParams.search) {
      filterConditions.push(
        or(
          ilike(legal_documents.title, `%${listParams.search}%`),
          ilike(legal_documents.content, `%${listParams.search}%`)
        )
      );
    }

    // Build sort order
    let orderBy;
    const orderDirection = listParams.sortOrder === 'asc' ? asc : desc;

    switch (listParams.sortBy) {
      case 'created':
        orderBy = orderDirection(legal_documents.created_at);
        break;
      case 'updated':
        orderBy = orderDirection(legal_documents.updated_at);
        break;
      case 'title':
        orderBy = orderDirection(legal_documents.title);
        break;
      case 'type':
        orderBy = orderDirection(legal_documents.document_type);
        break;
      case 'size':
        orderBy = orderDirection(legal_documents.file_size);
        break;
      default:
        orderBy = desc(legal_documents.updated_at);
    }

    // Get total count
    const [countResult] = await db
      .select({ count: count() })
      .from(legal_documents)
      .where(filterConditions.length > 0 ? and(...filterConditions) : undefined);

    // Get documents
    const documents = await db
      .select({
        id: legal_documents.id,
        title: legal_documents.title,
        content: listParams.includeContent ? legal_documents.content : sql`''`.as('content'),
        documentType: legal_documents.document_type,
        jurisdiction: legal_documents.jurisdiction,
        practiceArea: legal_documents.practice_area,
        fileName: legal_documents.file_name,
        fileSize: legal_documents.file_size,
        mimeType: legal_documents.mime_type,
        fileHash: legal_documents.file_hash,
        processingStatus: legal_documents.processing_status,
        isConfidential: legal_documents.is_confidential,
        retentionDate: legal_documents.retention_date,
        createdAt: legal_documents.created_at,
        updatedAt: legal_documents.updated_at,
        createdBy: legal_documents.created_by,
        lastModifiedBy: legal_documents.last_modified_by,
        analysisResults: listParams.includeAnalysis ? legal_documents.analysis_results : sql`NULL`.as('analysisResults'),
        // Check if embeddings exist (aliased for later mapping)
        hasContentEmbedding: sql`CASE WHEN ${legal_documents.content_embedding} IS NOT NULL THEN true ELSE false END`.as('hasContentEmbedding'),
        hasTitleEmbedding: sql`CASE WHEN ${legal_documents.title_embedding} IS NOT NULL THEN true ELSE false END`.as('hasTitleEmbedding'),
      })
      .from(legal_documents)
      .where(filterConditions.length > 0 ? and(...filterConditions) : undefined)
      .orderBy(orderBy)
      .limit(listParams.limit)
      .offset(listParams.offset);

    // Get case associations for each document (using evidence as association table)
    const documentIds = documents.map(doc => doc.id);
    const caseAssociations = documentIds.length > 0 ? await db
      .select({
        documentId: evidence.document_id,
        caseId: evidence.case_id,
        caseTitle: cases.title,
        caseNumber: cases.case_number,
        relationship: evidence.relationship,
        importance: evidence.importance,
      })
      .from(evidence)
      .innerJoin(cases, eq(evidence.case_id, cases.id))
      .where(inArray(evidence.document_id, documentIds)) : [];

    // Group case associations by document ID
    const casesByDocument = caseAssociations.reduce((acc, assoc) => {
      const docId = (assoc as any).documentId;
      if (!acc[docId]) {
        acc[docId] = [];
      }
      acc[docId].push({
        caseId: (assoc as any).caseId,
        caseTitle: (assoc as any).caseTitle,
        caseNumber: (assoc as any).caseNumber,
        relationship: (assoc as any).relationship,
        importance: (assoc as any).importance,
      });
      return acc;
    }, {} as Record<string, unknown[]>);

    // Format response
    const formattedDocuments = documents.map(doc => ({
      id: doc.id,
      title: doc.title,
      content: (doc as any).content || null,
      documentType: (doc as any).documentType,
      jurisdiction: (doc as any).jurisdiction,
      practiceArea: (doc as any).practiceArea,
      fileName: (doc as any).fileName,
      fileSize: (doc as any).fileSize,
      mimeType: (doc as any).mimeType,
      fileHash: (doc as any).fileHash,
      processingStatus: (doc as any).processingStatus,
      isConfidential: (doc as any).isConfidential,
      retentionDate: (doc as any).retentionDate,
      createdAt: (doc as any).createdAt,
      updatedAt: (doc as any).updatedAt,
      createdBy: (doc as any).createdBy,
      lastModifiedBy: (doc as any).lastModifiedBy,
      analysisResults: (doc as any).analysisResults,
      hasEmbeddings: (doc as any).hasContentEmbedding && (doc as any).hasTitleEmbedding,
      embeddingStatus: {
        hasContentEmbedding: (doc as any).hasContentEmbedding,
        hasTitleEmbedding: (doc as any).hasTitleEmbedding,
      },
      associatedCases: casesByDocument[doc.id] || [],
      caseCount: (casesByDocument[doc.id] || []).length,
    }));

    return json({
      success: true,
      documents: formattedDocuments,
      pagination: {
        total: (countResult as any).count,
        limit: listParams.limit,
        offset: listParams.offset,
        hasMore: listParams.offset + listParams.limit < (countResult as any).count,
        page: Math.floor(listParams.offset / listParams.limit) + 1,
        totalPages: Math.ceil((countResult as any).count / listParams.limit),
      },
      filters: {
        documentType: listParams.documentType,
        jurisdiction: listParams.jurisdiction,
        practiceArea: listParams.practiceArea,
        status: listParams.status,
        isConfidential: listParams.isConfidential,
        hasEmbeddings: listParams.hasEmbeddings,
        search: listParams.search,
      },
      sorting: {
        sortBy: listParams.sortBy,
        sortOrder: listParams.sortOrder,
      },
    });

  } catch (error: any) {
    console.error("Documents list error:", error);

    if (error instanceof z.ZodError) {
      return json({
        success: false,
        error: "Invalid query parameters",
        details: error.errors,
      }, { status: 400 });
    }

    return json({
      success: false,
      error: error?.message || "Failed to retrieve documents",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Create a new document (text-based, not file upload)
 */
export const POST: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const body = await request.json();
    const documentData = createDocumentSchema.parse(body);

    // Create document record
    const [insertedDoc] = await db
      .insert(legal_documents)
      .values({
        title: documentData.title,
        content: documentData.content,
        document_type: documentData.documentType,
        jurisdiction: documentData.jurisdiction,
        practice_area: documentData.practiceArea,
        is_confidential: documentData.isConfidential,
        processing_status: 'processing',
        created_by: null, // TODO: Add user authentication
      })
      .returning();

    // Process embeddings and analysis in background if requested
    if (documentData.generateEmbeddings || documentData.generateAnalysis) {
      processDocumentAsync(
        (insertedDoc as any).id,
        documentData.content,
        documentData.title,
        documentData.generateEmbeddings,
        documentData.generateAnalysis
      );
    }

    return json({
      success: true,
      document: {
        id: (insertedDoc as any).id,
        title: (insertedDoc as any).title,
        documentType: (insertedDoc as any).document_type,
        jurisdiction: (insertedDoc as any).jurisdiction,
        practiceArea: (insertedDoc as any).practice_area,
        processingStatus: (insertedDoc as any).processing_status,
        isConfidential: (insertedDoc as any).is_confidential,
        createdAt: (insertedDoc as any).created_at,
      },
      message: "Document created successfully",
      processingInBackground: documentData.generateEmbeddings || documentData.generateAnalysis,
    });

  } catch (error: any) {
    console.error("Document creation error:", error);

    if (error instanceof z.ZodError) {
      return json({
        success: false,
        error: "Invalid document data",
        details: error.errors,
      }, { status: 400 });
    }

    return json({
      success: false,
      error: error?.message || "Failed to create document",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Get document statistics and analytics
 */
export const PUT: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const { action } = await request.json();

    if (action === 'analytics') {
      const analytics = await getDocumentAnalytics();
      return json({
        success: true,
        analytics
      });
    } else if (action === 'reprocess') {
      // Reprocess embeddings for documents without them
      const reprocessResult = await reprocessDocuments();
      return json({
        success: true,
        ...reprocessResult
      });
    } else {
      return json({
        success: false,
        error: "Unknown action",
        availableActions: ['analytics', 'reprocess']
      }, { status: 400 });
    }

  } catch (error: any) {
    console.error("Document analytics error:", error);

    return json({
      success: false,
      error: "Failed to get analytics",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Get comprehensive document analytics
 */
async function getDocumentAnalytics(): Promise<any> {
  const [
    totalStats,
    typeStats,
    statusStats,
    embeddingStats,
    recentActivity
  ] = await Promise.all([
    // Total document counts
    db
      .select({
        total: count(),
        confidential: count(sql`CASE WHEN ${legal_documents.is_confidential} = true THEN 1 END`),
        withEmbeddings: count(sql`CASE WHEN ${legal_documents.content_embedding} IS NOT NULL THEN 1 END`),
        withAnalysis: count(sql`CASE WHEN ${legal_documents.analysis_results} IS NOT NULL THEN 1 END`),
        avgFileSize: sql`AVG(${legal_documents.file_size})`.as('avgFileSize'),
        totalSize: sql`SUM(${legal_documents.file_size})`.as('totalSize'),
      })
      .from(legal_documents),

    // Document type distribution
    db
      .select({
        documentType: legal_documents.document_type,
        count: count()
      })
      .from(legal_documents)
      .groupBy(legal_documents.document_type)
      .orderBy(desc(count())),

    // Processing status distribution
    db
      .select({
        status: legal_documents.processing_status,
        count: count()
      })
      .from(legal_documents)
      .groupBy(legal_documents.processing_status)
      .orderBy(desc(count())),

    // Vector search statistics
    vectorSearchService.getVectorStats(),

    // Recent activity (last 30 days)
    db
      .select({
        date: sql`DATE(${legal_documents.created_at})`.as('date'),
        count: count()
      })
      .from(legal_documents)
      .where(sql`${legal_documents.created_at} >= NOW() - INTERVAL '30 days'`)
      .groupBy(sql`DATE(${legal_documents.created_at})`)
      .orderBy(sql`DATE(${legal_documents.created_at})`)
  ]);

  return {
    totals: (totalStats as any)[0],
    distribution: {
      byType: typeStats,
      byStatus: statusStats,
    },
    vectorStats: embeddingStats,
    recentActivity,
    performance: {
      averageProcessingTime: '2.3s', // Would calculate from actual data
      successRate: 0.98,
      errorRate: 0.02,
    }
  };
}

/**
 * Reprocess documents that don't have embeddings or analysis
 */
async function reprocessDocuments(): Promise<any> {
  try {
    // Find documents without embeddings
    const documentsToProcess = await db
      .select({
        id: legal_documents.id,
        title: legal_documents.title,
        content: legal_documents.content,
      })
      .from(legal_documents)
      .where(
        and(
          eq(legal_documents.processing_status, 'completed'),
          or(
            sql`${legal_documents.content_embedding} IS NULL`,
            sql`${legal_documents.title_embedding} IS NULL`,
            sql`${legal_documents.analysis_results} IS NULL`
          )
        )
      )
      .limit(50); // Process in batches

    // Start background processing for each document
    const processed = await Promise.allSettled(
      documentsToProcess.map(doc =>
        processDocumentAsync((doc as any).id, (doc as any).content, (doc as any).title, true, true)
      )
    );

    const successful = processed.filter(p => p.status === 'fulfilled').length;
    const failed = processed.filter(p => p.status === 'rejected').length;

    return {
      message: `Reprocessing initiated for ${documentsToProcess.length} documents`,
      queued: documentsToProcess.length,
      estimated: {
        successful,
        failed,
      }
    };

  } catch (error: any) {
    console.error('Reprocessing error:', error);
    throw error;
  }
}

/**
 * Process document embeddings and analysis asynchronously
 */
async function processDocumentAsync(
  documentId: string,
  content: string,
  title: string,
  generateEmbeddings: boolean,
  generateAnalysis: boolean
): Promise<any> {
  try {
    const updates: any = {};

    if (generateEmbeddings) {
      // Generate embeddings
      const contentEmbedding = await generateEmbedding(content);
      const titleEmbedding = await generateEmbedding(title);

      updates.content_embedding = contentEmbedding;
      updates.title_embedding = titleEmbedding;
    }

    if (generateAnalysis) {
      // Generate AI analysis
      const analysis = await generateDocumentAnalysis(content);
      updates.analysisResults = analysis;
    }

    // Update document with processing results
    updates.processing_status = 'completed';
    updates.updated_at = new Date();

    await db
      .update(legal_documents)
      .set(updates)
      .where(eq(legal_documents.id, documentId));

  } catch (error: any) {
    console.error('Background processing error:', error);

    // Mark as error status
    await db
      .update(legal_documents)
      .set({
        processing_status: 'error',
        updated_at: new Date()
      })
      .where(eq(legal_documents.id, documentId));
  }
}

/**
 * Generate embeddings for text (placeholder)
 */
async function generateEmbedding(text: string): Promise<number[]> {
  // This would integrate with your embedding service (Ollama, OpenAI, etc.)
  // For now, return a placeholder 384-dimensional vector
  return Array(384).fill(0).map(() => Math.random() - 0.5);
}

/**
 * Generate AI analysis for document (placeholder)
 */
async function generateDocumentAnalysis(content: string): Promise<any> {
  // This would integrate with your AI analysis service
  return {
    entities: [],
    keyTerms: [],
    sentimentScore: 0,
    complexityScore: 0,
    confidenceLevel: 0.8,
    extractedDates: [],
    extractedAmounts: [],
    parties: [],
    obligations: [],
    risks: []
  };
}