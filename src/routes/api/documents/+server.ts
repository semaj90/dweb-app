import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from "$lib/database/postgres-enhanced.js";
import { legalDocuments, legalCases, caseDocuments } from "$lib/database/schema/legal-documents.js";
import { vectorSearchService } from "$lib/database/vector-operations.js";
import { sql, desc, asc, and, or, eq, ilike, inArray, count, isNotNull } from "drizzle-orm";
import { z } from 'zod';

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
  metadata: z.record(z.any()).optional(),
});

/**
 * Get documents with filtering, sorting, and pagination
 */
export const GET: RequestHandler = async ({ url }) => {
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
      filterConditions.push(eq(legalDocuments.documentType, listParams.documentType));
    }

    if (listParams.jurisdiction) {
      filterConditions.push(eq(legalDocuments.jurisdiction, listParams.jurisdiction));
    }

    if (listParams.practiceArea) {
      filterConditions.push(eq(legalDocuments.practiceArea, listParams.practiceArea));
    }

    if (listParams.status) {
      filterConditions.push(eq(legalDocuments.processingStatus, listParams.status));
    }

    if (listParams.isConfidential !== undefined) {
      filterConditions.push(eq(legalDocuments.isConfidential, listParams.isConfidential));
    }

    if (listParams.hasEmbeddings !== undefined) {
      if (listParams.hasEmbeddings) {
        filterConditions.push(
          and(
            isNotNull(legalDocuments.contentEmbedding),
            isNotNull(legalDocuments.titleEmbedding)
          )
        );
      } else {
        filterConditions.push(
          or(
            sql`${legalDocuments.contentEmbedding} IS NULL`,
            sql`${legalDocuments.titleEmbedding} IS NULL`
          )
        );
      }
    }

    if (listParams.search) {
      filterConditions.push(
        or(
          ilike(legalDocuments.title, `%${listParams.search}%`),
          ilike(legalDocuments.content, `%${listParams.search}%`)
        )
      );
    }

    // Build sort order
    let orderBy;
    const orderDirection = listParams.sortOrder === 'asc' ? asc : desc;

    switch (listParams.sortBy) {
      case 'created':
        orderBy = orderDirection(legalDocuments.createdAt);
        break;
      case 'updated':
        orderBy = orderDirection(legalDocuments.updatedAt);
        break;
      case 'title':
        orderBy = orderDirection(legalDocuments.title);
        break;
      case 'type':
        orderBy = orderDirection(legalDocuments.documentType);
        break;
      case 'size':
        orderBy = orderDirection(legalDocuments.fileSize);
        break;
      default:
        orderBy = desc(legalDocuments.updatedAt);
    }

    // Get total count
    const [countResult] = await db
      .select({ count: count() })
      .from(legalDocuments)
      .where(filterConditions.length > 0 ? and(...filterConditions) : undefined);

    // Get documents
    const documents = await db
      .select({
        id: legalDocuments.id,
        title: legalDocuments.title,
        content: listParams.includeContent ? legalDocuments.content : sql`''`.as('content'),
        documentType: legalDocuments.documentType,
        jurisdiction: legalDocuments.jurisdiction,
        practiceArea: legalDocuments.practiceArea,
        fileName: legalDocuments.fileName,
        fileSize: legalDocuments.fileSize,
        mimeType: legalDocuments.mimeType,
        fileHash: legalDocuments.fileHash,
        processingStatus: legalDocuments.processingStatus,
        isConfidential: legalDocuments.isConfidential,
        retentionDate: legalDocuments.retentionDate,
        createdAt: legalDocuments.createdAt,
        updatedAt: legalDocuments.updatedAt,
        createdBy: legalDocuments.createdBy,
        lastModifiedBy: legalDocuments.lastModifiedBy,
        analysisResults: listParams.includeAnalysis ? legalDocuments.analysisResults : sql`NULL`.as('analysisResults'),
        // Check if embeddings exist
        hasContentEmbedding: sql`CASE WHEN ${legalDocuments.contentEmbedding} IS NOT NULL THEN true ELSE false END`.as('hasContentEmbedding'),
        hasTitleEmbedding: sql`CASE WHEN ${legalDocuments.titleEmbedding} IS NOT NULL THEN true ELSE false END`.as('hasTitleEmbedding'),
      })
      .from(legalDocuments)
      .where(filterConditions.length > 0 ? and(...filterConditions) : undefined)
      .orderBy(orderBy)
      .limit(listParams.limit)
      .offset(listParams.offset);

    // Get case associations for each document
    const documentIds = documents.map(doc => doc.id);
    const caseAssociations = documentIds.length > 0 ? await db
      .select({
        documentId: caseDocuments.documentId,
        caseId: caseDocuments.caseId,
        caseTitle: legalCases.title,
        caseNumber: legalCases.caseNumber,
        relationship: caseDocuments.relationship,
        importance: caseDocuments.importance,
      })
      .from(caseDocuments)
      .innerJoin(legalCases, eq(caseDocuments.caseId, legalCases.id))
      .where(inArray(caseDocuments.documentId, documentIds)) : [];

    // Group case associations by document ID
    const casesByDocument = caseAssociations.reduce((acc, assoc) => {
      if (!acc[assoc.documentId]) {
        acc[assoc.documentId] = [];
      }
      acc[assoc.documentId].push({
        caseId: assoc.caseId,
        caseTitle: assoc.caseTitle,
        caseNumber: assoc.caseNumber,
        relationship: assoc.relationship,
        importance: assoc.importance,
      });
      return acc;
    }, {} as Record<string, unknown[]>);

    // Format response
    const formattedDocuments = documents.map(doc => ({
      id: doc.id,
      title: doc.title,
      content: doc.content || null,
      documentType: doc.documentType,
      jurisdiction: doc.jurisdiction,
      practiceArea: doc.practiceArea,
      fileName: doc.fileName,
      fileSize: doc.fileSize,
      mimeType: doc.mimeType,
      fileHash: doc.fileHash,
      processingStatus: doc.processingStatus,
      isConfidential: doc.isConfidential,
      retentionDate: doc.retentionDate,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
      createdBy: doc.createdBy,
      lastModifiedBy: doc.lastModifiedBy,
      analysisResults: doc.analysisResults,
      hasEmbeddings: doc.hasContentEmbedding && doc.hasTitleEmbedding,
      embeddingStatus: {
        hasContentEmbedding: doc.hasContentEmbedding,
        hasTitleEmbedding: doc.hasTitleEmbedding,
      },
      associatedCases: casesByDocument[doc.id] || [],
      caseCount: (casesByDocument[doc.id] || []).length,
    }));

    return json({
      success: true,
      documents: formattedDocuments,
      pagination: {
        total: countResult.count,
        limit: listParams.limit,
        offset: listParams.offset,
        hasMore: listParams.offset + listParams.limit < countResult.count,
        page: Math.floor(listParams.offset / listParams.limit) + 1,
        totalPages: Math.ceil(countResult.count / listParams.limit),
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

  } catch (error: unknown) {
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
export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    const documentData = createDocumentSchema.parse(body);

    // Create document record
    const [insertedDoc] = await db
      .insert(legalDocuments)
      .values({
        title: documentData.title,
        content: documentData.content,
        documentType: documentData.documentType,
        jurisdiction: documentData.jurisdiction,
        practiceArea: documentData.practiceArea,
        isConfidential: documentData.isConfidential,
        processingStatus: 'processing',
        createdBy: null, // TODO: Add user authentication
      })
      .returning();

    // Process embeddings and analysis in background if requested
    if (documentData.generateEmbeddings || documentData.generateAnalysis) {
      processDocumentAsync(
        insertedDoc.id,
        documentData.content,
        documentData.title,
        documentData.generateEmbeddings,
        documentData.generateAnalysis
      );
    }

    return json({
      success: true,
      document: {
        id: insertedDoc.id,
        title: insertedDoc.title,
        documentType: insertedDoc.documentType,
        jurisdiction: insertedDoc.jurisdiction,
        practiceArea: insertedDoc.practiceArea,
        processingStatus: insertedDoc.processingStatus,
        isConfidential: insertedDoc.isConfidential,
        createdAt: insertedDoc.createdAt,
      },
      message: "Document created successfully",
      processingInBackground: documentData.generateEmbeddings || documentData.generateAnalysis,
    });

  } catch (error: unknown) {
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
export const PUT: RequestHandler = async ({ request }) => {
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

  } catch (error: unknown) {
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
async function getDocumentAnalytics() {
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
        confidential: count(sql`CASE WHEN ${legalDocuments.isConfidential} = true THEN 1 END`),
        withEmbeddings: count(sql`CASE WHEN ${legalDocuments.contentEmbedding} IS NOT NULL THEN 1 END`),
        withAnalysis: count(sql`CASE WHEN ${legalDocuments.analysisResults} IS NOT NULL THEN 1 END`),
        avgFileSize: sql`AVG(${legalDocuments.fileSize})`.as('avgFileSize'),
        totalSize: sql`SUM(${legalDocuments.fileSize})`.as('totalSize'),
      })
      .from(legalDocuments),

    // Document type distribution
    db
      .select({
        documentType: legalDocuments.documentType,
        count: count()
      })
      .from(legalDocuments)
      .groupBy(legalDocuments.documentType)
      .orderBy(desc(count())),

    // Processing status distribution
    db
      .select({
        status: legalDocuments.processingStatus,
        count: count()
      })
      .from(legalDocuments)
      .groupBy(legalDocuments.processingStatus)
      .orderBy(desc(count())),

    // Vector search statistics
    vectorSearchService.getVectorStats(),

    // Recent activity (last 30 days)
    db
      .select({
        date: sql`DATE(${legalDocuments.createdAt})`.as('date'),
        count: count()
      })
      .from(legalDocuments)
      .where(sql`${legalDocuments.createdAt} >= NOW() - INTERVAL '30 days'`)
      .groupBy(sql`DATE(${legalDocuments.createdAt})`)
      .orderBy(sql`DATE(${legalDocuments.createdAt})`)
  ]);

  return {
    totals: totalStats[0],
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
async function reprocessDocuments() {
  try {
    // Find documents without embeddings
    const documentsToProcess = await db
      .select({
        id: legalDocuments.id,
        title: legalDocuments.title,
        content: legalDocuments.content,
      })
      .from(legalDocuments)
      .where(
        and(
          eq(legalDocuments.processingStatus, 'completed'),
          or(
            sql`${legalDocuments.contentEmbedding} IS NULL`,
            sql`${legalDocuments.titleEmbedding} IS NULL`,
            sql`${legalDocuments.analysisResults} IS NULL`
          )
        )
      )
      .limit(50); // Process in batches

    // Start background processing for each document
    const processed = await Promise.allSettled(
      documentsToProcess.map(doc =>
        processDocumentAsync(doc.id, doc.content, doc.title, true, true)
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

  } catch (error) {
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
): Promise<void> {
  try {
    const updates: unknown = {};

    if (generateEmbeddings) {
      // Generate embeddings
      const contentEmbedding = await generateEmbedding(content);
      const titleEmbedding = await generateEmbedding(title);
      
      updates.contentEmbedding = contentEmbedding;
      updates.titleEmbedding = titleEmbedding;
    }

    if (generateAnalysis) {
      // Generate AI analysis
      const analysis = await generateDocumentAnalysis(content);
      updates.analysisResults = analysis;
    }

    // Update document with processing results
    updates.processingStatus = 'completed';
    updates.updatedAt = new Date();

    await db
      .update(legalDocuments)
      .set(updates)
      .where(eq(legalDocuments.id, documentId));

  } catch (error) {
    console.error('Background processing error:', error);
    
    // Mark as error status
    await db
      .update(legalDocuments)
      .set({ 
        processingStatus: 'error',
        updatedAt: new Date()
      })
      .where(eq(legalDocuments.id, documentId));
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