import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from "$lib/server/db";
import { legalDocuments as legal_documents, insertLegalDocumentSchema } from '$lib/database/schema/legal-documents';
import { cognitiveCacheManager } from "$lib/services/cognitive-cache-integration";
import { eq } from "drizzle-orm";
import { getDatabaseHealth } from "$lib/server/db";
import crypto from 'crypto';
import { z } from 'zod';

// Upload schema for validation
const uploadSchema = z.object({
  title: z.string().min(1).max(500).optional(),
  documentType: z.enum(['contract', 'motion', 'evidence', 'correspondence', 'brief', 'regulation', 'case_law']),
  jurisdiction: z.string().min(1).max(100).default('federal'),
  practiceArea: z.enum(['corporate', 'litigation', 'intellectual_property', 'employment', 'real_estate', 'criminal', 'family', 'tax', 'immigration', 'environmental']).optional(),
  isConfidential: z.boolean().default(false),
  includeEmbeddings: z.boolean().default(true),
  generateAnalysis: z.boolean().default(true),
});

/**
 * Document Upload API Endpoint with Database Integration
 * Handles file upload, text extraction, vector embeddings, and database storage
 */
export const POST: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return json({ error: "No file provided" }, { status: 400 });
    }

    // Validate file constraints
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      return json({ error: "File too large. Maximum size is 50MB" }, { status: 400 });
    }

    const allowedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "text/plain",
      "text/rtf",
      "application/json",
    ];

    if (!allowedTypes.includes(file.type)) {
      return json({ 
        error: "Unsupported file type",
        supportedTypes: allowedTypes 
      }, { status: 400 });
    }

    // Parse and validate form data
    const uploadData = uploadSchema.parse({
      title: formData.get("title"),
      documentType: formData.get("documentType"),
      jurisdiction: formData.get("jurisdiction") || 'federal',
      practiceArea: formData.get("practiceArea"),
      isConfidential: formData.get("isConfidential") === "true",
      includeEmbeddings: formData.get("includeEmbeddings") !== "false",
      generateAnalysis: formData.get("generateAnalysis") !== "false",
    });

    // Generate file hash for deduplication
    const fileBuffer = await file.arrayBuffer();
    const fileHash = crypto.createHash('sha256').update(new Uint8Array(fileBuffer)).digest('hex');

    // Check cognitive cache for recent uploads
    const cacheKey = `document_upload_${fileHash}`;
    const cacheRequest = {
      key: cacheKey,
      type: 'legal-data' as const,
      context: {
        action: 'duplicate-check',
        fileHash,
        workflowStep: 'upload-validation',
        priority: 'high' as const
      }
    };

    const cachedResult = await cognitiveCacheManager.get(cacheRequest);
    if (cachedResult && cachedResult.confidence > 0.9) {
      return json({
        success: false,
        error: "Document already exists (cached)",
        duplicateId: cachedResult.data.id,
        duplicateTitle: cachedResult.data.title
      }, { status: 409 });
    }

    // Check for duplicate files in database
    const existingDoc = await db
      .select({ id: legal_documents.id, title: legal_documents.title })
      .from(legal_documents)
      .where(eq(legal_documents.file_hash, fileHash))
      .limit(1);

    if (existingDoc.length > 0) {
      // Cache the duplicate result for future requests
      await cognitiveCacheManager.set(cacheRequest, {
        id: existingDoc[0].id,
        title: existingDoc[0].title
      }, { distributeAcrossCaches: true, cognitiveValue: 0.95 });

      return json({
        success: false,
        error: "Document already exists",
        duplicateId: existingDoc[0].id,
        duplicateTitle: existingDoc[0].title
      }, { status: 409 });
    }

    // Extract text content from file
    const textContent = await extractTextFromFile(file, fileBuffer);
    
    if (!textContent || textContent.length < 10) {
      return json({ 
        error: "Unable to extract text content from file or content too short" 
      }, { status: 400 });
    }

    // Generate title if not provided
    const documentTitle = uploadData.title || generateTitleFromContent(textContent, file.name);

    // Check database health before insertion
    const dbHealth = await getDatabaseHealth();
    if (dbHealth.overall !== 'healthy') {
      return json({
        success: false,
        error: "Database temporarily unavailable",
        healthStatus: dbHealth
      }, { status: 503 });
    }

    // Create initial document record with proper schema mapping
    const documentData = {
      title: documentTitle,
      content: textContent,
      document_type: uploadData.documentType,
      jurisdiction: uploadData.jurisdiction,
      practice_area: uploadData.practiceArea,
      file_name: file.name,
      file_size: file.size,
      mime_type: file.type,
      file_hash: fileHash,
      is_confidential: uploadData.isConfidential,
      processing_status: 'processing' as const,
      created_by: null, // TODO: Add user authentication
      created_at: new Date(),
      updated_at: new Date()
    };

    // Insert document into database
    const [insertedDoc] = await db
      .insert(legal_documents)
      .values(documentData)
      .returning({ id: legal_documents.id });

    // Process embeddings and analysis in background if requested
    if (uploadData.includeEmbeddings || uploadData.generateAnalysis) {
      processDocumentAsync(insertedDoc.id, textContent, uploadData);
    }

    // Cache successful upload result
    const responseData = {
      success: true,
      document: {
        id: insertedDoc.id,
        title: documentTitle,
        documentType: uploadData.documentType,
        fileName: file.name,
        fileSize: file.size,
        processingStatus: 'processing',
        isConfidential: uploadData.isConfidential,
      },
      message: "Document uploaded successfully",
      processingInBackground: uploadData.includeEmbeddings || uploadData.generateAnalysis,
      meta: {
        timestamp: new Date().toISOString(),
        databaseHealth: dbHealth.overall,
        cached: false
      }
    };

    // Cache the upload result for monitoring and analytics
    await cognitiveCacheManager.set({
      key: `document_created_${insertedDoc.id}`,
      type: 'legal-data' as const,
      context: {
        userId: null, // TODO: Add user ID when auth is implemented
        action: 'document-upload-success',
        documentType: uploadData.documentType,
        priority: 'medium' as const
      }
    }, responseData, { distributeAcrossCaches: true, cognitiveValue: 0.8 });

    return json(responseData);

  } catch (error: any) {
    console.error("Document upload error:", error);

    if (error instanceof z.ZodError) {
      return json({
        success: false,
        error: "Invalid upload parameters",
        details: error.errors,
      }, { status: 400 });
    }

    return json({
      success: false,
      error: error?.message || "Document upload failed",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Extract text content from various file types
 */
async function extractTextFromFile(file: File, fileBuffer: ArrayBuffer): Promise<string> {
  const mimeType = file.type;

  try {
    if (mimeType === 'text/plain') {
      return new TextDecoder().decode(fileBuffer);
    }
    
    if (mimeType === 'application/json') {
      const jsonContent = JSON.parse(new TextDecoder().decode(fileBuffer));
      return JSON.stringify(jsonContent, null, 2);
    }

    if (mimeType === 'application/pdf') {
      // Use PDF.js or similar library for PDF text extraction
      // For now, return placeholder - implement actual PDF extraction
      return await extractPdfText(fileBuffer);
    }

    if (mimeType.includes('word') || mimeType.includes('officedocument')) {
      // Use mammoth.js or similar for Word document extraction
      return await extractWordText(fileBuffer);
    }

    throw new Error(`Unsupported file type for text extraction: ${mimeType}`);
  } catch (error: any) {
    console.error('Text extraction error:', error);
    throw new Error(`Failed to extract text from ${mimeType} file`);
  }
}

/**
 * Extract text from PDF files
 */
async function extractPdfText(buffer: ArrayBuffer): Promise<string> {
  try {
    // Import PDF.js dynamically
    const pdfjs = await import('pdfjs-dist/legacy/build/pdf.js');
    
    const pdf = await pdfjs.getDocument({ data: buffer }).promise;
    let fullText = '';

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map((item: any) => item.str)
        .join(' ');
      fullText += pageText + '\n';
    }

    return fullText.trim();
  } catch (error: any) {
    console.error('PDF extraction error:', error);
    throw new Error('Failed to extract text from PDF');
  }
}

/**
 * Extract text from Word documents
 */
async function extractWordText(buffer: ArrayBuffer): Promise<string> {
  try {
    // This would use a library like mammoth.js
    // For now, return a placeholder
    return "Word document text extraction not yet implemented";
  } catch (error: any) {
    console.error('Word extraction error:', error);
    throw new Error('Failed to extract text from Word document');
  }
}

/**
 * Generate a document title from content and filename
 */
function generateTitleFromContent(content: string, filename: string): string {
  // Remove file extension
  const baseName = filename.replace(/\.[^/.]+$/, "");
  
  // Try to extract a meaningful title from the first few lines
  const firstLines = content.split('\n').slice(0, 3);
  const potentialTitle = firstLines
    .find(line => line.trim().length > 10 && line.trim().length < 100);
  
  return potentialTitle?.trim() || baseName;
}

/**
 * Process document embeddings and analysis asynchronously
 */
async function processDocumentAsync(
  documentId: string, 
  content: string, 
  options: { includeEmbeddings: boolean; generateAnalysis: boolean }
): Promise<any> {
  try {
    const updates: any = {};

    if (options.includeEmbeddings) {
      // Generate embeddings using your embedding service
      const contentEmbedding = await generateEmbedding(content);
      const title = content.split('\n')[0] || '';
      const titleEmbedding = await generateEmbedding(title);
      
      updates.contentEmbedding = contentEmbedding;
      updates.titleEmbedding = titleEmbedding;
    }

    if (options.generateAnalysis) {
      // Generate AI analysis
      const analysis = await generateDocumentAnalysis(content);
      updates.analysisResults = analysis;
    }

    // Update document with processing results using proper schema
    updates.processing_status = 'completed';
    updates.updated_at = new Date();

    await db
      .update(legal_documents)
      .set(updates)
      .where(eq(legal_documents.id, documentId));

    // Clear relevant caches after successful processing
    await cognitiveCacheManager.set({
      key: `document_processed_${documentId}`,
      type: 'legal-data' as const,
      context: { action: 'processing-complete', documentId }
    }, { processingCompleted: true, timestamp: new Date().toISOString() });

  } catch (error: any) {
    console.error('Background processing error:', error);
    
    // Mark as error status using proper schema
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
 * Generate embeddings for text (placeholder - implement with your embedding service)
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

/**
 * Get document status and details
 */
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  try {
    const documentId = url.searchParams.get("id");

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }

    // Check cognitive cache first
    const docCacheRequest = {
      key: `document_${documentId}`,
      type: 'legal-data' as const,
      context: { action: 'document-retrieval', documentId }
    };

    const cachedDoc = await cognitiveCacheManager.get(docCacheRequest);
    if (cachedDoc && cachedDoc.confidence > 0.8) {
      return json({
        success: true,
        document: cachedDoc.data,
        meta: { loadSource: 'cache', cached: true }
      });
    }

    // Get document from database
    const [document] = await db
      .select()
      .from(legal_documents)
      .where(eq(legal_documents.id, documentId))
      .limit(1);

    if (!document) {
      return json({ error: "Document not found" }, { status: 404 });
    }

    const responseData = {
      success: true,
      document: {
        id: document.id,
        title: document.title,
        documentType: document.document_type,
        jurisdiction: document.jurisdiction,
        practiceArea: document.practice_area,
        fileName: document.file_name,
        fileSize: document.file_size,
        mimeType: document.mime_type,
        processingStatus: document.processing_status,
        isConfidential: document.is_confidential,
        hasEmbeddings: !!(document.content_embedding && document.title_embedding),
        hasAnalysis: !!document.analysis_results,
        createdAt: document.created_at,
        updatedAt: document.updated_at,
        // Include analysis results if available and not confidential
        analysisResults: !document.is_confidential ? document.analysis_results : null,
      },
      meta: { loadSource: 'database', cached: false }
    };

    // Cache the document for future requests
    await cognitiveCacheManager.set(docCacheRequest, responseData.document, {
      distributeAcrossCaches: true,
      cognitiveValue: 0.85
    });

    return json(responseData);
  } catch (error: any) {
    console.error("Document status check error:", error);

    return json({
      success: false,
      error: "Failed to get document status",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};

/**
 * Delete a document
 */
export const DELETE: RequestHandler = async ({ url }): Promise<any> => {
  try {
    const documentId = url.searchParams.get("id");

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }

    // Check if document exists
    const [document] = await db
      .select({ id: legal_documents.id, title: legal_documents.title })
      .from(legal_documents)
      .where(eq(legal_documents.id, documentId))
      .limit(1);

    if (!document) {
      return json({ error: "Document not found" }, { status: 404 });
    }

    // Delete the document (cascade will handle related records)
    await db
      .delete(legal_documents)
      .where(eq(legal_documents.id, documentId));

    // Clear all cached data for this document
    const cacheKeys = [
      `document_${documentId}`,
      `document_upload_${documentId}`,
      `document_created_${documentId}`,
      `document_processed_${documentId}`
    ];
    
    for (const key of cacheKeys) {
      await cognitiveCacheManager.set({
        key,
        type: 'legal-data' as const,
        context: { action: 'document-deleted', documentId }
      }, null); // Clear cache entry
    }

    return json({
      success: true,
      message: `Document "${document.title}" deleted successfully`,
      deletedId: documentId,
    });

  } catch (error: any) {
    console.error("Document deletion error:", error);

    return json({
      success: false,
      error: "Failed to delete document",
      details: process.env.NODE_ENV === "development" ? error : undefined,
    }, { status: 500 });
  }
};
