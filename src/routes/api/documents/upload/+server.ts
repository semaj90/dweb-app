import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from "$lib/database/postgres-enhanced.js";
import { legalDocuments, insertLegalDocumentSchema } from "$lib/database/schema/legal-documents.js";
import { vectorSearchService } from "$lib/database/vector-operations.js";
import { eq } from "drizzle-orm";
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
export const POST: RequestHandler = async ({ request }) => {
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

    // Check for duplicate files
    const existingDoc = await db
      .select({ id: legalDocuments.id, title: legalDocuments.title })
      .from(legalDocuments)
      .where(eq(legalDocuments.fileHash, fileHash))
      .limit(1);

    if (existingDoc.length > 0) {
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

    // Create initial document record
    const documentData = {
      title: documentTitle,
      content: textContent,
      documentType: uploadData.documentType,
      jurisdiction: uploadData.jurisdiction,
      practiceArea: uploadData.practiceArea,
      fileName: file.name,
      fileSize: file.size,
      mimeType: file.type,
      fileHash,
      isConfidential: uploadData.isConfidential,
      processingStatus: 'processing' as const,
      createdBy: null, // TODO: Add user authentication
    };

    // Insert document into database
    const [insertedDoc] = await db
      .insert(legalDocuments)
      .values(documentData)
      .returning({ id: legalDocuments.id });

    // Process embeddings and analysis in background if requested
    if (uploadData.includeEmbeddings || uploadData.generateAnalysis) {
      processDocumentAsync(insertedDoc.id, textContent, uploadData);
    }

    return json({
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
    });

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
  } catch (error) {
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
  } catch (error) {
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
  } catch (error) {
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
): Promise<void> {
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
export const GET: RequestHandler = async ({ url }) => {
  try {
    const documentId = url.searchParams.get("id");

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }

    // Get document from database
    const [document] = await db
      .select()
      .from(legalDocuments)
      .where(eq(legalDocuments.id, documentId))
      .limit(1);

    if (!document) {
      return json({ error: "Document not found" }, { status: 404 });
    }

    return json({
      success: true,
      document: {
        id: document.id,
        title: document.title,
        documentType: document.documentType,
        jurisdiction: document.jurisdiction,
        practiceArea: document.practiceArea,
        fileName: document.fileName,
        fileSize: document.fileSize,
        mimeType: document.mimeType,
        processingStatus: document.processingStatus,
        isConfidential: document.isConfidential,
        hasEmbeddings: !!(document.contentEmbedding && document.titleEmbedding),
        hasAnalysis: !!document.analysisResults,
        createdAt: document.createdAt,
        updatedAt: document.updatedAt,
        // Include analysis results if available and not confidential
        analysisResults: !document.isConfidential ? document.analysisResults : null,
      },
    });
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
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const documentId = url.searchParams.get("id");

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }

    // Check if document exists
    const [document] = await db
      .select({ id: legalDocuments.id, title: legalDocuments.title })
      .from(legalDocuments)
      .where(eq(legalDocuments.id, documentId))
      .limit(1);

    if (!document) {
      return json({ error: "Document not found" }, { status: 404 });
    }

    // Delete the document (cascade will handle related records)
    await db
      .delete(legalDocuments)
      .where(eq(legalDocuments.id, documentId));

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
