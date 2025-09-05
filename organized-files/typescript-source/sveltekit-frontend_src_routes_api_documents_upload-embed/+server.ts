// @ts-nocheck
// Document Upload + Embedding Pipeline API
// Integrates with your existing LangChain+Ollama+pgvector infrastructure

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/server/database';
import { documents, evidence, documentVectors } from '$lib/db/schema';
import { eq } from 'drizzle-orm';
import { OllamaEmbeddings } from '@langchain/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import crypto from 'crypto';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { existsSync } from 'fs';

// ============================================================================
// CONFIGURATION
// ============================================================================

const config = {
  ollamaBaseUrl: 'http://localhost:11434',
  embeddingModel: 'nomic-embed-text', // 384 dimensions
  chunkSize: 1000,
  chunkOverlap: 200,
  uploadDir: './uploads/documents',
  maxFileSize: 10 * 1024 * 1024, // 10MB
  allowedTypes: [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ]
};

// ============================================================================
// EMBEDDING SERVICE
// ============================================================================

class DocumentEmbeddingService {
  private embeddings: OllamaEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;

  constructor() {
    this.embeddings = new OllamaEmbeddings({
      baseUrl: config.ollamaBaseUrl,
      model: config.embeddingModel,
    });

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: config.chunkSize,
      chunkOverlap: config.chunkOverlap,
    });
  }

  async processDocument(documentId: string, content: string) {
    const startTime = Date.now();
    
    try {
      // Split text into chunks
      const chunks = await this.textSplitter.splitText(content);
      console.log(`📄 Split document into ${chunks.length} chunks`);

      // Generate embeddings for all chunks
      const embeddings = await Promise.all(
        chunks.map((chunk: any) => this.embeddings.embedQuery(chunk))
      );

      // Store chunks and embeddings in database
      const vectorRecords = chunks.map((chunk, index) => ({
        documentId,
        chunkIndex: index,
        content: chunk,
        embedding: embeddings[index],
        metadata: {
          chunkSize: chunk.length,
          model: config.embeddingModel,
          timestamp: new Date().toISOString()
        }
      }));

      await db.insert(documentVectors).values(vectorRecords);

      const processingTime = Date.now() - startTime;
      console.log(`✅ Processed document in ${processingTime}ms`);

      return {
        success: true,
        chunksCreated: chunks.length,
        processingTime,
        embeddings: embeddings.length,
        avgChunkSize: Math.round(chunks.reduce((sum, chunk) => sum + chunk.length, 0) / chunks.length)
      };

    } catch (error) {
      console.error('❌ Document processing failed:', error);
      throw error;
    }
  }
}

// ============================================================================
// TEXT EXTRACTION UTILITIES
// ============================================================================

async function extractText(file: File): Promise<string> {
  const mimeType = file.type;
  
  if (mimeType === 'text/plain' || mimeType === 'text/markdown') {
    return await file.text();
  }
  
  if (mimeType === 'application/pdf') {
    // In production, use a PDF library like pdf-parse
    // For now, return placeholder
    return `[PDF Content] ${file.name} - Size: ${file.size} bytes`;
  }
  
  if (mimeType.includes('word')) {
    // In production, use mammoth.js for Word docs
    return `[Word Document] ${file.name} - Size: ${file.size} bytes`;
  }
  
  throw new Error(`Unsupported file type: ${mimeType}`);
}

// ============================================================================
// API HANDLER
// ============================================================================

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const caseId = formData.get('caseId') as string;
    const evidenceId = formData.get('evidenceId') as string;
    const title = formData.get('title') as string || file.name;
    const userId = formData.get('userId') as string;

    // Validation
    if (!file) {
      throw error(400, 'No file provided');
    }

    if (!caseId || !userId) {
      throw error(400, 'Missing required fields: caseId, userId');
    }

    if (file.size > config.maxFileSize) {
      throw error(400, `File too large. Max size: ${config.maxFileSize / (1024 * 1024)}MB`);
    }

    if (!config.allowedTypes.includes(file.type)) {
      throw error(400, `Unsupported file type: ${file.type}`);
    }

    // Create upload directory if it doesn't exist
    if (!existsSync(config.uploadDir)) {
      await mkdir(config.uploadDir, { recursive: true });
    }

    // Generate file hash for integrity
    const buffer = await file.arrayBuffer();
    const hash = crypto.createHash('sha256').update(new Uint8Array(buffer)).digest('hex');
    
    // Save file
    const filename = `${Date.now()}-${file.name}`;
    const filePath = join(config.uploadDir, filename);
    await writeFile(filePath, new Uint8Array(buffer));

    // Extract text content
    const extractedText = await extractText(file);

    // Create document record
    const [document] = await db.insert(documents).values({
      caseId,
      evidenceId: evidenceId || null,
      filename: file.name,
      filePath,
      extractedText,
      createdBy: userId,
    }).returning();

    console.log(`📁 Created document record: ${document.id}`);

    // Process for embeddings
    const embeddingService = new DocumentEmbeddingService();
    const embedResult = await embeddingService.processDocument(document.id, extractedText);

    // Update document with analysis
    await db.update(documents)
      .set({
        analysis: {
          embedding: embedResult,
          fileSize: file.size,
          mimeType: file.type,
          hash,
          processingTimestamp: new Date().toISOString()
        }
      })
      .where(eq(documents.id, document.id));

    return json({
      success: true,
      document: {
        id: document.id,
        filename: document.filename,
        extractedText: extractedText.substring(0, 500) + '...', // Preview
      },
      embedding: embedResult,
      message: `Document processed successfully with ${embedResult.chunksCreated} chunks`
    });

  } catch (err) {
    console.error('❌ Upload error:', err);
    
    if (err instanceof Error) {
      throw error(500, `Upload failed: ${err.message}`);
    }
    
    throw error(500, 'Unknown upload error');
  }
};

// ============================================================================
// HEALTH CHECK
// ============================================================================

export const GET: RequestHandler = async () => {
  try {
    // Test Ollama connection
    const embeddings = new OllamaEmbeddings({
      baseUrl: config.ollamaBaseUrl,
      model: config.embeddingModel,
    });

    const testEmbedding = await embeddings.embedQuery("test");
    
    return json({
      status: 'healthy',
      config: {
        model: config.embeddingModel,
        chunkSize: config.chunkSize,
        dimensions: testEmbedding.length,
        uploadDir: config.uploadDir,
        maxFileSize: `${config.maxFileSize / (1024 * 1024)}MB`
      },
      ollama: {
        baseUrl: config.ollamaBaseUrl,
        connected: true,
        embeddingDimensions: testEmbedding.length
      }
    });

  } catch (err) {
    return json({
      status: 'unhealthy',
      error: err instanceof Error ? err.message : 'Unknown error',
      config: {
        model: config.embeddingModel,
        ollamaBaseUrl: config.ollamaBaseUrl
      }
    }, { status: 500 });
  }
};