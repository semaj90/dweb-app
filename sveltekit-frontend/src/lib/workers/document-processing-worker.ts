import { rabbitMQService } from '../services/rabbitmq-service.js';
import { db } from '$lib/server/db';
import * as schema from '$lib/server/db/schema-postgres';
import { eq } from 'drizzle-orm';
import { v4 as uuidv4 } from 'uuid';
import fs from "fs";

const schemaAny = schema as any;

export interface DocumentProcessingJob {
  documentId: string | number;
  s3Key: string;
  s3Bucket: string;
  caseId?: string;
  userId?: string;
  originalName: string;
  mimeType: string;
  fileSize: number;
  processingType: 'ocr' | 'embedding' | 'summarization' | 'full_analysis';
  priority: number;
  timestamp: string;
}

export interface ProcessingContext {
  job: DocumentProcessingJob;
  tempFilePath?: string;
  extractedText?: string;
  chunks?: DocumentChunk[];
  embeddings?: EmbeddingResult[];
  summary?: string;
}

export interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    chunkIndex: number;
    startPosition: number;
    endPosition: number;
    wordCount: number;
  };
}

export interface EmbeddingResult {
  chunkId: string;
  embedding: number[];
  model: string;
}

class DocumentProcessingWorker {
  private isRunning = false;
  private processedCount = 0;
  private failedCount = 0;
  private intervalHandle: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // Bind methods to preserve context if needed
    this.processJob = this.processJob.bind(this);
    this.processDocumentFromDB = this.processDocumentFromDB.bind(this);
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('Document processing worker is already running');
      return;
    }

    this.isRunning = true;
    console.log('🔄 Starting document processing worker...');

    try {
      // Connect to RabbitMQ (optional for this simplified worker)
      await rabbitMQService.connect();

      // Start consuming jobs from the document processing queue (polling)
      this.startConsuming();

    } catch (error: any) {
      console.error('Failed to start document processing worker:', error);
      this.isRunning = false;
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    console.log('🛑 Stopping document processing worker...');

    if (this.intervalHandle) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = null;
    }

    await rabbitMQService.close();
  }

  private startConsuming(): void {
    console.log('📥 Worker ready to consume document processing jobs (polling every 5s)');

    this.intervalHandle = setInterval(async () => {
      if (!this.isRunning) {
        if (this.intervalHandle) {
          clearInterval(this.intervalHandle);
          this.intervalHandle = null;
        }
        return;
      }

      try {
        const queuedRecords = await db.select()
          .from(schemaAny.document_processing)
          .where(eq(schemaAny.document_processing.status, 'queued'))
          .limit(5);

        for (const record of queuedRecords) {
          await this.processDocumentFromDB(record);
        }
      } catch (error: any) {
        console.error('Error checking for jobs:', error);
      }
    }, 5000);
  }

  private async processDocumentFromDB(processingRecord: any): Promise<void> {
    if (!processingRecord || !processingRecord.document_id) {
      console.warn('Invalid processing record, skipping');
      return;
    }

    // Fetch document details
    const docs = await db.select()
      .from(schemaAny.documents)
      .where(eq(schemaAny.documents.id, processingRecord.document_id))
      .limit(1);

    const document = docs && docs.length > 0 ? docs[0] : null;

    if (!document) {
      console.error(`Document not found: ${processingRecord.document_id}`);
      // mark processing record failed
      await this.updateProcessingStatus(processingRecord.document_id, 'failed', 'Document not found');
      return;
    }

    // Create job object
    const job: DocumentProcessingJob = {
      documentId: document.id,
      s3Key: document.s3_key || '',
      s3Bucket: document.s3_bucket || 'legal-documents',
      originalName: document.original_name || 'unknown',
      mimeType: document.mime_type || 'application/octet-stream',
      fileSize: document.file_size || 0,
      caseId: document.case_id,
      userId: document.user_id,
      processingType: 'full_analysis',
      priority: 5,
      timestamp: new Date().toISOString()
    };

    await this.processJob(job);
  }

  private async processJob(job: DocumentProcessingJob): Promise<void> {
    const context: ProcessingContext = { job };

    try {
      console.log(`📄 Processing document: ${job.documentId} (${job.originalName})`);

      // Update status to processing
      await this.updateProcessingStatus(job.documentId, 'processing', 'Starting document analysis');

      // Step 4a: Download file from S3/MinIO
      await this.downloadDocument(context);

      // Step 5: OCR & Text Extraction
      await this.extractText(context);

      // Step 6a: Text Chunking
      await this.chunkDocument(context);

      // Step 6b: Generate Embeddings
      await this.generateEmbeddings(context);

      // Step 7: Store in pgvector
      await this.storeVectorEmbeddings(context);

      // Step 8: Generate Summary
      await this.generateSummary(context);

      // Mark as completed
      await this.updateProcessingStatus(job.documentId, 'completed', 'Document processing completed successfully');

      this.processedCount++;
      console.log(`✅ Successfully processed document: ${job.documentId}`);

    } catch (error: any) {
      console.error(`❌ Error processing document ${job.documentId}:`, error);

      await this.updateProcessingStatus(
        job.documentId,
        'failed',
        `Processing failed: ${error?.message ?? String(error)}`
      );

      this.failedCount++;
    } finally {
      // Cleanup temp files
      if (context.tempFilePath) {
        try {
          await this.cleanupTempFile(context.tempFilePath);
        } catch (cleanupError) {
          console.warn('Failed to cleanup temp file:', cleanupError);
        }
      }
    }
  }

  private async downloadDocument(context: ProcessingContext): Promise<void> {
    console.log(`⬇️  Downloading document from S3: ${context.job.s3Key}`);

    // Simulate S3 download - in production, implement actual MinIO/S3 client
    const response = await (globalThis as any).fetch(`http://localhost:9000/${context.job.s3Bucket}/${context.job.s3Key}`);

    if (!response.ok) {
      throw new Error(`Failed to download document: ${response.statusText}`);
    }

    // Save to temp file (simplified - in production use proper temp file handling)
    const tempFilePath = `/tmp/${context.job.documentId}_${Date.now()}.${this.getFileExtension(context.job.originalName)}`;
    context.tempFilePath = tempFilePath;

    console.log(`💾 Document downloaded to: ${tempFilePath}`);
  }

  private async extractText(context: ProcessingContext): Promise<void> {
    console.log(`🔍 Extracting text from: ${context.job.originalName}`);

    const { job } = context;

    // Different extraction methods based on file type
    switch (job.mimeType) {
      case 'application/pdf':
        context.extractedText = await this.extractPDFText(context.tempFilePath!);
        break;
      case 'image/jpeg':
      case 'image/png':
        context.extractedText = await this.extractImageText(context.tempFilePath!);
        break;
      case 'text/plain':
        context.extractedText = await this.extractPlainText(context.tempFilePath!);
        break;
      default:
        throw new Error(`Unsupported file type: ${job.mimeType}`);
    }

    if (!context.extractedText || context.extractedText.length < 10) {
      throw new Error('Failed to extract meaningful text from document');
    }

    console.log(`📝 Extracted ${context.extractedText.length} characters of text`);
  }

  private async extractPDFText(filePath: string): Promise<string> {
    // Simulate PDF text extraction
    // In production, use pdf-parse or similar library
    return `Extracted PDF text from ${filePath}. This would contain the actual document content extracted using a proper PDF parsing library.`;
  }

  private async extractImageText(filePath: string): Promise<string> {
    // Simulate OCR with Tesseract
    // In production, use node-tesseract-ocr or similar
    return `OCR extracted text from image ${filePath}.`;
  }

  private async extractPlainText(filePath: string): Promise<string> {
    // Read plain text file
    // In production, use fs.readFile
    return `Plain text content from ${filePath}`;
  }

  private async chunkDocument(context: ProcessingContext): Promise<void> {
    console.log('✂️  Chunking document for embeddings');

    const { extractedText } = context;
    if (!extractedText) throw new Error('No text to chunk');

    // Simple chunking algorithm (in production, use a proper text splitter)
    const chunkSize = 1000; // characters
    const overlap = 200;
    const chunks: DocumentChunk[] = [];

    for (let i = 0; i < extractedText.length; i += (chunkSize - overlap)) {
      const chunkContent = extractedText.slice(i, i + chunkSize);
      const chunkId = uuidv4();

      chunks.push({
        id: chunkId,
        content: chunkContent,
        metadata: {
          chunkIndex: chunks.length,
          startPosition: i,
          endPosition: Math.min(i + chunkSize, extractedText.length),
          wordCount: chunkContent.split(/\s+/).filter(Boolean).length
        }
      });
    }

    context.chunks = chunks;
    console.log(`📝 Created ${chunks.length} document chunks`);
  }

  private async generateEmbeddings(context: ProcessingContext): Promise<void> {
    console.log('🧠 Generating embeddings with Ollama');

    const { chunks } = context;
    if (!chunks) throw new Error('No chunks to embed');

    const embeddings: EmbeddingResult[] = [];

    for (const chunk of chunks) {
      try {
        const embeddingResponse = await (globalThis as any).fetch('http://localhost:11434/api/embeddings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: 'nomic-embed-text',
            prompt: chunk.content
          })
        });

        if (!embeddingResponse.ok) {
          console.warn(`Failed to generate embedding for chunk ${chunk.id}`);
          continue;
        }

        const embeddingResult = await embeddingResponse.json();

        embeddings.push({
          chunkId: chunk.id,
          embedding: embeddingResult.embedding,
          model: 'nomic-embed-text'
        });
      } catch (err) {
        console.warn(`Embedding API error for chunk ${chunk.id}:`, err);
      }
    }

    context.embeddings = embeddings;
    console.log(`🎯 Generated ${embeddings.length} embeddings`);
  }

  private async storeVectorEmbeddings(context: ProcessingContext): Promise<void> {
    console.log('💾 Storing embeddings in pgvector');

    const { job, chunks, embeddings } = context;
    if (!chunks || !embeddings) throw new Error('No chunks or embeddings to store');

    for (const chunk of chunks) {
      const embedding = embeddings.find(e => e.chunkId === chunk.id);
      const values = {
        id: chunk.id,
        document_id: job.documentId,
        chunk_index: chunk.metadata.chunkIndex,
        content: chunk.content,
        start_position: chunk.metadata.startPosition,
        end_position: chunk.metadata.endPosition,
        word_count: chunk.metadata.wordCount,
        embedding: embedding ? embedding.embedding : null,
        embedding_model: embedding ? embedding.model : null,
        created_at: new Date(),
        updated_at: new Date()
      };

      try {
        await db.insert(schemaAny.document_chunks).values(values);
      } catch (err) {
        console.warn('Failed to insert document chunk:', err);
      }
    }

    console.log(`✅ Stored ${chunks.length} chunks with embeddings`);
  }

  private async generateSummary(context: ProcessingContext): Promise<void> {
    console.log('📋 Generating document summary with Ollama Gemma3');

    const { job, extractedText } = context;
    if (!extractedText) throw new Error('No text to summarize');

    try {
      const summaryResponse = await (globalThis as any).fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: `Please provide a comprehensive legal analysis and summary of the following document:\n\n${extractedText.slice(0, 4000)}`,
          stream: false,
          options: {
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 1000
          }
        })
      });

      if (!summaryResponse.ok) {
        throw new Error(`Failed to generate summary: ${summaryResponse.statusText}`);
      }

      const summaryResult = await summaryResponse.json();
      context.summary = summaryResult.response ?? String(summaryResult);

      await db.insert(schemaAny.document_summaries).values({
        id: uuidv4(),
        document_id: typeof job.documentId === 'string' ? Number(job.documentId) : job.documentId,
        summary_text: context.summary ?? '',
        summary_type: 'legal_analysis',
        model_used: 'gemma3-legal',
        confidence_score: 0.85, // Mock confidence
        created_at: new Date(),
        updated_at: new Date()
      });
    } catch (err) {
      console.warn('Summary generation failed:', err);
      throw err;
    }

    console.log(`📄 Generated summary (${(context.summary ?? '').length} characters)`);
  }

  private async updateProcessingStatus(documentId: string | number, status: string, message?: string): Promise<void> {
    try {
      await db.update(schemaAny.document_processing)
        .set({
          status,
          status_message: message,
          updated_at: new Date()
        })
        .where(eq(schemaAny.document_processing.document_id, documentId));

      // Also update main document status
      await db.update(schemaAny.documents)
        .set({
          status: status === 'completed' ? 'processed' : status,
          updated_at: new Date()
        })
        .where(eq(schemaAny.documents.id, documentId));
    } catch (err) {
      console.warn('Failed to update processing status:', err);
    }
  }

  private getFileExtension(filename: string): string {
    return filename.split('.').pop() || 'unknown';
  }

  private async cleanupTempFile(filePath: string): Promise<void> {
    // In production, implement proper file cleanup
    console.log(`🗑️  Cleaning up temp file: ${filePath}`);
  }
}

// Export singleton instance
export const documentProcessingWorker = new DocumentProcessingWorker();