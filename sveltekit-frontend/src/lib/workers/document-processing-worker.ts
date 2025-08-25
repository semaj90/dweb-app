import { rabbitMQService, type DocumentProcessingJob } from '../services/rabbitmq-service.js';
import { db } from '$lib/server/db';
import { documents, document_processing, document_chunks, document_summaries } from '$lib/server/db/schema-postgres';
import { eq } from 'drizzle-orm';
import { v4 as uuidv4 } from 'uuid';

interface ProcessingContext {
  job: DocumentProcessingJob;
  tempFilePath?: string;
  extractedText?: string;
  chunks?: DocumentChunk[];
  embeddings?: EmbeddingResult[];
  summary?: string;
}

interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    chunkIndex: number;
    startPosition: number;
    endPosition: number;
    wordCount: number;
  };
}

interface EmbeddingResult {
  chunkId: string;
  embedding: number[];
  model: string;
}

class DocumentProcessingWorker {
  private isRunning = false;
  private processedCount = 0;
  private failedCount = 0;
  
  constructor() {
    // Bind methods to preserve context
    this.processJob = this.processJob.bind(this);
  }
  
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('Document processing worker is already running');
      return;
    }
    
    this.isRunning = true;
    console.log('🔄 Starting document processing worker...');
    
    try {
      // Connect to RabbitMQ
      await rabbitMQService.connect();
      
      // Start consuming jobs from the document processing queue
      await this.startConsuming();
      
    } catch (error) {
      console.error('Failed to start document processing worker:', error);
      this.isRunning = false;
      throw error;
    }
  }
  
  async stop(): Promise<void> {
    this.isRunning = false;
    console.log('🛑 Stopping document processing worker...');
    
    await rabbitMQService.close();
  }
  
  private async startConsuming(): Promise<void> {
    // Note: This is a simplified implementation
    // In production, you would use a proper message consumer
    console.log('📥 Worker ready to consume document processing jobs');
    
    // For this implementation, we'll simulate periodic job checking
    // In a real RabbitMQ setup, this would be event-driven
    const checkInterval = setInterval(async () => {
      if (!this.isRunning) {
        clearInterval(checkInterval);
        return;
      }
      
      try {
        // Check for queued jobs in database
        const queuedJobs = await db.select()
          .from(document_processing)
          .where(eq(document_processing.status, 'queued'))
          .limit(5);
          
        for (const job of queuedJobs) {
          await this.processDocumentFromDB(job);
        }
        
      } catch (error) {
        console.error('Error checking for jobs:', error);
      }
    }, 5000); // Check every 5 seconds
  }
  
  private async processDocumentFromDB(processingRecord: any): Promise<void> {
    // Get document details
    const [document] = await db.select()
      .from(documents)
      .where(eq(documents.id, processingRecord.document_id))
      .limit(1);
      
    if (!document) {
      console.error(`Document not found: ${processingRecord.document_id}`);
      return;
    }
    
    // Create job object
    const job: DocumentProcessingJob = {
      documentId: document.id,
      s3Key: document.s3_key || '',
      s3Bucket: document.s3_bucket || 'legal-documents',
      originalName: document.original_name,
      mimeType: document.mime_type,
      fileSize: document.file_size,
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
      
    } catch (error) {
      console.error(`❌ Error processing document ${job.documentId}:`, error);
      
      await this.updateProcessingStatus(
        job.documentId, 
        'failed', 
        `Processing failed: ${error.message}`
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
    const response = await fetch(`http://localhost:9000/${context.job.s3Bucket}/${context.job.s3Key}`);
    
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
    return `Extracted PDF text from ${filePath}. This would contain the actual document content extracted using a proper PDF parsing library like pdf-parse or pdf2pic with OCR.`;
  }
  
  private async extractImageText(filePath: string): Promise<string> {
    // Simulate OCR with Tesseract
    // In production, use node-tesseract-ocr or similar
    return `OCR extracted text from image ${filePath}. This would contain text extracted using Tesseract OCR engine.`;
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
    
    // Simple chunking algorithm (in production, use LangChain RecursiveCharacterTextSplitter)
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
          wordCount: chunkContent.split(/\s+/).length
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
      // Call Ollama embedding API
      const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
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
    }
    
    context.embeddings = embeddings;
    console.log(`🎯 Generated ${embeddings.length} embeddings`);
  }
  
  private async storeVectorEmbeddings(context: ProcessingContext): Promise<void> {
    console.log('💾 Storing embeddings in pgvector');
    
    const { job, chunks, embeddings } = context;
    if (!chunks || !embeddings) throw new Error('No chunks or embeddings to store');
    
    // Store document chunks with embeddings in database
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = embeddings.find(e => e.chunkId === chunk.id);
      
      if (!embedding) continue;
      
      await db.insert(document_chunks).values({
        id: chunk.id,
        document_id: job.documentId,
        chunk_index: chunk.metadata.chunkIndex,
        content: chunk.content,
        start_position: chunk.metadata.startPosition,
        end_position: chunk.metadata.endPosition,
        word_count: chunk.metadata.wordCount,
        embedding: embedding.embedding,
        embedding_model: embedding.model,
        created_at: new Date(),
        updated_at: new Date()
      });
    }
    
    console.log(`✅ Stored ${chunks.length} chunks with embeddings`);
  }
  
  private async generateSummary(context: ProcessingContext): Promise<void> {
    console.log('📋 Generating document summary with Ollama Gemma3');
    
    const { job, extractedText } = context;
    if (!extractedText) throw new Error('No text to summarize');
    
    // Call Ollama for summarization using gemma3-legal model
    const summaryResponse = await fetch('http://localhost:11434/api/generate', {
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
    context.summary = summaryResult.response;
    
    // Store summary in database
    await db.insert(document_summaries).values({
      id: uuidv4(),
      document_id: job.documentId,
      summary_text: context.summary,
      summary_type: 'legal_analysis',
      model_used: 'gemma3-legal',
      confidence_score: 0.85, // Mock confidence
      created_at: new Date(),
      updated_at: new Date()
    });
    
    console.log(`📄 Generated summary (${context.summary.length} characters)`);
  }
  
  private async updateProcessingStatus(documentId: string, status: string, message?: string): Promise<void> {
    await db.update(document_processing)
      .set({
        status,
        status_message: message,
        updated_at: new Date()
      })
      .where(eq(document_processing.document_id, documentId));
      
    // Also update main document status
    await db.update(documents)
      .set({
        status: status === 'completed' ? 'processed' : status,
        updated_at: new Date()
      })
      .where(eq(documents.id, documentId));
  }
  
  private getFileExtension(filename: string): string {
    return filename.split('.').pop() || 'unknown';
  }
  
  private async cleanupTempFile(filePath: string): Promise<void> {
    // In production, implement proper file cleanup
    console.log(`🗑️  Cleaning up temp file: ${filePath}`);
  }
  
  // Health and stats methods
  getStats() {
    return {
      isRunning: this.isRunning,
      processedCount: this.processedCount,
      failedCount: this.failedCount,
      successRate: this.processedCount + this.failedCount > 0 
        ? (this.processedCount / (this.processedCount + this.failedCount)) * 100 
        : 0
    };
  }
}

// Export singleton instance
export const documentProcessingWorker = new DocumentProcessingWorker();

// Export types
export type { DocumentProcessingJob, ProcessingContext, DocumentChunk, EmbeddingResult };