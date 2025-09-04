/**
 * Type definitions for Document Ingest Integration
 * Supports both single and batch document processing
 */

export interface DocumentIngestRequest {
  title: string;
  content: string;
  case_id?: string;
  metadata?: Record<string, any>;
}

export interface BatchIngestRequest {
  documents: DocumentIngestRequest[];
}

export interface IngestResult {
  success: boolean;
  documentId: string;
  embeddingId: string;
  processingTime: number;
  metadata?: any;
}

export interface BatchIngestResult {
  success: boolean;
  batchId: string;
  processed: number;
  failed: number;
  successRate: string;
  results: Array<{
    documentId: string;
    embeddingId: string;
    processingTime: number;
  }>;
  performance?: any;
}

export interface ChunkingOptions {
  maxChunkSize?: number;
  overlap?: number;
  preserveSentences?: boolean;
  legalAware?: boolean;
}

export interface ChunkedDocument {
  content: string;
  index: number;
  metadata?: Record<string, any>;
}

export interface LegalSection {
  title: string;
  content: string;
  type: string;
  context: string;
}

export interface SimilarDocument {
  id: string;
  title: string;
  content: string;
  similarity: number;
  metadata?: Record<string, any>;
  embedding?: number[];
}