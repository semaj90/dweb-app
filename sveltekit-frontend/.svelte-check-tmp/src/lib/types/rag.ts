
/**
 * RAG (Retrieval Augmented Generation) Types
 * Standardized types for document processing and search
 */

export interface DocumentType {
  id: string;
  title: string;
  content: string;
  type: 'legal' | 'case' | 'evidence' | 'research' | 'document';
  metadata?: Record<string, any>;
}

export interface RAGSearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  type: DocumentType['type'];
  // Additional properties expected by optimization files
  document?: RAGDocument;
  explanation?: string;
  relevantChunks?: TextChunk[];
  highlights?: string[];
  legalRelevance?: {
    overall: number;
    factual: number;
    procedural: number;
    precedential: number;
    jurisdictional: number;
    confidence: number;
  };
  relevanceScore?: number;
  rank?: number;
  snippet?: string;
}

export interface RAGQuery {
  query: string;
  type?: DocumentType['type'];
  limit?: number;
  threshold?: number;
}

export interface RAGResponse {
  results: RAGSearchResult[];
  query: string;
  totalResults: number;
  processingTime: number;
}

// Fix for optimization files that expect "research" type
export type ExtendedDocumentType = DocumentType['type'] | 'research';

export interface OptimizedRAGDocument {
  id: string;
  content: string;
  type: ExtendedDocumentType;
  embedding?: number[];
  metadata: Record<string, any>;
}

// Type guards for safe type checking
export function isValidDocumentType(type: string): type is DocumentType['type'] {
  return ['legal', 'case', 'evidence', 'research', 'document'].includes(type);
}

export function normalizeDocumentType(type: string): DocumentType['type'] {
  if (type === 'research') return 'document';
  return isValidDocumentType(type) ? type : 'document';
}

// Enhanced document processing types
export interface DocumentChunk {
  id: string;
  documentId: string;
  content: string;
  embedding: number[];
  metadata: Record<string, any>;
  chunkIndex: number;
}

export interface VectorSearchOptions {
  limit?: number;
  threshold?: number;
  includeMetadata?: boolean;
  filters?: Record<string, any>;
}

export interface EmbeddingConfig {
  model: string;
  dimensions: number;
  provider: 'openai' | 'ollama' | 'local';
}

// Missing exports referenced by optimization files
export interface RAGDocument {
  id: string;
  title: string;
  content: string;
  type: DocumentType['type'];
  embedding?: number[];
  metadata: {
    source: string;
    type: string;
    jurisdiction: string;
    practiceArea: string[];
    confidentialityLevel: number;
    lastModified: Date;
    fileSize: number;
    language: string;
    tags: string[];
  };
  createdAt?: Date;
  updatedAt?: Date;
  version?: string;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  type: DocumentType['type'];
  metadata: Record<string, any>;
}

export interface TextChunk {
  id: string;
  content: string;
  documentId: string;
  index: number;
  metadata: Record<string, any>;
  // Additional properties for optimization compatibility
  startIndex?: number;
  endIndex?: number;
  score?: number;
  embeddings?: number[];
  chunkType?: string;
}

// RAG source types
export interface RAGSource {
  id: string;
  title: string;
  content: string;
  relevance: number;
  type: 'document' | 'evidence' | 'case' | 'legal' | 'research';
}
