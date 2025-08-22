// Legal Document Types
// Complete type definitions for legal documents

export interface LegalDocument {
  id: string;
  title: string;
  documentType: string;
  jurisdiction: string;
  court?: string;
  citation?: string;
  fullCitation?: string;
  docketNumber?: string;
  dateDecided?: Date;
  datePublished?: Date;
  content: string;
  summary?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  processingStatus: 'pending' | 'processing' | 'completed' | 'failed';
  analysisResults?: unknown;
  contentEmbedding?: number[];
  titleEmbedding?: number[];
  fileHash?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  practiceArea?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface NewLegalDocument {
  title: string;
  content: string;
  documentType: string;
  jurisdiction: string;
  court?: string;
  citation?: string;
  fullCitation?: string;
  docketNumber?: string;
  dateDecided?: Date;
  datePublished?: Date;
  summary?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  processingStatus?: 'pending' | 'processing' | 'completed' | 'failed';
  analysisResults?: unknown;
  contentEmbedding?: number[];
  titleEmbedding?: number[];
  fileHash?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  practiceArea?: string;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface DocumentAnalysis {
  entities: Array<{
    type: string;
    value: string;
    confidence: number;
  }>;
  keyTerms: string[];
  sentimentScore: number;
  complexityScore: number;
  confidenceLevel: number;
  extractedDates: string[];
  extractedAmounts: string[];
  parties: string[];
  obligations: string[];
  risks: Array<{
    type: string;
    severity: 'low' | 'medium' | 'high';
    description: string;
  }>;
  rawAnalysis?: string;
  processingTime?: number;
  agentUsed?: string;
  error?: string;
}

// Content embedding types for vector storage
export interface ContentEmbedding {
  id?: string;
  contentId: string;
  contentType: string;
  textContent: string;
  embedding?: string; // JSON stringified number[]
  metadata?: unknown;
  createdAt?: Date;
  documentId?: string;
  model?: string;
}
