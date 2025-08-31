// Enhanced RAG Service Type Definitions
// Centralized type definitions for AI-powered legal document processing

export interface RAGQuery {
  query: string;
  caseId?: string;
  documentIds?: string[];
  maxResults?: number;
  similarityThreshold?: number;
  includeMetadata?: boolean;
  filters?: {
    jurisdiction?: string;
    documentType?: string;
    practiceArea?: string;
    dateRange?: {
      start: string;
      end: string;
    };
  };
}

export interface SearchResult {
  documentId: string;
  title: string;
  content: string;
  similarity: number;
  rank: number;
  metadata?: {
    documentType: string;
    jurisdiction: string;
    practiceArea: string;
    createdAt: string;
    fileHash?: string;
  };
}

export interface RAGResponse {
  query: string;
  results: SearchResult[];
  response: string;
  timestamp: Date;
  processingTime: number;
  cacheHit?: boolean;
  metadata?: {
    totalDocuments: number;
    averageSimilarity: number;
    sources: string[];
    modelUsed: string;
    tokenUsage?: {
      prompt: number;
      completion: number;
      total: number;
    };
  };
}

export interface EnhancedRAGConfig {
  maxResults: number;
  similarityThreshold: number;
  embeddingModel: string;
  llmModel: string;
  cacheEnabled: boolean;
  cacheTtl: number;
  enableSemantic: boolean;
  enableContextual: boolean;
}

export interface DocumentAnalysis {
  documentId: string;
  entities: {
    type: string;
    value: string;
    confidence: number;
  }[];
  keyTerms: string[];
  sentimentScore: number;
  complexityScore: number;
  confidenceLevel: number;
  extractedDates: string[];
  extractedAmounts: string[];
  parties: string[];
  obligations: string[];
  risks: {
    type: string;
    severity: 'low' | 'medium' | 'high';
    description: string;
  }[];
}

export interface LegalAnalysisRequest {
  documentId?: string;
  content?: string;
  analysisType: 'document_analysis' | 'contract_review' | 'risk_assessment' | 'precedent_search' | 'compliance_check';
  options?: {
    includeRisks?: boolean;
    includePrecedents?: boolean;
    includeCompliance?: boolean;
    jurisdiction?: string;
    practiceArea?: string;
  };
}

export interface LegalAnalysisResponse {
  analysis: DocumentAnalysis;
  recommendations: {
    type: string;
    priority: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    actionRequired: boolean;
  }[];
  precedents?: {
    caseNumber: string;
    citation: string;
    relevance: 'high' | 'medium' | 'low';
    summary: string;
  }[];
  complianceIssues?: {
    regulation: string;
    status: 'compliant' | 'non-compliant' | 'needs_review';
    description: string;
  }[];
  processingTime: number;
  timestamp: Date;
}

// Service interface for enhanced RAG operations
export interface EnhancedRAGService {
  query(request: RAGQuery): Promise<RAGResponse>;
  analyzeDocument(request: LegalAnalysisRequest): Promise<LegalAnalysisResponse>;
  generateEmbedding(text: string): Promise<number[]>;
  findSimilarDocuments(documentId: string, limit?: number): Promise<SearchResult[]>;
  updateDocumentIndex(documentId: string, content: string): Promise<void>;
  getServiceHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    services: {
      database: boolean;
      vectorStore: boolean;
      llm: boolean;
      cache: boolean;
    };
    uptime: number;
  }>;
}

// Export type guards for runtime type checking
export function isRAGQuery(obj: any): obj is RAGQuery {
  return obj && typeof obj.query === 'string';
}

export function isRAGResponse(obj: any): obj is RAGResponse {
  return obj && 
    typeof obj.query === 'string' && 
    Array.isArray(obj.results) && 
    typeof obj.response === 'string';
}

export function isSearchResult(obj: any): obj is SearchResult {
  return obj && 
    typeof obj.documentId === 'string' && 
    typeof obj.title === 'string' && 
    typeof obj.content === 'string' && 
    typeof obj.similarity === 'number';
}