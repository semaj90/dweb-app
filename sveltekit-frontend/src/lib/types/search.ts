/**
 * Enhanced Search and Document Types with backward compatibility
 */

export interface SearchResult {
  id: string;
  title: string;
  content?: string;
  summary?: string;
  excerpt?: string;
  score: number;
  rank?: number;
  document?: unknown; // Backward compatibility
  metadata?: Record<string, any>;
  type?: string;
}

export interface SummaryResult {
  summary: string;
  keyPoints: string[];
  metadata: {
    documentsProcessed: number;
    processingTime: number;
    lambda: number;
    sentenceCount?: number; // Added for MMR compatibility
  };
  sources?: string[]; // Added for enhanced functionality
}

export interface SummaryRequest {
  documents: any[];
  maxSentences?: number;
  lambda?: number;
  type?: string;
}

export interface LegalDocument {
  id: string;
  title: string;
  content: string;
  type?: string;
  metadata?: Record<string, any>;
}

export interface SystemStatus {
  database: boolean;
  qdrant: boolean;
  embeddings: boolean;
  vectorSearch: boolean;
  redis?: boolean;
  ollama?: boolean;
  gpu?: boolean;
  cuda?: boolean;
  minio?: boolean;
  neo4j?: boolean;
  rabbitmq?: boolean;
  elasticsearch?: boolean;
  langchain?: boolean;
}

export interface TestResults {
  query: string;
  results: any[];
  timestamp: Date;
  performance: {
    duration: number;
    documentsSearched: number;
  };
  error?: unknown;
}

export interface TensorOperation {
  type: string;
  data: any;
  shape?: number[];
}

export interface MetricData {
  name: string;
  value: number;
  timestamp: Date;
  labels?: Record<string, string>;
}

// GPU-specific types
export interface GPUChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  embedding?: number[];
  metadata?: {
    model?: string;
    processingTime?: number;
    gpuUsed?: boolean;
    tokenCount?: number;
  };
}

export interface GPUProcessingStatus {
  gpuAvailable: boolean;
  cudaVersion?: string;
  gpuMemory?: {
    total: number;
    used: number;
    free: number;
  };
  activeJobs: number;
  queueLength: number;
}

export interface StreamingResponse {
  type: 'chunk' | 'complete' | 'error';
  content?: string;
  error?: string;
  metadata?: {
    tokensGenerated?: number;
    processingTimeMs?: number;
  };
}
