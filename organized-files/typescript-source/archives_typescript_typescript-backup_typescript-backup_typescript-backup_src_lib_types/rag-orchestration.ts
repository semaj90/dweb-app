/**
 * Type definitions for Production RAG Orchestration System
 */

export interface DocumentProcessingJob {
  jobId: string;
  uploadId: string;
  caseId: string;
  filename: string;
  storageUrl: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number; // 0-100
  
  stages: {
    upload: ProcessingStage;
    extraction: ProcessingStage;
    chunking: ProcessingStage;
    embedding: ProcessingStage;
    vectorIndexing: ProcessingStage;
    summarization: ProcessingStage;
  };
  
  // Data from processing stages
  extractedText?: string;
  textChunks?: TextChunk[];
  embeddings?: number[][];
  embeddingMetadata?: EmbeddingMetadata[];
  vectorIds?: string[];
  summary?: string;
  keyTerms?: string[];
  legalEntities?: LegalEntity[];
  metadata?: DocumentMetadata;
  
  // Timing and metrics
  startTime: number;
  endTime?: number;
  processingTime?: number;
  error?: string;
  metrics: ProcessingMetrics;
}

export interface ProcessingStage {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  startTime?: number;
  endTime?: number;
  processingTime?: number;
  error?: string;
}

export interface TextChunk {
  id: string;
  text: string;
  startIndex: number;
  endIndex: number;
  metadata: ChunkMetadata;
}

export interface ChunkMetadata {
  page?: number;
  section?: string;
  paragraph?: number;
  confidence?: number;
  language?: string;
  wordCount: number;
}

export interface EmbeddingMetadata {
  chunkId: string;
  model: string;
  dimensions: number;
  confidence: number;
  processingTime: number;
}

export interface DocumentMetadata {
  fileSize: number;
  mimeType: string;
  pageCount?: number;
  wordCount: number;
  language: string;
  extractionMethod: string;
  extractionConfidence: number;
  createdAt: string;
  modifiedAt: string;
}

export interface LegalEntity {
  type: 'person' | 'organization' | 'case' | 'statute' | 'date' | 'location';
  value: string;
  confidence: number;
  startIndex: number;
  endIndex: number;
  context?: string;
}

export interface RAGQueryResult {
  query: string;
  response: string;
  sources: RAGSource[];
  confidence: number;
  processingTime: number;
  cached: boolean;
  metadata: {
    model: string;
    totalTokens: number;
    retrievalTime: number;
    generationTime: number;
  };
}

export interface RAGSource {
  id: string;
  content: string;
  score: number;
  metadata: {
    filename: string;
    caseId: string;
    uploadId: string;
    chunkId: string;
    page?: number;
    section?: string;
  };
}

export interface ServiceHealthStatus {
  name: string;
  url: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  lastCheck: number;
  responseTime: number;
  error?: string;
}

export interface ProcessingMetrics {
  documentsProcessed: number;
  totalProcessingTime: number;
  averageProcessingTime: number;
  successRate: number;
  activeJobs: number;
  queueDepth: number;
}

export interface RAGPipelineState {
  coordinator: {
    status: 'initializing' | 'ready' | 'processing' | 'error';
    uptime: number;
    version: string;
  };
  services: Map<string, ServiceHealthStatus>;
  metrics: ProcessingMetrics;
  activeJobs: DocumentProcessingJob[];
  recentJobs: DocumentProcessingJob[];
}

export interface DocumentEmbeddingResult {
  embeddings: number[][];
  metadata: EmbeddingMetadata[];
  model: string;
  dimensions: number;
  processingTime: number;
}

export interface VectorIndexResult {
  vectorIds: string[];
  indexName: string;
  totalVectors: number;
  processingTime: number;
}

export interface DocumentSummaryResult {
  summary: string;
  keyTerms: string[];
  entities: LegalEntity[];
  confidence: number;
  model: string;
  processingTime: number;
}

// Event types for the coordinator
export interface CoordinatorEvents {
  'system:initialized': void;
  'system:error': { error: Error };
  'document:uploaded': { jobId: string; filename: string };
  'document:processed': DocumentProcessingJob;
  'job:progress': DocumentProcessingJob;
  'stage:completed': { jobId: string; stage: string; job: DocumentProcessingJob };
  'stage:failed': { jobId: string; stage: string; error: string };
  'service:health': { service: string; status: ServiceHealthStatus };
}

// Configuration interfaces
export interface RAGCoordinatorConfig {
  redis: {
    host: string;
    port: number;
    password?: string;
    db: number;
  };
  services: {
    [key: string]: {
      url: string;
      timeout: number;
      retries: number;
    };
  };
  processing: {
    concurrency: number;
    batchSize: number;
    chunkSize: number;
    chunkOverlap: number;
  };
  monitoring: {
    healthCheckInterval: number;
    metricsRetention: number;
  };
}

// Job queue interfaces
export interface JobQueue {
  name: string;
  priority: number;
  concurrency: number;
  retries: number;
  backoff: {
    type: 'fixed' | 'exponential';
    delay: number;
  };
}

export interface JobQueueStats {
  waiting: number;
  active: number;
  completed: number;
  failed: number;
  delayed: number;
  paused: number;
}

// Error types
export class RAGOrchestrationError extends Error {
  constructor(
    message: string,
    public stage: string,
    public jobId: string,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'RAGOrchestrationError';
  }
}

export class ServiceUnavailableError extends Error {
  constructor(
    public serviceName: string,
    public serviceUrl: string,
    message?: string
  ) {
    super(message || `Service ${serviceName} is unavailable`);
    this.name = 'ServiceUnavailableError';
  }
}