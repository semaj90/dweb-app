// Missing type definitions shim for common global types
// Progressive replacement: Using proper Svelte 5 TypeScript patterns

import type { 
  AITask as ProperAITask, 
  AIResponse as ProperAIResponse, 
  WorkerStatus as ProperWorkerStatus 
} from '$lib/types/svelte5-patterns';

// AI/LLM Types - Using proper definitions
declare global {
  type LLMProvider = 'ollama' | 'openai' | 'anthropic' | 'huggingface';
  type AITask = ProperAITask;
  type AIResponse<T = any> = ProperAIResponse<T>;
  type WorkerStatus = ProperWorkerStatus;
  type WorkerMessage = {
    taskId: string;
    type: 'status' | 'result' | 'error';
    data: unknown;
    timestamp: Date;
  };
  type EnhancedRAGEngine = {
    query: (input: string, options?: { limit?: number; threshold?: number }) => Promise<AIResponse>;
    embed: (text: string) => Promise<number[]>;
    similaritySearch: (query: string, options?: { limit?: number }) => Promise<any[]>;
  };
  type ErrorProcessingPipeline = {
    processErrors: (errors: Error[]) => Promise<{ processed: number; fixed: number }>;
    analyzeError: (error: Error) => Promise<{ category: string; severity: 'low' | 'medium' | 'high' }>;
  };
}

// API Request/Response Types - Using proper definitions
import type {
  CaseCreateRequest as ProperCaseCreateRequest,
  CaseUpdateRequest as ProperCaseUpdateRequest,
  CaseSearchRequest as ProperCaseSearchRequest,
  CaseSearchResponse as ProperCaseSearchResponse,
  EvidenceCreateRequest as ProperEvidenceCreateRequest,
  EvidenceSearchRequest as ProperEvidenceSearchRequest,
  BulkOperationResponse as ProperBulkOperationResponse,
  FormSubmissionResult as ProperFormSubmissionResult
} from '$lib/types/svelte5-patterns';

declare global {
  type CaseCreateRequest = ProperCaseCreateRequest;
  type CaseUpdateRequest = ProperCaseUpdateRequest;
  type CaseSearchRequest = ProperCaseSearchRequest;
  type CaseSearchResponse = ProperCaseSearchResponse;
  type EvidenceCreateRequest = ProperEvidenceCreateRequest;
  type EvidenceSearchRequest = ProperEvidenceSearchRequest;
  type CommandSearchRequest = {
    query: string;
    filters?: Record<string, unknown>;
    limit?: number;
  };
  type CommandSearchResponse = {
    results: Array<{ id: string; title: string; description: string; score: number }>;
    total: number;
  };
  type BulkOperationResponse = ProperBulkOperationResponse;
  type FormSubmissionResult<T = any> = ProperFormSubmissionResult<T>;
}

// Database Types - Using proper definitions
import type {
  CaseState,
  EvidenceState,
  VectorSearchResult as ProperVectorSearchResult,
  VectorSearchOptions as ProperVectorSearchOptions
} from '$lib/types/svelte5-patterns';

declare module '$lib/types/database' {
  export type LegalDocument = {
    id: string;
    title: string;
    content: string;
    fileType: 'PDF' | 'TEXT' | 'DOCX' | 'HTML';
    uploadedAt: Date;
    processedAt?: Date;
    metadata: Record<string, unknown>;
    embedding?: number[];
  };
  export type DocumentChunk = {
    id: string;
    documentId: string;
    chunkIndex: number;
    content: string;
    embedding?: number[];
    metadata: Record<string, unknown>;
  };
  export type UserAiQuery = {
    id: string;
    userId: string;
    query: string;
    response?: string;
    createdAt: Date;
    embedding?: number[];
  };
  export type AutoTag = {
    id: string;
    entityId: string;
    entityType: 'case' | 'evidence' | 'document';
    tag: string;
    confidence: number;
    createdAt: Date;
  };
  export type Case = CaseState;
  export type Evidence = EvidenceState;
  export type VectorSearchOptions = ProperVectorSearchOptions;
  export type VectorSearchResult = ProperVectorSearchResult;
}

// Service Types
declare global {
  type DocumentCache = any;
  type ReinforcementLearningCache = any;
  type PGVectorStore = {
    ensureTableInDatabase?: () => Promise<void>;
    similaritySearchWithScore?: (query: string, limit?: number) => Promise<Array<[any, number]>>;
    addDocuments?: (docs: any[]) => Promise<void>;
    delete?: (options?: any) => Promise<void>;
    [key: string]: any;
  };
  type QueryResult = {
    content: string;
    score: number;
    sources?: any[];
  };
}

// XState Types
declare global {
  type RecommendationMachineContext = {
    userContext?: any;
    [key: string]: any;
  };
  type ConcurrencyContext = any;
  type ConcurrencyTask = any;
  type WorkerResult = any;
}

// External Library Types
declare module '$lib/types' {
  // Duplicate removed: // Duplicate removed: export type Case = any;
}

declare global {
  type GGUFInferenceRequest = {
    prompt: string;
    maxTokens: number;
    temperature: number;
    topP: number;
    topK: number;
    repeatPenalty: number;
    stopTokens: string[];
    priority: any;
  };
}

// Row/Database result types
declare global {
  interface RowList<T> {
    rows: T;
  }
}

