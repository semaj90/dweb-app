// Missing type definitions shim for common global types
// These are permissive 'any' types to reduce TypeScript noise during migration

// AI/LLM Types
declare global {
  type LLMProvider = any;
  type AITask = any;
  type AIResponse<T = any> = any;
  type WorkerStatus = any;
  type WorkerMessage = any;
  type EnhancedRAGEngine = any;
  type ErrorProcessingPipeline = any;
}

// API Request/Response Types
declare global {
  type CaseCreateRequest = any;
  type CaseUpdateRequest = any;
  type CaseSearchRequest = any;
  type CaseSearchResponse = any;
  type EvidenceCreateRequest = any;
  type EvidenceSearchRequest = any;
  type CommandSearchRequest = any;
  type CommandSearchResponse = any;
  type BulkOperationResponse = {
    processed: number;
    [key: string]: any;
  };
  type FormSubmissionResult<T = any> = any;
}

// Database Types
declare module '$lib/types/database' {
  export type LegalDocument = any;
  export type DocumentChunk = any;
  export type UserAiQuery = any;
  export type AutoTag = any;
  export type Case = any;
  export type Evidence = any;
  export type VectorSearchOptions = any;
  export type VectorSearchResult = {
    id: string;
    content: string;
    similarity: number;
    metadata: any;
    sourceType: "document" | "evidence" | "case";
    rankingMatrix: number[][];
  };
}

// Service Types
declare global {
  type DocumentCache = any;
  type ReinforcementLearningCache = any;
  type PGVectorStore = {
    ensureTableInDatabase?: any;
    similaritySearchWithScore?: any;
    [key: string]: any;
  };
  type QueryResult = {
    content: string;
    score: number;
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
  export type Case = any;
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

export {};