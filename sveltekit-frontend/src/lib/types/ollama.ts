
export interface EmbeddingResponse {
  embedding: number[];
}

export interface GenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface StreamingGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
}

export interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details?: {
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

export interface OllamaTagsResponse {
  models: OllamaModel[];
}

export interface DocumentChunk {
  content: string;
  embedding: number[];
  metadata: {
    chunkIndex: number;
    totalChunks: number;
    documentId?: string;
    [key: string]: unknown;
  };
}

export interface EmbeddingMetadata {
  model: string;
  dimension: number;
  timestamp: Date;
  context?: Record<string, any>;
}

export interface AnalysisResult {
  type: 'summary' | 'entities' | 'sentiment' | 'classification';
  content: string;
  confidence?: number;
  metadata?: Record<string, any>;
}