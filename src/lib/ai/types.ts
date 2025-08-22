// Type definitions for AI services

export interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_duration?: number;
  eval_duration?: number;
  eval_count?: number;
}

export interface OllamaEmbedding {
  embedding: number[];
}

export interface ModelInfo {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    families: string[] | null;
    parameter_size: string;
    quantization_level: string;
  };
}

export interface Document {
  id: string;
  content: string;
  metadata: {
    title?: string;
    type?: string;
    source?: string;
    created_at?: Date;
    tags?: string[];
    [key: string]: unknown;
  };
  embedding?: number[];
}

export interface SearchResult {
  document: Document;
  score: number;
  highlights?: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

export interface LegalCase {
  id: string;
  title: string;
  description: string;
  client_name?: string;
  case_type: string;
  status: 'active' | 'closed' | 'pending';
  created_at: Date;
  updated_at: Date;
  documents: Document[];
  notes?: string;
  tags: string[];
}

export interface EmbeddingSearchOptions {
  limit?: number;
  threshold?: number;
  filter?: Record<string, any>;
}
