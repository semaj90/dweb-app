// AI Types for Gemma3 Legal AI Integration
// Comprehensive type definitions for chat, models, and streaming

export interface AIModel {
  name: string;
  size: string;
  modified: Date;
  description?: string;
  family?: string;
  format?: string;
  parameters?: string;
  quantization?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  streaming?: boolean;
  error?: boolean;
  metadata?: {
    tokens?: number;
    duration?: number;
    temperature?: number;
    context_length?: number;
  };
}

export interface OllamaResponse {
  response: string;
  model: string;
  created_at: Date;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface StreamResponse {
  response: string;
  model: string;
  done: boolean;
  created_at: Date;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  options?: OllamaOptions;
  system?: string;
  template?: string;
  context?: number[];
  raw?: boolean;
  keep_alive?: string;
}

export interface OllamaChatRequest {
  model: string;
  messages: OllamaMessage[];
  stream?: boolean;
  options?: OllamaOptions;
  keep_alive?: string;
  format?: "json";
}

export interface OllamaMessage {
  role: "user" | "assistant" | "system";
  content: string;
  images?: string[];
}

export interface OllamaOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  seed?: number;
  num_ctx?: number;
  num_predict?: number;
  stop?: string[];
  tfs_z?: number;
  typical_p?: number;
  repeat_last_n?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  mirostat?: number;
  mirostat_eta?: number;
  mirostat_tau?: number;
  penalize_newline?: boolean;
  numa?: boolean;
  num_thread?: number;
  num_gpu?: number;
  main_gpu?: number;
  low_vram?: boolean;
  f16_kv?: boolean;
}

export interface ModelInfo {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

export interface LegalAIContext {
  caseType?:
    | "criminal"
    | "civil"
    | "corporate"
    | "family"
    | "immigration"
    | "intellectual_property";
  jurisdiction?: string;
  practiceArea?: string;
  documentType?:
    | "contract"
    | "brief"
    | "motion"
    | "pleading"
    | "opinion"
    | "statute";
  urgency?: "low" | "medium" | "high" | "critical";
  confidentiality?: "public" | "confidential" | "attorney_client_privileged";
}

export interface LegalAIRequest {
  message: string;
  context?: LegalAIContext;
  attachments?: FileAttachment[];
  citations?: boolean;
  format?: "plain" | "markdown" | "legal_memo" | "brief_format";
}

export interface FileAttachment {
  id: string;
  name: string;
  type: string;
  size: number;
  content?: string;
  url?: string;
  metadata?: {
    pages?: number;
    wordCount?: number;
    language?: string;
  };
}

export interface LegalCitation {
  type: "case" | "statute" | "regulation" | "secondary";
  title: string;
  citation: string;
  year?: number;
  court?: string;
  jurisdiction?: string;
  url?: string;
  relevance: "primary" | "secondary" | "supporting";
}

export interface AIServiceStatus {
  healthy: boolean;
  model: string;
  endpoint: string;
  latency?: number;
  lastCheck: Date;
  version?: string;
  capabilities?: string[];
}

export interface ChatSession {
  id: string;
  title: string;
  created: Date;
  updated: Date;
  messages: ChatMessage[];
  model: string;
  context?: LegalAIContext;
  metadata?: {
    totalTokens: number;
    totalDuration: number;
    messageCount: number;
  };
}

export interface RAGDocument {
  id: string;
  title: string;
  content: string;
  type:
    | "case_law"
    | "statute"
    | "regulation"
    | "article"
    | "brief"
    | "contract";
  metadata: {
    jurisdiction?: string;
    year?: number;
    court?: string;
    citations?: string[];
    tags?: string[];
    category?: string;
  };
  embeddings?: number[];
  created: Date;
  updated: Date;
}

export interface RAGSearchResult {
  document: RAGDocument;
  score: number;
  highlights?: string[];
  reasoning?: string;
}

export interface AIError extends Error {
  code:
    | "MODEL_NOT_FOUND"
    | "CONNECTION_ERROR"
    | "RATE_LIMITED"
    | "INVALID_REQUEST"
    | "UNKNOWN";
  model?: string;
  endpoint?: string;
  details?: any;
  retryable: boolean;
}

// Event types for real-time updates
export interface ChatEvent {
  type: "message" | "typing" | "error" | "model_changed" | "session_created";
  sessionId: string;
  data: any;
  timestamp: Date;
}

// Configuration types
export interface AIModelConfig {
  name: string;
  displayName: string;
  description: string;
  capabilities: string[];
  parameters: {
    maxTokens: number;
    temperature: number;
    topP: number;
    topK: number;
    repeatPenalty: number;
  };
  pricing?: {
    inputTokens: number;
    outputTokens: number;
    currency: string;
  };
}

export interface LegalAIConfig {
  models: AIModelConfig[];
  defaultModel: string;
  fallbackModel: string;
  features: {
    streaming: boolean;
    citations: boolean;
    ragIntegration: boolean;
    documentAnalysis: boolean;
    contextInjection: boolean;
  };
  limits: {
    maxMessageLength: number;
    maxContextLength: number;
    maxSessionDuration: number;
    maxConcurrentRequests: number;
  };
}

// Utility type helpers
export type MessageRole = ChatMessage["role"];
export type ModelStatus = "available" | "loading" | "error" | "offline";
export type StreamingState =
  | "idle"
  | "connecting"
  | "streaming"
  | "complete"
  | "error";

// API response wrappers
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: Date;
  requestId: string;
}

export interface PaginatedResponse<T> extends APIResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };
}
