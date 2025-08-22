// Chat types for Legal AI platform
export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  sessionId?: string;
  metadata?: {
    model?: string;
    confidence?: number;
    totalDuration?: number;
    loadDuration?: number;
    promptEvalCount?: number;
    evalCount?: number;
    sources?: string[];
    processingTime?: number;
    [key: string]: unknown;
  };
}

export interface ChatSession {
  id: string;
  model: string;
  createdAt: Date;
  updatedAt?: Date;
  messageCount?: number;
  isActive?: boolean;
  metadata?: {
    userAgent?: string;
    context?: string;
    tags?: string[];
    [key: string]: unknown;
  };
}

export interface ChatState {
  messages: ChatMessage[];
  currentSession: ChatSession | null;
  isLoading: boolean;
  error: string | null;
}

export interface OllamaResponse {
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

export interface ChatRequest {
  message: string;
  context?: ChatMessage[];
  sessionId?: string;
  model?: string;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  confidence?: number;
  sources?: string[];
  processingTime?: number;
  model: string;
  metadata?: unknown;
}
