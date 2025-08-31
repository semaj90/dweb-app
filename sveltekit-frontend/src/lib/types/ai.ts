
export interface AIResponse {
  confidence?: number;
  keyTerms?: string[];
  processingTime?: number;
  gpuProcessed?: boolean;
  legalRisk?: string;
  [key: string]: unknown;
}

export interface VectorSearchResult {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
  source?: {
    type: string;
    name: string;
    url: string;
  };
  highlights?: string[];
  confidence?: number;
}

export interface SemanticEntity {
  id?: string;
  text: string;
  type: string;
  confidence: number;
  start?: number;
  end?: number;
  metadata?: Record<string, unknown>;
}

// Context7 integration types
export interface OrchestrationOptions {
  enabled: boolean;
  priority: 'low' | 'medium' | 'high';
  timeout?: number;
  retries?: number;
}

export interface MCPToolRequest {
  tool: string;
  args: Record<string, unknown>;
  context?: Record<string, unknown>;
}

export interface EnhancedRAGEngine {
  query: (prompt: string, options?: Record<string, unknown>) => Promise<AIResponse>;
  search: (query: string) => Promise<VectorSearchResult[]>;
  analyze: (content: string) => Promise<SemanticEntity[]>;
}
