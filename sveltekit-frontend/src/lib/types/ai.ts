// @ts-nocheck
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
