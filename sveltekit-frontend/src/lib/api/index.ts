/**
 * @fileoverview Barrel Export for Legal AI API Routes
 * Comprehensive TypeScript API routing system for SvelteKit integration
 * with Go microservice, Ollama LLM, and Redis caching layers
 */

// =====================================
// Core API Types & Interfaces
// =====================================

export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: "success" | "error" | "loading";
  timestamp: string;
  requestId: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };
}

export interface LegalDocument {
  id: string;
  caseId: string;
  title: string;
  content: string;
  extractedText: string;
  documentType: "legal" | "evidence" | "contract" | "brief";
  jurisdiction: string;
  uploadedAt: string;
  processedAt?: string;
  embeddings?: number[];
  metadata: Record<string, any>;
}

export interface SemanticSearchQuery {
  query: string;
  limit?: number;
  offset?: number;
  filters?: {
    documentType?: string[];
    jurisdiction?: string;
    caseId?: string;
    dateRange?: { start: Date; end: Date };
    confidenceThreshold?: number;
  };
  includeEmbeddings?: boolean;
  rerank?: boolean;
}

export interface ChatMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  metadata?: {
    caseId?: string;
    relevantDocuments?: string[];
    processingTime?: number;
    tokenCount?: number;
    model?: string;
  };
}

// =====================================
// Authentication & User Management
// =====================================

export interface AuthUser {
  id: string;
  email: string;
  role: "admin" | "prosecutor" | "detective" | "user";
  permissions: string[];
  lastLogin?: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: "yorha-dark" | "yorha-light" | "professional";
  language: "en" | "es" | "fr";
  notifications: {
    email: boolean;
    browser: boolean;
    mobile: boolean;
  };
  dashboard: {
    layout: "grid" | "list";
    widgets: string[];
  };
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData extends LoginCredentials {
  name: string;
  role: "prosecutor" | "detective" | "user";
  organization?: string;
}

// =====================================
// Document Processing API
// =====================================

export interface DocumentUploadOptions {
  caseId: string;
  documentType: LegalDocument["documentType"];
  title?: string;
  jurisdiction?: string;
  metadata?: Record<string, any>;
  processImmediately?: boolean;
  generateEmbeddings?: boolean;
}

export interface DocumentProcessingResult {
  documentId: string;
  extractedText: string;
  wordCount: number;
  language: string;
  confidence: number;
  processingTime: number;
  embeddings?: {
    model: string;
    dimensions: number;
    vectors: number[];
  };
  analysis?: {
    entities: Array<{ text: string; type: string; confidence: number }>;
    summary: string;
    topics: string[];
    sentiment?: { score: number; label: "positive" | "negative" | "neutral" };
  };
}

// =====================================
// Vector Search & Embeddings
// =====================================

export interface VectorSearchOptions {
  query: string | number[];
  collection: "documents" | "cases" | "precedents";
  topK?: number;
  threshold?: number;
  metadataFilters?: Record<string, any>;
  includePayload?: boolean;
  includeVectors?: boolean;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  payload?: Record<string, any>;
  vector?: number[];
  document?: Partial<LegalDocument>;
}

export interface EmbeddingRequest {
  text: string | string[];
  model?: "nomic-embed-text" | "all-minilm-l6-v2" | "custom";
  normalize?: boolean;
  dimensions?: number;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model: string;
  dimensions: number;
  tokensUsed: number;
  processingTime: number;
}

// =====================================
// Real-time Chat & Streaming
// =====================================

export interface ChatSessionOptions {
  caseId?: string;
  model?: "llama3.1:8b" | "gemma2:9b" | "mixtral:8x7b";
  temperature?: number;
  maxTokens?: number;
  contextWindow?: number;
  systemPrompt?: string;
  ragEnabled?: boolean;
  ragSources?: string[];
}

export interface StreamingChatResponse {
  type: "token" | "complete" | "error" | "thinking";
  content?: string;
  metadata?: {
    tokensGenerated?: number;
    processingTime?: number;
    ragContext?: Array<{ id: string; relevance: number; snippet: string }>;
    error?: string;
  };
  done: boolean;
}

// =====================================
// Case Management
// =====================================

export interface LegalCase {
  id: string;
  title: string;
  description: string;
  status: "active" | "closed" | "archived" | "pending";
  priority: "low" | "medium" | "high" | "critical";
  jurisdiction: string;
  assignedTo: string[];
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  documents: LegalDocument[];
  evidence: Evidence[];
  notes: CaseNote[];
  timeline: CaseTimelineEntry[];
  metadata: Record<string, any>;
}

export interface Evidence {
  id: string;
  caseId: string;
  type: "physical" | "digital" | "document" | "testimony" | "photo" | "video";
  title: string;
  description: string;
  source: string;
  collectedAt: string;
  collectedBy: string;
  chainOfCustody: ChainOfCustodyEntry[];
  analysis?: {
    findings: string[];
    confidence: number;
    methodology: string;
  };
  files: Array<{
    id: string;
    filename: string;
    size: number;
    mimeType: string;
  }>;
}

export interface CaseNote {
  id: string;
  caseId: string;
  authorId: string;
  title: string;
  content: string;
  type: "general" | "analysis" | "action" | "reminder";
  private: boolean;
  createdAt: string;
  updatedAt: string;
  tags: string[];
}

export interface CaseTimelineEntry {
  id: string;
  caseId: string;
  type:
    | "document_added"
    | "evidence_collected"
    | "note_created"
    | "status_changed"
    | "assignment_changed";
  title: string;
  description: string;
  actorId: string;
  timestamp: string;
  metadata: Record<string, any>;
}

export interface ChainOfCustodyEntry {
  id: string;
  evidenceId: string;
  action: "collected" | "transferred" | "analyzed" | "stored" | "accessed";
  actor: string;
  location: string;
  timestamp: string;
  notes?: string;
  digitalSignature?: string;
}

// =====================================
// API Client Configuration
// =====================================

export interface ApiClientConfig {
  baseUrl: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  headers?: Record<string, string>;
  interceptors?: {
    request?: Array<(config: any) => any>;
    response?: Array<(response: any) => any>;
    error?: Array<(error: any) => any>;
  };
}

export interface ServiceEndpoints {
  sveltekit: {
    baseUrl: string;
    port: number;
    healthCheck: string;
  };
  goMicroservice: {
    baseUrl: string;
    port: number;
    grpcPort?: number;
    healthCheck: string;
    endpoints: {
      documents: string;
      search: string;
      embeddings: string;
      chat: string;
    };
  };
  ollama: {
    baseUrl: string;
    port: number;
    healthCheck: string;
    endpoints: {
      generate: string;
      chat: string;
      embeddings: string;
      models: string;
    };
  };
  redis: {
    host: string;
    port: number;
    password?: string;
    db?: number;
  };
  postgresql: {
    host: string;
    port: number;
    database: string;
    username: string;
    password: string;
    ssl?: boolean;
  };
  qdrant?: {
    baseUrl: string;
    port: number;
    apiKey?: string;
    collections: {
      documents: string;
      cases: string;
      precedents: string;
    };
  };
}

// =====================================
// Barrel Exports - API Services
// =====================================

// Authentication Services
export * from "./services/auth-service";
export * from "./services/user-service";

// Document Processing Services
export * from "./services/document-service";
export * from "./services/processing-service";
export * from "./services/upload-service";

// Search & AI Services
export * from "./services/chat-service";
export * from "./services/embedding-service";
export * from "./services/ollama-service";
export * from "./services/search-service";
export * from "./services/vector-service";

// Case Management Services
export * from "./services/case-service";
export * from "./services/evidence-service";
export * from "./services/note-service";

// Infrastructure Services
export * from "./services/cache-service";
export * from "./services/health-service";
export * from "./services/metrics-service";

// API Clients & Utilities
export * from "./clients/api-client";
export * from "./clients/sse-client";
export * from "./clients/websocket-client";
export * from "./utils/api-helpers";
export * from "./utils/error-handlers";
export * from "./utils/rate-limiter";

// =====================================
// Route Constants
// =====================================

export const API_ROUTES = {
  // Authentication
  AUTH: {
    LOGIN: "/api/auth/login",
    LOGOUT: "/api/auth/logout",
    REGISTER: "/api/auth/register",
    REFRESH: "/api/auth/refresh",
    PROFILE: "/api/auth/profile",
    CHANGE_PASSWORD: "/api/auth/change-password",
  },

  // Document Management
  DOCUMENTS: {
    LIST: "/api/documents",
    UPLOAD: "/api/documents/upload",
    GET: "/api/documents/:id",
    UPDATE: "/api/documents/:id",
    DELETE: "/api/documents/:id",
    DOWNLOAD: "/api/documents/:id/download",
    PROCESS: "/api/documents/:id/process",
    EXTRACT_TEXT: "/api/documents/extract-text",
  },

  // Search & AI
  SEARCH: {
    SEMANTIC: "/api/search/semantic",
    FULLTEXT: "/api/search/fulltext",
    HYBRID: "/api/search/hybrid",
    SUGGESTIONS: "/api/search/suggestions",
  },

  CHAT: {
    SESSIONS: "/api/chat/sessions",
    STREAM: "/api/chat/stream",
    HISTORY: "/api/chat/sessions/:id/history",
    CLEAR: "/api/chat/sessions/:id/clear",
  },

  // Vector Operations
  VECTORS: {
    EMBED: "/api/vectors/embed",
    SEARCH: "/api/vectors/search",
    COLLECTIONS: "/api/vectors/collections",
    UPSERT: "/api/vectors/upsert",
    DELETE: "/api/vectors/delete",
  },

  // Case Management
  CASES: {
    LIST: "/api/cases",
    CREATE: "/api/cases",
    GET: "/api/cases/:id",
    UPDATE: "/api/cases/:id",
    DELETE: "/api/cases/:id",
    DOCUMENTS: "/api/cases/:id/documents",
    EVIDENCE: "/api/cases/:id/evidence",
    NOTES: "/api/cases/:id/notes",
    TIMELINE: "/api/cases/:id/timeline",
    EXPORT: "/api/cases/:id/export",
  },

  // Evidence
  EVIDENCE: {
    LIST: "/api/evidence",
    CREATE: "/api/evidence",
    GET: "/api/evidence/:id",
    UPDATE: "/api/evidence/:id",
    DELETE: "/api/evidence/:id",
    CHAIN_OF_CUSTODY: "/api/evidence/:id/custody",
    ANALYSIS: "/api/evidence/:id/analysis",
  },

  // System & Health
  SYSTEM: {
    HEALTH: "/api/health",
    METRICS: "/api/metrics",
    STATUS: "/api/status",
    CONFIG: "/api/config",
    LOGS: "/api/logs",
  },

  // External Services
  EXTERNAL: {
    OLLAMA_HEALTH: "http://localhost:11434/api/version",
    GO_SERVICE_HEALTH: "http://localhost:8080/health",
    REDIS_PING: "redis://localhost:6379/ping",
    POSTGRES_HEALTH: "postgresql://localhost:5432/health",
  },
} as const;

// =====================================
// HTTP Status Codes
// =====================================

export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  RATE_LIMITED: 429,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503,
} as const;

// =====================================
// Default Configurations
// =====================================

export const DEFAULT_CONFIG: ServiceEndpoints = {
  sveltekit: {
    baseUrl: "http://localhost:3000",
    port: 3000,
    healthCheck: "/health",
  },
  goMicroservice: {
    baseUrl: "http://localhost:8080",
    port: 8080,
    grpcPort: 50051,
    healthCheck: "/health",
    endpoints: {
      documents: "/api/v1/documents",
      search: "/api/v1/search",
      embeddings: "/api/v1/embeddings",
      chat: "/api/v1/chat",
    },
  },
  ollama: {
    baseUrl: "http://localhost:11434",
    port: 11434,
    healthCheck: "/api/version",
    endpoints: {
      generate: "/api/generate",
      chat: "/api/chat",
      embeddings: "/api/embeddings",
      models: "/api/tags",
    },
  },
  redis: {
    host: "localhost",
    port: 6379,
    db: 0,
  },
  postgresql: {
    host: "localhost",
    port: 5432,
    database: "legal_ai_db",
    username: "legal_admin",
    password: "123456",
  },
  qdrant: {
    baseUrl: "http://localhost:6333",
    port: 6333,
    collections: {
      documents: "legal_documents",
      cases: "legal_cases",
      precedents: "legal_precedents",
    },
  },
};

// =====================================
// Runtime Environment Detection
// =====================================

export const RUNTIME_ENV = {
  isDevelopment: import.meta.env.NODE_ENV === "development",
  isProduction: import.meta.env.NODE_ENV === "production",
  isBrowser: typeof window !== "undefined",
  isServer: typeof window === "undefined",
  isTest: import.meta.env.NODE_ENV === "test",
} as const;

export default {
  API_ROUTES,
  HTTP_STATUS,
  DEFAULT_CONFIG,
  RUNTIME_ENV,
};
