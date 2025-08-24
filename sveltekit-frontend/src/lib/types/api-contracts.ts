// End-to-End Type-Safe API Contracts
// Shared types between client and server for complete type safety

import type { z } from 'zod';
import type { Case, Evidence, User, LegalDocument } from '../server/db/schema-postgres';
import type { StandardApiResponse } from '../server/api/response';

// Re-export StandardApiResponse for external use
export type { StandardApiResponse };

// ==================== API CONTRACT TYPES ====================

// Base API response wrapper
export interface ApiContract<TRequest = any, TResponse = any> {
  request: TRequest;
  response: StandardApiResponse<TResponse>;
}

// ==================== CASE API CONTRACTS ====================

export namespace CaseAPI {
  // GET /api/cases - List and search cases
  export interface ListRequest {
    query?: string;
    status?: string[];
    priority?: string[];
    assignedTo?: string;
    dateStart?: string; // ISO date
    dateEnd?: string; // ISO date
    page?: number;
    limit?: number;
    useVectorSearch?: boolean;
  }

  export interface ListResponse {
    cases: Case[];
    pagination: {
      page: number;
      limit: number;
      total: number;
      hasNext: boolean;
      hasPrev: boolean;
    };
    search?: {
      term: string;
      resultsCount: number;
      vectorSearchUsed: boolean;
    } | null;
  }

  export type List = ApiContract<ListRequest, ListResponse>;

  // POST /api/cases - Create new case
  export interface CreateRequest {
    title: string;
    description?: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    status?: 'open' | 'investigating' | 'pending' | 'closed' | 'archived';
    incidentDate?: string; // ISO date
    location?: string;
    jurisdiction?: string;
  }

  export interface CreateResponse {
    case: Case;
    message: string;
  }

  export type Create = ApiContract<CreateRequest, CreateResponse>;

  // PUT /api/cases?id=<id> - Update case
  export interface UpdateRequest {
    title?: string;
    description?: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
    location?: string;
    jurisdiction?: string;
  }

  export interface UpdateResponse {
    case: Case;
    message: string;
  }

  export type Update = ApiContract<UpdateRequest, UpdateResponse>;

  // GET /api/cases?id=<id> - Get single case
  export interface GetRequest {
    id: string;
  }

  export interface GetResponse {
    case: Case & {
      evidence: Evidence[];
      createdByUser?: User;
      leadProsecutorUser?: User;
    };
  }

  export type Get = ApiContract<GetRequest, GetResponse>;
}

// ==================== EVIDENCE API CONTRACTS ====================

export namespace EvidenceAPI {
  // GET /api/evidence - List and search evidence
  export interface ListRequest {
    query?: string;
    caseId?: string;
    evidenceTypes?: string[];
    tags?: string[];
    dateStart?: string; // ISO date
    dateEnd?: string; // ISO date
    page?: number;
    limit?: number;
    useVectorSearch?: boolean;
  }

  export interface ListResponse {
    evidence: Evidence[];
    pagination: {
      page: number;
      limit: number;
      total: number;
      hasNext: boolean;
      hasPrev: boolean;
    };
    search?: {
      term: string;
      resultsCount: number;
      vectorSearchUsed: boolean;
    } | null;
  }

  export type List = ApiContract<ListRequest, ListResponse>;

  // POST /api/evidence - Create new evidence
  export interface CreateRequest {
    caseId?: string;
    title: string;
    description?: string;
    evidenceType: 'document' | 'photograph' | 'video' | 'audio' | 'physical' | 'digital' | 
                   'testimony' | 'forensic' | 'dna' | 'fingerprint' | 'weapon' | 'drug';
    fileType?: string;
    fileUrl?: string;
    fileName?: string;
    fileSize?: number;
    mimeType?: string;
    hash?: string;
    tags?: string[];
    collectedAt?: string; // ISO date
    collectedBy?: string;
    location?: string;
  }

  export interface CreateResponse {
    evidence: Evidence;
    message: string;
  }

  export type Create = ApiContract<CreateRequest, CreateResponse>;

  // PUT /api/evidence?id=<id> - Update evidence
  export interface UpdateRequest {
    title?: string;
    description?: string;
    evidenceType?: 'document' | 'photograph' | 'video' | 'audio' | 'physical' | 'digital' | 
                   'testimony' | 'forensic' | 'dna' | 'fingerprint' | 'weapon' | 'drug';
    tags?: string[];
    isAdmissible?: boolean;
  }

  export interface UpdateResponse {
    evidence: Evidence;
    message: string;
  }

  export type Update = ApiContract<UpdateRequest, UpdateResponse>;

  // DELETE /api/evidence?id=<id> - Delete evidence
  export interface DeleteRequest {
    id: string;
    reason?: string;
  }

  export interface DeleteResponse {
    message: string;
    evidenceId: string;
  }

  export type Delete = ApiContract<DeleteRequest, DeleteResponse>;
}

// ==================== AI/CHAT API CONTRACTS ====================

export namespace ChatAPI {
  export interface Message {
    id: string;
    content: string;
    role: 'user' | 'assistant' | 'system';
    timestamp: string; // ISO date
    metadata?: {
      model?: string;
      confidence?: number;
      processingTime?: number;
      sources?: VectorSearchResult[];
    };
  }

  export interface VectorSearchResult {
    id: string;
    content: string;
    score: number;
    type: 'case' | 'evidence' | 'statute' | 'document';
    metadata?: Record<string, any>;
  }

  // POST /api/ai/enhanced-chat - AI Chat
  export interface ChatRequest {
    query: string;
    context?: {
      userRole?: string;
      caseId?: string;
      documentIds?: string[];
      sessionContext?: unknown;
      enableLegalBERT?: boolean;
      enableRAG?: boolean;
      maxDocuments?: number;
    };
    settings?: {
      enhancementLevel?: 'basic' | 'standard' | 'advanced' | 'comprehensive';
      includeConfidenceScores?: boolean;
      enableStreamingResponse?: boolean;
      model?: string;
      temperature?: number;
      maxTokens?: number;
    };
  }

  export interface ChatResponse {
    response: string;
    synthesizedInput?: unknown;
    legalAnalysis?: unknown;
    ragResults?: unknown;
    confidence: number;
    processingTime: number;
    metadata: {
      model: string;
      tokensUsed?: number;
      enabledFeatures: string[];
      fallbacksUsed?: string[];
      cacheHits?: string[];
    };
    recommendations?: string[];
    contextualPrompts?: unknown[];
  }

  export type Chat = ApiContract<ChatRequest, ChatResponse>;
}

// ==================== VECTOR SEARCH API CONTRACTS ====================

export namespace VectorSearchAPI {
  export interface SearchRequest {
    query: string;
    type?: 'semantic' | 'similarity' | 'hybrid';
    limit?: number;
    threshold?: number;
    filters?: Record<string, any>;
  }

  export interface SearchResponse {
    results: VectorSearchResult[];
    total: number;
    query: string;
    took: string;
    metadata?: {
      model?: string;
      threshold?: number;
      vectorSearchUsed?: boolean;
    };
  }

  export interface VectorSearchResult {
    id: string;
    content: string;
    score: number;
    type: 'case' | 'evidence' | 'statute' | 'document';
    metadata?: Record<string, any>;
  }

  export type Search = ApiContract<SearchRequest, SearchResponse>;
}

// ==================== SYSTEM HEALTH API CONTRACTS ====================

export namespace HealthAPI {
  export interface HealthRequest {
    detailed?: boolean;
  }

  export interface ServiceStatus {
    status: 'healthy' | 'degraded' | 'unhealthy';
    [key: string]: any;
  }

  export interface HealthResponse {
    status: 'healthy' | 'degraded' | 'unhealthy';
    timestamp: string;
    version: string;
    uptime: number;
    environment: string;
    services: {
      database: ServiceStatus;
      ollama: ServiceStatus;
      enhancedRAG: ServiceStatus;
      uploadService: ServiceStatus;
      memory: ServiceStatus;
      cache: ServiceStatus;
    };
    performance: {
      responseTime: number;
      status: 'healthy' | 'degraded';
    };
    errors?: string[];
    system?: {
      nodeVersion: string;
      platform: string;
      arch: string;
      cpuUsage: NodeJS.CpuUsage;
      env: Record<string, string>;
    };
  }

  export type Health = ApiContract<HealthRequest, HealthResponse>;

  // POST /api/health - System maintenance
  export interface MaintenanceRequest {
    action: 'clear_cache' | 'force_gc' | 'test_database';
  }

  export interface MaintenanceResponse {
    message: string;
    [key: string]: any;
  }

  export type Maintenance = ApiContract<MaintenanceRequest, MaintenanceResponse>;
}

// ==================== TYPE UTILITY FUNCTIONS ====================

// Extract request type from API contract
export type RequestOf<T extends ApiContract<any, any>> = T['request'];

// Extract response type from API contract
export type ResponseOf<T extends ApiContract<any, any>> = T['response'];

// Extract data type from API response
export type DataOf<T extends ApiContract<any, StandardApiResponse<any>>> = 
  T['response'] extends StandardApiResponse<infer U> ? U : never;

// ==================== CLIENT API HELPERS ====================

// Type-safe API client function generator
export type ApiClient<T extends ApiContract<any, any>> = (
  request: RequestOf<T>
) => Promise<ResponseOf<T>>;

// API endpoint configuration
export interface ApiEndpoint<T extends ApiContract<any, any>> {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  path: string;
  contract: T;
}

// ==================== API ENDPOINT DEFINITIONS ====================

export const API_ENDPOINTS = {
  // Case endpoints
  cases: {
    list: { method: 'GET', path: '/api/cases' } as ApiEndpoint<CaseAPI.List>,
    create: { method: 'POST', path: '/api/cases' } as ApiEndpoint<CaseAPI.Create>,
    update: { method: 'PUT', path: '/api/cases' } as ApiEndpoint<CaseAPI.Update>,
    get: { method: 'GET', path: '/api/cases' } as ApiEndpoint<CaseAPI.Get>,
  },
  
  // Evidence endpoints
  evidence: {
    list: { method: 'GET', path: '/api/evidence' } as ApiEndpoint<EvidenceAPI.List>,
    create: { method: 'POST', path: '/api/evidence' } as ApiEndpoint<EvidenceAPI.Create>,
    update: { method: 'PUT', path: '/api/evidence' } as ApiEndpoint<EvidenceAPI.Update>,
    delete: { method: 'DELETE', path: '/api/evidence' } as ApiEndpoint<EvidenceAPI.Delete>,
  },
  
  // AI/Chat endpoints
  ai: {
    chat: { method: 'POST', path: '/api/ai/enhanced-chat' } as ApiEndpoint<ChatAPI.Chat>,
  },
  
  // Vector search endpoints
  vectorSearch: {
    search: { method: 'POST', path: '/api/vector-search' } as ApiEndpoint<VectorSearchAPI.Search>,
  },
  
  // Health endpoints
  health: {
    check: { method: 'GET', path: '/api/health' } as ApiEndpoint<HealthAPI.Health>,
    maintenance: { method: 'POST', path: '/api/health' } as ApiEndpoint<HealthAPI.Maintenance>,
  },
} as const;

// ==================== VALIDATION SCHEMAS ====================

// Re-export validation schemas from the server for client-side validation
// Note: These would typically be imported from a shared validation library
export type ValidationSchema<T> = z.ZodType<T>;
