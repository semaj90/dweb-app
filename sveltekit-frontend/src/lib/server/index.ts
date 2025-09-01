// ---
// CRITICAL FIX: Comprehensive Barrel Export File
//
// As identified in SVELTEKIT-ARCHITECTURE-SUMMARY.md, this file was a critical missing piece.
// It provides a single, centralized export point for all server-side modules, services,
// database schemas, and types. This simplifies imports across the application and improves
// maintainability.
//
// e.g., import { apiClient, db, documents } from '$lib/server';
// ---

// --- Database & ORM ---
export * from './db/index';
export * from './db/schema-postgres';
export * from './db/schema-types';
export { db } from './db/connection';

// --- Evidence Management ---
export * from './evidence.service';

// --- API Response Utilities ---
export * from './api/response';
export { withApiHandler, parseRequestBody, CommonErrors, createPagination } from './api/response';

// --- Microservice Communication ---
export * from './api-client.service';

// --- Storage Services ---
export * from './storage/minio-service';

// --- Authentication & Authorization ---
export * from './auth/session';
export * from './auth/jwt';
export * from './auth/permissions';

// --- AI & ML Services ---
export * from './ai/ollama-service';
export * from './ai/enhanced-rag-service';
export * from './ai/vector-service';
export * from './ai/embedding-service';

// --- GPU & CUDA Integration ---
export * from './gpu/cuda-orchestrator';
export * from './gpu/tensor-processor';
export * from './gpu/webgpu-bridge';

// --- Legal Domain Services ---
export * from './legal/document-processor';
export * from './legal/case-manager';
export * from './legal/evidence-analyzer';
export * from './legal/precedent-search';

// --- Communication Protocols ---
export * from './protocols/quic-client';
export * from './protocols/grpc-client';
export * from './protocols/websocket-manager';
export * from './protocols/nats-messaging';

// --- Caching Layer ---
export * from './cache/redis-client';
export * from './cache/memory-cache';
export * from './cache/vector-cache';

// --- Health & Monitoring ---
export * from './health/service-monitor';
export * from './health/performance-tracker';
export * from './health/system-diagnostics';

// --- Search & Indexing ---
export * from './search/qdrant-client';
export * from './search/neo4j-client';
export * from './search/elasticsearch-client';

// --- File Processing ---
export * from './files/upload-handler';
export * from './files/document-parser';
export * from './files/metadata-extractor';

// --- Security & Validation ---
export * from './security/input-sanitizer';
export * from './security/rate-limiter';
export * from './security/encryption';

// --- Workflow & State Management ---
export * from './workflow/xstate-manager';
export * from './workflow/job-queue';
export * from './workflow/task-scheduler';

// --- Core Application Types ---
export type {
  // Database Types
  User, Case, Evidence, Document, Citation,
  
  // API Types
  ApiResponse, ApiError, ApiMetadata,
  
  // Legal Domain Types
  LegalDocument, CaseType, EvidenceType,
  
  // AI/ML Types
  EmbeddingVector, SemanticAnalysis, RAGResponse,
  
  // System Types
  ServiceHealth, PerformanceMetrics, SystemStatus,
  
  // Communication Types
  WebSocketMessage, GRPCRequest, QuicPayload
} from './types';

// --- Utility Functions ---
export { 
  generateId, 
  validateInput, 
  sanitizeHtml,
  formatResponse,
  handleError,
  logRequest,
  measurePerformance
} from './utils';

// --- Constants & Configuration ---
export { 
  API_ENDPOINTS,
  SERVICE_PORTS,
  PROTOCOL_CONFIG,
  PERFORMANCE_THRESHOLDS,
  SECURITY_SETTINGS
} from './config';

// --- Default Exports for Common Services ---
export { default as apiClient } from './api-client.service';
export { default as evidenceService } from './evidence.service';
export { default as minioService } from './storage/minio-service';
export { default as ollamaService } from './ai/ollama-service';
export { default as redisClient } from './cache/redis-client';
export { default as healthMonitor } from './health/service-monitor';

// --- Re-exports from External Dependencies ---
export { z } from 'zod';
export { eq, and, or, desc, asc, sql } from 'drizzle-orm';
export { randomUUID } from 'crypto';