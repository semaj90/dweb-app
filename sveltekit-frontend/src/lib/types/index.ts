// ---
// SINGLE SOURCE OF TRUTH FOR TYPES
//
// This file establishes a centralized repository for all shared interfaces and types
// used across the SvelteKit application, addressing the root cause of hundreds of
// TypeScript errors related to type mismatches and missing properties.
// ---

// --- Core Service & API Types ---

export type ServiceStatus = 'operational' | 'degraded' | 'offline' | 'unknown';

export interface APIResponse<T> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
    };
    metadata?: {
        timestamp: string;
        processingTimeMs: number;
    };
}

// --- AI & Worker Manager Types (Fixes ~50+ errors) ---

export type AITaskType = "generate" | "analyze" | "embed" | "search" | "embedding" | "analysis" | "classification" | "summarization";
export type WorkerMessageType = "error" | "status" | "result" | "task" | "TASK_STARTED" | "TASK_COMPLETED" | "TASK_ERROR" | "TASK_CANCELLED" | "STATUS_UPDATE";

export interface AITask {
  taskId: string;
  type: AITaskType;
  providerId: string; // e.g., 'ollama', 'openai'
  model: string;
  prompt: string;
  timestamp: number;
  priority: 'low' | 'medium' | 'high';
  // Additional properties for specific tasks can be added
  [key: string]: any;
}

export interface WorkerStatus {
  status: 'idle' | 'processing' | 'error';
  activeRequests: number;
  queueLength: number;
  providers: { id: string, status: ServiceStatus }[];
  maxConcurrent: number;
  uptime: number;
  totalProcessed: number;
  errors: number;
  performance: {
    avgTaskTime: number;
    tasksPerMinute: number;
  };
  lastActivity: Date;
}

export interface WorkerMessage {
  taskId?: string;
  type: WorkerMessageType;
  data?: unknown;
  payload?: AITask | APIResponse<unknown> | WorkerStatus;
}

// --- Copilot & RAG Types (Fixes ~20+ errors) ---

export type CopilotSource = "context7_mcp" | "enhanced_local_index" | "basic_index";

export interface CopilotIndexEntry {
  // Base properties
  id: string;
  content: string;
  score: number;

  // Unified metadata properties
  source: CopilotSource;
  type: 'case' | 'document' | 'evidence' | 'statute';
  jurisdiction: string;
  practiceArea: string[];
  confidentialityLevel: number;
  lastModified: Date;
  fileSize: number;
  language: string;
  tags: string[];
}

export interface SimilarityResult {
    id: string;
    documentId: string;
    documentType: string;
    chunkIndex: number;
    content: string;
    score: number;
}

// --- Database & Document Types ---
export interface LegalDocument {
    id: string;
    caseId: string;
    title: string;
    content: string;
    createdAt: string;
    updatedAt: string;
    metadata: Record<string, any>;
    embedding?: number[];
}

// --- NES-Style Canvas Engine & AI Prediction Types ---

export interface CanvasState {
  id: string;
  animation: string;
  frame: number;
  fabricJSON: object; // Represents the serialized fabric.js canvas state
  metadata: {
    duration?: number;
    transitions?: string[];
    userContext?: string[];
    confidence?: number;
  };
}

export interface CanvasAnimation {
    name: string;
    frames: CanvasState[];
    loop: boolean;
}

export interface UserActivityLog {
    timestamp: number;
    action: string; // e.g., 'canvas_click', 'animation_changed'
    context: Record<string, any>;
    sessionId: string;
}

// --- XState & Machine Types ---
export interface GlobalAppContext {
    user: { id: string; email: string; } | null;
    activeCaseId: string | null;
    theme: 'light' | 'dark';
}

export type GlobalAppEvent =
  | { type: 'LOGIN'; user: { id: string; email: string } }
  | { type: 'LOGOUT' }
  | { type: 'SET_CASE'; caseId: string }
  | { type: 'SET_THEME'; theme: 'light' | 'dark' };

// --- Legal Case & Evidence Types ---

export interface LegalCase {
  id: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'in_progress' | 'closed' | 'archived';
  createdAt: string;
  updatedAt: string;
  userId: string;
  metadata: Record<string, any>;
}

export interface Evidence {
  id: string;
  caseId: string;
  title: string;
  description: string;
  type: 'document' | 'image' | 'video' | 'audio' | 'other';
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  uploadedAt: string;
  metadata: Record<string, any>;
  embedding?: number[];
}

// --- Vector & Search Types ---

export interface VectorSearchRequest {
  query: string;
  filters?: Record<string, any>;
  limit?: number;
  threshold?: number;
}

export interface VectorSearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
}

// --- Form & Validation Types ---

export interface FormValidationError {
  field: string;
  message: string;
  code?: string;
}

export interface FormState<T> {
  data: T;
  errors: FormValidationError[];
  isValid: boolean;
  isSubmitting: boolean;
}

// --- Notification & Alert Types ---

export type NotificationType = 'info' | 'success' | 'warning' | 'error';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: number;
  autoClose?: boolean;
  duration?: number;
}

// --- UI Component Props Types ---

export interface ButtonProps {
  variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  href?: string;
  type?: 'button' | 'submit' | 'reset';
  onclick?: (event: MouseEvent) => void;
  class?: string;
}

export interface ModalProps {
  show: boolean;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closable?: boolean;
  onClose?: () => void;
  class?: string;
}

// --- Authentication Types ---

export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  role: 'user' | 'admin' | 'attorney' | 'paralegal';
  permissions: string[];
  createdAt: string;
  lastLoginAt?: string;
}

export interface AuthSession {
  user: User;
  token: string;
  expiresAt: string;
  refreshToken?: string;
}

// --- File Upload Types ---

export interface FileUploadRequest {
  file: File;
  caseId?: string;
  description?: string;
  metadata?: Record<string, any>;
}

export interface FileUploadResponse {
  id: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  url: string;
  uploadedAt: string;
}

// --- WebSocket & Real-time Types ---

export type WebSocketMessageType = 'ping' | 'pong' | 'subscribe' | 'unsubscribe' | 'notification' | 'update';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  payload?: any;
  timestamp: number;
  requestId?: string;
}

// --- API Types Re-export ---
// Export all centralized API types for easy access

export type {
  // Core API Response Types
  ApiResponse,
  ApiError,
  HealthStatus,
  ServiceHealth,
  SystemInfo,

  // AI & Chat Types
  ChatRequest,
  ChatResponse,
  AIAnalysisRequest,
  AIAnalysisResponse,

  // Vector Search & RAG Types
  VectorSearchRequest as APIVectorSearchRequest,
  VectorSearchResponse,
  RAGRequest,
  RAGResponse,

  // Case Management Types
  Case,
  CreateCaseRequest,
  UpdateCaseRequest,
  CasesListResponse,

  // Document Types
  Document,
  DocumentUploadRequest,
  DocumentUploadResponse,
  DocumentSearchRequest,
  DocumentSearchResponse,

  // Evidence Types
  Evidence as APIEvidence,
  CreateEvidenceRequest,

  // User & Auth Types
  User as APIUser,
  AuthResponse,

  // Analytics & Metrics Types
  DashboardStats,
  AnalyticsRequest,
  AnalyticsResponse,

  // Cluster & Admin Types
  ClusterStatus,
  ScaleRequest,
  ServiceEvent,

  // Search & Query Types
  SearchRequest as APISearchRequest,
  SearchResponse as APISearchResponse,

  // Pagination Types
  PaginationParams,
  PaginatedResponse,

  // Error Handling Types
  ValidationError,
  ApiValidationError,

  // Request Context Types
  RequestContext,

  // Type Guards
  isApiError,
  isApiResponse,
  isPaginatedResponse,

  // Enhanced Production Types
  APIResponse as EnhancedAPIResponse,
  ServiceTier,
  ProtocolEndpoint,
  DatabaseEndpoint,
  MessagingEndpoint,
  FrontendEndpoint,
  ServiceEndpoints,
  HealthCheckResult,
  ClusterMetrics,
  PerformanceMetrics,
  EnhancedRAGRequest,
  EnhancedRAGResponse,
  EnhancedUploadRequest,
  EnhancedUploadResponse,
  DimensionalCacheRequest,
  DimensionalCacheResponse,
  XStateRequest,
  XStateResponse,
  ModuleRequest,
  ModuleResponse,
  RecommendationRequest,
  RecommendationResponse,
  SystemHealthResponse,
  ServiceDiscoveryResponse,
  NATSMessageRequest,
  NATSMessageResponse,
  NATSSubscriptionRequest,
  NATSSubscriptionResponse,
  APIRequestContext,
  APIErrorResponse,
  ProtocolRouter,
  EnhancedAPIHandler,
  MultiProtocolRequestOptions
} from './api.js';
