import type { User } from './user';


// Evidence AI Analysis Type
export interface EvidenceAIAnalysis {
  // Core analysis metrics (required by the user)
  validationScore?: number; // 0-100 scale
  riskLevel?: "low" | "medium" | "high" | "critical";
  complexityLevel?: "simple" | "moderate" | "complex" | "highly_complex";
  
  // Analysis results
  summary?: string;
  relevanceScore?: number;
  keyFindings?: string[];
  legalImplications?: string[];
  recommendations?: string[];
  risks?: string[];
  tags?: string[];
  confidence?: number; // 0-1 scale
  
  // Processing metadata
  analysisMetrics?: AnalysisMetrics;
  processingTime?: number;
  model?: string;
  analyzedAt?: string; // ISO date string
  version?: number;
  
  // Allow additional properties for backward compatibility
  [key: string]: unknown;
}

export interface AnalysisMetrics {
  contentLength?: number;
  processingSteps?: number;
  confidenceDistribution?: Record<string, number>;
  qualityScore?: number; // 0-100 scale
  completenessScore?: number; // 0-100 scale
  accuracyIndicators?: string[];
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}
// AI Chat API Types
export interface ChatMessage {
  id: string;
  content: string;
  role: "user" | "assistant" | "system";
  timestamp: Date;
  sources?: VectorSearchResult[];
  metadata?: {
    model?: string;
    temperature?: number;
    tokensUsed?: number;
    references?: string[];
    emotionalTone?: string;
    proactive?: boolean;
    reactions?: Record<string, boolean>;
    provider?: string;
    confidence?: number;
    executionTime?: number;
    fromCache?: boolean;
  };
}
export interface AIResponse {
  answer: string;
  sources?: VectorSearchResult[];
  provider?: string;
  model?: string;
  confidence?: number;
  executionTime?: number;
  fromCache?: boolean;
  metadata?: Record<string, any>;
}
export interface ConversationHistory {
  id: string;
  title: string;
  messages: ChatMessage[];
  timestamp: number;
  role?: "user" | "assistant" | "system";
  content?: string;
  metadata?: Record<string, any>;
}
export interface ChatRequest {
  messages: ChatMessage[];
  context?: {
    caseId?: string;
    currentPage?: string;
    userId?: string;
  };
  proactiveMode?: boolean;
  settings?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
  };
}
export interface ChatResponse {
  content: string;
  role: "assistant";
  metadata?: {
    model?: string;
    temperature?: number;
    tokensUsed?: number;
    references?: string[];
    emotionalTone?: string;
    proactive?: boolean;
    processingTime?: number;
  };
}
// Evidence API Types
export interface EvidenceUploadRequest {
  caseId: string;
  title: string;
  description?: string;
  type: "document" | "image" | "video" | "audio" | "other";
  url?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}
export interface EvidenceUploadResponse {
  id: string;
  uploadUrl?: string;
  processingStatus?: "pending" | "processing" | "completed" | "failed";
}
// Evidence Types
export interface Evidence {
  id: string;
  caseId: string | null;
  criminalId: string | null;
  title: string;
  description: string | null;
  evidenceType: string;
  fileType: string | null;
  subType: string | null;
  fileUrl: string | null;
  fileName: string | null;
  fileSize: number | null;
  mimeType: string | null;
  hash: string | null;
  tags: string[];
  chainOfCustody: any[];
  collectedAt: Date | null;
  collectedBy: string | null;
  location: string | null;
  labAnalysis: Record<string, any>;
  aiAnalysis: EvidenceAIAnalysis;
  aiTags: string[];
  aiSummary: string | null;
  summary: string | null;
  isAdmissible: boolean;
  confidentialityLevel: string;
  canvasPosition: Record<string, any>;
  uploadedBy: string | null;
  uploadedAt: Date;
  updatedAt: Date;
  // Additional fields commonly used in components
  type?: string;
}
// Simplified Evidence type for UI components
export interface EvidenceItem {
  id: string;
  title: string;
  description: string | null;
  evidenceType: string;
  fileType: string | null;
  aiAnalysis?: EvidenceAIAnalysis;
  summary?: string | null;
  canvasPosition?: Record<string, any>;
  // Additional fields for compatibility
  type?: string;
  collectedAt?: Date | null;
}
// Search API Types
export interface SearchRequest {
  query: string;
  type?: "cases" | "evidence" | "statutes" | "all";
  filters?: {
    caseId?: string;
    dateRange?: {
      start: Date;
      end: Date;
    };
    tags?: string[];
    status?: string[];
  };
  pagination?: {
    page: number;
    limit: number;
  };
}
export interface SearchResponse<T = any> {
  results: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}
// User API Types
export interface UserProfile {
  id: string;
  email: string;
  name?: string;
  firstName?: string;
  lastName?: string;
  role?: string;
  avatar?: string;
  preferences?: {
    theme?: "light" | "dark" | "system";
    notifications?: boolean;
    language?: string;
  };
}
export interface UserUpdateRequest {
  name?: string;
  firstName?: string;
  lastName?: string;
  preferences?: UserProfile["preferences"];
}
// File Upload Types
export interface FileUploadRequest {
  file: File;
  caseId?: string;
  type?: string;
  metadata?: Record<string, any>;
}
export interface FileUploadResponse {
  id: string;
  url: string;
  filename: string;
  size: number;
  mimeType: string;
  uploadedAt: Date;
}
// Vector Search Types
export interface VectorSearchRequest {
  query: string;
  type?: "semantic" | "similarity" | "hybrid";
  limit?: number;
  threshold?: number;
  filters?: Record<string, any>;
}
export interface VectorSearchResult {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, any>;
  type: "case" | "evidence" | "statute" | "document";
}
// Embedding Types
export interface EmbeddingRequest {
  text: string;
  model?: string;
}
export interface EmbeddingResponse {
  embedding: number[];
  model: string;
  usage?: {
    promptTokens?: number;
    totalTokens?: number;
  };
}
// Citation Types
export interface Citation {
  id: string;
  title: string;
  content: string;
  source: string;
  category: string;
  tags: string[];
  dateAdded?: Date;
  createdAt: Date;
  updatedAt: Date;
  isBookmarked?: boolean;
  isFavorite?: boolean;
  notes?: string;
  relevanceScore?: number;
  contextData?: {
    caseId?: string;
    evidenceId?: string;
    userId?: string;
    [key: string]: unknown;
  };
  metadata?: {
    author?: string;
    year?: number;
    court?: string;
    jurisdiction?: string;
    caseNumber?: string;
    url?: string;
  };
  // Additional fields commonly used in components
  savedAt?: Date;
}
// Error Response Types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: Date;
}
// Generic API Handler Types
export type ApiHandler<TRequest = any, TResponse = any> = (
  request: TRequest,
) => Promise<ApiResponse<TResponse>>;

export type ApiErrorHandler = (error: Error) => ApiResponse<never>;

// Case Management Types
export interface Case {
  id: string;
  caseNumber: string;
  title: string;
  name?: string;
  description?: string;
  incidentDate?: Date;
  location?: string;
  priority: "low" | "medium" | "high" | "critical";
  status: "open" | "investigating" | "pending" | "closed" | "archived";
  category?: string;
  dangerScore?: number;
  estimatedValue?: number;
  jurisdiction?: string;
  leadProsecutor?: string;
  assignedTeam?: string[];
  tags?: string[];
  aiSummary?: string;
  aiTags?: string[];
  aiAnalysis?: CaseAIAnalysis;
  metadata?: Record<string, any>;
  createdBy?: string;
  createdAt: Date;
  updatedAt: Date;
  closedAt?: Date;
  // Additional fields commonly used in components
  openedAt?: Date;
  defendantName?: string;
  courtDate?: Date;
  evidenceCount?: number;
}

// AI Analysis Types
export interface EvidenceAIAnalysis {
  // Core analysis metrics
  validationScore?: number; // 0-100 scale
  riskLevel?: "low" | "medium" | "high" | "critical";
  complexityLevel?: "simple" | "moderate" | "complex" | "highly_complex";
  
  // Analysis results
  keyFindings?: string[];
  legalImplications?: string[];
  recommendations?: string[];
  risks?: string[];
  confidence?: number; // 0-1 scale
  
  // Processing metadata
  analysisMetrics?: AnalysisMetrics;
  model?: string;
  processingTime?: number; // milliseconds
  analyzedAt?: string; // ISO date string
  version?: number;
  
  // Content analysis
  entities?: NamedEntity[];
  topics?: Topic[];
  sentiment?: SentimentAnalysis;
  
  // Legal-specific analysis
  legalPrecedents?: LegalPrecedent[];
  statuteReferences?: StatuteReference[];
  caseConnections?: CaseConnection[];
}

export interface CaseAIAnalysis {
  // Core analysis metrics
  validationScore?: number; // 0-100 scale
  riskLevel?: "low" | "medium" | "high" | "critical";
  complexityLevel?: "simple" | "moderate" | "complex" | "highly_complex";
  
  // Case-specific analysis
  strengthAssessment?: StrengthAssessment;
  timelineAnalysis?: TimelineAnalysis;
  evidenceGaps?: EvidenceGap[];
  prosecutionStrategy?: ProsecutionStrategy;
  
  // Analysis results
  keyFindings?: string[];
  legalImplications?: string[];
  recommendations?: string[];
  risks?: string[];
  confidence?: number; // 0-1 scale
  
  // Processing metadata
  analysisMetrics?: AnalysisMetrics;
  model?: string;
  processingTime?: number; // milliseconds
  analyzedAt?: string; // ISO date string
  version?: number;
}

export interface AnalysisMetrics {
  contentLength?: number;
  processingSteps?: number;
  confidenceDistribution?: Record<string, number>;
  qualityScore?: number; // 0-100 scale
  completenessScore?: number; // 0-100 scale
  accuracyIndicators?: string[];
}

export interface NamedEntity {
  text: string;
  type: "person" | "organization" | "location" | "date" | "statute" | "case" | "other";
  confidence: number;
  startIndex?: number;
  endIndex?: number;
  metadata?: Record<string, any>;
}

export interface Topic {
  name: string;
  relevance: number; // 0-1 scale
  keywords: string[];
  description?: string;
}

export interface SentimentAnalysis {
  overall: "positive" | "negative" | "neutral";
  confidence: number; // 0-1 scale
  emotions?: Record<string, number>;
}

export interface LegalPrecedent {
  caseTitle: string;
  citation: string;
  relevance: number; // 0-1 scale
  jurisdiction?: string;
  year?: number;
  keyConcepts: string[];
  applicability?: "directly_applicable" | "analogous" | "distinguishable";
}

export interface StatuteReference {
  title: string;
  code: string;
  section?: string;
  relevance: number; // 0-1 scale
  jurisdiction?: string;
  applicability?: "directly_applicable" | "related" | "background";
}

export interface CaseConnection {
  caseId: string;
  connectionType: "similar" | "related" | "precedent" | "conflicting";
  strength: number; // 0-1 scale
  description?: string;
  sharedElements: string[];
}

export interface StrengthAssessment {
  overall: "strong" | "moderate" | "weak";
  evidenceQuality: number; // 0-100 scale
  legalFoundation: number; // 0-100 scale
  prosecutabilityScore: number; // 0-100 scale
  challenges: string[];
  strengths: string[];
}

export interface TimelineAnalysis {
  eventCount: number;
  timespan: {
    start: string; // ISO date
    end: string; // ISO date
  };
  keyEvents: TimelineEvent[];
  gaps: TimelineGap[];
  consistency: number; // 0-100 scale
}

export interface TimelineEvent {
  date: string; // ISO date
  description: string;
  importance: "critical" | "important" | "minor";
  evidenceIds: string[];
  confidence: number; // 0-1 scale
}

export interface TimelineGap {
  startDate: string; // ISO date
  endDate: string; // ISO date
  description: string;
  importance: "critical" | "important" | "minor";
  investigationPriority: number; // 0-100 scale
}

export interface EvidenceGap {
  type: "witness" | "document" | "physical" | "digital" | "expert" | "other";
  description: string;
  importance: "critical" | "important" | "minor";
  suggestedActions: string[];
  deadline?: string; // ISO date
}

export interface ProsecutionStrategy {
  approach: "aggressive" | "standard" | "cautious";
  mainArguments: string[];
  supportingEvidence: string[];
  anticipatedDefenses: string[];
  recommendedActions: StrategicAction[];
  timeline: StrategicTimeline;
}

export interface StrategicAction {
  action: string;
  priority: "high" | "medium" | "low";
  deadline?: string; // ISO date
  assignedTo?: string;
  dependencies?: string[];
  estimatedEffort?: "low" | "medium" | "high";
}

export interface StrategicTimeline {
  phases: StrategicPhase[];
  milestones: Milestone[];
  criticalPath: string[];
}

export interface StrategicPhase {
  name: string;
  description: string;
  duration: number; // days
  dependencies?: string[];
  deliverables: string[];
}

export interface Milestone {
  name: string;
  date: string; // ISO date
  description: string;
  importance: "critical" | "important" | "minor";
}

// ============================================================================
// PRODUCTION API SYSTEM - SVELTEKIT 2 UNIFIED INTEGRATION
// ============================================================================

// Core API Response Interface (Enhanced)
export interface APIResponse<T = any> extends ApiResponse<T> {
  metadata?: Record<string, any>;
  requestId?: string;
  processingTime?: number;
  timestamp?: string;
}

// Service Tier Enumeration for Protocol Selection
export enum ServiceTier {
  ULTRA_FAST = 'ULTRA_FAST',    // < 5ms (QUIC)
  HIGH_PERF = 'HIGH_PERF',      // < 15ms (gRPC)  
  STANDARD = 'STANDARD',        // < 50ms (HTTP)
  REALTIME = 'REALTIME'         // WebSocket events
}

// Multi-Protocol Service Configuration
export interface ProtocolEndpoint {
  http?: string;
  grpc?: string;
  quic?: string;
  websocket?: string;
  primary?: string;     // For multi-instance services (Ollama)
  secondary?: string;   // Fallback instances
  embeddings?: string;  // Specialized endpoints
  health?: string;      // Health check path
  tier?: ServiceTier;
  status: 'active' | 'experimental' | 'deprecated' | 'maintenance';
}

// Database Service Configuration  
export interface DatabaseEndpoint {
  host: string;
  port: number;
  database?: string;
  status: 'active' | 'error' | 'maintenance';
}

// Messaging Service Configuration (NATS, etc.)
export interface MessagingEndpoint {
  server?: string;
  websocket?: string;
  monitor?: string;
  health?: string;
  status: 'active' | 'error' | 'maintenance';
}

// Frontend Service Configuration
export interface FrontendEndpoint {
  http?: string;
  dev?: string;
  status: 'active' | 'maintenance';
}

// Complete Service Endpoints Map (37 Go Services + Infrastructure)
export interface ServiceEndpoints {
  // Core AI Services (Tier 1) - Always Running
  enhancedRAG: ProtocolEndpoint;
  uploadService: ProtocolEndpoint;
  documentProcessor: ProtocolEndpoint;
  grpcServer: ProtocolEndpoint;
  
  // AI Enhancement Services (Tier 2) - Advanced Features  
  advancedCUDA: ProtocolEndpoint;
  dimensionalCache: ProtocolEndpoint;
  xstateManager: ProtocolEndpoint;
  moduleManager: ProtocolEndpoint;
  recommendationEngine: ProtocolEndpoint;
  
  // Specialized AI Services
  enhancedSemanticArchitecture: ProtocolEndpoint;
  enhancedLegalAI: ProtocolEndpoint;
  enhancedMulticore: ProtocolEndpoint;
  liveAgentEnhanced: ProtocolEndpoint;
  
  // File & Document Services
  ginUpload: ProtocolEndpoint;
  summarizerService: ProtocolEndpoint;
  aiSummary: ProtocolEndpoint;
  
  // Multi-Core Ollama Cluster
  ollama: ProtocolEndpoint;
  
  // Database Services
  postgresql: DatabaseEndpoint;
  redis: DatabaseEndpoint; 
  qdrant: ProtocolEndpoint;
  neo4j?: ProtocolEndpoint;
  
  // Messaging & Communication
  nats: MessagingEndpoint;
  
  // Infrastructure & Monitoring Services
  clusterManager: ProtocolEndpoint;
  loadBalancer: ProtocolEndpoint;
  gpuIndexerService: ProtocolEndpoint;
  contextErrorPipeline: ProtocolEndpoint;
  simdHealth: ProtocolEndpoint;
  
  // Development & Testing
  simpleServer: ProtocolEndpoint;
  testServer: ProtocolEndpoint;
  
  // Frontend
  sveltekit: FrontendEndpoint;
}

// Enhanced Health Check Interface
export interface HealthCheckResult {
  status: 'healthy' | 'unhealthy' | 'error' | 'timeout';
  responseTime?: number;
  endpoint?: string;
  error?: string;
  lastCheck: string;
  metadata?: {
    version?: string;
    uptime?: number;
    connections?: number;
    memoryUsage?: string;
    cpuUsage?: string;
  };
}

// Cluster Metrics from Windows Native Process Monitoring
export interface ClusterMetrics {
  spawned: Record<string, number>;
  deferredActive: number;
  deferredTotal: number;
  lastAllocation?: {
    type: string;
    port: number;
    timestamp: string;
  };
  events: Array<{
    type: string;
    message: string;
    timestamp: string;
    metadata?: Record<string, any>;
  }>;
  workers: Array<{
    type: string;
    pid?: number;
    port: number;
    uptimeSec: number;
    status: 'running' | 'starting' | 'error' | 'stopped';
  }>;
  deferredQueue: Array<{
    type: string;
    attempts: number;
    lastAttempt: string;
    reason?: string;
  }>;
}

// Performance Metrics Interface
export interface PerformanceMetrics {
  protocols: {
    QUIC: string;
    gRPC: string; 
    HTTP: string;
    WebSocket: string;
  };
  resources: {
    cpu: string;
    memory: string;
    gpu?: string;
    storage: string;
  };
  performance: {
    averageResponseTime: string;
    uptime: string;
    throughput: string;
  };
  timestamp: string;
}

// Enhanced RAG Request/Response for Production
export interface EnhancedRAGRequest extends VectorSearchRequest {
  context?: string;
  useCache?: boolean;
  userId?: string;
  sessionId?: string;
  caseId?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface EnhancedRAGResponse extends APIResponse {
  results: Array<{
    content: string;
    score: number;
    metadata: Record<string, any>;
    source?: string;
    chunkIndex?: number;
  }>;
  answer?: string;
  totalResults: number;
  processingTime: number;
  model: string;
  cached: boolean;
  confidence?: number;
}

// Document Upload with Enhanced Processing
export interface EnhancedUploadRequest extends FileUploadRequest {
  extractText?: boolean;
  performOCR?: boolean;
  generateEmbeddings?: boolean;
  analyzeContent?: boolean;
  userId?: string;
  sessionId?: string;
}

export interface EnhancedUploadResponse extends APIResponse {
  documentId: string;
  filename: string;
  size: number;
  contentType: string;
  uploadTime: string;
  processingStatus: 'pending' | 'processing' | 'completed' | 'failed';
  extractedText?: string;
  ocrResults?: {
    text: string;
    confidence: number;
    language: string;
  };
  embeddings?: {
    model: string;
    dimensions: number;
    generated: boolean;
  };
  analysis?: EvidenceAIAnalysis;
  metadata: Record<string, any>;
}

// Dimensional Caching for Advanced Features
export interface DimensionalCacheRequest {
  key: string;
  embeddings?: number[][];
  attention?: number[][];
  metadata?: Record<string, any>;
  ttl?: number; // Time to live in seconds
  userId?: string;
}

export interface DimensionalCacheResponse extends APIResponse {
  key: string;
  hit: boolean;
  embeddings?: number[][];
  attention?: number[][];
  metadata?: Record<string, any>;
  cacheStats?: {
    hitRate: number;
    size: number;
    capacity: number;
    evictions: number;
  };
}

// XState Idle Detection & Queue Management
export interface XStateRequest {
  action: 'transition' | 'queue' | 'status' | 'health';
  state?: 'idle' | 'active' | 'computing' | 'offline' | 'error';
  jobData?: {
    id: string;
    type: 'computation' | 'analysis' | 'processing' | 'rag' | 'upload';
    priority: 'low' | 'medium' | 'high' | 'critical';
    payload: Record<string, any>;
    userId?: string;
  };
  userId?: string;
}

export interface XStateResponse extends APIResponse {
  currentState: string;
  queueStatus?: {
    pending: number;
    processing: number;
    completed: number;
    failed: number;
    total: number;
  };
  idleTime?: number;
  lastActivity?: string;
  machineConfig?: {
    states: string[];
    transitions: Record<string, string[]>;
  };
}

// Modular Hot-Swappable Experience System
export interface ModuleRequest {
  action: 'load' | 'unload' | 'switch' | 'list' | 'health';
  moduleId?: string;
  userId?: string;
  moduleConfig?: Record<string, any>;
  preserveSession?: boolean;
}

export interface ModuleResponse extends APIResponse {
  modules?: Array<{
    id: string;
    name: string;
    version: string;
    status: 'loaded' | 'unloaded' | 'loading' | 'error';
    capabilities: string[];
    metadata?: Record<string, any>;
  }>;
  activeModule?: string;
  switchTime?: number;
  memoryUsage?: string;
}

// Self-Prompting & AI-Powered Recommendations
export interface RecommendationRequest {
  userId: string;
  context?: string;
  type: 'resume' | 'suggest' | 'trending' | 'related' | 'corrections';
  query?: string;
  limit?: number;
  caseId?: string;
  sessionId?: string;
}

export interface RecommendationResponse extends APIResponse {
  recommendations?: Array<{
    text: string;
    type: 'suggestion' | 'correction' | 'continuation' | 'related';
    confidence: number;
    metadata?: Record<string, any>;
  }>;
  context?: string;
  lastActivity?: string;
  corrected?: string;
  relatedSearches?: string[];
  userPattern?: {
    mostUsedFeatures: string[];
    preferredSearchTerms: string[];
    averageSessionTime: number;
  };
}

// System Health & Comprehensive Monitoring
export interface SystemHealthResponse extends APIResponse {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  healthScore: number;
  services: Record<string, HealthCheckResult>;
  summary: {
    total: number;
    healthy: number;
    unhealthy: number;
    experimental: number;
  };
  deployment: string;
  infrastructure: {
    platform: 'Windows Native';
    docker: false;
    gpu: string;
    memory: string;
    storage: string;
  };
}

// Service Discovery with Protocol Information
export interface ServiceDiscoveryResponse extends APIResponse {
  services: Array<{
    name: string;
    config: ProtocolEndpoint | DatabaseEndpoint | MessagingEndpoint | FrontendEndpoint;
    protocols: string[];
    tier: ServiceTier | 'DATABASE' | 'MESSAGING' | 'FRONTEND';
    port?: number;
    health?: string;
  }>;
  total: number;
  active: number;
  experimental: number;
  protocolSupport: {
    HTTP: number;
    gRPC: number;
    QUIC: number;  
    WebSocket: number;
  };
  deployment: {
    type: 'Windows Native';
    docker: false;
    processes: number;
  };
}

// NATS Messaging Integration Types
export interface NATSMessageRequest {
  subject: string;
  data: any;
  headers?: Record<string, string>;
  timeout?: number;
  correlationId?: string;
  userId?: string;
}

export interface NATSMessageResponse extends APIResponse {
  messageId: string;
  subject: string;
  published: boolean;
  timestamp: string;
  correlationId?: string;
}

export interface NATSSubscriptionRequest {
  subjects: string[];
  queueGroup?: string;
  userId?: string;
  deliverPolicy?: 'all' | 'last' | 'new';
}

export interface NATSSubscriptionResponse extends APIResponse {
  subscriptions: Array<{
    subject: string;
    queueGroup?: string;
    active: boolean;
    messageCount: number;
  }>;
  connectionStatus: 'connected' | 'disconnected' | 'reconnecting';
}

// Request Context for SvelteKit Integration
export interface APIRequestContext {
  userId?: string;
  sessionId?: string;
  userAgent?: string;
  clientIP?: string;
  requestId?: string;
  correlationId?: string;
  startTime: number;
  caseId?: string;
  permissions?: string[];
}

// Enhanced Error Response
export interface APIErrorResponse extends APIResponse {
  error: string;
  code?: string;
  details?: Record<string, any>;
  requestId?: string;
  timestamp: string;
  retryable?: boolean;
  suggestedActions?: string[];
}

// Protocol Router for Multi-Protocol Service Access
export interface ProtocolRouter {
  route<T extends keyof ServiceEndpoints>(
    service: T,
    endpoint: string,
    options?: RequestInit & { protocol?: 'auto' | 'http' | 'grpc' | 'quic' | 'websocket' }
  ): Promise<Response>;
  
  healthCheck(service: keyof ServiceEndpoints): Promise<HealthCheckResult>;
  
  getOptimalProtocol(service: keyof ServiceEndpoints): 'http' | 'grpc' | 'quic' | 'websocket';
  
  getServiceConfig<T extends keyof ServiceEndpoints>(service: T): ServiceEndpoints[T];
  
  getAllServices(): Array<{
    name: keyof ServiceEndpoints;
    config: ServiceEndpoints[keyof ServiceEndpoints];
    protocols: string[];
  }>;
}

// Utility Type for API Route Handlers with Enhanced Context
export type EnhancedAPIHandler<TRequest = any, TResponse = APIResponse> = (
  request: TRequest,
  context: APIRequestContext
) => Promise<TResponse>;

// Multi-Protocol Request Options
export interface MultiProtocolRequestOptions extends Omit<RequestInit, 'cache' | 'priority'> {
  protocol?: 'auto' | 'http' | 'grpc' | 'quic' | 'websocket';
  timeout?: number;
  retries?: number;
  fallback?: boolean;
  cache?: boolean;
  priority?: 'low' | 'medium' | 'high' | 'critical';
}
