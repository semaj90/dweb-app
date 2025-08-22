
/**
 * Unified Types - Consolidates duplicate type exports across the application
 * Replaces scattered type definitions with single source of truth
 */

// ===== OLLAMA & AI TYPES =====

export interface EmbeddingResponse {
  embedding: number[];
  model: string;
  prompt_tokens?: number;
  total_duration?: number;
}

export interface GenerateResponse {
  response: string;
  model: string;
  created_at: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaModel {
  name: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    families?: string[];
    parameter_size: string;
    quantization_level: string;
  };
  modified_at: string;
}

export interface OllamaHealthCheck {
  status: 'healthy' | 'unhealthy';
  embedModel: boolean;
  llmModel: boolean;
  models: string[];
  error?: string;
}

// ===== RAG SYSTEM TYPES =====

export interface RAGQuery {
  query: string;
  caseId?: string;
  filters?: {
    documentType?: string;
    jurisdiction?: string;
    dateRange?: { start: Date; end: Date };
  };
  limit?: number;
  threshold?: number;
}

export interface RAGSource {
  id: string;
  content: string;
  title: string;
  documentType: 'contract' | 'evidence' | 'legal_brief' | 'correspondence' | 'case_law';
  embedding?: number[];
  metadata?: {
    caseId?: string;
    evidenceId?: string;
    jurisdiction?: string;
    confidence?: number;
    tags?: string[];
  };
}

export interface RAGResult {
  source: RAGSource;
  score: number;
  relevance: string;
  highlights?: string[];
}

export interface RAGSearchResponse {
  results: RAGResult[];
  query: string;
  totalResults: number;
  processingTime: number;
  model: string;
}

// ===== LEGAL DOMAIN TYPES =====

export interface LegalCase {
  id: string;
  title: string;
  description: string;
  status: 'active' | 'closed' | 'pending' | 'archived';
  priority: 'low' | 'medium' | 'high' | 'critical';
  caseType: 'criminal' | 'civil' | 'family' | 'corporate' | 'regulatory';
  jurisdiction: 'federal' | 'state' | 'local' | 'international';
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  assignedTo?: string[];
  dueDate?: Date;
  tags?: string[];
}

export interface Evidence {
  id: string;
  caseId: string;
  title: string;
  description?: string;
  type: 'document' | 'image' | 'video' | 'audio' | 'physical' | 'digital';
  filename?: string;
  fileSize?: number;
  mimeType?: string;
  extractedText?: string;
  embedding?: number[];
  aiAnalysis?: {
    summary?: string;
    entities?: string[];
    sentiment?: string;
    confidence?: number;
    tags?: string[];
  };
  createdAt: Date;
  createdBy: string;
  chainOfCustody?: ChainOfCustodyEntry[];
}

export interface ChainOfCustodyEntry {
  id: string;
  evidenceId: string;
  userId: string;
  action: 'created' | 'accessed' | 'modified' | 'transferred' | 'analyzed';
  timestamp: Date;
  location?: string;
  notes?: string;
  ipAddress?: string;
  userAgent?: string;
}

export interface LegalDocument {
  id: string;
  caseId: string;
  evidenceId?: string;
  title: string;
  content: string;
  documentType: 'contract' | 'brief' | 'motion' | 'order' | 'correspondence' | 'evidence';
  extractedText?: string;
  embedding?: number[];
  entities?: LegalEntity[];
  clauses?: LegalClause[];
  riskScore?: number;
  confidenceScore?: number;
  createdAt: Date;
  processedAt?: Date;
}

export interface LegalEntity {
  id: string;
  type: 'person' | 'organization' | 'location' | 'date' | 'monetary' | 'legal_reference';
  text: string;
  startPos: number;
  endPos: number;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface LegalClause {
  id: string;
  type: 'liability' | 'indemnification' | 'termination' | 'confidentiality' | 'payment' | 'dispute_resolution';
  text: string;
  startPos: number;
  endPos: number;
  riskLevel: 'low' | 'medium' | 'high';
  analysis?: string;
}

// ===== USER & AUTHENTICATION TYPES =====

export interface User {
  id: string;
  email: string;
  username: string;
  fullName: string;
  role: 'admin' | 'prosecutor' | 'detective' | 'legal_assistant' | 'user';
  department?: string;
  jurisdiction?: string;
  permissions: Permission[];
  preferences?: UserPreferences;
  createdAt: Date;
  lastLogin?: Date;
  isActive: boolean;
}

export interface Permission {
  resource: string;
  actions: ('read' | 'write' | 'delete' | 'admin')[];
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  notifications: {
    email: boolean;
    push: boolean;
    sms: boolean;
  };
  ui: {
    density: 'compact' | 'comfortable' | 'spacious';
    sidebarCollapsed: boolean;
  };
}

export interface AuthSession {
  id: string;
  userId: string;
  token: string;
  refreshToken?: string;
  expiresAt: Date;
  ipAddress?: string;
  userAgent?: string;
  isActive: boolean;
}

// ===== API & RESPONSE TYPES =====

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  requestId?: string;
}

export interface PaginatedResponse<T = any> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: unknown;
}

export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
  timestamp: string;
  requestId?: string;
  validationErrors?: ValidationError[];
}

// ===== SEARCH & FILTERING TYPES =====

export interface SearchOptions {
  query?: string;
  filters?: Record<string, any>;
  sort?: {
    field: string;
    direction: 'asc' | 'desc';
  };
  pagination?: {
    page: number;
    limit: number;
  };
}

export interface SearchResult<T = any> {
  item: T;
  score?: number;
  highlights?: string[];
  metadata?: Record<string, any>;
}

export interface SearchResponse<T = any> {
  results: SearchResult<T>[];
  total: number;
  query: string;
  filters?: Record<string, any>;
  processingTime: number;
  suggestions?: string[];
}

// ===== CACHING & PERFORMANCE TYPES =====

export interface CacheEntry<T = any> {
  key: string;
  value: T;
  metadata: {
    createdAt: Date;
    expiresAt?: Date;
    accessCount: number;
    lastAccessed: Date;
    size: number;
    tags?: string[];
  };
}

export interface CacheStats {
  totalEntries: number;
  totalSize: number;
  hitRate: number;
  missRate: number;
  evictionCount: number;
  averageAccessTime: number;
}

// ===== AI ANALYSIS TYPES =====

export interface AIAnalysisRequest {
  content: string;
  type: 'summary' | 'entities' | 'sentiment' | 'classification' | 'risk_assessment';
  context?: {
    caseId?: string;
    documentType?: string;
    jurisdiction?: string;
  };
  options?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
  };
}

export interface AIAnalysisResponse {
  result: string | object;
  confidence: number;
  model: string;
  processingTime: number;
  tokensUsed?: number;
  metadata?: {
    entities?: LegalEntity[];
    sentiment?: {
      score: number;
      label: 'positive' | 'negative' | 'neutral';
    };
    riskScore?: number;
    tags?: string[];
  };
}

// ===== XSTATE & WORKFLOW TYPES =====

export interface WorkflowContext {
  caseId?: string;
  userId: string;
  currentStep: string;
  progress: number;
  errors: string[];
  data: Record<string, any>;
  aiRecommendations?: string[];
  confidence?: number;
}

export interface WorkflowEvent {
  type: string;
  payload?: unknown;
  timestamp: Date;
  userId: string;
}

// ===== MCP & INTEGRATION TYPES =====

export interface MCPRequest {
  tool: string;
  parameters: Record<string, any>;
  context?: Record<string, any>;
}

export interface MCPResponse<T = any> {
  success: boolean;
  result?: T;
  error?: string;
  metadata?: {
    processingTime: number;
    model?: string;
    tokens?: number;
  };
}

export interface Context7Integration {
  stackAnalysis: boolean;
  bestPractices: boolean;
  integrationSuggestions: boolean;
  libraryDocs: boolean;
  performanceOptimization: boolean;
}

// ===== FORM & VALIDATION TYPES =====

export interface FormField {
  name: string;
  type: 'text' | 'email' | 'password' | 'number' | 'date' | 'select' | 'textarea' | 'file';
  label: string;
  placeholder?: string;
  required?: boolean;
  validation?: {
    pattern?: string;
    min?: number;
    max?: number;
    minLength?: number;
    maxLength?: number;
  };
  options?: { value: string; label: string }[];
}

export interface FormSchema {
  fields: FormField[];
  validation?: Record<string, any>;
}

// ===== COMPONENT PROPS TYPES =====

export interface BaseComponentProps {
  class?: string;
  id?: string;
  'data-testid'?: string;
}

export interface DialogProps extends BaseComponentProps {
  open?: boolean;
  title?: string;
  description?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'default' | 'destructive' | 'success';
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
  loading?: boolean;
  type?: 'button' | 'submit' | 'reset';
}

// ===== TYPE GUARDS =====

export function isEmbeddingResponse(obj: any): obj is EmbeddingResponse {
  return obj && typeof obj === 'object' && Array.isArray(obj.embedding);
}

export function isRAGResult(obj: any): obj is RAGResult {
  return obj && typeof obj === 'object' && obj.source && typeof obj.score === 'number';
}

export function isApiError(obj: any): obj is ApiError {
  return obj && typeof obj === 'object' && typeof obj.code === 'string' && typeof obj.message === 'string';
}

export function isLegalCase(obj: any): obj is LegalCase {
  return obj && typeof obj === 'object' && typeof obj.id === 'string' && typeof obj.title === 'string';
}

export function isUser(obj: any): obj is User {
  return obj && typeof obj === 'object' && typeof obj.id === 'string' && typeof obj.email === 'string';
}