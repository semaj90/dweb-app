// Core types for the Legal AI application

export interface Evidence {
  id: string;
  title: string;
  description?: string;
  content?: string;
  type: string;
  evidenceType?: string;
  fileUrl?: string;
  fileName?: string;
  fileSize?: number;
  caseId?: string;
  uploadedAt?: Date;
  createdAt?: Date;
  posX?: number;
  posY?: number;
  width?: number;
  height?: number;
  metadata?: any;
  evidenceId?: string; // Additional property for API compatibility
  embedding?: number[]; // Vector embedding data
  analysis?: {
    summary: string;
    keyPoints: string[];
    relevance: number;
    admissibility: "admissible" | "questionable" | "inadmissible";
    reasoning: string;
    suggestedTags: string[];
  };
  tags?: string[];
  similarEvidence?: { id: string; title: string; similarity: number }[];
  thumbnailUrl?: string;
  aiSummary?: string;
  hash?: string;
  collectedBy?: string;
}

export interface Report {
  id: string;
  title: string;
  content: string;
  reportType?: string;
  caseId?: string;
  createdAt?: Date;
  updatedAt?: Date;
  summary?: string;
  wordCount?: number;
  estimatedReadTime?: number;
  tags?: string[];
  status?: "draft" | "published" | "archived";
}

export interface CanvasState {
  id: string;
  caseId: string;
  canvasData: string;
  thumbnailUrl?: string;
  version?: number;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface Case {
  id: string;
  title: string;
  description?: string;
  status: "closed" | "open" | "pending" | "archived" | "investigating";
  priority?: string;
  createdAt?: Date;
  updatedAt?: Date;
  openedAt?: Date;
  caseNumber?: string;
  defendantName?: string;
  evidenceCount?: number;
  courtDate?: Date;
  summary?: {
    riskAssessment: {
      level: "LOW" | "MEDIUM" | "HIGH";
      score: number;
    };
  };
}

export interface User {
  id: string;
  email: string;
  name?: string;
  firstName?: string;
  lastName?: string;
  role?: string;
  isActive?: boolean;
  avatarUrl?: string;
  permissions?: string[];
  preferences?: any;
}

// UI Component Types
export type ButtonVariant =
  | "default"
  | "outline"
  | "primary"
  | "secondary"
  | "danger"
  | "ghost"
  | "crimson"
  | "nier"
  | "gold"
  | "destructive"
  | "success"
  | "warning"
  | "info"
  | "link";

export type ButtonSize = "xs" | "sm" | "md" | "lg" | "xl";

// AI and Analysis Types
export interface AIMessage {
  id: string;
  role: "user" | "system" | "assistant";
  content: string;
  timestamp: number;
  sources?: {
    id: string;
    title: string;
    content: string;
    score: number;
    type: string;
  }[];
  metadata?: {
    provider: "hybrid" | "local" | "cloud";
    model: string;
    confidence: number;
    executionTime: number;
    fromCache: boolean;
    gpu?: string;
  };
}

export interface AnalysisResults {
  classification: any;
  keyEntities: { text: string; type: string; confidence: number }[];
  similarity: number;
  summary: string;
  riskAssessment: string;
  error?: string;
}

// Re-export for backward compatibility
export type { Evidence as EvidenceType };
export type { Report as ReportType };
export type { CanvasState as CanvasStateType };

// Case Scoring Types
export interface CaseScoringRequest {
  caseId?: string;
  case_id?: string;
  criteria?: ScoringCriteria;
  scoring_criteria?: ScoringCriteria;
  evidence?: Evidence[];
  case_data?: {
    title?: string;
    description?: string;
    evidence?: Evidence[];
    defendants?: string[];
    jurisdiction?: string;
  };
  additionalData?: any;
  temperature?: number;
}

export interface CaseScoringResult {
  caseId: string;
  case_id?: string;
  score: number;
  breakdown: {
    [key: string]: number;
  };
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  recommendations: string[];
  timestamp: Date;
  confidence?: number;
  scoring_criteria?: ScoringCriteria;
  ai_analysis?: any;
  processing_time?: number;
}

export interface ScoringCriteria {
  weights: {
    evidenceQuality: number;
    caseComplexity: number;
    legalPrecedent: number;
    evidenceVolume: number;
  };
  thresholds: {
    low: number;
    medium: number;
    high: number;
  };
  // Legacy snake_case properties for backward compatibility
  evidence_strength?: number;
  witness_reliability?: number;
  legal_precedent?: number;
  public_interest?: number;
  case_complexity?: number;
  resource_requirements?: number;
}

// Vector Search Types
export interface VectorSearchResult {
  id: string;
  score: number;
  payload: any;
  vector?: number[];
  // Additional properties for compatibility
  similarity?: number;
  content?: unknown;
  title?: unknown;
  type?: unknown;
  metadata?: unknown;
  case_id?: unknown;
  created_at?: unknown;
  relevance_score?: unknown;
}

export interface DocumentVector {
  id: string;
  vector: number[];
  payload: any;
  metadata?: any;
  // Additional properties used in the codebase
  content?: string;
  title?: string;
  type?: string;
  case_id?: string;
  relevance_score?: number;
}

export interface SearchOptions {
  limit?: number;
  threshold?: number;
  type?: string;
  collection?: string;
  minScore?: number;
  caseId?: string;
  // Additional properties used in the codebase
  filter?: any;
  includePayload?: boolean;
  includeVector?: boolean;
  documentType?: string;
  userId?: string;
  score_threshold?: number;
}

export interface CollectionInfo {
  name: string;
  vectorsCount: number;
  pointsCount: number;
  config: any;
}

export interface BatchUpsertResult {
  operation_id: number;
  status: "acknowledged" | "completed" | "failed";
  result?: any;
  // Additional properties for compatibility
  successful?: boolean;
}

export interface ConversationMessage {
  conversationId: string;
  messageId: string;
  content: string;
  userId?: string;
  role?: "user" | "assistant" | "system";
  timestamp?: Date;
}

export interface ChatMessage {
  conversationId: string;
  messageId: string;
  content: string;
  userId?: string;
  role?: "user" | "assistant" | "system";
  timestamp?: Date;
}

export interface CaseSummary {
  caseId: string;
  content: string;
  embedding?: number[];
  metadata?: any;
}

// Database Types
export interface Database {
  cases: Case[];
  evidence: Evidence[];
  users: User[];
  reports: Report[];
  canvasStates: CanvasState[];
}

// API Types
export interface API {
  cases: {
    getAll: () => Promise<Case[]>;
    getById: (id: string) => Promise<Case>;
    create: (data: Partial<Case>) => Promise<Case>;
    update: (id: string, data: Partial<Case>) => Promise<Case>;
    delete: (id: string) => Promise<void>;
  };
  evidence: {
    getAll: () => Promise<Evidence[]>;
    getById: (id: string) => Promise<Evidence>;
    create: (data: Partial<Evidence>) => Promise<Evidence>;
    update: (id: string, data: Partial<Evidence>) => Promise<Evidence>;
    delete: (id: string) => Promise<void>;
  };
  users: {
    getAll: () => Promise<User[]>;
    getById: (id: string) => Promise<User>;
    create: (data: Partial<User>) => Promise<User>;
    update: (id: string, data: Partial<User>) => Promise<User>;
    delete: (id: string) => Promise<void>;
  };
}
