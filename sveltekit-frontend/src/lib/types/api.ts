import type { User } from "$lib/types/user";

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
  [key: string]: any;
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
    [key: string]: any;
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
