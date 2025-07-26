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
  | "destructive";

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
