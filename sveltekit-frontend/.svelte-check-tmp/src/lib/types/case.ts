// Legal Case Management Types
export interface Case {
  id: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'draft' | 'active' | 'under_review' | 'closed' | 'archived';
  caseType: 'civil' | 'criminal' | 'corporate' | 'family' | 'immigration' | 'other';
  clientId?: string;
  assignedLawyerId?: string;
  jurisdiction?: string;
  courtLevel?: 'district' | 'appellate' | 'supreme';
  dateCreated: Date;
  dateModified: Date;
  dueDate?: Date;
  estimatedHours?: number;
  actualHours?: number;
  metadata?: Record<string, unknown>;
}

export interface CaseForm {
  id?: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'draft' | 'active' | 'under_review' | 'closed' | 'archived';
  caseType: 'civil' | 'criminal' | 'corporate' | 'family' | 'immigration' | 'other';
  clientId?: string;
  assignedLawyerId?: string;
  jurisdiction?: string;
  courtLevel?: 'district' | 'appellate' | 'supreme';
  dateCreated?: Date;
  dateModified?: Date;
  dueDate?: Date;
  estimatedHours?: number;
  actualHours?: number;
  metadata?: Record<string, unknown>;
}

export interface CaseFormValidationErrors {
  title?: string;
  description?: string;
  priority?: string;
  caseType?: string;
  clientId?: string;
  jurisdiction?: string;
  [key: string]: string | undefined;
}

export interface CaseFormState {
  data: CaseForm;
  errors: CaseFormValidationErrors;
  isSubmitting: boolean;
  isValid: boolean;
  isDirty: boolean;
  touchedFields: Set<keyof CaseForm>;
}

// Case Timeline and Events
export interface TimelineEvent {
  id: string;
  caseId: string;
  type: 'filing' | 'hearing' | 'evidence_added' | 'document_received' | 'communication' | 'milestone' | 'deadline';
  title: string;
  description?: string;
  date: Date;
  participants?: string[];
  documents?: string[];
  metadata?: Record<string, unknown>;
  createdBy: string;
  createdAt: Date;
}

// Case Analytics and Metrics  
export interface CaseMetrics {
  caseId: string;
  totalDocuments: number;
  totalEvidence: number;
  totalHours: number;
  timelineEvents: number;
  collaborators: number;
  lastActivity: Date;
  progressPercentage: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  confidenceScore: number;
}

// Legal Context and Relationships
export interface LegalContext {
  caseId: string;
  precedents: LegalPrecedent[];
  statutes: LegalStatute[];
  regulations: LegalRegulation[];
  jurisdiction: JurisdictionInfo;
  practiceArea: string[];
}

export interface LegalPrecedent {
  id: string;
  title: string;
  court: string;
  date: Date;
  citationKey: string;
  relevanceScore: number;
  summary: string;
  url?: string;
}

export interface LegalStatute {
  id: string;
  title: string;
  code: string;
  section: string;
  jurisdiction: string;
  effectiveDate: Date;
  summary: string;
  relevanceScore: number;
}

export interface LegalRegulation {
  id: string;
  title: string;
  agency: string;
  cfr?: string;
  effectiveDate: Date;
  summary: string;
  relevanceScore: number;
}

export interface JurisdictionInfo {
  name: string;
  type: 'federal' | 'state' | 'local';
  code: string;
  courts: CourtInfo[];
}

export interface CourtInfo {
  name: string;
  level: 'trial' | 'appellate' | 'supreme';
  jurisdiction: string;
  address?: string;
  website?: string;
}