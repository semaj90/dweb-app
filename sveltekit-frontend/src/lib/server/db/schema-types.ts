/**
 * Database Schema Types for Legal AI Platform
 * 
 * Comprehensive type definitions for Drizzle ORM schema with legal-specific
 * entities, relationships, and compliance metadata.
 * 
 * Features:
 * - Legal case management types
 * - Evidence chain-of-custody types
 * - Attorney-client privilege types
 * - Court filing and deadline types
 * - Compliance and audit types
 * - Document classification types
 * - User role and permission types
 * 
 * @author Legal AI Platform Team
 * @version 2.1.0
 * @lastModified 2025-01-20
 */

import type { InferSelectModel, InferInsertModel } from "drizzle-orm";
import { 
  cases, 
  evidence, 
  reports, 
  users, 
  criminals,
  personsOfInterest,
  legalDocuments,
  notes,
  caseAssignments,
  evidenceChainOfCustody,
  complianceAudits,
  legalHolds,
  courtFilings,
  clientCommunications,
  billableTime,
  caseTimeline,
  documentReviews,
  privilegeLog,
  conflictChecks,
  caseCollaborations
} from './unified-schema';

// ===== CORE ENTITY TYPES =====

/**
 * Legal Case Entity Types
 */
export type Case = InferSelectModel<typeof cases>;
export type CaseInsert = InferInsertModel<typeof cases>;

/**
 * Evidence Entity Types with Chain of Custody
 */
export type Evidence = InferSelectModel<typeof evidence>;
export type EvidenceInsert = InferInsertModel<typeof evidence>;

/**
 * Legal Report Entity Types
 */
export type Report = InferSelectModel<typeof reports>;
export type ReportInsert = InferInsertModel<typeof reports>;

/**
 * User/Attorney Entity Types with Roles
 */
export type User = InferSelectModel<typeof users>;
export type UserInsert = InferInsertModel<typeof users>;

/**
 * Criminal/Defendant Entity Types
 */
export type Criminal = InferSelectModel<typeof criminals>;
export type CriminalInsert = InferInsertModel<typeof criminals>;

/**
 * Person of Interest Entity Types
 */
export type PersonOfInterest = InferSelectModel<typeof personsOfInterest>;
export type PersonOfInterestInsert = InferInsertModel<typeof personsOfInterest>;

/**
 * Legal Document Entity Types with Privilege Protection
 */
export type LegalDocument = InferSelectModel<typeof legalDocuments>;
export type LegalDocumentInsert = InferInsertModel<typeof legalDocuments>;

/**
 * Case Note Entity Types
 */
export type Note = InferSelectModel<typeof notes>;
export type NoteInsert = InferInsertModel<typeof notes>;

// ===== EXTENDED ENTITY TYPES =====

/**
 * Case Assignment Entity Types
 */
export type CaseAssignment = InferSelectModel<typeof caseAssignments>;
export type CaseAssignmentInsert = InferInsertModel<typeof caseAssignments>;

/**
 * Evidence Chain of Custody Entity Types
 */
export type EvidenceChainOfCustody = InferSelectModel<typeof evidenceChainOfCustody>;
export type EvidenceChainOfCustodyInsert = InferInsertModel<typeof evidenceChainOfCustody>;

/**
 * Compliance Audit Entity Types
 */
export type ComplianceAudit = InferSelectModel<typeof complianceAudits>;
export type ComplianceAuditInsert = InferInsertModel<typeof complianceAudits>;

/**
 * Legal Hold Entity Types
 */
export type LegalHold = InferSelectModel<typeof legalHolds>;
export type LegalHoldInsert = InferInsertModel<typeof legalHolds>;

/**
 * Court Filing Entity Types
 */
export type CourtFiling = InferSelectModel<typeof courtFilings>;
export type CourtFilingInsert = InferInsertModel<typeof courtFilings>;

/**
 * Client Communication Entity Types
 */
export type ClientCommunication = InferSelectModel<typeof clientCommunications>;
export type ClientCommunicationInsert = InferInsertModel<typeof clientCommunications>;

/**
 * Billable Time Entity Types
 */
export type BillableTime = InferSelectModel<typeof billableTime>;
export type BillableTimeInsert = InferInsertModel<typeof billableTime>;

/**
 * Case Timeline Entity Types
 */
export type CaseTimeline = InferSelectModel<typeof caseTimeline>;
export type CaseTimelineInsert = InferInsertModel<typeof caseTimeline>;

/**
 * Document Review Entity Types
 */
export type DocumentReview = InferSelectModel<typeof documentReviews>;
export type DocumentReviewInsert = InferInsertModel<typeof documentReviews>;

/**
 * Privilege Log Entity Types
 */
export type PrivilegeLog = InferSelectModel<typeof privilegeLog>;
export type PrivilegeLogInsert = InferInsertModel<typeof privilegeLog>;

/**
 * Conflict Check Entity Types
 */
export type ConflictCheck = InferSelectModel<typeof conflictChecks>;
export type ConflictCheckInsert = InferInsertModel<typeof conflictChecks>;

/**
 * Case Collaboration Entity Types
 */
export type CaseCollaboration = InferSelectModel<typeof caseCollaborations>;
export type CaseCollaborationInsert = InferInsertModel<typeof caseCollaborations>;

// ===== RELATIONSHIP TYPES =====

/**
 * Comprehensive Case with all related entities
 */
export type CaseWithRelations = Case & {
  evidence?: Evidence[];
  reports?: Report[];
  personsOfInterest?: PersonOfInterest[];
  legalDocuments?: LegalDocument[];
  notes?: Note[];
  leadProsecutor?: User;
  createdBy?: User;
  assignments?: CaseAssignment[];
  timeline?: CaseTimeline[];
  courtFilings?: CourtFiling[];
  complianceAudits?: ComplianceAudit[];
  legalHolds?: LegalHold[];
  collaborations?: CaseCollaboration[];
  billableTime?: BillableTime[];
  conflictChecks?: ConflictCheck[];
};

/**
 * Evidence with chain of custody and relations
 */
export type EvidenceWithRelations = Evidence & {
  case?: Case;
  uploadedBy?: User;
  chainOfCustody?: EvidenceChainOfCustody[];
  reviews?: DocumentReview[];
  legalHolds?: LegalHold[];
  complianceAudits?: ComplianceAudit[];
};

/**
 * Report with comprehensive relations
 */
export type ReportWithRelations = Report & {
  case?: Case;
  createdBy?: User;
  lastEditedBy?: User;
  reviews?: DocumentReview[];
  timeline?: CaseTimeline[];
  billableTime?: BillableTime[];
};

/**
 * Legal Document with privilege and review information
 */
export type LegalDocumentWithRelations = LegalDocument & {
  case?: Case;
  uploadedBy?: User;
  reviews?: DocumentReview[];
  privilegeEntries?: PrivilegeLog[];
  legalHolds?: LegalHold[];
  chainOfCustody?: EvidenceChainOfCustody[];
  complianceAudits?: ComplianceAudit[];
};

/**
 * User with role and case assignments
 */
export type UserWithRelations = User & {
  assignedCases?: CaseAssignment[];
  createdCases?: Case[];
  uploadedEvidence?: Evidence[];
  createdReports?: Report[];
  billableTime?: BillableTime[];
  clientCommunications?: ClientCommunication[];
  documentReviews?: DocumentReview[];
  privilegeEntries?: PrivilegeLog[];
  conflictChecks?: ConflictCheck[];
};

/**
 * Criminal/Defendant with case relations
 */
export type CriminalWithRelations = Criminal & {
  cases?: Case[];
  evidence?: Evidence[];
  timeline?: CaseTimeline[];
  conflictChecks?: ConflictCheck[];
};

/**
 * Person of Interest with case relations
 */
export type PersonOfInterestWithRelations = PersonOfInterest & {
  cases?: Case[];
  timeline?: CaseTimeline[];
  conflictChecks?: ConflictCheck[];
};

// ===== SPECIALIZED TYPES =====

/**
 * Case Status Enumeration
 */
export type CaseStatus = 
  | 'open'
  | 'under_review'
  | 'in_litigation'
  | 'settlement_negotiations'
  | 'awaiting_trial'
  | 'trial_in_progress'
  | 'closed'
  | 'dismissed'
  | 'settled'
  | 'archived';

/**
 * Evidence Type Enumeration
 */
export type EvidenceType = 
  | 'document'
  | 'photograph'
  | 'video'
  | 'audio'
  | 'physical_evidence'
  | 'digital_evidence'
  | 'testimony'
  | 'expert_opinion'
  | 'forensic_report'
  | 'chain_of_custody'
  | 'other';

/**
 * Confidentiality Level Enumeration
 */
export type ConfidentialityLevel = 
  | 'public'
  | 'confidential'
  | 'privileged'
  | 'attorney_client'
  | 'work_product'
  | 'expert_witness'
  | 'settlement_privileged';

/**
 * User Role Enumeration
 */
export type UserRole = 
  | 'admin'
  | 'partner'
  | 'associate'
  | 'paralegal'
  | 'investigator'
  | 'client'
  | 'expert_witness'
  | 'court_reporter'
  | 'judge'
  | 'opposing_counsel'
  | 'third_party';

/**
 * Document Review Status
 */
export type ReviewStatus = 
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'privileged'
  | 'responsive'
  | 'non_responsive'
  | 'redacted'
  | 'withheld'
  | 'produced';

/**
 * Billable Time Category
 */
export type BillableCategory = 
  | 'research'
  | 'drafting'
  | 'review'
  | 'communication'
  | 'meeting'
  | 'court_appearance'
  | 'deposition'
  | 'investigation'
  | 'trial_prep'
  | 'administrative'
  | 'travel';

/**
 * Compliance Status
 */
export type ComplianceStatus = 
  | 'compliant'
  | 'non_compliant'
  | 'under_review'
  | 'remediated'
  | 'waived'
  | 'pending_approval';

/**
 * Filing Status
 */
export type FilingStatus = 
  | 'draft'
  | 'pending_review'
  | 'ready_to_file'
  | 'filed'
  | 'accepted'
  | 'rejected'
  | 'amended'
  | 'withdrawn';

// ===== UTILITY TYPES =====

/**
 * Database Entity with Timestamps
 */
export type EntityWithTimestamps = {
  id: string;
  createdAt: Date;
  updatedAt: Date;
};

/**
 * Entity with Audit Trail
 */
export type EntityWithAudit = EntityWithTimestamps & {
  createdBy: string;
  lastEditedBy?: string;
  version: number;
  auditLog?: ComplianceAudit[];
};

/**
 * Entity with Legal Compliance
 */
export type EntityWithCompliance = EntityWithAudit & {
  confidentialityLevel: ConfidentialityLevel;
  legalHold?: boolean;
  retentionDate?: Date;
  destructionDate?: Date;
  complianceStatus: ComplianceStatus;
};

/**
 * Search Result with Metadata
 */
export type SearchResult<T> = {
  entity: T;
  score: number;
  highlights?: string[];
  metadata?: Record<string, any>;
};

/**
 * Paginated Results
 */
export type PaginatedResults<T> = {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPrevPage: boolean;
};

/**
 * Legal AI Analysis Result
 */
export type LegalAnalysisResult = {
  entityId: string;
  entityType: string;
  analysisType: string;
  confidence: number;
  findings: string[];
  recommendations: string[];
  legalCitations?: string[];
  precedents?: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  complianceIssues?: string[];
  createdAt: Date;
  createdBy: string;
};

/**
 * Document Classification Result
 */
export type DocumentClassification = {
  documentId: string;
  documentType: string;
  confidentialityLevel: ConfidentialityLevel;
  privilege: boolean;
  responsiveness: boolean;
  legalIssues?: string[];
  keywords?: string[];
  entities?: string[];
  confidence: number;
  reviewRequired: boolean;
  redactionRequired: boolean;
};

/**
 * Chain of Custody Entry
 */
export type ChainOfCustodyEntry = {
  id: string;
  evidenceId: string;
  custodian: string;
  action: 'received' | 'transferred' | 'analyzed' | 'stored' | 'destroyed';
  timestamp: Date;
  location: string;
  condition: string;
  notes?: string;
  signature?: string;
  witnessSignature?: string;
  integrityHash?: string;
};

// ===== RE-EXPORT SCHEMA TABLES =====

export {
  cases,
  evidence,
  reports,
  users,
  criminals,
  personsOfInterest,
  legalDocuments,
  notes,
  caseAssignments,
  evidenceChainOfCustody,
  complianceAudits,
  legalHolds,
  courtFilings,
  clientCommunications,
  billableTime,
  caseTimeline,
  documentReviews,
  privilegeLog,
  conflictChecks,
  caseCollaborations
} from './unified-schema';

// ===== TYPE GUARDS =====

/**
 * Type guard for checking if entity has legal compliance
 */
export function hasLegalCompliance(entity: any): entity is EntityWithCompliance {
  return entity && 
         typeof entity.confidentialityLevel === 'string' &&
         typeof entity.complianceStatus === 'string';
}

/**
 * Type guard for checking if entity has audit trail
 */
export function hasAuditTrail(entity: any): entity is EntityWithAudit {
  return entity && 
         typeof entity.createdBy === 'string' &&
         typeof entity.version === 'number';
}

/**
 * Type guard for checking if entity is privileged
 */
export function isPrivileged(entity: { confidentialityLevel?: string }): boolean {
  return entity.confidentialityLevel === 'attorney_client' || 
         entity.confidentialityLevel === 'work_product' ||
         entity.confidentialityLevel === 'privileged';
}

/**
 * Type guard for checking if entity requires legal hold
 */
export function requiresLegalHold(entity: any): boolean {
  return entity && entity.legalHold === true;
}

// ===== DEFAULT EXPORTS =====

export default {
  // Core types
  Case,
  Evidence,
  Report,
  User,
  Criminal,
  PersonOfInterest,
  LegalDocument,
  Note,
  
  // Extended types
  CaseAssignment,
  EvidenceChainOfCustody,
  ComplianceAudit,
  LegalHold,
  CourtFiling,
  ClientCommunication,
  BillableTime,
  CaseTimeline,
  DocumentReview,
  PrivilegeLog,
  ConflictCheck,
  CaseCollaboration,
  
  // Relationship types
  CaseWithRelations,
  EvidenceWithRelations,
  ReportWithRelations,
  LegalDocumentWithRelations,
  UserWithRelations,
  CriminalWithRelations,
  PersonOfInterestWithRelations,
  
  // Utility types
  EntityWithTimestamps,
  EntityWithAudit,
  EntityWithCompliance,
  SearchResult,
  PaginatedResults,
  LegalAnalysisResult,
  DocumentClassification,
  ChainOfCustodyEntry,
  
  // Type guards
  hasLegalCompliance,
  hasAuditTrail,
  isPrivileged,
  requiresLegalHold
};
