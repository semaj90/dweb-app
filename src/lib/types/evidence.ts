// Enhanced Evidence System Types (fixed)

export interface EvidenceNode {
  id: string;
  title: string;
  // Keep node-level type lightweight (string key) for simple lookups in UI
  type: EvidenceTypeKey;
  position: Position;
  data: EvidenceData;
  connections?: string[];
  metadata?: EvidenceMetadata;
}

export interface Position {
  x: number;
  y: number;
}

// Strong discriminated union for evidence kinds. Use `kind` to discriminate.
export type EvidenceType =
  | { kind: 'physical'; description?: string; weightGrams?: number }
  | { kind: 'digital'; storageProvider?: string; path?: string }
  | { kind: 'document'; pageCount?: number; docType?: string; language?: string }
  | { kind: 'photo'; width?: number; height?: number; cameraModel?: string }
  | { kind: 'video'; durationSeconds?: number; codec?: string }
  | { kind: 'audio'; durationSeconds?: number; codec?: string }
  | { kind: 'testimony'; witnessName?: string; recordedAt?: string }
  | { kind: 'forensic'; labReportId?: string; method?: string }
  | { kind: 'witness'; witnessId?: string }
  | { kind: 'expert'; expertId?: string; field?: string };

// Backwards-compatible key type for lightweight use (UI, indexes)
export type EvidenceTypeKey = EvidenceType['kind'];

export interface EvidenceData {
  id: string;
  caseId: string;
  criminalId?: string;
  title: string;
  description?: string;
  // simple key for indexing/search
  evidenceType: EvidenceTypeKey;
  // optional detailed discriminated object
  evidenceTypeDetails?: EvidenceType;
  fileType?: string;
  subType?: string;
  fileUrl?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  // raw binary payload (for gRPC/QUIC transfers) — optional and may be base64-encoded in JSON
  rawBytes?: Uint8Array | null;
  hash?: string;
  tags?: string[];
  chainOfCustody?: ChainOfCustodyEntry[];
  collectedAt?: string; // ISO8601 instead of Date
  collectedBy?: string;
  location?: string;
  labAnalysis?: LabAnalysis;
  aiAnalysis?: AIAnalysis;
  aiTags?: string[];
  aiSummary?: string;
  summary?: string;
  isAdmissible?: boolean;
  confidentialityLevel?: 'public' | 'restricted' | 'confidential' | 'secret';
  canvasPosition?: Position;
  // flexible metadata bag for vendor-specific or downstream extensions
  metadata?: Record<string, unknown>;
  uploadedBy?: string;
  uploadedAt?: string;
  updatedAt?: string;
}

export interface ChainOfCustodyEntry {
  id: string;
  timestamp: string;
  handler: string;
  action: 'collected' | 'transferred' | 'analyzed' | 'stored' | 'accessed';
  location: string;
  notes?: string;
  signature?: string;
}

export interface LabAnalysis {
  technician?: string;
  method?: string;
  results?: Record<string, unknown>;
  confidence?: number;
  timestamp?: string;
  notes?: string;
}

export interface AIAnalysis {
  model?: string;
  confidence?: number;
  entities?: Entity[];
  sentiment?: number;
  classification?: string;
  keywords?: string[];
  summary?: string;
  relationships?: Relationship[];
  timestamp?: string;
  processingTime?: number;
  gpuAccelerated?: boolean;
}

export interface Entity {
  text: string;
  type: EntityType;
  confidence: number;
  position?: { start: number; end: number };
  metadata?: Record<string, unknown>;
}

export type EntityType =
  | 'person'
  | 'organization'
  | 'location'
  | 'date'
  | 'time'
  | 'money'
  | 'weapon'
  | 'vehicle'
  | 'substance'
  | 'legal_term'
  | 'case_number';

export interface Relationship {
  from: string;
  to: string;
  type: RelationshipType;
  confidence: number;
  metadata?: Record<string, unknown>;
}

export type RelationshipType =
  | 'related_to'
  | 'contradicts'
  | 'supports'
  | 'sequence'
  | 'caused_by'
  | 'leads_to'
  | 'contains'
  | 'mentions'
  | 'weak'
  | 'strong';

export interface EvidenceMetadata {
  vectorEmbedding?: number[];
  semanticTags?: string[];
  processingStatus?: ProcessingStatus;
  qualityScore?: number;
  authenticity?: AuthenticityCheck;
  duplicates?: string[];
}

export type ProcessingStatus =
  | 'pending'
  | 'processing'
  | 'analyzed'
  | 'indexed'
  | 'error'
  | 'completed';

export interface AuthenticityCheck {
  verified: boolean;
  method: string;
  confidence: number;
  timestamp: string;
  notes?: string;
}

export interface CanvasData {
  canvasJson: Record<string, unknown>;
  evidenceNodes: EvidenceNode[];
  nodeRelationships: NodeRelationship[];
  caseId?: string;
  userId?: string;
  timestamp: string;
  metadata?: CanvasMetadata;
}

export interface NodeRelationship {
  id?: string;
  from: string;
  to: string;
  type: RelationshipType;
  confidence?: number;
  metadata?: Record<string, unknown>;
  createdAt?: string;
  createdBy?: string;
}

export interface CanvasMetadata {
  version: string;
  created: string;
  lastModified: string;
  author: string;
  permissions?: CanvasPermissions;
}

export interface CanvasPermissions {
  read: string[];
  write: string[];
  admin: string[];
}

// Enhanced Search Types
export interface EvidenceSearchQuery {
  query: string;
  caseId?: string;
  evidenceType?: EvidenceTypeKey;
  searchMode: SearchMode;
  limit?: number;
  offset?: number;
  filters?: SearchFilters;
}

export type SearchMode = 'text' | 'content' | 'semantic' | 'hybrid' | 'ai_enhanced';

export interface SearchFilters {
  dateRange?: { start: string; end: string };
  confidentiality?: string[];
  tags?: string[];
  hasAnalysis?: boolean;
  isAdmissible?: boolean;
  minConfidence?: number;
}

export interface EvidenceSearchResult {
  id: string;
  caseId: string;
  title: string;
  description?: string;
  evidenceType: EvidenceTypeKey;
  fileName?: string;
  fileUrl?: string;
  tags?: string[];
  summary?: string;
  uploadedAt?: string;
  similarity: number;
  searchType: SearchMode;
  contentMatch?: string;
  highlights?: SearchHighlight[];
}

export interface SearchHighlight {
  field: string;
  text: string;
  start: number;
  end: number;
}

// Processing Types
export interface ProcessingRequest {
  evidenceId: string;
  steps: ProcessingStep[];
  options: ProcessingOptions;
  userId?: string;
}

export type ProcessingStep =
  | 'ocr'
  | 'embedding'
  | 'analysis'
  | 'classification'
  | 'entity_extraction'
  | 'similarity'
  | 'indexing';

export interface ProcessingOptions {
  useGPUAcceleration?: boolean;
  priority?: 'low' | 'normal' | 'high' | 'urgent';
  notify?: boolean;
  saveIntermediateResults?: boolean;
  overrideExisting?: boolean;
}

export interface ProcessingResult {
  jobId: string;
  status: ProcessingStatus;
  progress: number;
  step: ProcessingStep;
  stepProgress?: number;
  results?: Record<string, unknown>;
  error?: string;
  startTime: string;
  endTime?: string;
  processingTime?: number;
  gpuAccelerated?: boolean;
}

// WebGPU and WASM Types
export interface WebGPUCapabilities {
  available: boolean;
  device?: any; // GPUDevice
  adapter?: any; // GPUAdapter
  features: string[];
  limits: Record<string, number>; // GPULimits
}

export interface WASMModule {
  name: string;
  loaded: boolean;
  instance?: WebAssembly.Instance;
  exports?: Record<string, unknown>;
}

// Real-time Types
export interface RealtimeUpdate {
  type: 'evidence_update' | 'canvas_update' | 'processing_update';
  caseId: string;
  evidenceId?: string;
  data: Record<string, unknown>;
  timestamp: string;
  userId: string;
}

// Response shape for API endpoints that return evidence objects
export interface EvidenceResponse {
  id: string;
  data: EvidenceData;
  status: ProcessingStatus | 'created' | 'accepted' | 'rejected';
  message?: string;
  warnings?: string[];
  createdAt: string;
  updatedAt?: string;
}

