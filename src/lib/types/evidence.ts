// Enhanced Evidence System Types
export interface EvidenceNode {
  id: string;
  title: string;
  type: EvidenceType;
  position: Position;
  data: EvidenceData;
  connections?: string[];
  metadata?: EvidenceMetadata;
}

export interface Position {
  x: number;
  y: number;
}

export type EvidenceType = 
  | 'physical'
  | 'digital' 
  | 'document'
  | 'photo'
  | 'video'
  | 'audio'
  | 'testimony'
  | 'forensic'
  | 'witness'
  | 'expert';

export interface EvidenceData {
  id: string;
  caseId: string;
  criminalId?: string;
  title: string;
  description?: string;
  evidenceType: EvidenceType;
  fileType?: string;
  subType?: string;
  fileUrl?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  hash?: string;
  tags?: string[];
  chainOfCustody?: ChainOfCustodyEntry[];
  collectedAt?: Date;
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
  uploadedBy?: string;
  uploadedAt?: Date;
  updatedAt?: Date;
}

export interface ChainOfCustodyEntry {
  id: string;
  timestamp: Date;
  handler: string;
  action: 'collected' | 'transferred' | 'analyzed' | 'stored' | 'accessed';
  location: string;
  notes?: string;
  signature?: string;
}

export interface LabAnalysis {
  id?: string;
  technician?: string;
  method?: string;
  results?: any;
  confidence?: number;
  timestamp?: Date;
  notes?: string;
}

export interface AIAnalysis {
  id?: string;
  model?: string;
  confidence?: number;
  entities?: Entity[];
  sentiment?: number;
  classification?: string;
  keywords?: string[];
  summary?: string;
  relationships?: Relationship[];
  timestamp?: Date;
  processingTime?: number;
  gpuAccelerated?: boolean;
}

export interface Entity {
  text: string;
  type: EntityType;
  confidence: number;
  position?: { start: number; end: number };
  metadata?: any;
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
  metadata?: any;
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
  timestamp: Date;
  notes?: string;
}

export interface CanvasData {
  canvas_json: any;
  evidence_nodes: EvidenceNode[];
  node_relationships: NodeRelationship[];
  case_id?: string;
  user_id?: string;
  timestamp: string;
  metadata?: CanvasMetadata;
}

export interface NodeRelationship {
  id?: string;
  from: string;
  to: string;
  type: RelationshipType;
  confidence?: number;
  metadata?: any;
  createdAt?: Date;
  createdBy?: string;
}

export interface CanvasMetadata {
  version: string;
  created: Date;
  lastModified: Date;
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
  evidenceType?: EvidenceType;
  searchMode: SearchMode;
  limit?: number;
  offset?: number;
  filters?: SearchFilters;
}

export type SearchMode = 'text' | 'content' | 'semantic' | 'hybrid' | 'ai_enhanced';

export interface SearchFilters {
  dateRange?: { start: Date; end: Date };
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
  evidenceType: EvidenceType;
  fileName?: string;
  fileUrl?: string;
  tags?: string[];
  summary?: string;
  uploadedAt?: Date;
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
  results?: any;
  error?: string;
  startTime: Date;
  endTime?: Date;
  processingTime?: number;
  gpuAccelerated?: boolean;
}

// WebGPU and WASM Types
export interface WebGPUCapabilities {
  available: boolean;
  device?: GPUDevice;
  adapter?: GPUAdapter;
  features: string[];
  limits: GPULimits;
}

export interface WASMModule {
  name: string;
  loaded: boolean;
  instance?: WebAssembly.Instance;
  exports?: any;
}

// Real-time Types
export interface RealtimeUpdate {
  type: 'evidence_update' | 'canvas_update' | 'processing_update';
  caseId: string;
  evidenceId?: string;
  data: any;
  timestamp: Date;
  userId: string;
}

// Export all for easy importing
export type {
  // Main types
  EvidenceNode,
  EvidenceData,
  EvidenceMetadata,
  CanvasData,
  NodeRelationship,
  
  // Analysis types
  AIAnalysis,
  Entity,
  Relationship,
  
  // Search types
  EvidenceSearchQuery,
  EvidenceSearchResult,
  SearchFilters,
  
  // Processing types
  ProcessingRequest,
  ProcessingResult,
  ProcessingOptions,
  
  // Real-time types
  RealtimeUpdate,
  
  // GPU/WASM types
  WebGPUCapabilities,
  WASMModule
};