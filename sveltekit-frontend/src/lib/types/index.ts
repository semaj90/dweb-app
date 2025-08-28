
// === DATABASE TYPES ===
// Re-export Drizzle database types for clean imports
export type {
  // Table Select Types
  SelectUser,
  SelectSession,
  SelectCase,
  SelectEvidence, 
  SelectLegalDocument,
  SelectDocumentChunk,
  
  // Table Insert Types
  InsertUser,
  InsertSession,
  InsertCase,
  InsertEvidence,
  InsertLegalDocument,
  InsertDocumentChunk,
  
  // Database Utilities
  QueryResult,
  DatabaseConfig,
  UserRole,
  CaseStatus,
  EvidenceType,
  DocumentType,
  TableName
} from '$lib/server/db/index.js';

// Legacy interfaces for compatibility
export interface Database {
  [key: string]: unknown;
}

export interface API {
  [key: string]: unknown;
}

export interface Config {
  [key: string]: unknown;
}

// UI Component Types
export type ButtonVariant = 'primary' | 'secondary' | 'destructive' | 'outline' | 'ghost' | 'link' | 'danger' | 'success' | 'warning' | 'info' | 'default' | 'nier' | 'crimson' | 'gold';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'xl';

// Evidence Types
export interface Evidence {
  id: string;
  title: string;
  description?: string;
  type: 'document' | 'image' | 'video' | 'audio' | 'physical' | 'digital';
  evidenceType: 'document' | 'image' | 'video' | 'audio';
  caseId: string;
  uploadedBy: string;
  uploadedAt: string | Date;
  fileUrl?: string;
  fileName?: string;
  mimeType?: string;
  fileSize?: number;
  metadata?: Record<string, any>;
  tags?: string[];
}

// Session User (simplified version for auth)
export interface SessionUser {
  id: string;
  email: string;
  name: string;
  firstName: string;
  lastName: string;
  role: string;
}

export interface Report {
  id: string;
  title: string;
  content: string;
  summary: string;
  createdAt: string;
  updatedAt: string;
  status: 'draft' | 'completed' | 'archived';
  type: 'case' | 'evidence' | 'legal' | 'analysis';
  reportType: string;
  wordCount: number;
  estimatedReadTime: number;
  tags: string[];
  metadata?: Record<string, any>;
}

// User interface moved to ./user.ts for centralized management

// Enhanced type definitions for barrel store compatibility
// Re-export all types from organized type files

// Core Domain Types - Safe exports without conflicts
export { User, UserSession } from './user.js';
export { Case, CaseForm, CaseFormState, CaseMetrics } from './case.js';

// Export Evidence interface defined above
export type { Evidence };

// Component Props (Svelte 5 runes compatible)
export * from './component-props.js';

// XState Types - Centralized state management types
export type { AIAssistantEvent, AIAssistantContext, ConversationEntry } from './xstate.js';

// AI & ML Types - Safe exports
export { type AIAnalysisResult, type AIModelConfig, type VectorSearchOptions } from './ai-types.js';
export { type ChatMessage, type ChatSession, type StreamingResponse } from './ai-chat.js';

// Global type references for enhanced compatibility
/// <reference path="./webgpu.d.ts" />
/// <reference path="./webassembly-enhanced.d.ts" />
/// <reference path="./drizzle-enhanced.d.ts" />
/// <reference path="./env-enhanced.d.ts" />
