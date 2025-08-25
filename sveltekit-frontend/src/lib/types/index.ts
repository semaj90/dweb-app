
// Core types stub for frontend services
export interface Database {
  [key: string]: unknown;
}

export interface API {
  [key: string]: unknown;
}

// Additional types can be added here as needed
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

export interface User {
  id: string;
  email: string;
  name: string;
  firstName: string;
  lastName: string;
  role: 'admin' | 'prosecutor' | 'detective' | 'user';
  createdAt: string; // ISO string
  updatedAt: string; // ISO string
  avatarUrl?: string;
  isActive: boolean;
  emailVerified: boolean;
  preferences?: Record<string, any>;
  // Optional legacy / alternate casing fields encountered in some sources
  created_at?: string;
  last_login?: string;
}

// Enhanced type definitions for barrel store compatibility
export * from './webgpu.d.ts';
export * from './webassembly-enhanced.d.ts'; 
export * from './drizzle-enhanced.d.ts';
export * from './env-enhanced.d.ts';

// Global type references for enhanced compatibility
/// <reference path="./webgpu.d.ts" />
/// <reference path="./webassembly-enhanced.d.ts" />
/// <reference path="./drizzle-enhanced.d.ts" />
/// <reference path="./env-enhanced.d.ts" />
