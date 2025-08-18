// Enhanced Search Result Types
// Fixes for SearchResult interface and related types

import type { LegalDocument } from './legal-document';

export interface SearchResult {
  score: number;
  rank: number;
  id: string;
  title: string;
  content?: string;
  summary?: string;
  excerpt?: string;
  metadata?: Record<string, any>;
  type?: string;
  createdAt?: Date;
  updatedAt?: Date;
  document: LegalDocument; // Required property that was missing
}

export interface MetricData {
  name: string;
  value: number;
  timestamp?: Date;
  type?: 'counter' | 'gauge' | 'histogram';
  labels?: Record<string, string>;
}

export interface SearchParams {
  query: string;
  searchType?: 'semantic' | 'full-text' | 'hybrid';
  limit?: number;
  offset?: number;
  filters?: SearchFilters;
}

export interface SearchFilters {
  documentType?: string;
  jurisdiction?: string;
  practiceArea?: string;
  dateRange?: {
    start: Date;
    end: Date;
  };
  tags?: string[];
}

export interface RerankedResults {
  rerankedResults: SearchResult[];
  originalResults: SearchResult[];
  reranking_time: number;
  method: 'cross-encoder' | 'mmr' | 'bm25';
}

export interface SearchSession {
  id?: string;
  query: string;
  searchType?: string;
  queryEmbedding?: string; // JSON string of number[]
  results?: unknown;
  resultCount?: number;
  createdAt?: Date;
}

export interface EmbeddingRecord {
  id?: string;
  contentId: string;
  contentType: string;
  textContent: string;
  embedding?: string; // JSON string of number[]
  metadata?: unknown;
  createdAt?: Date;
  model?: string;
}

// Utility types for better type safety
export type DocumentType = 'contract' | 'motion' | 'brief' | 'statute' | 'regulation' | 'case' | 'general';
export type JurisdictionType = 'federal' | 'state' | 'local' | 'international';
export type PracticeAreaType = 'criminal' | 'civil' | 'corporate' | 'family' | 'immigration' | 'tax' | 'general';
