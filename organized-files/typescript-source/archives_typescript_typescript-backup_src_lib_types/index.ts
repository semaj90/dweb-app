// Types Barrel File
// Central export point for all type definitions

// Core types
export * from './ai';
export * from './canvas';
export * from './legal-document';
export * from './logging';
export * from './progress';

// Evidence types (primary source)
export * from './evidence';

// Neo4j types (excluding conflicting EvidenceNode)
export type {
  Neo4jNode,
  Neo4jRelationship,
  Neo4jPath,
  Neo4jQueryResult,
  CaseNode,
  PrecedentNode
} from './neo4j';

// Search types (excluding conflicting SearchFilters which exists in evidence.ts)
export type {
  SearchResult as SearchTypeResult,
  MetricData,
  SearchParams,
  RerankedResults,
  SearchSession,
  EmbeddingRecord,
  DocumentType,
  JurisdictionType,
  PracticeAreaType
} from './search-types';

// Processing types
export interface ProcessedDocument {
  id: string;
  title: string;
  content: string;
  extractedData: {
    entities: Array<{ type: string; value: string }>;
    keyTerms: string[];
    summary: string;
  };
  processingResult: {
    status: 'completed' | 'failed' | 'pending';
    confidence: number;
    processingTime: number;
  };
}

// Import global type definitions (they're automatically available)
import './webgpu.d.ts';
import './webassembly.d.ts';