// TypeScript declarations for missing types across the codebase
// This file resolves many TS2304 "Cannot find name" errors

// Engine/Graphics types
declare class ShaderCache {
  static get(key: string): any;
  static set(key: string, value: any): void;
}

declare class MatrixTransformLib {
  static createTransform(): any;
  static multiply(a: any, b: any): any;
}

// Docker/Optimization types  
declare class DockerResourceOptimizer {
  static optimizeMemory(): Promise<any>;
  static getCurrentUsage(): Promise<any>;
}

// RAG/Search types
declare interface RAGSearchResult {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, any>;
}

declare interface TextChunk {
  text: string;
  index: number;
  metadata?: Record<string, any>;
}

declare interface RAGDocument {
  id: string;
  content: string;
  embedding?: number[];
  metadata?: Record<string, any>;
}

// Store types
declare const enhancedRAGStore: {
  search: (query: string) => Promise<RAGSearchResult[]>;
  add: (doc: RAGDocument) => Promise<void>;
};

declare const documentVectors: any;

// Routing types
declare interface DynamicRouteConfig {
  path: string;
  component: any;
  metadata?: Record<string, any>;
}

declare interface GeneratedRoute {
  path: string;
  handler: any;
}

declare function registerDynamicRoute(config: DynamicRouteConfig): GeneratedRoute;

// Document processing types
declare interface DocumentProcessingOptions {
  type: 'pdf' | 'docx' | 'txt';
  extractImages?: boolean;
  ocrEnabled?: boolean;
}

// Context7/MCP types
declare function createContext7MCPIntegration(): any;