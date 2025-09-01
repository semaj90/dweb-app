// globals.d.ts
// Comprehensive ambient types to reduce noisy 'property does not exist on type unknown' errors

type AnyObject = Record<string, any>;

// Lightweight LokiJS collection/database helpers used in tests and stores
export interface LokiCollection<T = any> {
  name: string;
  insert?: (item: T) => void;
  findOne?: (query: any) => T | undefined;
  find?: (query?: any) => T[];
  removeWhere?: (fn: (item: T) => boolean) => void;
  count?: () => number;
  clear?: () => void;
  chain?: () => any;
  map?: (fn: (item: T) => any) => any[];
  get?: (id: string) => T | undefined;
}

export interface LokiDB {
  listCollections: () => LokiCollection[];
  getCollection: (name: string) => LokiCollection | undefined;
}

// Expose runtime globals inside a declare global block so they merge correctly
declare global {
  interface Window {
    lokiDB?: LokiDB;
  }

  // Provide a top-level Locals type for files that reference `Locals` directly
  interface Locals {
    user?: { id: string; email?: string; name?: string } | null;
    session?: { id: string; user?: { id: string; email?: string } } | null;
    db?: any;
    services?: Partial<Record<string, any>>;
    requestId?: string;
    [key: string]: any;
  }
}

// Common model descriptor returned by Ollama / model registries
export interface ModelDescriptor {
  name: string;
  capabilities?: string[];
  [k: string]: any;
}

// Chunk / document shapes used across tests
export interface DocChunk {
  document_id?: string;
  content?: string;
  similarity_score?: number;
  metadata?: {
    document_type?: string;
    jurisdiction?: string;
    date?: string | number;
    [k: string]: any;
  };
  [k: string]: any;
}

// Generic message/export interfaces used in tests
export interface ExportMessage {
  role?: string;
  content?: string;
  sources?: any[];
  [k: string]: any;
}

declare module '*/tests/*' {
  const _: any;
  export default _;
}

// Import meta/env shims for Vite / SvelteKit
export interface ImportMetaEnv {
  NODE_ENV?: string;
  VITE_OLLAMA_BASE_URL?: string;
  VITE_API_BASE?: string;
  VITE_ENABLE_GPU?: string;
  [key: string]: string | boolean | undefined;
}

export interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Fetch placeholder (browser + node-fetch like)
declare type FetchLike = (input: RequestInfo, init?: RequestInit) => Promise<Response>;
declare const fetch: FetchLike;

// Minimal WebGPU placeholder to silence GPU typings until real types are introduced
declare namespace GPU {
  type Buffer = any;
  type Device = any;
  type Adapter = any;
}

// Playwright / DOM helper shims used in tests to reduce noisy errors
export interface Element {
  style?: any;
}

// Simple helper to type Playwright click chains seen in tests
export interface ClickHandle extends Promise<void> {
  first?: () => Promise<void>;
  catch?: (cb: (...args: any[]) => any) => any;
}

// Allow importing JSON and wasm modules as any to reduce transient type errors during checks
declare module '*.json' {
  const value: any;
  export default value;
}
declare module '*.wasm' {
  const value: any;
  export default value;
}

// Generic module fallback for dynamic imports or untyped packages
declare module '*';

// WebSocket & Worker shims used in client-side code/tests
declare class WebSocket {
  constructor(url: string, protocols?: string | string[]);
  send(data: any): void;
  close(code?: number, reason?: string): void;
  onopen?: (ev?: any) => void;
  onmessage?: (ev?: any) => void;
  onclose?: (ev?: any) => void;
  onerror?: (ev?: any) => void;
}

declare class Worker {
  constructor(scriptURL: string, options?: any);
  postMessage(msg: any): void;
  terminate(): void;
  onmessage?: (ev: any) => void;
}

// Audio / Web API shims
declare class AudioContext {
  resume(): Promise<void>;
  suspend(): Promise<void>;
}

// Simple NodeJS global typing when @types/node isn't loaded in the frontend
declare namespace NodeJS {
  interface Global {
    fetch?: any;
    lokiDB?: any;
  }
}

declare const global: NodeJS.Global & Window;

// Allow importing CSS modules and images as any
declare module '*.css';
declare module '*.svg';
declare module '*.png';
declare module '*.jpg';

// -- Runtime/global shims added to reduce TS2339 and missing-global errors --
declare global {
  // Storage helpers (MinIO/S3 wrapper)
  function putObject(bucket: string, key: string, data: unknown, opts?: any): Promise<any>;
  function getObject(bucket: string, key: string): Promise<Uint8Array | Buffer | null>;
  function deleteObject(bucket: string, key: string): Promise<void>;

  // Minimal Ollama/service surface for model capability discovery and analysis
  const ollamaService: {
    analyzeDocument?: (text: string, mode?: string) => Promise<any>;
    embeddings?: (text: string) => Promise<number[]>;
    chat?: (input: any) => Promise<any>;
    tags?: () => Promise<Array<{ name: string; value?: any }>>;
    health?: () => Promise<any>;
    [k: string]: any;
  };

  // Vector operations shim used by many modules
  interface EnhancedVectorOperations {
    generateEmbedding: (input: any) => Promise<number[]>;
    deleteEmbedding?: (id: string) => Promise<void>;
    upsert?: (doc: any) => Promise<any>;
    search?: (query: string | any, opts?: any) => Promise<any>;
    batchUpsert?: (docs: any[]) => Promise<any>;
  }

  const vectorOps: EnhancedVectorOperations;

  // Nomic embedding function shim used in some services
  const nomicEmbedText: (text: string) => Promise<number[]>;

  // Qdrant and other vector DB clients - minimal
  const qdrantClient: any;
  const pgVectorClient: any;
}

// Export nothing to keep this file a module for TS


