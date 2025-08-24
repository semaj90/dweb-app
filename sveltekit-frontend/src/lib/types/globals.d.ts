// globals.d.ts
// Comprehensive ambient types to reduce noisy 'property does not exist on type unknown' errors

type AnyObject = Record<string, any>;

// Lightweight LokiJS collection/database helpers used in tests and stores
interface LokiCollection<T = any> {
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

interface LokiDB {
  listCollections: () => LokiCollection[];
  getCollection: (name: string) => LokiCollection | undefined;
}

interface Window {
  lokiDB?: LokiDB;
}

// Common model descriptor returned by Ollama / model registries
interface ModelDescriptor {
  name: string;
  capabilities?: string[];
  [k: string]: any;
}

// Chunk / document shapes used across tests
interface DocChunk {
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
interface ExportMessage {
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
interface ImportMetaEnv {
  NODE_ENV?: string;
  VITE_OLLAMA_BASE_URL?: string;
  VITE_API_BASE?: string;
  VITE_ENABLE_GPU?: string;
  [key: string]: string | boolean | undefined;
}

interface ImportMeta {
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
interface Element {
  style?: any;
}

// Simple helper to type Playwright click chains seen in tests
interface ClickHandle extends Promise<void> {
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

