/**
 * 🎯 COMPREHENSIVE TYPE DEFINITIONS
 * 
 * Complete type definitions for all packages in our tech stack,
 * following Svelte 5 best practices from svelte-complete documentation.
 */

// ===== SVELTE 5 CORE TYPES =====

// Rune types
export interface StateRune<T> {
  current: T;
}

export interface DerivedRune<T> {
  current: T;
}

export interface EffectRune {
  (): void | (() => void);
}

export interface PropsRune<T extends Record<string, any>> {
  (): T;
}

export interface BindableRune<T> {
  (initial?: T): T;
}

// Snippet types (from Svelte 5)
export interface Snippet<Parameters extends readonly any[] = []> {
  (...args: Parameters): {
    render(): string;
    setup?(): void;
    teardown?(): void;
  };
}

// Component types
export interface Component<Props extends Record<string, any> = {}> {
  (props: Props): {
    render(): string;
    setup?(): void;
    teardown?(): void;
  };
}

export type ComponentProps<T> = T extends Component<infer P> ? P : never;

// Action types
export interface ActionReturn<Parameter = any> {
  update?: (parameter: Parameter) => void;
  destroy?: () => void;
}

// Transition types
export interface TransitionConfig {
  delay?: number;
  duration?: number;
  easing?: (t: number) => number;
  css?: (t: number, u: number) => string;
  tick?: (t: number, u: number) => void;
}

// Animation types
export interface AnimationConfig {
  delay?: number;
  duration?: number;
  easing?: (t: number) => number;
  css?: (t: number, u: number) => string;
  tick?: (t: number, u: number) => void;
}

// ===== SVELTEKIT 2 TYPES =====

// Page and layout load functions
export interface LoadEvent {
  params: Record<string, string>;
  url: URL;
  request: Request;
  cookies: {
    get(name: string): string | undefined;
    set(name: string, value: string, options?: any): void;
  };
  locals: Record<string, any>;
  parent(): Promise<Record<string, any>>;
  depends(...deps: string[]): void;
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
}

export interface PageLoad<Data = any> {
  (event: LoadEvent): Promise<Data> | Data;
}

export interface LayoutLoad<Data = any> {
  (event: LoadEvent): Promise<Data> | Data;
}

// Server-side request handlers
export interface RequestEvent {
  params: Record<string, string>;
  url: URL;
  request: Request;
  cookies: {
    get(name: string): string | undefined;
    set(name: string, value: string, options?: any): void;
    delete(name: string, options?: any): void;
  };
  locals: Record<string, any>;
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
  platform?: any;
}

export interface RequestHandler<Data = any> {
  (event: RequestEvent): Promise<Response> | Response;
}

// SvelteKit hooks
export interface Handle {
  (input: { event: RequestEvent; resolve: any }): Promise<Response>;
}

export interface HandleError {
  (input: { error: any; event: RequestEvent }): any;
}

export interface HandleFetch {
  (input: { event: RequestEvent; request: Request; fetch: typeof fetch }): Promise<Response>;
}

// SvelteKit stores
export interface PageStore {
  url: URL;
  params: Record<string, string>;
  route: { id: string | null };
  data: Record<string, any>;
  error: any;
  state: Record<string, any>;
  form: any;
}

export type NavigatingStore = {
  from?: { params: Record<string, string>; url: URL };
  to?: { params: Record<string, string>; url: URL };
  type?: 'link' | 'popstate' | 'goto';
} | null;

export interface UpdatedStore {
  current: boolean;
  check(): Promise<boolean>;
}

// ===== DATABASE TYPES =====

// SQL and query types
export interface SQL<T = unknown> {
  queryChunks: readonly string[];
  params: readonly unknown[];
  typings?: { [key: string]: string };
  shouldInlineParams?: boolean;
  sql: string;
}

export interface QueryResult<T = any> {
  rows: T[];
  rowCount: number;
  command?: string;
  fields?: Array<{ name: string; dataTypeID: number }>;
}

export interface DatabaseConnection {
  query<T = any>(sql: string, params?: any[]): Promise<QueryResult<T>>;
  transaction<T>(callback: (client: DatabaseConnection) => Promise<T>): Promise<T>;
  end(): Promise<void>;
}

// Enhanced Postgres connection type to fix import issues
export interface PostgresConnection {
  (options?: PostgresOptions): Sql<{}>;
  (url: string, options?: PostgresOptions): Sql<{}>;
}

export interface PostgresOptions {
  host?: string;
  port?: number;
  database?: string;
  username?: string;
  password?: string;
  max?: number;
  idle_timeout?: number;
  ssl?: boolean | string | object;
  prepare?: boolean;
  connect_timeout?: number;
  onnotice?: (notice: any) => void;
  onparameter?: (key: string, value: any) => void;
  onconnect?: () => Promise<void>;
  [key: string]: any;
}

// Drizzle ORM specific types
export interface DrizzleConfig {
  schema?: Record<string, any>;
  logger?: boolean | any;
  mode?: 'default' | 'planetscale';
}

// Enhanced Drizzle column functions to fix untyped function calls
export interface DrizzleColumnHelpers {
  pgTable: any;
  serial: any;
  text: any;
  varchar: any;
  integer: any;
  boolean: any;
  timestamp: any;
  json: any;
  jsonb: any;
  uuid: any;
  real: any;
  doublePrecision: any;
  vector: any;
  primaryKey: any;
  foreignKey: any;
  unique: any;
  index: any;
  // Query operators
  eq: any;
  ne: any;
  gt: any;
  gte: any;
  lt: any;
  lte: any;
  isNull: any;
  isNotNull: any;
  inArray: any;
  notInArray: any;
  like: any;
  ilike: any;
  between: any;
  notBetween: any;
  exists: any;
  notExists: any;
  and: any;
  or: any;
  not: any;
  sql: any;
}

export interface DrizzleTable<T extends Record<string, any> = Record<string, any>> {
  _: {
    name: string;
    columns: T;
    schema?: string;
    baseName: string;
  };
}

export interface DrizzleColumn<T = any> {
  name: string;
  dataType: string;
  columnType: string;
  data?: T;
  notNull?: boolean;
  hasDefault?: boolean;
  enumValues?: readonly string[];
}

// Vector database types
export interface EmbeddingVector {
  id: string;
  values: number[];
  metadata?: Record<string, any>;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  values?: number[];
  metadata?: Record<string, any>;
  document?: any;
}

export interface VectorSearchOptions {
  limit?: number;
  threshold?: number;
  includeValues?: boolean;
  includeMetadata?: boolean;
  filter?: Record<string, any>;
}

// ===== AI/ML TYPES =====

// Ollama types
export interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  format?: 'json' | string;
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repeat_penalty?: number;
    seed?: number;
    num_predict?: number;
    stop?: string[];
  };
  system?: string;
  template?: string;
  context?: number[];
  raw?: boolean;
  keep_alive?: string;
}

export interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaEmbeddingRequest {
  model: string;
  prompt: string;
  options?: any;
  keep_alive?: string;
}

export interface OllamaEmbeddingResponse {
  embedding: number[];
}

export interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    parameter_size: string;
    quantization_level?: string;
  };
}

// RAG types
export interface RAGDocument {
  id: string;
  content: string;
  metadata: Record<string, any>;
  embedding?: number[];
}

export interface RAGQuery {
  query: string;
  context?: RAGDocument[];
  options?: {
    model?: string;
    embedModel?: string;
    contextLimit?: number;
    temperature?: number;
    maxTokens?: number;
  };
}

export interface RAGResponse {
  response: string;
  context: RAGDocument[];
  sources: any[];
  confidence: number;
  processingTime: number;
  model?: string;
  metadata?: Record<string, any>;
}

// ===== CACHE TYPES =====

// Enhanced cache configuration (fixing our previous errors)
export interface CacheConfiguration {
  layers: CacheLayerConfig[];
  defaultTtl: number;
  defaultTTL?: number; // Alias for defaultTtl
  maxMemoryUsage: number;
  enableCompression: boolean;
  enableIntelligentTierSelection?: boolean;
  enableAnalytics?: boolean;
  enablePredictiveLoading?: boolean;
  enableCoherence?: boolean;
  compressionThreshold?: number;
  metricsInterval?: number;
  analyticsInterval?: number;
}

export interface CacheLayerConfig {
  type: 'memory' | 'redis' | 'postgres' | 'vector' | 'filesystem' | 'cdn' | 'browser';
  priority: number;
  capacity: number;
  ttl: number;
  enabled?: boolean;
  options?: Record<string, any>;
}

export interface CacheEntry {
  value: any;
  metadata: Record<string, any>;
  ttl: number;
  createdAt: number;
  lastAccessed?: number;
  accessCount?: number;
  size?: number;
  compressed?: boolean;
}

export interface CacheMetrics {
  hits: number;
  misses: number;
  errors?: number;
  gets?: number;
  sets?: number;
  deletes?: number;
  totalOperations: number;
  totalOperationTime?: number;
  hitsByLayer?: Record<string, number>;
  writesByLayer?: Record<string, number>;
  hitRate?: number;
  averageOperationTime?: number;
}

export interface CacheAnalytics {
  accessPatterns?: Map<string, any>;
  hotKeys?: Set<string>;
  coldKeys?: Set<string>;
  performanceMetrics?: Record<string, any>;
  usageStats?: Record<string, any>;
}

export interface CacheStats {
  totalEntries: number;
  memoryUsage: number;
  hitRate: number;
  size?: number;
}

export interface CacheStrategy {
  readStrategy?: 'cache-first' | 'network-first' | 'cache-only' | 'network-only';
  writeStrategy?: 'write-through' | 'write-behind' | 'write-around';
  evictionStrategy?: 'lru' | 'lfu' | 'fifo' | 'ttl';
  replicationStrategy?: 'none' | 'master-slave' | 'peer-to-peer';
}

export interface CachePolicy {
  evictionStrategy?: 'lru' | 'lfu' | 'fifo' | 'ttl';
  maxSize: number;
  ttl: number;
  compressionEnabled: boolean;
}

// ===== LOKIJS ENHANCED TYPES =====

// Enhanced LokiJS types to fix missing exports
export interface Collection<T = any> {
  insert(obj: T | T[]): T | T[];
  find(query?: any): T[];
  findOne(query?: any): T | null;
  update(obj: T): T;
  remove(obj: T | T[]): void;
  chain(): any;
  count(query?: any): number;
  data: T[];
  name: string;
}

export interface LokiMemoryAdapter {
  loadDatabase(dbname: string, callback: (data: any) => void): void;
  saveDatabase(dbname: string, dbstring: string, callback?: () => void): void;
  deleteDatabase(dbname: string, callback?: () => void): void;
}

export interface Loki {
  addCollection<T>(name: string, options?: any): Collection<T>;
  getCollection<T>(name: string): Collection<T> | null;
  removeCollection(name: string): void;
  loadDatabase(options?: any): void;
  saveDatabase(callback?: (err?: any) => void): void;
  close(callback?: () => void): void;
  serialize(): string;
}

// ===== REDIS ENHANCED TYPES =====

// Enhanced Redis options to fix configuration errors
export interface EnhancedRedisOptions {
  host?: string;
  port?: number;
  password?: string;
  db?: number;
  maxRetriesPerRequest?: number;
  enableReadyCheck?: boolean;
  lazyConnect?: boolean;
  retryStrategy?: (times: number) => number;
  reconnectOnError?: (err: any) => boolean;
  enableOfflineQueue?: boolean;
  commandTimeout?: number;
  keyPrefix?: string;
  cacheTtl?: number;
  [key: string]: any;
}

// ===== TESTING TYPES =====

export interface TestContext {
  name: string;
  timeout?: number;
  skip?: boolean;
  only?: boolean;
}

export interface ExpectationResult {
  pass: boolean;
  message: string;
}

export interface MockFunction<T extends (...args: any[]) => any = (...args: any[]) => any> {
  (...args: Parameters<T>): ReturnType<T>;
  mockImplementation(fn: T): this;
  mockReturnValue(value: ReturnType<T>): this;
  mockResolvedValue(value: Awaited<ReturnType<T>>): this;
  mockRejectedValue(error: any): this;
  mockClear(): this;
  mockReset(): this;
  mockRestore(): this;
  calls: Parameters<T>[];
  results: { type: 'return' | 'throw'; value: any }[];
}

// ===== ENVIRONMENT TYPES =====

export interface EnvironmentConfig {
  // Database
  DATABASE_URL: string;
  POSTGRES_URL?: string;
  REDIS_URL?: string;
  
  // AI services
  OLLAMA_URL?: string;
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  
  // Vector databases
  QDRANT_URL?: string;
  QDRANT_API_KEY?: string;
  
  // Object storage
  MINIO_URL?: string;
  MINIO_ACCESS_KEY?: string;
  MINIO_SECRET_KEY?: string;
  
  // Graph database
  NEO4J_URL?: string;
  NEO4J_USERNAME?: string;
  NEO4J_PASSWORD?: string;
  
  // Application
  NODE_ENV: 'development' | 'production' | 'test';
  PORT?: string;
  HOST?: string;
}

// ===== UTILITY TYPES =====

// Generic utility types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends (infer U)[]
    ? DeepPartial<U>[]
    : T[P] extends Record<string, any>
    ? DeepPartial<T[P]>
    : T[P];
};

export type RequiredBy<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type Nullable<T> = T | null | undefined;

export type NonNullable<T> = T extends null | undefined ? never : T;

// Function utility types
export type AsyncFunction<T = any> = (...args: any[]) => Promise<T>;

export type EventHandler<T = Event> = (event: T) => void | Promise<void>;

// Class utility types
export type ClassValue = 
  | string 
  | number 
  | boolean 
  | undefined 
  | null
  | { [key: string]: any }
  | ClassValue[];

// ===== GLOBAL AUGMENTATIONS =====

// Global type augmentations for missing functionality
declare global {
  // Enhanced window interface
  interface Window {
    comprehensivePackageBarrelStore?: any;
    webkitSpeechRecognition?: any;
    SpeechRecognition?: any;
  }
  
  // Enhanced console interface
  interface Console {
    trace(...args: any[]): void;
    group(label?: string): void;
    groupCollapsed(label?: string): void;
    groupEnd(): void;
    time(label?: string): void;
    timeEnd(label?: string): void;
  }

  // Enhanced Node.js process interface  
  namespace NodeJS {
    interface ProcessEnv extends EnvironmentConfig {
      [key: string]: string | undefined;
    }
    
    interface Process {
      env: ProcessEnv;
      versions: {
        node: string;
        v8: string;
        [key: string]: string;
      };
    }
  }

  // WebGPU interface enhancements
  interface GPUDevice {
    destroy?(): void;
    addEventListener?(type: string, listener: (event: any) => void): void;
    removeEventListener?(type: string, listener: (event: any) => void): void;
    dispatchEvent?(event: Event): boolean;
  }

  // WebAssembly enhancements
  namespace WebAssembly {
    interface ImportValue {
      (): any;
    }
    
    interface Imports {
      [key: string]: any;
    }
  }
}

// Re-export all types for convenience
export type {
  // Svelte 5 types
  StateRune, DerivedRune, EffectRune, PropsRune, BindableRune,
  
  // SvelteKit 2 types
  LoadEvent, PageStore, NavigatingStore, UpdatedStore,
  
  // Cache types
  CacheConfiguration, CacheLayerConfig, CacheEntry, CacheMetrics, 
  CacheAnalytics, CacheStats, CacheStrategy, CachePolicy,
  
  // AI/ML types
  OllamaGenerateRequest, OllamaGenerateResponse, OllamaEmbeddingRequest, 
  OllamaEmbeddingResponse, OllamaModel, RAGDocument, RAGQuery, RAGResponse,
  
  // Testing types
  TestContext, ExpectationResult, MockFunction,
  
  // Environment types
  EnvironmentConfig,
  
  // Utility types
  DeepPartial, RequiredBy, OptionalBy, Nullable, AsyncFunction, 
  EventHandler, ClassValue
};