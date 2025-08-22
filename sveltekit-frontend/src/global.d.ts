/// <reference types="svelte" />
// Relax Svelte component and HTML attribute typings for monorepo-wide incremental fixes.
// This makes all imported .svelte components accept any props (safe shim for large legacy codebases)
// and allows arbitrary HTML attributes (UnoCSS attributify, custom data-* attributes, etc.).

// Svelte 5 Snippet type support - fix for TS2694 error
declare module 'svelte' {
  export interface Snippet<T = void> {
    (this: void, ...args: T extends void ? [] : [T]): unknown;
  }
}

declare module '*.svelte' {
  import { SvelteComponentTyped } from 'svelte';
  // any props, any events, any slots
  export default class Component extends SvelteComponentTyped<Record<string, any>, Record<string, any>, Record<string, any>> {}
}
/**
 * Allow arbitrary HTML attributes (UnoCSS attributify, data-*, etc.) on Svelte elements.
 * Support both svelte2tsx's svelteHTML and svelte.JSX augmentation.
 */
declare namespace svelteHTML {
  interface HTMLAttributes<T> {
    [key: string]: unknown;
  }
}

declare namespace svelte.JSX {
  interface HTMLAttributes<T> {
    [key: string]: unknown;
  }
}

// Missing dependency type stubs for arktype and other libs
declare module '@ark/schema' {
  export const ArkEnv: any;
  export type ArkConfig = any;
  export type BaseRoot = any;
  export type TypeNode = any;
  export type InternalTypeNode = any;
  export const writeInvalidOperandMessage: (...args: any[]) => any;
  export const writeMissingSubmoduleMessage: (...args: any[]) => any;
  export const writeUnresolvableMessage: (...args: any[]) => any;
  export const writeIndivisibleMessage: (...args: any[]) => any;
}

declare module '@ark/util' {
  export type Dict = Record<string, any>;
  export type array = any[];
  export type Key = string | number | symbol;
  export type conform<T, U> = T;
  export type show<T> = T;
  export type of<T> = T;
  export type Primitive = string | number | boolean | null | undefined;
  export const throwParseError: (...args: any[]) => never;
  export const throwInternalError: (...args: any[]) => never;
  export const isArray: (value: any) => value is any[];
}

declare module '@opentelemetry/api' {
  export interface Tracer {
    startSpan(name: string, options?: any): any;
  }
  export const trace: {
    getTracer(name: string, version?: string): Tracer;
  };
}

// Fix @vinejs/vine default property errors
declare module 'validator' {
  interface IsEmailOptions {
    [key: string]: any;
  }
  interface IsURLOptions {
    [key: string]: any;
  }
  interface IsIBANOptions {
    [key: string]: any;
  }
  
  interface ValidatorStatic {
    isEmail(str: string, options?: IsEmailOptions): boolean;
    isURL(str: string, options?: IsURLOptions): boolean;
    isIBAN(str: string, options?: IsIBANOptions): boolean;
    default?: ValidatorStatic;
  }
}

// Fix environment variable issues
declare module '$env/static/private' {
  export const DATABASE_URL: string;
  export const OLLAMA_URL: string;
  export const OLLAMA_API_URL: string;
  export const REDIS_URL: string;
  export const MINIO_ENDPOINT: string;
  export const NEO4J_URI: string;
  export const env: Record<string, string | undefined>;
}

// Fix collection type for LokiJS
declare global {
  interface Collection<T = any> {
    insert(doc: T): T;
    find(query?: any): T[];
    findOne(query?: any): T | null;
    update(doc: T): T;
    remove(doc: T): void;
    removeWhere(query: any): void;
  }
  
  // WebGPU types fallback
  const GPUBufferUsage: {
    STORAGE: number;
    COPY_SRC: number;
    UNIFORM: number;
    COPY_DST: number;
    MAP_READ: number;
  };
  
  const GPUMapMode: {
    READ: number;
    WRITE: number;
  };
  
  interface GPUDevice {
    createBuffer(descriptor: any): any;
    createComputePipeline(descriptor: any): any;
    createBindGroup(descriptor: any): any;
    queue: {
      submit(commandBuffers: any[]): void;
      writeBuffer(buffer: any, offset: number, data: any): void;
    };
  }
}

// Enhanced API response types
declare namespace API {
  interface BaseResponse<T = any> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
  }

  interface PaginatedResponse<T = any> extends BaseResponse<T[]> {
    pagination?: {
      page: number;
      limit: number;
      total: number;
      totalPages: number;
    };
  }
}

// XState machine context types
declare namespace XState {
  interface BaseMachineContext {
    error?: MachineError;
    loading?: boolean;
    retryCount?: number;
    lastUpdated?: Date;
  }

  interface MachineError {
    message: string;
    code?: string;
    type: 'validation' | 'network' | 'service' | 'permission' | 'unknown';
    recoverable?: boolean;
    details?: Record<string, unknown>;
    timestamp: Date;
  }

  interface DocumentContext extends BaseMachineContext {
    documentId?: string;
    content?: string;
    metadata?: Record<string, any>;
    uploadProgress?: number;
    processingStage?: 'uploading' | 'processing' | 'indexing' | 'complete' | 'failed';
  }

  interface ChatContext extends BaseMachineContext {
    messages?: Array<{
      id: string;
      content: string;
      role: 'user' | 'assistant' | 'system';
      timestamp: Date;
      metadata?: Record<string, unknown>;
    }>;
    currentMessage?: string;
    sessionId?: string;
    isTyping?: boolean;
  }

  interface UploadContext extends BaseMachineContext {
    file?: File;
    fileId?: string;
    progress?: number;
    url?: string;
    metadata?: {
      filename: string;
      size: number;
      type: string;
      hash?: string;
    };
  }

  // Machine event types
  interface MachineEvent {
    type: string;
    [key: string]: unknown;
  }

  interface ErrorEvent extends MachineEvent {
    type: 'ERROR';
    error: MachineError;
  }

  interface RetryEvent extends MachineEvent {
    type: 'RETRY';
    maxRetries?: number;
  }

  interface ResetEvent extends MachineEvent {
    type: 'RESET';
    preserveData?: boolean;
  }
}

// Common utility types
declare namespace Utils {
  type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
  };

  type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

  type NonEmptyArray<T> = [T, ...T[]];
}

// Fix common module issues
declare module 'bull' {
  export default class Queue {
    constructor(name: string, opts?: any);
    add(data: any, opts?: any): Promise<any>;
    process(handler: (job: any) => Promise<any>): void;
    on(event: string, handler: (...args: any[]) => void): void;
  }
}

declare module 'lokijs' {
  export default class Loki {
    constructor(filename?: string, options?: any);
    addCollection(name: string, options?: any): Collection<any>;
    getCollection(name: string): Collection<any> | null;
    saveDatabase(): void;
    loadDatabase(callback?: () => void): void;
  }
}

// Enhanced file upload types
declare namespace Upload {
  interface FileMetadata {
    filename: string;
    size: number;
    type: string;
    lastModified: number;
    hash?: string;
  }

  interface UploadProgress {
    loaded: number;
    total: number;
    percentage: number;
    speed?: number;
    timeRemaining?: number;
  }

  interface UploadResult {
    success: boolean;
    fileId?: string;
    url?: string;
    error?: string;
    metadata?: FileMetadata;
  }
}
