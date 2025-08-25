/// <reference types="@sveltejs/kit" />
/// <reference types="svelte" />
/// <reference types="vite/client" />

// Define SessionUser interface inline to avoid import issues
interface SessionUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  isActive: boolean;
}


declare global {
  namespace App {
    interface Error {
      code?: string;
      id?: string;
      errorId?: string;
    }

    interface Locals {
      user: SessionUser | null;
      session: string | null;
      featureFlags?: Record<string, boolean>;
      apiContext?: unknown;
      serviceHealth?: unknown;
      services: {
        ollama?: any;
        postgres?: any;
        redis?: any;
        vector?: any;
        ai?: any;
        workflows?: any;
        initialized?: boolean;
      };
      requestId: string;
      startTime: number;
      // Additional properties for legal AI platform
      jurisdiction?: string;
      practiceArea?: string[];
      clientContext?: any;
      metrics?: {
        startTime: number;
        dbQueries: number;
        cacheHits: number;
        apiCalls: number;
      };
    }

    interface PageData {
      user?: SessionUser | null;
    }

    interface Platform {
      env?: {
        REDIS_URL: string;
        RABBITMQ_URL: string;
        NEO4J_URL: string;
        DATABASE_URL: string;
        OLLAMA_URL: string;
        NODE_ENV: string;
      };
    }
  }

  interface Window {
    fs: {
      readFile: (
        path: string,
        options?: { encoding?: string },
      ) => Promise<Uint8Array | string>;
    };
    __MATRIX_UI__: Record<string, unknown>;
    __TAURI__?: Record<string, unknown>;
    electronAPI?: Record<string, unknown>;
    
    // Testing framework globals
    describe?: (name: string, fn: () => void) => void;
    it?: (name: string, fn: () => void | Promise<void>) => void;
    test?: (name: string, fn: () => void | Promise<void>) => void;
    expect?: (value: any) => any;
    beforeEach?: (fn: () => void | Promise<void>) => void;
    afterEach?: (fn: () => void | Promise<void>) => void;
    beforeAll?: (fn: () => void | Promise<void>) => void;
    afterAll?: (fn: () => void | Promise<void>) => void;
    
    // Barrel store for missing functions
    barrelStore?: any;
    
    // WebGPU polyfill
    GPU?: any;
  }

  interface HTMLElement {
    inert?: boolean;
  }
}

// Additional global type declarations for missing functions
declare global {
  // Testing framework globals (for Node.js/Jest environment)
  const describe: (name: string, fn: () => void) => void;
  const it: (name: string, fn: () => void | Promise<void>) => void;
  const test: (name: string, fn: () => void | Promise<void>) => void;
  const expect: (value: any) => any;
  const beforeEach: (fn: () => void | Promise<void>) => void;
  const afterEach: (fn: () => void | Promise<void>) => void;
  const beforeAll: (fn: () => void | Promise<void>) => void;
  const afterAll: (fn: () => void | Promise<void>) => void;
  
  // Enhanced GPUDevice interface
  interface GPUDevice {
    destroy?(): void;
    addEventListener?(type: string, listener: (event: any) => void): void;
    removeEventListener?(type: string, listener: (event: any) => void): void;
  }
  
  // GPU error event interfaces
  interface GPUUncapturedErrorEvent {
    type: 'uncapturederror';
    error: any;
    timestamp: number;
  }
  
  interface GPUError {
    message: string;
  }
  
  // Loki.js enhanced interfaces
  namespace Loki {
    interface Collection<T = any> {
      remove?(doc: T | T[]): void;
      removeWhere?(query: any): number;
      data: T[];
      name: string;
    }
    
    interface LokiConstructor {
      LokiMemoryAdapter?: new () => any;
    }
    
    interface Database {
      removeCollection?(name: string): void;
    }
  }
}

// Common database entity interfaces
export interface DatabaseEntity {
  id?: string | number;
  created_at?: string | Date;
  updated_at?: string | Date;
  case_id?: string;
  document_id?: string;
  user_id?: string;
  metadata?: Record<string, any>;
}

// Chat/Message interfaces with required properties
export interface ChatMessage extends DatabaseEntity {
  message: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string | Date;
  sources?: Array<{
    title: string;
    url?: string;
    relevance: number;
  }>;
}

// Cache entry interface with required properties
export interface CacheEntry {
  key: string;
  value: any;
  createdAt: number;
  expiresAt: number;
  lastAccessed?: number;
  accessCount?: number;
  size?: number;
}

export {};
