
// Global type definitions for TypeScript error resolution
declare global {
  interface Window {
    fs?: unknown;
  }
  
  namespace svelteHTML {
    interface HTMLAttributes<T> {
      'data-testid'?: string;
    }
  }
}

// Extend module declarations for better type safety
declare module '@qdrant/js-client-rest' {
  export interface QdrantClient {
    upsert(collection: string, options: any): Promise<any>;
    search(collection: string, request: any): Promise<any>;
    getCollections(): Promise<any>;
    getCollection(name: string): Promise<any>;
    createCollection(name: string, options: any): Promise<any>;
  }
  
  export interface PointStruct {
    id: string | number;
    vector: number[];
    payload?: Record<string, any>;
  }
  
  export interface Filter {
    must?: Array<Record<string, any>>;
    should?: Array<Record<string, any>>;
  }
  
  export interface SearchRequest {
    vector: number[];
    limit?: number;
    score_threshold?: number;
    with_payload?: boolean;
    filter?: Filter;
  }
}

