// Missing dependency types for TypeScript compatibility

// Svelte store types
declare module 'svelte/store' {
  export interface Readable<T> {
    subscribe(fn: (value: T) => void): () => void;
  }
  
  export interface Writable<T> extends Readable<T> {
    set(value: T): void;
    update(fn: (value: T) => T): void;
  }
}

// QDrant client types
declare module '@qdrant/js-client-rest' {
  export class QdrantClient {
    constructor(options: { url: string; apiKey?: string });
    getCollection(collectionName: string): Promise<any>;
    createCollection(collectionName: string, config: any): Promise<any>;
    getCollections(): Promise<any>;
    upsert(collectionName: string, points: any): Promise<any>;
    search(collectionName: string, searchRequest: any): Promise<any>;
    delete(collectionName: string, pointsSelector: any): Promise<any>;
  }
}

declare module '@langchain/community/vectorstores/qdrant' {
  export class QdrantVectorStore {
    static fromExistingCollection(embeddings: any, config: any): Promise<QdrantVectorStore>;
    similaritySearch(query: string, k?: number): Promise<any[]>;
    addDocuments(documents: any[]): Promise<void>;
  }
}

// Global QDrant instances for runtime
declare global {
  const QdrantClient: typeof import('@qdrant/js-client-rest').QdrantClient;
  const QdrantVectorStore: typeof import('@langchain/community/vectorstores/qdrant').QdrantVectorStore;
}

// LangChain types
declare module 'langchain/schema/output_parser' {
  export class StringOutputParser {
    parse(text: string): string;
    stream(input: AsyncIterable<any>): AsyncIterable<string>;
  }
}

declare module 'langchain/schema/runnable' {
  export function formatDocumentsAsString(docs: any[]): string;
}

declare module '@langchain/core/output_parsers' {
  export class StringOutputParser {
    parse(text: string): string;
    stream(input: AsyncIterable<any>): AsyncIterable<string>;
  }
}

declare module '@langchain/core/utils/document' {
  export function formatDocumentsAsString(docs: any[]): string;
}

declare module '@langchain/core/documents' {
  export interface Document {
    pageContent: string;
    metadata: Record<string, any>;
  }
}

// Ollama LangChain types
declare module '@langchain/ollama' {
  export interface ChatOllamaInput {
    model: string;
    baseUrl?: string;
    temperature?: number;
    topP?: number;
    topK?: number;
    numPredict?: number;
    maxTokens?: number;  // Added this missing property
    stop?: string[];
    format?: string;
    keepAlive?: string;
    headers?: Record<string, string>;
    timeout?: number;
    stream?: boolean;
  }

  export interface OllamaEmbeddingsParams {
    model: string;
    baseUrl?: string;
    keepAlive?: string;
    headers?: Record<string, string>;
  }

  export class ChatOllama {
    constructor(config: ChatOllamaInput);
    invoke(input: string): Promise<any>;
    stream(input: string): AsyncIterable<any>;
  }

  export class OllamaEmbeddings {
    constructor(config: OllamaEmbeddingsParams);
    embedQuery(query: string): Promise<number[]>;
    embedDocuments(documents: string[]): Promise<number[][]>;
  }
}

// Additional QDrant and vector store types
declare module '@langchain/community/vectorstores/pgvector' {
  export class PGVectorStore {
    static fromExistingTable(embeddings: any, config: any): Promise<PGVectorStore>;
    static fromDocuments(documents: any[], embeddings: any, config: any): Promise<PGVectorStore>;
    similaritySearch(query: string, k?: number): Promise<any[]>;
    addDocuments(documents: any[]): Promise<void>;
    delete(options: { ids?: string[] }): Promise<void>;
  }
}

// Node.js stream types for non-Node environments
declare global {
  interface Readable<T = any> {
    subscribe?(fn: (value: T) => void): () => void;
    on?(event: string, listener: (...args: any[]) => void): this;
    pipe?(destination: any): any;
    read?(size?: number): any;
  }
}

declare module 'stream' {
  export class Readable<T = any> {
    on(event: string, listener: (...args: any[]) => void): this;
    pipe(destination: any): any;
    read(size?: number): any;
  }
}

// Vitest types
declare module 'vitest' {
  export const vi: {
    fn: (impl?: (...args: any[]) => any) => any;
  };
}

// PGVector types
declare module '@langchain/community/vectorstores/pgvector' {
  export class PGVectorStore {
    // Add methods as needed
  }
}

// Transformers.js types
declare module '@xenova/transformers' {
  export interface Pipeline {
    processor?: any;
    (input: string | string[]): Promise<any>;
  }
  
  export interface PretrainedOptions {
    device?: string;
    dtype?: string;
    cache_dir?: string;
    local_files_only?: boolean;
  }

  export const env: {
    useBrowserCache: boolean;
    allowLocalModels: boolean;
    allowRemoteModels: boolean;
    backends: {
      onnx: {
        wasm: {
          numThreads: number;
        };
      };
    };
  };

  export function pipeline(
    task: string,
    model?: string,
    options?: PretrainedOptions
  ): Promise<Pipeline>;

  export interface FeatureExtractionPipeline extends Pipeline {}
  export interface TextGenerationPipeline extends Pipeline {}
}

// LokiJS types - Complete interface definitions
declare module 'lokijs' {
  export interface LokiObj {
    $loki: number;
    meta: {
      revision: number;
      created: number;
      version: number;
    };
  }

  export interface LokiCollection<T = any> {
    insert(doc: T): T & LokiObj;
    insertOne(doc: T): T & LokiObj;
    find(query?: Partial<T> | ((obj: T & LokiObj) => boolean)): (T & LokiObj)[];
    findOne(query?: Partial<T> | ((obj: T & LokiObj) => boolean)): (T & LokiObj) | null;
    findAndRemove(query?: Partial<T> | ((obj: T & LokiObj) => boolean)): (T & LokiObj)[];
    where(query: (obj: T & LokiObj) => boolean): (T & LokiObj)[];
    remove(doc: T & LokiObj): void;
    removeWhere(query: (obj: T & LokiObj) => boolean): void;
    update(doc: T & LokiObj): T & LokiObj;
    count(query?: any): number;
    clear(): void;
    chain(): LokiCollectionChain<T>;
    simplesort(prop: keyof (T & LokiObj), desc?: boolean): LokiCollection<T>;
    compoundsort(sorts: Array<[keyof (T & LokiObj), boolean?]>): LokiCollection<T>;
    data: (T & LokiObj)[];
  }

  export interface LokiCollectionChain<T = any> {
    find(query?: any): LokiCollectionChain<T>;
    where(filter: (obj: T & LokiObj) => boolean): LokiCollectionChain<T>;
    simplesort(property: keyof (T & LokiObj), desc?: boolean): LokiCollectionChain<T>;
    limit(qty: number): LokiCollectionChain<T>;
    data(): (T & LokiObj)[];
  }

  export interface LokiDatabase {
    addCollection<T = any>(name: string, options?: {
      unique?: string[];
      indices?: string[];
      asyncListeners?: boolean;
      transactional?: boolean;
      autoupdate?: boolean;
      exact?: string[];
    }): LokiCollection<T>;
    getCollection<T = any>(name: string): LokiCollection<T> | null;
    removeCollection(name: string): void;
    saveDatabase(callback?: (err?: any) => void): void;
    loadDatabase(options?: any, callback?: (err?: any) => void): void;
    serialize(): string;
    deserialize(serializedDb: string): void;
    listCollections(): Array<{ name: string; type: string; count: number }>;
    getName(): string;
    close(callback?: () => void): void;
    filename: string;
    options: any;
  }

  export interface LokiMemoryAdapter {
    new (): any;
    loadDatabase(dbname: string, callback: (data: string | null) => void): void;
    saveDatabase(dbname: string, dbstring: string, callback: (err?: Error) => void): void;
    deleteDatabase(dbname: string, callback: (err?: Error) => void): void;
  }

  export interface LokiFsAdapter {
    new (): any;
    loadDatabase(dbname: string, callback: (data: string | null) => void): void;
    saveDatabase(dbname: string, dbstring: string, callback: (err?: Error) => void): void;
    deleteDatabase(dbname: string, callback: (err?: Error) => void): void;
  }

  export class LokiConstructor {
    constructor(filename?: string | null, options?: {
      adapter?: any;
      autoload?: boolean;
      autoloadCallback?: (err?: any) => void;
      autosave?: boolean;
      autosaveInterval?: number;
      persistenceMethod?: 'fs' | 'localStorage' | 'memory' | 'adapter';
      destructureDelimiter?: string;
      serializationMethod?: 'normal' | 'pretty' | 'destructured';
      throttledSaves?: boolean;
    });
    
    // Database methods
    addCollection<T = any>(name: string, options?: {
      unique?: string[];
      indices?: string[];
      asyncListeners?: boolean;
      transactional?: boolean;
      autoupdate?: boolean;
      exact?: string[];
    }): LokiCollection<T>;
    getCollection<T = any>(name: string): LokiCollection<T> | null;
    removeCollection(name: string): void;
    saveDatabase(callback?: (err?: any) => void): void;
    loadDatabase(options?: any, callback?: (err?: any) => void): void;
    serialize(): string;
    deserialize(serializedDb: string): void;
    listCollections(): Array<{ name: string; type: string; count: number }>;
    getName(): string;
    close(callback?: () => void): void;
    filename: string;
    options: any;

    // Static adapter properties
    static LokiFsAdapter: new () => LokiFsAdapter;
    static LokiFSAdapter: new () => LokiFsAdapter;
    static LokiMemoryAdapter: new () => LokiMemoryAdapter;
  }
  
  const loki: typeof LokiConstructor;
  export default loki;
}

// Performance types for test environments
declare global {
  namespace NodeJS {
    interface Global {
      performance: {
        now(): number;
      };
      gc?: () => void;
    }
  }
  
  interface Window {
    gc?: () => void;
  }
  
  // Mock classes for complex modules
  class ShaderCache {
    constructor(context: any, options?: any);
    precompileForSprite?(sprite: any): void;
    applyTransforms?(transforms: any): void;
  }
  
  class MatrixTransformLib {
    constructor(options?: any);
    generateCSSTransforms?(matrix: any): string;
  }
  
  class DockerResourceOptimizer {
    constructor(options?: any);
    cacheWithCompression?(data: any): void;
    dispose?(): void;
  }
}

// bits-ui module types
declare module 'bits-ui' {
  // Dialog exports
  export namespace Dialog {
    export const Root: any;
    export const Portal: any;
    export const Overlay: any;
    export const Content: any;
    export const Header: any;
    export const Title: any;
    export const Description: any;
    export const Close: any;
    export const Trigger: any;
  }

  // Select exports
  export namespace Select {
    export const Root: any;
    export const Trigger: any;
    export const Value: any;
    export const Content: any;
    export const Item: any;
    export const ItemText: any;
    export const ItemIndicator: any;
    export const Input: any;
    export const Group: any;  // Added missing Group export
    export const Label: any;
    export const Separator: any;
  }

  // Combobox exports
  export namespace Combobox {
    export const Root: any;
    export const Input: any;
    export const Trigger: any;
    export const Content: any;
    export const Item: any;
    export const ItemText: any;
    export const ItemIndicator: any;
    export const HiddenInput: any;
  }

  // DatePicker exports
  export namespace DatePicker {
    export const Root: any;
    export const Trigger: any;
    export const Content: any;
    export const Header: any;
    export const Heading: any;
    export const PrevButton: any;
    export const NextButton: any;
    export const Grid: any;
    export const GridHead: any;
    export const GridBody: any;
    export const GridRow: any;
    export const HeadCell: any;
    export const Cell: any;
    export const Day: any;
    export const TimeField: any;
    export const TimeSegment: any;
  }

  // Toast exports
  export namespace Toast {
    export const Provider: any;
    export const Root: any;
    export const Content: any;
    export const Title: any;
    export const Description: any;
    export const Action: any;
    export const Close: any;
    export const Viewport: any;
  }

  // Common types
  export interface CommonProps {
    class?: string;
    children?: any;
  }
}

// Fabric.js namespace exports
declare module 'fabric' {
  export namespace fabric {
    export class Canvas {
      constructor(element: HTMLCanvasElement | string, options?: any);
      add(object: any): Canvas;
      remove(object: any): Canvas;
      getObjects(): any[];
      clear(): Canvas;
      renderAll(): Canvas;
      toJSON(): any;
      loadFromJSON(json: any, callback?: () => void): void;
      getElement(): HTMLCanvasElement;
      getContext(): CanvasRenderingContext2D;
    }
    
    export class Object {
      constructor(options?: any);
      set(key: string, value: any): Object;
      get(key: string): any;
      toJSON(): any;
    }
    
    export class Circle extends Object {
      constructor(options?: any);
    }
    
    export class Line extends Object {
      constructor(points: number[], options?: any);
    }
    
    export class Text extends Object {
      constructor(text: string, options?: any);
    }
  }
}

// Export conflicts resolution
declare module '$lib/mcp-context72-get-library-docs' {
  export function resolveLibraryId(name: string): Promise<string>;
  export function getLibraryDocs(id: string, options?: any): Promise<any>;
}

declare module '$lib/utils' {
  export function fetchWithTimeout(url: string, options?: RequestInit & { timeout?: number }): Promise<Response>;
  export function cn(...classes: (string | undefined | null | boolean)[]): string;
}

declare module '$lib/utils/cn' {
  export function cn(...classes: (string | undefined | null | boolean)[]): string;
  export function legalCn(...classes: (string | undefined | null | boolean)[]): string;
  export function confidenceClass(confidence: number): string;
  export function priorityClass(priority: 'low' | 'medium' | 'high' | 'critical'): string;
}

// Path utility types
declare module 'path' {
  export function join(...paths: string[]): string;
  export function resolve(...paths: string[]): string;
  export function dirname(path: string): string;
  export function basename(path: string, ext?: string): string;
}

// Form schemas module
declare module '$lib/schemas/forms' {
  export const DocumentUploadSchema: any;
  export const CaseCreationSchema: any;
  export const SearchQuerySchema: any;
  export const AIAnalysisSchema: any;
}

// Database schema types
declare module '$lib/database/enhanced-schema' {
  export const vector: (name: string, options: { dimensions: number }) => any;
}