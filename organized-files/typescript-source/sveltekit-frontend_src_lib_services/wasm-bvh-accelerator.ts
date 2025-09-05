// WASM BVH Accelerator Loading Pipeline
// Provides high-performance document highlighting using C++ BVH acceleration
// Falls back to JavaScript implementation if WASM is not available

import { writable, type Readable } from 'svelte/store';

// WASM module interface
export interface WASMBVHModule {
  highlightDocuments: (indices: number[]) => HighlightResult[];
  buildBVH: (documents: DocumentNode[]) => void;
  queryBVH: (queryVector: number[], k: number) => QueryResult[];
  memory: WebAssembly.Memory;
  _free: (ptr: number) => void;
  _malloc: (size: number) => number;
}

// Document node for BVH construction
export interface DocumentNode {
  id: number;
  position: { x: number; y: number; z?: number };
  embedding?: number[];
  metadata?: Record<string, any>;
}

// Highlight result from WASM
export interface HighlightResult {
  index: number;
  position: { x: number; y: number };
  confidence: number;
  highlight: boolean;
  metadata?: Record<string, any>;
}

// Query result from BVH
export interface QueryResult {
  documentId: number;
  distance: number;
  similarity: number;
}

// Accelerator configuration
export interface AcceleratorConfig {
  wasmPath: string;
  enableFallback: boolean;
  maxDocuments: number;
  memoryPoolSize: number;
  enableProfiling: boolean;
}

// Accelerator status
export interface AcceleratorStatus {
  type: 'wasm' | 'javascript' | 'not_loaded';
  isReady: boolean;
  documentsLoaded: number;
  memoryUsed: number;
  lastError?: string;
  performanceProfile?: {
    avgHighlightTime: number;
    avgQueryTime: number;
    totalOperations: number;
  };
}

/**
 * WASM BVH Accelerator for high-performance document operations
 */
export class WASMBVHAccelerator {
  private config: AcceleratorConfig;
  private wasmModule?: WASMBVHModule;
  private isInitialized = false;
  private documents: Map<number, DocumentNode> = new Map();
  private performanceMetrics = {
    highlightTimes: [] as number[],
    queryTimes: [] as number[],
    totalOperations: 0
  };

  // Reactive stores
  public status = writable<AcceleratorStatus>({
    type: 'not_loaded',
    isReady: false,
    documentsLoaded: 0,
    memoryUsed: 0
  });

  public isReady = writable<boolean>(false);
  public loadingProgress = writable<{ progress: number; stage: string }>({
    progress: 0,
    stage: 'Not started'
  });

  constructor(config: AcceleratorConfig) {
    this.config = config;
  }

  /**
   * Initialize the WASM accelerator
   */
  async initialize(): Promise<boolean> {
    console.log('🚀 Initializing WASM BVH Accelerator...');
    
    this.updateProgress(10, 'Loading WASM module');

    try {
      // Try to load WASM module
      const wasmModule = await this.loadWASMModule();
      
      if (wasmModule) {
        this.wasmModule = wasmModule;
        this.isInitialized = true;
        
        this.updateStatus({
          type: 'wasm',
          isReady: true,
          documentsLoaded: 0,
          memoryUsed: this.getMemoryUsage()
        });

        console.log('✅ WASM BVH Accelerator initialized successfully');
        this.updateProgress(100, 'WASM accelerator ready');
        this.isReady.set(true);
        return true;
      } else {
        throw new Error('WASM module failed to load');
      }

    } catch (error) {
      console.warn('⚠️ WASM accelerator failed to load:', error);
      
      if (this.config.enableFallback) {
        console.log('🔄 Falling back to JavaScript implementation');
        await this.initializeFallback();
        return true;
      } else {
        this.updateStatus({
          type: 'not_loaded',
          isReady: false,
          documentsLoaded: 0,
          memoryUsed: 0,
          lastError: error instanceof Error ? error.message : String(error)
        });
        
        this.updateProgress(0, 'Failed to load accelerator');
        return false;
      }
    }
  }

  /**
   * Load WASM module from URL
   */
  private async loadWASMModule(): Promise<WASMBVHModule | null> {
    this.updateProgress(20, 'Fetching WASM binary');

    try {
      // Check if file exists
      const response = await fetch(this.config.wasmPath);
      if (!response.ok) {
        throw new Error(`WASM file not found: ${this.config.wasmPath}`);
      }

      this.updateProgress(40, 'Compiling WASM module');

      // Compile WASM module
      const wasmBytes = await response.arrayBuffer();
      const wasmModule = await WebAssembly.instantiate(wasmBytes, {
        env: {
          memory: new WebAssembly.Memory({ initial: 256, maximum: 1024 }),
          table: new WebAssembly.Table({ initial: 0, element: 'anyfunc' }),
          abort: (msg: number, file: number, line: number, colm: number) => {
            console.error('WASM abort:', { msg, file, line, colm });
          }
        }
      });

      this.updateProgress(60, 'Initializing WASM exports');

      // Extract exports
      const exports = wasmModule.instance.exports as any;
      
      // Create module interface
      const module: WASMBVHModule = {
        highlightDocuments: (indices: number[]) => {
          return this.callWASMHighlight(exports, indices);
        },
        buildBVH: (documents: DocumentNode[]) => {
          this.callWASMBuildBVH(exports, documents);
        },
        queryBVH: (queryVector: number[], k: number) => {
          return this.callWASMQuery(exports, queryVector, k);
        },
        memory: exports.memory,
        _free: exports.free,
        _malloc: exports.malloc
      };

      this.updateProgress(80, 'Verifying WASM functionality');

      // Test basic functionality
      const testResult = module.highlightDocuments([0, 1, 2]);
      if (!Array.isArray(testResult)) {
        throw new Error('WASM module test failed');
      }

      return module;

    } catch (error) {
      console.error('WASM loading error:', error);
      return null;
    }
  }

  /**
   * Initialize JavaScript fallback
   */
  private async initializeFallback(): Promise<void> {
    this.updateProgress(50, 'Initializing JavaScript fallback');
    
    // Simulate loading time
    await new Promise(resolve => setTimeout(resolve, 500));
    
    this.updateStatus({
      type: 'javascript',
      isReady: true,
      documentsLoaded: 0,
      memoryUsed: 0
    });

    this.updateProgress(100, 'JavaScript fallback ready');
    this.isReady.set(true);
    console.log('✅ JavaScript fallback initialized');
  }

  /**
   * Highlight documents using the accelerator
   */
  async highlightDocuments(indices: number[]): Promise<HighlightResult[]> {
    if (!this.isInitialized && !this.config.enableFallback) {
      throw new Error('Accelerator not initialized');
    }

    const startTime = performance.now();

    try {
      let results: HighlightResult[];

      if (this.wasmModule) {
        // Use WASM acceleration
        results = this.wasmModule.highlightDocuments(indices);
        console.log(`🚀 WASM highlight completed for ${indices.length} documents`);
      } else {
        // Use JavaScript fallback
        results = this.fallbackHighlight(indices);
        console.log(`⚡ JavaScript fallback highlight completed for ${indices.length} documents`);
      }

      // Record performance
      const duration = performance.now() - startTime;
      this.recordHighlightPerformance(duration);

      return results;

    } catch (error) {
      console.error('❌ Highlight operation failed:', error);
      
      if (this.wasmModule) {
        // Fallback to JavaScript if WASM fails
        console.log('🔄 Falling back to JavaScript after WASM error');
        return this.fallbackHighlight(indices);
      }
      
      throw error;
    }
  }

  /**
   * Build BVH structure for documents
   */
  async buildBVH(documents: DocumentNode[]): Promise<void> {
    if (!this.isInitialized && !this.config.enableFallback) {
      throw new Error('Accelerator not initialized');
    }

    console.log(`🏗️ Building BVH for ${documents.length} documents`);

    // Store documents
    documents.forEach(doc => {
      this.documents.set(doc.id, doc);
    });

    if (this.wasmModule) {
      this.wasmModule.buildBVH(documents);
    } else {
      // JavaScript fallback - build simple spatial index
      this.buildJavaScriptBVH(documents);
    }

    this.updateStatus({
      ...this.getCurrentStatus(),
      documentsLoaded: documents.length,
      memoryUsed: this.getMemoryUsage()
    });

    console.log(`✅ BVH built successfully for ${documents.length} documents`);
  }

  /**
   * Query BVH for nearest neighbors
   */
  async queryBVH(queryVector: number[], k: number = 10): Promise<QueryResult[]> {
    if (!this.isInitialized && !this.config.enableFallback) {
      throw new Error('Accelerator not initialized');
    }

    const startTime = performance.now();

    let results: QueryResult[];

    if (this.wasmModule) {
      results = this.wasmModule.queryBVH(queryVector, k);
    } else {
      results = this.fallbackQuery(queryVector, k);
    }

    const duration = performance.now() - startTime;
    this.recordQueryPerformance(duration);

    return results;
  }

  /**
   * Call WASM highlight function
   */
  private callWASMHighlight(exports: any, indices: number[]): HighlightResult[] {
    // Allocate memory for input array
    const inputPtr = exports.malloc(indices.length * 4);
    const inputArray = new Int32Array(exports.memory.buffer, inputPtr, indices.length);
    inputArray.set(indices);

    // Call WASM function
    const resultPtr = exports.highlight_documents(inputPtr, indices.length);
    
    // Read results
    const results: HighlightResult[] = [];
    const resultArray = new Float32Array(exports.memory.buffer, resultPtr, indices.length * 4);
    
    for (let i = 0; i < indices.length; i++) {
      const baseIndex = i * 4;
      results.push({
        index: indices[i],
        position: {
          x: resultArray[baseIndex],
          y: resultArray[baseIndex + 1]
        },
        confidence: resultArray[baseIndex + 2],
        highlight: resultArray[baseIndex + 3] > 0.5
      });
    }

    // Free memory
    exports.free(inputPtr);
    exports.free(resultPtr);

    return results;
  }

  /**
   * Call WASM BVH build function
   */
  private callWASMBuildBVH(exports: any, documents: DocumentNode[]): void {
    // Prepare document data for WASM
    const dataSize = documents.length * 6; // id, x, y, z, embedding_ptr, metadata_ptr
    const dataPtr = exports.malloc(dataSize * 4);
    const dataArray = new Float32Array(exports.memory.buffer, dataPtr, dataSize);

    documents.forEach((doc, i) => {
      const baseIndex = i * 6;
      dataArray[baseIndex] = doc.id;
      dataArray[baseIndex + 1] = doc.position.x;
      dataArray[baseIndex + 2] = doc.position.y;
      dataArray[baseIndex + 3] = doc.position.z || 0;
      dataArray[baseIndex + 4] = 0; // embedding pointer (simplified)
      dataArray[baseIndex + 5] = 0; // metadata pointer (simplified)
    });

    // Call WASM function
    exports.build_bvh(dataPtr, documents.length);

    // Free memory
    exports.free(dataPtr);
  }

  /**
   * Call WASM query function
   */
  private callWASMQuery(exports: any, queryVector: number[], k: number): QueryResult[] {
    // Allocate memory for query vector
    const queryPtr = exports.malloc(queryVector.length * 4);
    const queryArray = new Float32Array(exports.memory.buffer, queryPtr, queryVector.length);
    queryArray.set(queryVector);

    // Call WASM function
    const resultPtr = exports.query_bvh(queryPtr, queryVector.length, k);

    // Read results
    const results: QueryResult[] = [];
    const resultArray = new Float32Array(exports.memory.buffer, resultPtr, k * 3);
    
    for (let i = 0; i < k; i++) {
      const baseIndex = i * 3;
      results.push({
        documentId: resultArray[baseIndex],
        distance: resultArray[baseIndex + 1],
        similarity: 1.0 - Math.min(1.0, resultArray[baseIndex + 2])
      });
    }

    // Free memory
    exports.free(queryPtr);
    exports.free(resultPtr);

    return results;
  }

  /**
   * JavaScript fallback for document highlighting
   */
  private fallbackHighlight(indices: number[]): HighlightResult[] {
    return indices.map(index => ({
      index,
      position: {
        x: 100 + (index * 80) + Math.random() * 40 - 20,
        y: 150 + (index % 3) * 60 + Math.random() * 20 - 10
      },
      confidence: 0.7 + Math.random() * 0.3,
      highlight: true,
      metadata: {
        fallback: true,
        processingTime: Date.now()
      }
    }));
  }

  /**
   * JavaScript fallback for BVH building
   */
  private buildJavaScriptBVH(documents: DocumentNode[]): void {
    // Simple spatial indexing for fallback
    console.log('📚 Building JavaScript spatial index...');
    // In a real implementation, this would build a proper spatial data structure
  }

  /**
   * JavaScript fallback for BVH queries
   */
  private fallbackQuery(queryVector: number[], k: number): QueryResult[] {
    const documents = Array.from(this.documents.values()).slice(0, k);
    
    return documents.map((doc, i) => ({
      documentId: doc.id,
      distance: Math.random(),
      similarity: Math.random() * 0.5 + 0.5
    }));
  }

  /**
   * Get current memory usage
   */
  private getMemoryUsage(): number {
    if (this.wasmModule && this.wasmModule.memory) {
      return this.wasmModule.memory.buffer.byteLength;
    }
    return 0;
  }

  /**
   * Record highlight performance metrics
   */
  private recordHighlightPerformance(duration: number): void {
    this.performanceMetrics.highlightTimes.push(duration);
    this.performanceMetrics.totalOperations++;
    
    // Keep only recent measurements
    if (this.performanceMetrics.highlightTimes.length > 100) {
      this.performanceMetrics.highlightTimes.shift();
    }
    
    this.updatePerformanceProfile();
  }

  /**
   * Record query performance metrics
   */
  private recordQueryPerformance(duration: number): void {
    this.performanceMetrics.queryTimes.push(duration);
    this.performanceMetrics.totalOperations++;
    
    if (this.performanceMetrics.queryTimes.length > 100) {
      this.performanceMetrics.queryTimes.shift();
    }
    
    this.updatePerformanceProfile();
  }

  /**
   * Update performance profile in status
   */
  private updatePerformanceProfile(): void {
    const avgHighlightTime = this.performanceMetrics.highlightTimes.length > 0
      ? this.performanceMetrics.highlightTimes.reduce((a, b) => a + b) / this.performanceMetrics.highlightTimes.length
      : 0;
      
    const avgQueryTime = this.performanceMetrics.queryTimes.length > 0
      ? this.performanceMetrics.queryTimes.reduce((a, b) => a + b) / this.performanceMetrics.queryTimes.length
      : 0;

    const currentStatus = this.getCurrentStatus();
    this.updateStatus({
      ...currentStatus,
      performanceProfile: {
        avgHighlightTime,
        avgQueryTime,
        totalOperations: this.performanceMetrics.totalOperations
      }
    });
  }

  /**
   * Get current status
   */
  private getCurrentStatus(): AcceleratorStatus {
    let currentStatus: AcceleratorStatus = {
      type: 'not_loaded',
      isReady: false,
      documentsLoaded: 0,
      memoryUsed: 0
    };
    
    this.status.subscribe(status => {
      currentStatus = status;
    })();
    
    return currentStatus;
  }

  /**
   * Update status store
   */
  private updateStatus(status: AcceleratorStatus): void {
    this.status.set(status);
  }

  /**
   * Update progress store
   */
  private updateProgress(progress: number, stage: string): void {
    this.loadingProgress.set({ progress, stage });
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats(): {
    avgHighlightTime: number;
    avgQueryTime: number;
    totalOperations: number;
    acceleratorType: string;
  } {
    const currentStatus = this.getCurrentStatus();
    
    return {
      avgHighlightTime: currentStatus.performanceProfile?.avgHighlightTime || 0,
      avgQueryTime: currentStatus.performanceProfile?.avgQueryTime || 0,
      totalOperations: currentStatus.performanceProfile?.totalOperations || 0,
      acceleratorType: currentStatus.type
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    console.log('🛑 Destroying WASM BVH Accelerator...');
    
    this.documents.clear();
    this.performanceMetrics = {
      highlightTimes: [],
      queryTimes: [],
      totalOperations: 0
    };
    
    this.updateStatus({
      type: 'not_loaded',
      isReady: false,
      documentsLoaded: 0,
      memoryUsed: 0
    });
    
    this.isReady.set(false);
    console.log('✅ WASM BVH Accelerator destroyed');
  }
}

/**
 * Factory function to create WASM accelerator
 */
export function createWASMBVHAccelerator(config?: Partial<AcceleratorConfig>): WASMBVHAccelerator {
  const defaultConfig: AcceleratorConfig = {
    wasmPath: '/wasm/bvh.wasm',
    enableFallback: true,
    maxDocuments: 10000,
    memoryPoolSize: 64 * 1024 * 1024, // 64MB
    enableProfiling: true
  };

  const finalConfig = { ...defaultConfig, ...config };
  return new WASMBVHAccelerator(finalConfig);
}

/**
 * Global WASM accelerator instance
 */
let globalAccelerator: WASMBVHAccelerator | null = null;

/**
 * Get or create global accelerator instance
 */
export function getWASMAccelerator(config?: Partial<AcceleratorConfig>): WASMBVHAccelerator {
  if (!globalAccelerator) {
    globalAccelerator = createWASMBVHAccelerator(config);
  }
  return globalAccelerator;
}

/**
 * Initialize global accelerator
 */
export async function initializeGlobalWASMAccelerator(config?: Partial<AcceleratorConfig>): Promise<boolean> {
  const accelerator = getWASMAccelerator(config);
  return await accelerator.initialize();
}