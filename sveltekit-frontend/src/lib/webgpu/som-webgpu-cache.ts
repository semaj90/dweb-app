// 🚀 WebGPU-Accelerated SOM Semantic Cache
// Real-time PageRank with loki.js-style IndexDB integration

import Loki from 'lokijs';
// LokiJS types may not be available; use loose typing for collections
type Collection<T> = any;

export interface NPMError {
  message: string;
  file: string;
  line: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  type: string;
  timestamp: string;
  context?: string[];
  dependencies?: string[];
}

export interface IntelligentTodo {
  id: string;
  priority: number;
  category: string;
  title: string;
  description: string;
  estimated_effort: number; // nanoseconds
  dependencies: string[];
  suggested_fixes: string[];
  related_errors: NPMError[];
  confidence: number;
  tags: string[];
  created_at: string;
  metadata: Record<string, any>;
}

// Respect environment flag to enable/disable WebGPU features in dev
const ENABLE_GPU = (() => {
  try {
    const v = process?.env?.ENABLE_GPU;
    if (typeof v === 'string') return v.toLowerCase() !== 'false' && v !== '0';
  } catch (e: any) { }
  return true;
})();

export class WebGPUSOMCache {
  private device: GPUDevice | null = null;
  private lokiDB: Loki;
  private indexDB: IDBDatabase | null = null;
  private todosCollection: Collection<any>;
  private errorsCollection: Collection<any>;
  private cacheCollection: Collection<any>;

  // WebGPU compute shaders for semantic operations
  private similarityShader = `
    @group(0) @binding(0) var<storage, read> query_vector: array<f32>;
    @group(0) @binding(1) var<storage, read> document_vectors: array<f32>;
    @group(0) @binding(2) var<storage, read_write> similarities: array<f32>;
    @group(0) @binding(3) var<uniform> metadata: array<u32, 4>; // [vector_dim, num_docs, 0, 0]

    @compute @workgroup_size(64)
    fn compute_similarity(@builtin(global_invocation_id) id: vec3<u32>) {
      let doc_id = id.x;
      let vector_dim = metadata[0];
      let num_docs = metadata[1];

      if (doc_id >= num_docs) { return; }

      var dot_product = 0.0;
      var query_norm = 0.0;
      var doc_norm = 0.0;

      for (var i = 0u; i < vector_dim; i++) {
        let q_val = query_vector[i];
        let d_val = document_vectors[doc_id * vector_dim + i];

        dot_product += q_val * d_val;
        query_norm += q_val * q_val;
        doc_norm += d_val * d_val;
      }

      let cosine_sim = dot_product / (sqrt(query_norm) * sqrt(doc_norm));
      similarities[doc_id] = cosine_sim;
    }
  `;

  private pageRankShader = `
    @group(0) @binding(0) var<storage, read> adjacency_matrix: array<f32>;
    @group(0) @binding(1) var<storage, read_write> pagerank_scores: array<f32>;
    @group(0) @binding(2) var<storage, read_write> new_scores: array<f32>;
    @group(0) @binding(3) var<uniform> params: array<f32, 4>; // [num_nodes, damping, 0, 0]

    @compute @workgroup_size(64)
    fn pagerank_iteration(@builtin(global_invocation_id) id: vec3<u32>) {
      let node_id = id.x;
      let num_nodes = u32(params[0]);
      let damping = params[1];

      if (node_id >= num_nodes) { return; }

      var rank_sum = 0.0;
      for (var i = 0u; i < num_nodes; i++) {
        let edge_weight = adjacency_matrix[i * num_nodes + node_id];
        if (edge_weight > 0.0) {
          // Calculate out-degree
          var out_degree = 0.0;
          for (var j = 0u; j < num_nodes; j++) {
            out_degree += adjacency_matrix[i * num_nodes + j];
          }
          if (out_degree > 0.0) {
            rank_sum += pagerank_scores[i] * edge_weight / out_degree;
          }
        }
      }

      new_scores[node_id] = (1.0 - damping) / f32(num_nodes) + damping * rank_sum;
    }
  `;

  private errorEmbeddingShader = `
    @group(0) @binding(0) var<storage, read> error_text: array<u32>; // Encoded text
    @group(0) @binding(1) var<storage, read_write> embeddings: array<f32>;
    @group(0) @binding(2) var<uniform> config: array<u32, 4>; // [text_length, embedding_dim, 0, 0]

    @compute @workgroup_size(32)
    fn compute_error_embedding(@builtin(global_invocation_id) id: vec3<u32>) {
      let embedding_id = id.x;
      let text_length = config[0];
      let embedding_dim = config[1];

      if (embedding_id >= embedding_dim) { return; }

      var value = 0.0;

      // Simple bag-of-words embedding with positional encoding
      for (var i = 0u; i < text_length; i++) {
        let char_code = error_text[i];
        let position_weight = 1.0 / (1.0 + f32(i) * 0.1);
        let char_contribution = f32(char_code) / 255.0 * position_weight;

        // Hash to embedding dimension
        let hash = (char_code * 17u + i * 31u) % embedding_dim;
        if (hash == embedding_id) {
          value += char_contribution;
        }
      }

      // Normalize
      embeddings[embedding_id] = tanh(value);
    }
  `;

  constructor() {
    this.lokiDB = new Loki('som-cache.db', {
      autoload: true,
      autoloadCallback: () => this.initializeCollections(),
      autosave: true,
      autosaveInterval: 4000,
    });
  }

  private initializeCollections() {
    this.todosCollection =
      this.lokiDB.getCollection('todos') ||
      this.lokiDB.addCollection('todos', {
        indices: ['priority', 'category', 'confidence'],
        unique: ['id'],
      });

    this.errorsCollection =
      this.lokiDB.getCollection('errors') ||
      this.lokiDB.addCollection('errors', {
        indices: ['severity', 'category', 'file'],
        unique: ['id'],
      });

    this.cacheCollection =
      this.lokiDB.getCollection('cache') ||
      this.lokiDB.addCollection('cache', {
        indices: ['key', 'timestamp'],
        ttl: 300000, // 5 minutes
      });
  }

  async initializeWebGPU(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.warn('WebGPU not supported, falling back to CPU processing');
        return false;
      }

      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!adapter) {
        console.warn('No WebGPU adapter found');
        return false;
      }

      this.device = await adapter.requestDevice({
        requiredFeatures: ['shader-f16'] as GPUFeatureName[],
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        },
      });

      console.log('🚀 WebGPU initialized for SOM semantic caching');
      return true;
    } catch (error: any) {
      console.error('WebGPU initialization failed:', error);
      return false;
    }
  }

  async initializeIndexDB(): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('SOMSemanticCache', 1);

      request.onerror = () => {
        console.error('IndexDB initialization failed');
        resolve(false);
      };

      request.onsuccess = (event: any) => {
        this.indexDB = (event.target as IDBOpenDBRequest).result;
        console.log('📄 IndexDB initialized for persistent caching');
        resolve(true);
      };

      request.onupgradeneeded = (event: any) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create todos store
        if (!db.objectStoreNames.contains('todos')) {
          const todosStore = db.createObjectStore('todos', { keyPath: 'id' });
          todosStore.createIndex('priority', 'priority', { unique: false });
          todosStore.createIndex('category', 'category', { unique: false });
          todosStore.createIndex('timestamp', 'created_at', { unique: false });
        }

        // Create errors store
        if (!db.objectStoreNames.contains('errors')) {
          const errorsStore = db.createObjectStore('errors', { keyPath: 'id' });
          errorsStore.createIndex('severity', 'severity', { unique: false });
          errorsStore.createIndex('file', 'file', { unique: false });
        }

        // Create cache store
        if (!db.objectStoreNames.contains('cache')) {
          const cacheStore = db.createObjectStore('cache', { keyPath: 'key' });
          cacheStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };
    });
  }

  async processNPMCheckErrors(npmOutput: string): Promise<IntelligentTodo[]> {
    // Check cache first
    const cacheKey = this.generateCacheKey(npmOutput);
    const cached = this.getCachedResult(cacheKey);
    if (cached) {
      console.log('📋 Retrieved cached SOM analysis');
      return cached;
    }

    // Extract errors from npm output
    const errors = this.parseNPMErrors(npmOutput);

    // Generate embeddings using WebGPU if available
    const errorEmbeddings = this.device
      ? await this.computeErrorEmbeddingsGPU(errors)
      : this.computeErrorEmbeddingsCPU(errors);

    // Send to Go SOM analyzer (simulated - replace with actual HTTP call)
    const intelligentTodos = await this.callGoSOMAnalyzer(errors);

    // Apply WebGPU PageRank refinement
    const rankedTodos = this.device
      ? await this.refineRankingWithWebGPU(intelligentTodos)
      : intelligentTodos;

    // Cache results
    this.cacheResult(cacheKey, rankedTodos);

    // Store in IndexDB for persistence
    await this.persistTodos(rankedTodos);

    return rankedTodos;
  }

  private parseNPMErrors(npmOutput: string): NPMError[] {
    const errors: NPMError[] = [];
    const lines = npmOutput.split('\n');

    for (const line of lines) {
      if (line.includes('error') || line.includes('Error')) {
        // Parse TypeScript-style errors
        const match = line.match(/(.+\.ts)\((\d+),\d+\): (.+)/);
        if (match) {
          errors.push({
            message: match[3],
            file: match[1],
            line: parseInt(match[2]),
            severity: this.determineSeverity(match[3]),
            category: this.determineCategory(match[3]),
            type: 'error',
            timestamp: new Date().toISOString(),
            context: [line],
          });
        }
      }
    }

    return errors;
  }

  private determineSeverity(message: string): 'low' | 'medium' | 'high' | 'critical' {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('critical') || lowerMessage.includes('fatal')) return 'critical';
    if (lowerMessage.includes('error') || lowerMessage.includes('cannot')) return 'high';
    if (lowerMessage.includes('warning') || lowerMessage.includes('deprecated')) return 'medium';
    return 'low';
  }

  private determineCategory(message: string): string {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('type') || lowerMessage.includes('property')) return 'typescript';
    if (lowerMessage.includes('import') || lowerMessage.includes('module')) return 'import';
    if (lowerMessage.includes('syntax') || lowerMessage.includes('unexpected')) return 'syntax';
    if (lowerMessage.includes('service') || lowerMessage.includes('fetch')) return 'service';
    if (lowerMessage.includes('build') || lowerMessage.includes('compile')) return 'build';
    return 'general';
  }

  private async computeErrorEmbeddingsGPU(errors: NPMError[]): Promise<Float32Array[]> {
    if (!this.device) return [];

    const embeddings: Float32Array[] = [];
    const embeddingDim = 128;

    // Create compute pipeline
    const shaderModule = this.device.createShaderModule({
      code: this.errorEmbeddingShader,
    });

    const computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'compute_error_embedding',
      },
    });

    for (const error of errors) {
      const textData = new TextEncoder().encode(error.message);
      const paddedText = new Uint32Array(256); // Fixed size
      for (let i = 0; i < Math.min(textData.length, 256); i++) {
        paddedText[i] = textData[i];
      }

      // Create buffers
      const textBuffer = this.device.createBuffer({
        size: paddedText.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const embeddingBuffer = this.device.createBuffer({
        size: embeddingDim * 4, // 4 bytes per float32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const configBuffer = this.device.createBuffer({
        size: 16, // 4 uint32s
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const resultBuffer = this.device.createBuffer({
        size: embeddingDim * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      // Write data to buffers
      this.device.queue.writeBuffer(textBuffer, 0, paddedText);
      this.device.queue.writeBuffer(
        configBuffer,
        0,
        new Uint32Array([textData.length, embeddingDim, 0, 0])
      );

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: textBuffer } },
          { binding: 1, resource: { buffer: embeddingBuffer } },
          { binding: 2, resource: { buffer: configBuffer } },
        ],
      });

      // Run compute
      const encoder = this.device.createCommandEncoder();
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(Math.ceil(embeddingDim / 32));
      computePass.end();

      encoder.copyBufferToBuffer(embeddingBuffer, 0, resultBuffer, 0, embeddingDim * 4);
      this.device.queue.submit([encoder.finish()]);

      // Read result
      await resultBuffer.mapAsync(GPUMapMode.READ);
      const embedding = new Float32Array(resultBuffer.getMappedRange());
      embeddings.push(embedding.slice());
      resultBuffer.unmap();

      // Cleanup
      textBuffer.destroy();
      embeddingBuffer.destroy();
      configBuffer.destroy();
      resultBuffer.destroy();
    }

    return embeddings;
  }

  private computeErrorEmbeddingsCPU(errors: NPMError[]): Float32Array[] {
    // Fallback CPU implementation
    return errors.map((error) => {
      const embedding = new Float32Array(128);
      const text = error.message.toLowerCase();

      for (let i = 0; i < text.length && i < 128; i++) {
        embedding[i] = text.charCodeAt(i) / 255.0;
      }

      return embedding;
    });
  }

  private async callGoSOMAnalyzer(errors: NPMError[]): Promise<IntelligentTodo[]> {
    // In a real implementation, this would call the Go SOM analyzer
    // For now, simulate the response structure
    try {
      const response = await fetch('http://localhost:8080/api/som/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ errors }),
      });

      if (!response.ok) {
        throw new Error(`SOM analyzer failed: ${response.status}`);
      }

      return await response.json();
    } catch (error: any) {
      console.warn('Go SOM analyzer unavailable, using mock data');
      return this.generateMockTodos(errors);
    }
  }

  private generateMockTodos(errors: NPMError[]): IntelligentTodo[] {
    // Generate mock intelligent todos based on the SOM analyzer output format
    const categories = new Map<string, NPMError[]>();

    errors.forEach((error) => {
      if (!categories.has(error.category)) {
        categories.set(error.category, []);
      }
      categories.get(error.category)!.push(error);
    });

    const todos: IntelligentTodo[] = [];
    let todoId = 0;

    categories.forEach((categoryErrors, category) => {
      const severity = categoryErrors.reduce(
        (max, error) =>
          this.getSeverityWeight(error.severity) > this.getSeverityWeight(max)
            ? error.severity
            : max,
        'low'
      );

      todos.push({
        id: `mock-todo-${todoId++}`,
        priority: this.getSeverityWeight(severity) + Math.random() * 0.1,
        category,
        title: `Fix ${categoryErrors.length} ${category} ${categoryErrors.length === 1 ? 'error' : 'errors'}`,
        description: `Address ${category} issues in ${new Set(categoryErrors.map((e: any) => e.file)).size} files`,
        estimated_effort: categoryErrors.length * 15 * 60 * 1000000000, // 15 minutes per error in nanoseconds
        dependencies: [],
        suggested_fixes: this.generateSuggestedFixes(category),
        related_errors: categoryErrors,
        confidence: 0.8 + Math.random() * 0.2,
        tags: [category, severity],
        created_at: new Date().toISOString(),
        metadata: {
          error_count: categoryErrors.length,
          files_affected: new Set(categoryErrors.map((e: any) => e.file)).size,
        },
      });
    });

    return todos.sort((a, b) => b.priority - a.priority);
  }

  private getSeverityWeight(severity: string): number {
    const weights = { critical: 1.0, high: 0.8, medium: 0.5, low: 0.2 };
    return weights[severity] || 0.2;
  }

  private generateSuggestedFixes(category: string): string[] {
    const fixes = {
      typescript: [
        'Add missing type declarations',
        'Fix import statements',
        'Update tsconfig.json',
      ],
      import: ['Check module paths', 'Install missing dependencies', 'Update import syntax'],
      syntax: ['Fix syntax errors', 'Check parentheses and brackets', 'Review code formatting'],
      service: ['Check service connectivity', 'Verify configuration', 'Restart services'],
      build: ['Clear build cache', 'Update dependencies', 'Check build configuration'],
      general: ['Review error messages', 'Check documentation', 'Apply standard fixes'],
    };
    return fixes[category] || fixes.general;
  }

  private async refineRankingWithWebGPU(todos: IntelligentTodo[]): Promise<IntelligentTodo[]> {
    if (!this.device || todos.length === 0) return todos;

    const numNodes = todos.length;
    const adjacencyMatrix = new Float32Array(numNodes * numNodes);

    // Build adjacency matrix based on todo relationships
    for (let i = 0; i < numNodes; i++) {
      for (let j = 0; j < numNodes; j++) {
        if (i !== j) {
          const similarity = this.calculateTodoSimilarity(todos[i], todos[j]);
          adjacencyMatrix[i * numNodes + j] = similarity;
        }
      }
    }

    // Initial PageRank scores
    const pageRankScores = new Float32Array(numNodes);
    pageRankScores.fill(1.0 / numNodes);

    // Create WebGPU resources
    const shaderModule = this.device.createShaderModule({
      code: this.pageRankShader,
    });

    const computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'pagerank_iteration',
      },
    });

    // Create buffers
    const adjacencyBuffer = this.device.createBuffer({
      size: adjacencyMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const scoresBuffer = this.device.createBuffer({
      size: pageRankScores.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const newScoresBuffer = this.device.createBuffer({
      size: pageRankScores.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const resultBuffer = this.device.createBuffer({
      size: pageRankScores.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Write initial data
    this.device.queue.writeBuffer(adjacencyBuffer, 0, adjacencyMatrix);
    this.device.queue.writeBuffer(scoresBuffer, 0, pageRankScores);
    this.device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([numNodes, 0.85, 0, 0]));

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: adjacencyBuffer } },
        { binding: 1, resource: { buffer: scoresBuffer } },
        { binding: 2, resource: { buffer: newScoresBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    // Run PageRank iterations
    for (let iter = 0; iter < 20; iter++) {
      const encoder = this.device.createCommandEncoder();
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(Math.ceil(numNodes / 64));
      computePass.end();

      // Copy new scores back to current scores
      encoder.copyBufferToBuffer(newScoresBuffer, 0, scoresBuffer, 0, pageRankScores.byteLength);
      this.device.queue.submit([encoder.finish()]);

      await this.device.queue.onSubmittedWorkDone();
    }

    // Read final results
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(scoresBuffer, 0, resultBuffer, 0, pageRankScores.byteLength);
    this.device.queue.submit([encoder.finish()]);

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const finalScores = new Float32Array(resultBuffer.getMappedRange());

    // Apply refined scores to todos
    const refinedTodos = todos.map((todo, index) => ({
      ...todo,
      priority: finalScores[index] * 0.3 + todo.priority * 0.7, // Blend WebGPU ranking with original
    }));

    resultBuffer.unmap();

    // Cleanup
    adjacencyBuffer.destroy();
    scoresBuffer.destroy();
    newScoresBuffer.destroy();
    paramsBuffer.destroy();
    resultBuffer.destroy();

    return refinedTodos.sort((a, b) => b.priority - a.priority);
  }

  private calculateTodoSimilarity(todo1: IntelligentTodo, todo2: IntelligentTodo): number {
    let similarity = 0;

    // Category similarity
    if (todo1.category === todo2.category) similarity += 0.4;

    // Tag overlap
    const tags1 = new Set(todo1.tags);
    const tags2 = new Set(todo2.tags);
    const tagIntersection = new Set([...tags1].filter((x) => tags2.has(x)));
    const tagUnion = new Set([...tags1, ...tags2]);
    if (tagUnion.size > 0) {
      similarity += 0.3 * (tagIntersection.size / tagUnion.size);
    }

    // File overlap in related errors
    const files1 = new Set(todo1.related_errors.map((e: any) => e.file));
    const files2 = new Set(todo2.related_errors.map((e: any) => e.file));
    const fileIntersection = new Set([...files1].filter((x) => files2.has(x)));
    if (files1.size > 0 || files2.size > 0) {
      similarity += 0.3 * (fileIntersection.size / Math.max(files1.size, files2.size));
    }

    return Math.min(similarity, 1.0);
  }

  private generateCacheKey(input: string): string {
    // Simple hash function for cache key
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `som_analysis_${Math.abs(hash)}`;
  }

  private getCachedResult(key: string): IntelligentTodo[] | null {
    const cached = this.cacheCollection.findOne({ key });
    if (cached && Date.now() - cached.timestamp < 300000) {
      // 5 minutes
      return cached.result;
    }
    return null;
  }

  private cacheResult(key: string, result: IntelligentTodo[]): void {
    this.cacheCollection.removeWhere({ key });
    this.cacheCollection.insert({
      key,
      result,
      timestamp: Date.now(),
    });
  }

  private async persistTodos(todos: IntelligentTodo[]): Promise<void> {
    if (!this.indexDB) return;

    const transaction = this.indexDB.transaction(['todos'], 'readwrite');
    const store = transaction.objectStore('todos');

    for (const todo of todos) {
      await store.put(todo);
    }

    // Also update LokiJS
    todos.forEach((todo) => {
      this.todosCollection.removeWhere({ id: todo.id });
      this.todosCollection.insert(todo);
    });
  }

  async getTodosByPriority(minPriority = 0): Promise<IntelligentTodo[]> {
    return this.todosCollection
      .find({ priority: { $gte: minPriority } })
      .sort((todo) => -todo.priority);
  }

  async getTodosByCategory(category: string): Promise<IntelligentTodo[]> {
    return this.todosCollection.find({ category }).sort((todo) => -todo.priority);
  }

  async clearCache(): Promise<void> {
    this.cacheCollection.removeDataOnly();

    if (this.indexDB) {
      const transaction = this.indexDB.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      await store.clear();
    }
  }

  dispose(): void {
    this.lokiDB.close();
    if (this.indexDB) {
      this.indexDB.close();
    }
  }
}

// Usage example
export async function initializeSOMCache(): Promise<WebGPUSOMCache> {
  const cache = new WebGPUSOMCache();

  const webGPUEnabled = await cache.initializeWebGPU();
  const indexDBEnabled = await cache.initializeIndexDB();

  console.log(`🧠 SOM Cache initialized - WebGPU: ${webGPUEnabled}, IndexDB: ${indexDBEnabled}`);

  return cache;
}

// Export for use in Svelte components
// Class is already exported above; no duplicate export
