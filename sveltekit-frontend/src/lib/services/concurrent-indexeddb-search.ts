import Fuse from '$lib/utils/fuse-import';

interface SearchableDocument {
  id: string;
  content: string;
  path: string;
  type: 'error' | 'component' | 'api' | 'config';
  metadata: {
    language: string;
    lastModified: number;
    size: number;
    embedding?: number[];
  };
}

interface SearchRequest {
  query: string;
  filters?: {
    type?: string[];
    language?: string[];
    dateRange?: [number, number];
  };
  options?: {
    threshold?: number;
    maxResults?: number;
    includeEmbeddings?: boolean;
  };
}

interface SearchWorkerMessage {
  type: 'search' | 'index' | 'clear';
  data: any;
  workerId: string;
}

export class ConcurrentIndexedDBSearch {
  private db: IDBDatabase | null = null;
  private fuse: Fuse<SearchableDocument> | null = null;
  private workers: Worker[] = [];
  private workerPool: number = 4;
  private isInitialized = false;
  private documents: SearchableDocument[] = [];

  constructor() {
    if (typeof window !== 'undefined') {
      this.initializeWorkerPool();
    }
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      if (typeof window !== 'undefined') {
        await this.openDatabase();
        await this.loadDocuments();
        this.initializeFuse();
        if (this.workers.length === 0) {
          this.initializeWorkerPool();
        }
      }
      this.isInitialized = true;
      console.log('✅ Concurrent IndexedDB Search initialized');
    } catch (error) {
      console.error('❌ Failed to initialize IndexedDB search:', error);
      throw error;
    }
  }

  private async openDatabase(): Promise<void> {
    if (typeof window === 'undefined') {
      console.log('⚠️ IndexedDB not available in SSR mode');
      return;
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open('LegalAISearchDB', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        if (!db.objectStoreNames.contains('documents')) {
          const store = db.createObjectStore('documents', { keyPath: 'id' });
          store.createIndex('type', 'type', { unique: false });
          store.createIndex('language', 'metadata.language', { unique: false });
          store.createIndex('lastModified', 'metadata.lastModified', { unique: false });
        }

        if (!db.objectStoreNames.contains('embeddings')) {
          const embeddingStore = db.createObjectStore('embeddings', { keyPath: 'id' });
          embeddingStore.createIndex('documentId', 'documentId', { unique: false });
        }
      };
    });
  }

  private async loadDocuments(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['documents'], 'readonly');
      const store = transaction.objectStore('documents');
      const request = store.getAll();

      request.onsuccess = () => {
        this.documents = request.result;
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  private initializeFuse(): void {
    const fuseOptions = {
      keys: [
        'content',
        'path',
        'type',
        'metadata.language'
      ],
      threshold: 0.3,
      includeScore: true,
      includeMatches: true,
      shouldSort: true,
      fieldNormWeight: 1.0,
      distance: 100,
      location: 0
    };

    this.fuse = new Fuse(this.documents, fuseOptions);
  }

  private initializeWorkerPool(): void {
    const workerCode = `
      self.onmessage = function(e) {
        const { type, data, workerId } = e.data;
        
        switch (type) {
          case 'search':
            performSearch(data, workerId);
            break;
          case 'index':
            updateIndex(data, workerId);
            break;
          case 'clear':
            clearCache(workerId);
            break;
        }
      };

      function performSearch(searchData, workerId) {
        const { query, documents, options } = searchData;
        
        try {
          const startTime = performance.now();
          const results = searchInDocuments(query, documents, options);
          const endTime = performance.now();
          
          self.postMessage({
            workerId,
            type: 'searchResult',
            data: {
              results,
              processingTime: endTime - startTime,
              documentCount: documents.length
            }
          });
        } catch (error) {
          self.postMessage({
            workerId,
            type: 'error',
            data: { error: error.message }
          });
        }
      }

      function searchInDocuments(query, documents, options) {
        const threshold = options.threshold || 0.3;
        const maxResults = options.maxResults || 50;
        
        return documents
          .map((doc, index) => {
            let score = 1;
            const lowerQuery = query.toLowerCase();
            
            if (doc.content.toLowerCase().includes(lowerQuery)) {
              score = 0.1;
            } else if (doc.path.toLowerCase().includes(lowerQuery)) {
              score = 0.3;
            } else if (doc.type.toLowerCase().includes(lowerQuery)) {
              score = 0.5;
            }
            
            return { item: doc, refIndex: index, score };
          })
          .filter(result => result.score <= threshold)
          .sort((a, b) => a.score - b.score)
          .slice(0, maxResults);
      }

      function updateIndex(indexData, workerId) {
        self.postMessage({
          workerId,
          type: 'indexUpdated',
          data: { success: true, documentsIndexed: indexData.length }
        });
      }

      function clearCache(workerId) {
        self.postMessage({
          workerId,
          type: 'cacheCleared',
          data: { success: true }
        });
      }
    `;

    const workerBlob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(workerBlob);

    for (let i = 0; i < this.workerPool; i++) {
      const worker = new Worker(workerUrl);
      worker.onmessage = this.handleWorkerMessage.bind(this);
      this.workers.push(worker);
    }

    URL.revokeObjectURL(workerUrl);
  }

  private handleWorkerMessage(event: MessageEvent): void {
    const { workerId, type, data } = event.data;
    
    switch (type) {
      case 'searchResult':
        this.handleSearchResult(workerId, data);
        break;
      case 'indexUpdated':
        console.log(`✅ Worker ${workerId} indexed ${data.documentsIndexed} documents`);
        break;
      case 'error':
        console.error(`❌ Worker ${workerId} error:`, data.error);
        break;
    }
  }

  private handleSearchResult(workerId: string, data: any): void {
    console.log(`🔍 Worker ${workerId} search completed in ${data.processingTime.toFixed(2)}ms`);
    console.log(`📊 Processed ${data.documentCount} documents, found ${data.results.length} matches`);
  }

  async search(request: SearchRequest): Promise<SearchableDocument[]> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const startTime = performance.now();
      
      let results: SearchableDocument[] = [];

      if (this.fuse) {
        const fuseResults = this.fuse.search(request.query);
        results = fuseResults.map(result => result.item);
      } else {
        results = await this.performWorkerSearch(request);
      }

      if (request.filters) {
        results = this.applyFilters(results, request.filters);
      }

      const maxResults = request.options?.maxResults || 50;
      const finalResults = results.slice(0, maxResults);

      const endTime = performance.now();
      console.log(`🎯 Search completed in ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`📊 Found ${finalResults.length} results for query: "${request.query}"`);

      return finalResults;
    } catch (error) {
      console.error('❌ Search error:', error);
      return [];
    }
  }

  private async performWorkerSearch(request: SearchRequest): Promise<SearchableDocument[]> {
    const documentsPerWorker = Math.ceil(this.documents.length / this.workerPool);
    const searchPromises: Promise<SearchableDocument[]>[] = [];

    for (let i = 0; i < this.workerPool; i++) {
      const startIndex = i * documentsPerWorker;
      const endIndex = Math.min(startIndex + documentsPerWorker, this.documents.length);
      const workerDocuments = this.documents.slice(startIndex, endIndex);

      if (workerDocuments.length > 0) {
        const promise = this.searchWithWorker(i, request.query, workerDocuments, request.options);
        searchPromises.push(promise);
      }
    }

    const workerResults = await Promise.all(searchPromises);
    return workerResults.flat();
  }

  private searchWithWorker(
    workerIndex: number,
    query: string,
    documents: SearchableDocument[],
    options?: SearchRequest['options']
  ): Promise<SearchableDocument[]> {
    return new Promise((resolve) => {
      const worker = this.workers[workerIndex];
      const workerId = `worker-${workerIndex}`;

      const messageHandler = (event: MessageEvent) => {
        if (event.data.workerId === workerId && event.data.type === 'searchResult') {
          worker.removeEventListener('message', messageHandler);
          resolve(event.data.data.results.map((r: any) => r.item));
        }
      };

      worker.addEventListener('message', messageHandler);

      worker.postMessage({
        type: 'search',
        data: { query, documents, options },
        workerId
      });
    });
  }

  private applyFilters(documents: SearchableDocument[], filters: SearchRequest['filters']): SearchableDocument[] {
    let filtered = documents;

    if (filters.type) {
      filtered = filtered.filter(doc => filters.type!.includes(doc.type));
    }

    if (filters.language) {
      filtered = filtered.filter(doc => filters.language!.includes(doc.metadata.language));
    }

    if (filters.dateRange) {
      const [start, end] = filters.dateRange;
      filtered = filtered.filter(doc => 
        doc.metadata.lastModified >= start && doc.metadata.lastModified <= end
      );
    }

    return filtered;
  }

  async indexDocument(document: SearchableDocument): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['documents'], 'readwrite');
      const store = transaction.objectStore('documents');
      
      const request = store.put(document);
      
      request.onsuccess = () => {
        this.documents.push(document);
        this.initializeFuse();
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  async indexDocuments(documents: SearchableDocument[]): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const batchSize = 100;
    const batches = [];
    
    for (let i = 0; i < documents.length; i += batchSize) {
      batches.push(documents.slice(i, i + batchSize));
    }

    console.log(`📚 Indexing ${documents.length} documents in ${batches.length} batches`);

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      
      await new Promise<void>((resolve, reject) => {
        const transaction = this.db!.transaction(['documents'], 'readwrite');
        const store = transaction.objectStore('documents');
        
        let completed = 0;
        
        for (const document of batch) {
          const request = store.put(document);
          
          request.onsuccess = () => {
            completed++;
            if (completed === batch.length) {
              resolve();
            }
          };
          
          request.onerror = () => reject(request.error);
        }
      });

      console.log(`✅ Batch ${i + 1}/${batches.length} indexed (${batch.length} documents)`);
    }

    this.documents = documents;
    this.initializeFuse();
    console.log(`🎯 All ${documents.length} documents indexed successfully`);
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text:latest',
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status}`);
      }

      const result = await response.json();
      return result.embedding;
    } catch (error) {
      console.error('❌ Embedding generation failed:', error);
      return [];
    }
  }

  async semanticSearch(query: string, options?: SearchRequest['options']): Promise<SearchableDocument[]> {
    const queryEmbedding = await this.generateEmbedding(query);
    
    if (queryEmbedding.length === 0) {
      return this.search({ query, options });
    }

    const semanticResults = this.documents
      .filter(doc => doc.metadata.embedding && doc.metadata.embedding.length > 0)
      .map(doc => ({
        document: doc,
        similarity: this.cosineSimilarity(queryEmbedding, doc.metadata.embedding!)
      }))
      .filter(result => result.similarity > (options?.threshold || 0.7))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, options?.maxResults || 20)
      .map(result => result.document);

    console.log(`🧠 Semantic search found ${semanticResults.length} results`);
    return semanticResults;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async hybridSearch(request: SearchRequest): Promise<{
    fuzzyResults: SearchableDocument[];
    semanticResults: SearchableDocument[];
    combinedResults: SearchableDocument[];
  }> {
    const [fuzzyResults, semanticResults] = await Promise.all([
      this.search(request),
      this.semanticSearch(request.query, request.options)
    ]);

    const combinedMap = new Map<string, SearchableDocument>();
    
    fuzzyResults.forEach(doc => combinedMap.set(doc.id, doc));
    semanticResults.forEach(doc => combinedMap.set(doc.id, doc));
    
    const combinedResults = Array.from(combinedMap.values());

    return {
      fuzzyResults,
      semanticResults,
      combinedResults
    };
  }

  async indexTypeScriptErrors(errors: { code: string; message: string; file: string; line: number }[]): Promise<void> {
    const documents: SearchableDocument[] = errors.map((error, index) => ({
      id: `error-${index}-${Date.now()}`,
      content: `${error.code}: ${error.message}`,
      path: error.file,
      type: 'error' as const,
      metadata: {
        language: 'typescript',
        lastModified: Date.now(),
        size: error.message.length,
        embedding: undefined
      }
    }));

    console.log(`📝 Indexing ${documents.length} TypeScript errors...`);
    
    const documentsWithEmbeddings = await Promise.all(
      documents.map(async (doc) => ({
        ...doc,
        metadata: {
          ...doc.metadata,
          embedding: await this.generateEmbedding(doc.content)
        }
      }))
    );

    await this.indexDocuments(documentsWithEmbeddings);
  }

  async searchErrors(query: string): Promise<SearchableDocument[]> {
    return this.search({
      query,
      filters: { type: ['error'] },
      options: { threshold: 0.2, maxResults: 100 }
    });
  }

  async getErrorStats(): Promise<{
    totalErrors: number;
    byLanguage: Record<string, number>;
    byType: Record<string, number>;
    recentErrors: number;
  }> {
    const errorDocs = this.documents.filter(doc => doc.type === 'error');
    const recentThreshold = Date.now() - (24 * 60 * 60 * 1000);
    
    const byLanguage: Record<string, number> = {};
    const byType: Record<string, number> = {};
    
    errorDocs.forEach(doc => {
      byLanguage[doc.metadata.language] = (byLanguage[doc.metadata.language] || 0) + 1;
      byType[doc.type] = (byType[doc.type] || 0) + 1;
    });

    return {
      totalErrors: errorDocs.length,
      byLanguage,
      byType,
      recentErrors: errorDocs.filter(doc => doc.metadata.lastModified > recentThreshold).length
    };
  }

  async clearDatabase(): Promise<void> {
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['documents', 'embeddings'], 'readwrite');
      
      const documentsStore = transaction.objectStore('documents');
      const embeddingsStore = transaction.objectStore('embeddings');
      
      const clearDocuments = documentsStore.clear();
      const clearEmbeddings = embeddingsStore.clear();
      
      transaction.oncomplete = () => {
        this.documents = [];
        this.initializeFuse();
        console.log('🗑️ Database cleared successfully');
        resolve();
      };
      
      transaction.onerror = () => reject(transaction.error);
    });
  }

  destroy(): void {
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    
    this.isInitialized = false;
    console.log('🛑 Concurrent IndexedDB Search destroyed');
  }
}

export const concurrentSearch = new ConcurrentIndexedDBSearch();