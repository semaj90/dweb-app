/**
 * Service Worker for AI Summaries API - Async Processing & Caching
 * Handles: Chunking, Background Sync, Streaming, Cache Management
 * Integrates with: NVIDIA CUDA, Triton, Local LLM Workers
 */

const CACHE_NAME = 'legal-ai-summaries-v1';
const API_CACHE_NAME = 'legal-ai-api-cache-v1';

// Cache strategies for different resource types
const CACHE_STRATEGIES = {
  summaries: 'networkFirst',
  embeddings: 'cacheFirst',
  models: 'cacheFirst',
  userActivity: 'networkFirst'
};

// Background sync configuration
const BACKGROUND_SYNC_TAG = 'summaries-sync';

// IndexedDB configuration for offline storage
const DB_NAME = 'LegalAISummariesDB';
const DB_VERSION = 1;
const STORE_NAME = 'summariesStore';

class SummariesServiceWorker {
  constructor() {
    this.initializeDB();
    this.setupEventListeners();
    this.initializeNVIDIAIntegration();
  }

  setupEventListeners() {
    // Install event
    self.addEventListener('install', (event) => {
      console.log('Summaries SW: Installing');
      event.waitUntil(this.onInstall());
    });

    // Activate event
    self.addEventListener('activate', (event) => {
      console.log('Summaries SW: Activating');
      event.waitUntil(this.onActivate());
    });

    // Fetch event with intelligent caching
    self.addEventListener('fetch', (event) => {
      if (this.shouldHandleRequest(event.request)) {
        event.respondWith(this.handleFetch(event.request));
      }
    });

    // Background sync for offline summary generation
    self.addEventListener('sync', (event) => {
      if (event.tag === BACKGROUND_SYNC_TAG) {
        event.waitUntil(this.handleBackgroundSync());
      }
    });

    // Message handling for real-time communication
    self.addEventListener('message', (event) => {
      this.handleMessage(event);
    });

    // Push notifications for completed summaries
    self.addEventListener('push', (event) => {
      event.waitUntil(this.handlePushNotification(event));
    });
  }

  async onInstall() {
    // Pre-cache essential resources
    const cache = await caches.open(CACHE_NAME);
    const essentialResources = [
      '/api/summaries',
      '/api/embed',
      '/static/models/embeddings-config.json'
    ];
    
    await cache.addAll(essentialResources);
    await self.skipWaiting();
  }

  async onActivate() {
    // Clean up old caches
    const cacheNames = await caches.keys();
    const oldCaches = cacheNames.filter(name => 
      name.startsWith('legal-ai-') && name !== CACHE_NAME && name !== API_CACHE_NAME
    );

    await Promise.all(oldCaches.map(cache => caches.delete(cache)));
    await self.clients.claim();
  }

  shouldHandleRequest(request) {
    const url = new URL(request.url);
    return (
      url.pathname.startsWith('/api/summaries') ||
      url.pathname.startsWith('/api/embed') ||
      url.pathname.startsWith('/api/ai/') ||
      url.pathname.includes('models/') ||
      url.pathname.includes('embeddings/')
    );
  }

  async handleFetch(request) {
    const url = new URL(request.url);
    const strategy = this.getCacheStrategy(url.pathname);

    switch (strategy) {
      case 'networkFirst':
        return this.networkFirstStrategy(request);
      case 'cacheFirst':
        return this.cacheFirstStrategy(request);
      case 'staleWhileRevalidate':
        return this.staleWhileRevalidateStrategy(request);
      default:
        return fetch(request);
    }
  }

  getCacheStrategy(pathname) {
    if (pathname.includes('/summaries')) return CACHE_STRATEGIES.summaries;
    if (pathname.includes('/embed')) return CACHE_STRATEGIES.embeddings;
    if (pathname.includes('/models/')) return CACHE_STRATEGIES.models;
    if (pathname.includes('/activity/')) return CACHE_STRATEGIES.userActivity;
    return 'networkFirst';
  }

  async networkFirstStrategy(request) {
    try {
      const networkResponse = await fetch(request);
      
      if (networkResponse.ok) {
        const cache = await caches.open(API_CACHE_NAME);
        await cache.put(request, networkResponse.clone());
      }
      
      return networkResponse;
    } catch (error) {
      console.log('SW: Network failed, falling back to cache');
      const cachedResponse = await caches.match(request);
      
      if (cachedResponse) {
        return cachedResponse;
      }
      
      // Queue for background sync if offline
      await this.queueForBackgroundSync(request);
      return this.createOfflineResponse();
    }
  }

  async cacheFirstStrategy(request) {
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    try {
      const networkResponse = await fetch(request);
      
      if (networkResponse.ok) {
        const cache = await caches.open(CACHE_NAME);
        await cache.put(request, networkResponse.clone());
      }
      
      return networkResponse;
    } catch (error) {
      return this.createOfflineResponse();
    }
  }

  async staleWhileRevalidateStrategy(request) {
    const cachedResponse = await caches.match(request);
    
    // Fetch in background to update cache
    const fetchPromise = fetch(request).then(response => {
      if (response.ok) {
        const cache = caches.open(API_CACHE_NAME);
        cache.then(c => c.put(request, response.clone()));
      }
      return response;
    });
    
    return cachedResponse || fetchPromise;
  }

  async handleBackgroundSync() {
    console.log('SW: Handling background sync for summaries');
    
    const db = await this.openDB();
    const transaction = db.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const pendingRequests = await store.getAll();

    for (const request of pendingRequests) {
      try {
        await this.processPendingSummary(request);
        
        // Remove from pending queue
        const deleteTransaction = db.transaction([STORE_NAME], 'readwrite');
        const deleteStore = deleteTransaction.objectStore(STORE_NAME);
        await deleteStore.delete(request.id);
        
      } catch (error) {
        console.error('SW: Failed to process pending summary:', error);
      }
    }
  }

  async processPendingSummary(request) {
    // Process summary request with chunking support
    const chunkSize = 2000; // Configurable chunk size
    const chunks = this.chunkContent(request.content, chunkSize);
    
    const results = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkRequest = {
        ...request,
        content: chunk,
        chunkIndex: i,
        totalChunks: chunks.length
      };
      
      try {
        // Process with local LLM or queue for NVIDIA GPU processing
        const result = await this.processWithOptimalBackend(chunkRequest);
        results.push(result);
        
        // Notify client of progress
        await this.notifyClients({
          type: 'chunkComplete',
          progress: (i + 1) / chunks.length,
          chunkIndex: i,
          result: result
        });
        
      } catch (error) {
        console.error(`SW: Failed to process chunk ${i}:`, error);
      }
    }
    
    // Combine results and create final summary
    const finalSummary = await this.combineChunkResults(results);
    
    // Notify clients of completion
    await this.notifyClients({
      type: 'summaryComplete',
      summary: finalSummary,
      requestId: request.id
    });
    
    return finalSummary;
  }

  async processWithOptimalBackend(request) {
    // Choose optimal processing backend based on availability
    
    // 1. Try NVIDIA GPU with Triton if available
    if (await this.isNVIDIAAvailable()) {
      return this.processWithNVIDIA(request);
    }
    
    // 2. Try local Ollama
    if (await this.isOllamaAvailable()) {
      return this.processWithOllama(request);
    }
    
    // 3. Fallback to CPU-based processing
    return this.processWithCPU(request);
  }

  async isNVIDIAAvailable() {
    try {
      // Check if NVIDIA container is responding
      const response = await fetch('http://localhost:8001/v2/health', {
        timeout: 5000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async isOllamaAvailable() {
    try {
      const response = await fetch('http://localhost:11434/api/tags', {
        timeout: 5000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async processWithNVIDIA(request) {
    // Process using NVIDIA Triton Inference Server
    const tritonEndpoint = 'http://localhost:8001/v2/models/gemma3/infer';
    
    const payload = {
      inputs: [{
        name: 'text_input',
        datatype: 'BYTES',
        shape: [1],
        data: [request.content]
      }],
      parameters: {
        temperature: 0.3,
        max_tokens: request.depth === 'forensic' ? 1000 : 500
      }
    };
    
    const response = await fetch(tritonEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    
    return {
      content: result.outputs[0].data[0],
      backend: 'nvidia-triton',
      processingTime: result.model_stats?.inference_time || 0,
      tokens: result.outputs[0].shape[0]
    };
  }

  async processWithOllama(request) {
    // Process using local Ollama
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3:7b-instruct-q4_K_M',
        prompt: request.content,
        options: {
          temperature: 0.3,
          num_predict: request.depth === 'forensic' ? 1000 : 500
        },
        stream: false
      })
    });
    
    const result = await response.json();
    
    return {
      content: result.response,
      backend: 'ollama',
      processingTime: result.total_duration / 1000000, // Convert to ms
      tokens: result.eval_count
    };
  }

  async processWithCPU(request) {
    // Fallback CPU processing (simplified text analysis)
    const sentences = request.content.split(/[.!?]+/).filter(s => s.trim().length > 10);
    const summary = sentences.slice(0, 3).join('. ') + '.';
    
    return {
      content: summary,
      backend: 'cpu-fallback',
      processingTime: 100,
      tokens: summary.split(' ').length
    };
  }

  chunkContent(content, chunkSize) {
    const chunks = [];
    for (let i = 0; i < content.length; i += chunkSize) {
      chunks.push(content.substring(i, i + chunkSize));
    }
    return chunks;
  }

  async combineChunkResults(results) {
    // Combine chunk results into coherent summary
    const combinedContent = results.map(r => r.content).join('\n\n');
    const totalTokens = results.reduce((sum, r) => sum + (r.tokens || 0), 0);
    const totalTime = results.reduce((sum, r) => sum + (r.processingTime || 0), 0);
    
    return {
      content: combinedContent,
      metadata: {
        totalChunks: results.length,
        totalTokens,
        totalProcessingTime: totalTime,
        backends: [...new Set(results.map(r => r.backend))]
      }
    };
  }

  async queueForBackgroundSync(request) {
    const db = await this.openDB();
    const transaction = db.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    
    const summaryRequest = {
      id: Date.now().toString(),
      url: request.url,
      method: request.method,
      body: await request.text(),
      timestamp: Date.now()
    };
    
    await store.add(summaryRequest);
    
    // Register for background sync
    await self.registration.sync.register(BACKGROUND_SYNC_TAG);
  }

  async handleMessage(event) {
    const { type, data } = event.data;
    
    switch (type) {
      case 'PROCESS_SUMMARY':
        await this.processSummaryMessage(data);
        break;
      case 'CLEAR_CACHE':
        await this.clearCache();
        break;
      case 'GET_CACHE_STATUS':
        await this.sendCacheStatus(event.ports[0]);
        break;
    }
  }

  async processSummaryMessage(data) {
    // Process summary request directly from main thread
    const result = await this.processWithOptimalBackend(data);
    
    await this.notifyClients({
      type: 'summaryResult',
      requestId: data.id,
      result: result
    });
  }

  async notifyClients(message) {
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage(message);
    });
  }

  async handlePushNotification(event) {
    const data = event.data?.json() || {};
    
    if (data.type === 'summaryComplete') {
      await self.registration.showNotification('Legal AI Summary Ready', {
        body: 'Your requested summary has been completed.',
        icon: '/icons/ai-summary-icon.png',
        badge: '/icons/badge.png',
        data: data
      });
    }
  }

  createOfflineResponse() {
    return new Response(JSON.stringify({
      error: 'Offline',
      message: 'Request queued for processing when online',
      queued: true
    }), {
      status: 202,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async initializeDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      };
    });
  }

  async openDB() {
    return this.initializeDB();
  }

  async clearCache() {
    await caches.delete(CACHE_NAME);
    await caches.delete(API_CACHE_NAME);
  }

  async sendCacheStatus(port) {
    const cacheNames = await caches.keys();
    const status = {
      caches: cacheNames.length,
      summariesCache: await caches.has(CACHE_NAME),
      apiCache: await caches.has(API_CACHE_NAME)
    };
    
    port.postMessage(status);
  }

  async initializeNVIDIAIntegration() {
    // Initialize NVIDIA integration if available
    try {
      const response = await fetch('http://localhost:8001/v2/health');
      if (response.ok) {
        console.log('SW: NVIDIA Triton integration available');
        this.nvidiaAvailable = true;
      }
    } catch {
      console.log('SW: NVIDIA Triton not available, using fallback');
      this.nvidiaAvailable = false;
    }
  }
}

// Initialize the service worker
const summariesSW = new SummariesServiceWorker();

console.log('Legal AI Summaries Service Worker loaded successfully');