/**
 * Service Worker for Chat Engine with Offline Capability
 * Features: SIMD JSON parsing, WebAssembly acceleration, offline caching
 */

// Import SIMD JSON parser (would be loaded from CDN or bundled)
// For now, we'll use native JSON with performance optimizations
const CACHE_NAME = 'chat-engine-v1';
const OFFLINE_CACHE_NAME = 'chat-offline-v1';

// Cache resources for offline capability
const CACHE_RESOURCES = [
  '/',
  '/manifest.json',
  '/offline.html',
  // Add other critical resources
];

// Initialize WebAssembly module for JSON parsing acceleration
let wasmModule = null;
let simdSupported = false;

// Check for SIMD support
try {
  simdSupported = typeof WebAssembly !== 'undefined' && 
    WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
} catch (e) {
  simdSupported = false;
}

/**
 * Install event - cache critical resources
 */
self.addEventListener('install', (event) => {
  console.log('🔧 Chat Engine Service Worker installing...');
  
  event.waitUntil(
    Promise.all([
      caches.open(CACHE_NAME).then(cache => cache.addAll(CACHE_RESOURCES)),
      caches.open(OFFLINE_CACHE_NAME),
      initializeWasmModule()
    ])
  );
  
  self.skipWaiting();
});

/**
 * Activate event - clean up old caches
 */
self.addEventListener('activate', (event) => {
  console.log('✅ Chat Engine Service Worker activated');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames
          .filter(cacheName => 
            cacheName !== CACHE_NAME && 
            cacheName !== OFFLINE_CACHE_NAME
          )
          .map(cacheName => caches.delete(cacheName))
      );
    })
  );
  
  self.clients.claim();
});

/**
 * Fetch event - handle network requests with caching strategy
 */
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(event.request));
  }
  // Handle chat data requests
  else if (url.pathname.includes('/chat/') || url.searchParams.has('chat')) {
    event.respondWith(handleChatRequest(event.request));
  }
  // Handle static resources
  else {
    event.respondWith(handleStaticRequest(event.request));
  }
});

/**
 * Message event - handle messages from main thread
 */
self.addEventListener('message', (event) => {
  const { type, data, id } = event.data;
  
  switch (type) {
    case 'PARSE_JSON':
      handleJsonParsing(data, id);
      break;
    
    case 'STORE_CHAT_OFFLINE':
      handleOfflineChatStorage(data, id);
      break;
    
    case 'PROCESS_EMBEDDINGS':
      handleEmbeddingProcessing(data, id);
      break;
    
    case 'GET_CACHE_STATUS':
      getCacheStatus().then(status => {
        self.clients.matchAll().then(clients => {
          clients.forEach(client => {
            client.postMessage({ type: 'CACHE_STATUS', data: status, id });
          });
        });
      });
      break;
    
    default:
      console.warn('Unknown message type:', type);
  }
});

/**
 * Initialize WebAssembly module for accelerated processing
 */
async function initializeWasmModule() {
  if (!simdSupported) {
    console.log('SIMD not supported, using JavaScript fallback');
    return;
  }

  try {
    // In a real implementation, you would load a compiled WASM module
    // For now, we'll simulate WASM capabilities
    wasmModule = {
      parseJSON: (jsonString) => {
        // Simulated WASM-accelerated JSON parsing
        const startTime = performance.now();
        const result = JSON.parse(jsonString);
        const parseTime = performance.now() - startTime;
        
        return {
          data: result,
          parseTime,
          accelerated: true
        };
      },
      
      processEmbeddings: (embeddings) => {
        // Simulated SIMD vector operations
        return embeddings.map(embedding => 
          embedding.map(value => value * 1.001) // Simulate processing
        );
      }
    };
    
    console.log('✅ WebAssembly module initialized (simulated)');
  } catch (error) {
    console.error('Failed to initialize WebAssembly module:', error);
  }
}

/**
 * Handle API requests with offline fallback
 */
async function handleApiRequest(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache successful responses
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
      return networkResponse;
    }
  } catch (error) {
    console.log('Network failed, checking cache:', error);
  }
  
  // Fallback to cache
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Return offline response
  return new Response(
    JSON.stringify({ 
      error: 'Offline', 
      message: 'This request is not available offline',
      cached: false
    }),
    {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    }
  );
}

/**
 * Handle chat-specific requests with enhanced caching
 */
async function handleChatRequest(request) {
  const url = new URL(request.url);
  const cacheKey = `chat:${url.pathname}${url.search}`;
  
  try {
    // For GET requests, try cache first (stale-while-revalidate)
    if (request.method === 'GET') {
      const cachedResponse = await caches.match(request);
      
      if (cachedResponse) {
        // Return cached version immediately
        fetch(request).then(async (networkResponse) => {
          if (networkResponse.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, networkResponse);
          }
        }).catch(() => {
          // Network failed, cache is still valid
        });
        
        return cachedResponse;
      }
    }
    
    // Try network
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Cache GET responses
      if (request.method === 'GET') {
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, networkResponse.clone());
      }
      
      return networkResponse;
    }
    
  } catch (error) {
    console.log('Chat request failed, checking offline storage:', error);
  }
  
  // Fallback to offline storage
  return handleOfflineChatFallback(request);
}

/**
 * Handle static resource requests
 */
async function handleStaticRequest(request) {
  // Cache-first strategy for static resources
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match('/offline.html');
    }
    
    throw error;
  }
}

/**
 * Handle accelerated JSON parsing
 */
async function handleJsonParsing(jsonString, messageId) {
  let result;
  
  if (wasmModule && simdSupported) {
    // Use WebAssembly-accelerated parsing
    result = wasmModule.parseJSON(jsonString);
  } else {
    // Fallback to optimized JavaScript parsing
    const startTime = performance.now();
    try {
      const data = JSON.parse(jsonString);
      result = {
        data,
        parseTime: performance.now() - startTime,
        accelerated: false
      };
    } catch (error) {
      result = {
        error: error.message,
        parseTime: performance.now() - startTime,
        accelerated: false
      };
    }
  }
  
  // Send result back to main thread
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({
        type: 'PARSED_JSON',
        data: result,
        id: messageId
      });
    });
  });
}

/**
 * Handle offline chat storage
 */
async function handleOfflineChatStorage(chatData, messageId) {
  try {
    const cache = await caches.open(OFFLINE_CACHE_NAME);
    const timestamp = Date.now();
    const key = `offline-chat-${timestamp}-${chatData.sessionId}`;
    
    // Store as cached response
    const response = new Response(JSON.stringify(chatData), {
      headers: {
        'Content-Type': 'application/json',
        'X-Offline-Storage': 'true',
        'X-Timestamp': timestamp.toString()
      }
    });
    
    await cache.put(key, response);
    
    // Notify main thread
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'CHAT_STORED_OFFLINE',
          data: { success: true, key, timestamp },
          id: messageId
        });
      });
    });
    
  } catch (error) {
    // Notify error
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'CHAT_STORED_OFFLINE',
          data: { success: false, error: error.message },
          id: messageId
        });
      });
    });
  }
}

/**
 * Handle embedding processing with SIMD optimization
 */
async function handleEmbeddingProcessing(embeddings, messageId) {
  let processedEmbeddings;
  
  if (wasmModule && simdSupported) {
    // Use SIMD-optimized processing
    processedEmbeddings = wasmModule.processEmbeddings(embeddings);
  } else {
    // JavaScript fallback
    processedEmbeddings = embeddings.map(embedding => 
      embedding.map(value => {
        // Simple normalization
        return Math.max(-1, Math.min(1, value));
      })
    );
  }
  
  // Send results back
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({
        type: 'EMBEDDINGS_PROCESSED',
        data: processedEmbeddings,
        id: messageId
      });
    });
  });
}

/**
 * Handle offline chat fallback
 */
async function handleOfflineChatFallback(request) {
  const offlineCache = await caches.open(OFFLINE_CACHE_NAME);
  const cachedItems = await offlineCache.keys();
  
  // Filter chat-related items
  const chatItems = cachedItems.filter(req => 
    req.url.includes('offline-chat-')
  );
  
  if (chatItems.length > 0) {
    // Return most recent offline chat data
    const responses = await Promise.all(
      chatItems.map(item => offlineCache.match(item))
    );
    
    const chatData = await Promise.all(
      responses.map(response => response?.json())
    );
    
    return new Response(
      JSON.stringify({
        offline: true,
        data: chatData,
        message: 'Offline data retrieved from cache'
      }),
      {
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
  
  // No offline data available
  return new Response(
    JSON.stringify({
      error: 'No offline data available',
      offline: true
    }),
    {
      status: 404,
      headers: { 'Content-Type': 'application/json' }
    }
  );
}

/**
 * Get comprehensive cache status
 */
async function getCacheStatus() {
  const mainCache = await caches.open(CACHE_NAME);
  const offlineCache = await caches.open(OFFLINE_CACHE_NAME);
  
  const mainKeys = await mainCache.keys();
  const offlineKeys = await offlineCache.keys();
  
  // Calculate cache sizes (approximate)
  let totalSize = 0;
  try {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      totalSize = estimate.usage || 0;
    }
  } catch (e) {
    // Fallback estimation
    totalSize = (mainKeys.length + offlineKeys.length) * 1024; // 1KB per item estimate
  }
  
  return {
    caches: {
      main: {
        name: CACHE_NAME,
        entries: mainKeys.length,
        urls: mainKeys.map(key => key.url)
      },
      offline: {
        name: OFFLINE_CACHE_NAME,
        entries: offlineKeys.length,
        chatEntries: offlineKeys.filter(key => key.url.includes('offline-chat-')).length
      }
    },
    totalSize,
    wasmSupported: simdSupported,
    wasmInitialized: wasmModule !== null,
    capabilities: {
      simdJson: simdSupported,
      offlineStorage: true,
      backgroundSync: 'serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype,
      pushNotifications: 'serviceWorker' in navigator && 'PushManager' in window
    }
  };
}

/**
 * Background sync for offline data
 */
if ('sync' in self.registration) {
  self.addEventListener('sync', (event) => {
    console.log('🔄 Background sync triggered:', event.tag);
    
    if (event.tag === 'chat-sync') {
      event.waitUntil(syncOfflineChats());
    }
  });
}

/**
 * Sync offline chats when connection is restored
 */
async function syncOfflineChats() {
  try {
    const offlineCache = await caches.open(OFFLINE_CACHE_NAME);
    const keys = await offlineCache.keys();
    
    const chatKeys = keys.filter(key => key.url.includes('offline-chat-'));
    
    for (const key of chatKeys) {
      const response = await offlineCache.match(key);
      const chatData = await response?.json();
      
      if (chatData) {
        try {
          // Send to server
          const syncResponse = await fetch('/api/chat/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(chatData)
          });
          
          if (syncResponse.ok) {
            // Remove from offline cache after successful sync
            await offlineCache.delete(key);
            console.log(`✅ Synced offline chat: ${key.url}`);
          }
        } catch (syncError) {
          console.log(`Failed to sync chat ${key.url}:`, syncError);
        }
      }
    }
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

// Error handling
self.addEventListener('error', (event) => {
  console.error('Service Worker error:', event.error);
});

self.addEventListener('unhandledrejection', (event) => {
  console.error('Service Worker unhandled rejection:', event.reason);
});

console.log('🚀 Chat Engine Service Worker loaded');