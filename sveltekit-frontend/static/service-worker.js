// Service Worker for managing Ollama token streams and legal AI optimizations
// Handles caching, background sync, and token stream management

const CACHE_NAME = 'legal-ai-v1';
const STATIC_CACHE = 'legal-ai-static-v1';
const API_CACHE = 'legal-ai-api-v1';
const EMBEDDING_CACHE = 'legal-ai-embeddings-v1';

// Cache strategies for different resource types
const CACHE_STRATEGIES = {
  static: 'cache-first',
  api: 'network-first',
  embeddings: 'cache-first',
  ollama: 'network-only'
};

// Files to cache for offline functionality
const STATIC_ASSETS = [
  '/',
  '/app.html',
  '/static/workers/simd-json-worker.js',
  '/static/workers/vector-search-worker.js',
  '/static/workers/embedding-worker.js',
  '/static/favicon.png'
];

// Token stream management
const activeStreams = new Map();
const streamBuffer = new Map();

/**
 * Service Worker Installation
 */
self.addEventListener('install', (event) => {
  console.log('Legal AI Service Worker installing...');
  
  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE).then(cache => cache.addAll(STATIC_ASSETS)),
      caches.open(API_CACHE),
      caches.open(EMBEDDING_CACHE)
    ])
  );
  
  self.skipWaiting();
});

/**
 * Service Worker Activation
 */
self.addEventListener('activate', (event) => {
  console.log('Legal AI Service Worker activated');
  
  event.waitUntil(
    // Clean up old caches
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== STATIC_CACHE && 
              cacheName !== API_CACHE && 
              cacheName !== EMBEDDING_CACHE) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  
  self.clients.claim();
});

/**
 * Fetch Event Handler - Core caching and optimization logic
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Handle different request types
  if (url.pathname.startsWith('/api/ollama/stream')) {
    // Handle Ollama streaming requests
    event.respondWith(handleOllamaStream(request));
  } else if (url.pathname.startsWith('/api/embeddings')) {
    // Handle embedding requests with caching
    event.respondWith(handleEmbeddingRequest(request));
  } else if (url.pathname.startsWith('/api/')) {
    // Handle API requests with network-first strategy
    event.respondWith(handleApiRequest(request));
  } else if (request.destination === 'document') {
    // Handle page requests
    event.respondWith(handlePageRequest(request));
  } else {
    // Handle static assets
    event.respondWith(handleStaticRequest(request));
  }
});

/**
 * Handle Ollama streaming responses with token buffering
 */
async function handleOllamaStream(request) {
  try {
    const streamId = generateStreamId();
    const response = await fetch(request);
    
    if (!response.ok) {
      throw new Error(`Ollama stream failed: ${response.statusText}`);
    }
    
    // Create a readable stream to intercept and buffer tokens
    const readable = new ReadableStream({
      start(controller) {
        const reader = response.body.getReader();
        const buffer = [];
        
        activeStreams.set(streamId, {
          controller,
          buffer,
          startTime: Date.now()
        });
        
        function pump() {
          return reader.read().then(({ done, value }) => {
            if (done) {
              // Stream complete - notify clients and cleanup
              self.clients.matchAll().then(clients => {
                clients.forEach(client => {
                  client.postMessage({
                    type: 'stream_complete',
                    streamId: streamId,
                    totalTokens: buffer.length,
                    duration: Date.now() - activeStreams.get(streamId).startTime
                  });
                });
              });
              
              activeStreams.delete(streamId);
              controller.close();
              return;
            }
            
            // Process token data
            const tokenData = processTokenChunk(value);
            if (tokenData) {
              buffer.push(tokenData);
              
              // Notify clients of new token
              self.clients.matchAll().then(clients => {
                clients.forEach(client => {
                  client.postMessage({
                    type: 'stream_token',
                    streamId: streamId,
                    token: tokenData,
                    progress: buffer.length
                  });
                });
              });
            }
            
            controller.enqueue(value);
            return pump();
          });
        }
        
        return pump();
      }
    });
    
    return new Response(readable, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        ...response.headers,
        'X-Stream-ID': streamId
      }
    });
    
  } catch (error) {
    console.error('Ollama stream error:', error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle embedding requests with intelligent caching
 */
async function handleEmbeddingRequest(request) {
  const cache = await caches.open(EMBEDDING_CACHE);
  const cachedResponse = await cache.match(request);
  
  // Check if we have a valid cached embedding
  if (cachedResponse) {
    const cacheAge = Date.now() - new Date(cachedResponse.headers.get('sw-cached-at')).getTime();
    
    // Cache embeddings for 24 hours
    if (cacheAge < 24 * 60 * 60 * 1000) {
      console.log('Serving cached embedding');
      return cachedResponse;
    }
  }
  
  try {
    // Fetch new embedding
    const response = await fetch(request);
    
    if (response.ok) {
      // Clone response for caching
      const responseToCache = response.clone();
      
      // Add cache timestamp
      const headers = new Headers(responseToCache.headers);
      headers.set('sw-cached-at', new Date().toISOString());
      
      const cachedResponse = new Response(responseToCache.body, {
        status: responseToCache.status,
        statusText: responseToCache.statusText,
        headers: headers
      });
      
      cache.put(request, cachedResponse);
    }
    
    return response;
  } catch (error) {
    // Return cached response if network fails
    if (cachedResponse) {
      console.log('Network failed, serving stale embedding cache');
      return cachedResponse;
    }
    
    return new Response(JSON.stringify({ error: 'Embedding service unavailable' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle API requests with network-first strategy
 */
async function handleApiRequest(request) {
  const cache = await caches.open(API_CACHE);
  
  try {
    const response = await fetch(request);
    
    // Cache successful GET requests
    if (response.ok && request.method === 'GET') {
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Try cache if network fails
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      console.log('Network failed, serving cached API response');
      return cachedResponse;
    }
    
    return new Response(JSON.stringify({ error: 'Service unavailable' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle page requests with cache-first strategy
 */
async function handlePageRequest(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const response = await fetch(request);
    
    if (response.ok) {
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Return offline page if available
    return cache.match('/') || new Response('Offline', { status: 503 });
  }
}

/**
 * Handle static asset requests
 */
async function handleStaticRequest(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const response = await fetch(request);
    
    if (response.ok) {
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    return new Response('Asset not available offline', { status: 503 });
  }
}

/**
 * Process token chunks from Ollama stream
 */
function processTokenChunk(chunk) {
  try {
    const text = new TextDecoder().decode(chunk);
    const lines = text.split('\n').filter(line => line.trim());
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonData = line.slice(6);
        const tokenData = JSON.parse(jsonData);
        
        if (tokenData.token) {
          return {
            token: tokenData.token,
            timestamp: Date.now(),
            done: tokenData.done || false
          };
        }
      }
    }
  } catch (error) {
    console.warn('Error processing token chunk:', error);
  }
  
  return null;
}

/**
 * Generate unique stream ID
 */
function generateStreamId() {
  return `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Handle background sync for offline actions
 */
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-legal-sync') {
    event.waitUntil(performBackgroundSync());
  }
});

/**
 * Perform background sync of legal data
 */
async function performBackgroundSync() {
  try {
    console.log('Performing background sync for legal data');
    
    // Sync cached legal documents
    const cache = await caches.open(API_CACHE);
    const keys = await cache.keys();
    
    for (const request of keys) {
      if (request.url.includes('/api/legal-documents') || 
          request.url.includes('/api/cases')) {
        try {
          const freshResponse = await fetch(request);
          if (freshResponse.ok) {
            cache.put(request, freshResponse);
          }
        } catch (error) {
          console.warn('Background sync failed for:', request.url);
        }
      }
    }
    
    // Notify clients of sync completion
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'background_sync_complete',
          timestamp: Date.now()
        });
      });
    });
    
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

/**
 * Handle push notifications for legal updates
 */
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'Legal AI notification',
    icon: '/static/icon-192.png',
    badge: '/static/badge-72.png',
    tag: 'legal-ai-notification',
    data: {
      url: '/',
      timestamp: Date.now()
    }
  };
  
  event.waitUntil(
    self.registration.showNotification('Legal AI Update', options)
  );
});

/**
 * Handle notification clicks
 */
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  event.waitUntil(
    clients.openWindow(event.notification.data.url || '/')
  );
});

/**
 * Message handler for client communication
 */
self.addEventListener('message', (event) => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'get_stream_stats':
      event.ports[0].postMessage({
        activeStreams: activeStreams.size,
        streamIds: Array.from(activeStreams.keys())
      });
      break;
      
    case 'clear_cache':
      clearCache(data.cacheType).then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    case 'preload_legal_data':
      preloadLegalData(data).then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    default:
      console.log('Unknown message type:', type);
  }
});

/**
 * Clear specific cache
 */
async function clearCache(cacheType) {
  const cacheNames = {
    static: STATIC_CACHE,
    api: API_CACHE,
    embeddings: EMBEDDING_CACHE,
    all: [STATIC_CACHE, API_CACHE, EMBEDDING_CACHE]
  };
  
  const names = Array.isArray(cacheNames[cacheType]) 
    ? cacheNames[cacheType] 
    : [cacheNames[cacheType]];
  
  for (const name of names) {
    if (name) {
      await caches.delete(name);
    }
  }
}

/**
 * Preload legal data for offline access
 */
async function preloadLegalData(urls) {
  const cache = await caches.open(API_CACHE);
  
  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        cache.put(url, response);
      }
    } catch (error) {
      console.warn('Failed to preload:', url);
    }
  }
}

console.log('Legal AI Service Worker loaded');