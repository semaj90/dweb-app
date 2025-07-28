/**
 * Neural Sprite Cache Service Worker
 * Provides cross-tab sprite caching and intelligent prefetching
 */

const CACHE_NAME = 'neural-sprite-cache-v1';
const MAX_CACHE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_SPRITE_COUNT = 1000;

let currentCacheSize = 0;
let spriteCount = 0;

// In-memory sprite cache for ultra-fast access
const memoryCache = new Map();

self.addEventListener('install', (event) => {
  console.log('Neural Sprite Cache Service Worker installing...');
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  console.log('Neural Sprite Cache Service Worker activated');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

self.addEventListener('message', (event) => {
  const { type, spriteId, sprite } = event.data;
  const port = event.ports[0];

  switch (type) {
    case 'GET_SPRITE':
      handleGetSprite(spriteId, port);
      break;
      
    case 'CACHE_SPRITE':
      handleCacheSprite(sprite);
      break;
      
    case 'CLEAR_CACHE':
      handleClearCache();
      break;
      
    case 'GET_CACHE_STATS':
      handleGetCacheStats(port);
      break;
  }
});

async function handleGetSprite(spriteId, port) {
  try {
    // 1. Check memory cache first
    if (memoryCache.has(spriteId)) {
      const sprite = memoryCache.get(spriteId);
      sprite.lastAccessed = Date.now();
      port.postMessage({ success: true, data: sprite.data });
      return;
    }

    // 2. Check Cache API
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(`/sprite/${spriteId}`);
    
    if (response) {
      const spriteData = await response.json();
      
      // Store in memory for next access
      memoryCache.set(spriteId, {
        data: spriteData,
        size: JSON.stringify(spriteData).length,
        lastAccessed: Date.now(),
        accessCount: 1
      });
      
      port.postMessage({ success: true, data: spriteData });
      return;
    }
    
    // Sprite not found
    port.postMessage({ success: false, data: null });
    
  } catch (error) {
    console.error('Error getting sprite:', error);
    port.postMessage({ success: false, data: null });
  }
}

async function handleCacheSprite(sprite) {
  try {
    const spriteData = JSON.stringify(sprite);
    const size = spriteData.length;
    
    // Check cache limits
    if (currentCacheSize + size > MAX_CACHE_SIZE || spriteCount >= MAX_SPRITE_COUNT) {
      await evictOldSprites(size);
    }
    
    // Store in Cache API
    const cache = await caches.open(CACHE_NAME);
    const response = new Response(spriteData, {
      headers: {
        'Content-Type': 'application/json',
        'X-Sprite-Size': size.toString(),
        'X-Cached-At': Date.now().toString()
      }
    });
    
    await cache.put(`/sprite/${sprite.id}`, response);
    
    // Store in memory cache
    memoryCache.set(sprite.id, {
      data: sprite,
      size: size,
      lastAccessed: Date.now(),
      accessCount: 1
    });
    
    currentCacheSize += size;
    spriteCount++;
    
    // Broadcast cache update to all clients
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({
        type: 'SPRITE_CACHED',
        data: { spriteId: sprite.id, size }
      });
    });
    
  } catch (error) {
    console.error('Error caching sprite:', error);
  }
}

async function evictOldSprites(requiredSize) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();
    
    // Get cache metadata
    const cacheEntries = await Promise.all(
      requests.map(async (request) => {
        const response = await cache.match(request);
        const size = parseInt(response.headers.get('X-Sprite-Size') || '0');
        const cachedAt = parseInt(response.headers.get('X-Cached-At') || '0');
        
        return {
          request,
          size,
          cachedAt,
          spriteId: request.url.split('/sprite/')[1]
        };
      })
    );
    
    // Sort by access pattern (LRU + size)
    const sortedEntries = cacheEntries.sort((a, b) => {
      const aMemory = memoryCache.get(a.spriteId);
      const bMemory = memoryCache.get(b.spriteId);
      
      const aScore = (aMemory?.accessCount || 1) / Math.log(a.size + 1);
      const bScore = (bMemory?.accessCount || 1) / Math.log(b.size + 1);
      
      return aScore - bScore;
    });
    
    // Evict until we have enough space
    let freedSize = 0;
    let evictedCount = 0;
    
    for (const entry of sortedEntries) {
      if (freedSize >= requiredSize && currentCacheSize - freedSize < MAX_CACHE_SIZE * 0.8) {
        break;
      }
      
      await cache.delete(entry.request);
      memoryCache.delete(entry.spriteId);
      
      freedSize += entry.size;
      evictedCount++;
    }
    
    currentCacheSize -= freedSize;
    spriteCount -= evictedCount;
    
    // Notify clients of cache eviction
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({
        type: 'CACHE_EVICTED',
        data: { evictedCount, freedSize }
      });
    });
    
    console.log(`Evicted ${evictedCount} sprites, freed ${freedSize} bytes`);
    
  } catch (error) {
    console.error('Error evicting sprites:', error);
  }
}

async function handleClearCache() {
  try {
    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();
    
    await Promise.all(requests.map(request => cache.delete(request)));
    
    memoryCache.clear();
    currentCacheSize = 0;
    spriteCount = 0;
    
    // Notify clients
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({
        type: 'CACHE_CLEARED',
        data: { success: true }
      });
    });
    
    console.log('Neural sprite cache cleared');
    
  } catch (error) {
    console.error('Error clearing cache:', error);
  }
}

async function handleGetCacheStats(port) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();
    
    const stats = {
      spriteCount: spriteCount,
      cacheSize: currentCacheSize,
      memoryCacheSize: memoryCache.size,
      cacheName: CACHE_NAME,
      maxCacheSize: MAX_CACHE_SIZE,
      maxSpriteCount: MAX_SPRITE_COUNT,
      cacheUtilization: currentCacheSize / MAX_CACHE_SIZE,
      timestamp: Date.now()
    };
    
    port.postMessage({ success: true, data: stats });
    
  } catch (error) {
    console.error('Error getting cache stats:', error);
    port.postMessage({ success: false, data: null });
  }
}

// Periodic cleanup of memory cache
setInterval(() => {
  const now = Date.now();
  const MEMORY_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
  
  for (const [spriteId, entry] of memoryCache.entries()) {
    if (now - entry.lastAccessed > MEMORY_CACHE_TTL) {
      memoryCache.delete(spriteId);
    }
  }
}, 60 * 1000); // Run every minute

// Preload frequently accessed sprites
self.addEventListener('fetch', (event) => {
  // Intercept requests to sprite endpoints for intelligent prefetching
  if (event.request.url.includes('/api/sprites/frequently-used')) {
    event.respondWith(handleFrequentSpritesRequest(event.request));
  }
});

async function handleFrequentSpritesRequest(request) {
  try {
    // Get the actual response
    const response = await fetch(request);
    const spriteIds = await response.clone().json();
    
    // Prefetch these sprites in background
    prefetchSprites(spriteIds.slice(0, 10)); // Top 10 most frequent
    
    return response;
  } catch (error) {
    console.error('Error handling frequent sprites request:', error);
    return new Response('[]', { 
      status: 200, 
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

async function prefetchSprites(spriteIds) {
  // Background prefetching to warm up the cache
  for (const spriteId of spriteIds) {
    if (!memoryCache.has(spriteId)) {
      try {
        const response = await fetch(`/api/sprites/${spriteId}`);
        if (response.ok) {
          const sprite = await response.json();
          await handleCacheSprite(sprite);
        }
      } catch (error) {
        console.warn(`Failed to prefetch sprite ${spriteId}:`, error);
      }
    }
  }
}