import { build, files, version } from '$service-worker';

// Create a unique cache name for this deployment
const CACHE = `cache-${version}`;

const ASSETS = [
  ...build, // the app itself
  ...files  // everything in static
];

self.addEventListener('install', (event) => {
  // Create a new cache and add all files to it
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => {
        (self as any).skipWaiting();
      })
  );
});

self.addEventListener('activate', (event) => {
  // Remove previous cached data from disk
  event.waitUntil(
    caches.keys().then(async (keys) => {
      for (const key of keys) {
        if (key !== CACHE) await caches.delete(key);
      }
      (self as any).clients.claim();
    })
  );
});

self.addEventListener('fetch', (event) => {
  // ignore POST requests etc
  if (event.request.method !== 'GET') return;

  event.respondWith(
    (async () => {
      const url = new URL(event.request.url);

      // Try to get the response from a cache.
      const cache = await caches.open(CACHE);
      
      // Handle API requests - always try network first for fresh data
      if (url.pathname.startsWith('/api/')) {
        try {
          const response = await fetch(event.request);
          return response;
        } catch {
          // If network fails, try cache for GET requests
          const cached = await cache.match(event.request);
          if (cached) return cached;
          
          // Return offline fallback
          return new Response(JSON.stringify({ 
            error: 'Offline', 
            message: 'Network unavailable' 
          }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
          });
        }
      }

      // Handle static assets - cache first
      if (ASSETS.includes(url.pathname)) {
        const cached = await cache.match(event.request);
        if (cached) return cached;
      }

      // For everything else, try the network first, falling back to cache
      try {
        const response = await fetch(event.request);
        
        // Cache successful responses for GET requests
        if (response.status === 200) {
          cache.put(event.request, response.clone());
        }
        
        return response;
      } catch {
        const cached = await cache.match(event.request);
        if (cached) return cached;
        
        return new Response('Offline', { status: 503 });
      }
    })()
  );
});

// Handle AI/Legal specific caching for offline functionality
self.addEventListener('message', (event: ExtendableMessageEvent) => {
  if (event.data && event.data.type === 'CACHE_LEGAL_DATA') {
    event.waitUntil(
      caches.open('legal-data').then((cache) => {
        return cache.put(`/api/legal/cache/${event.data.caseId}`, 
          new Response(JSON.stringify(event.data.data), {
            headers: { 'Content-Type': 'application/json' }
          })
        );
      })
    );
  }
});

// Background sync for evidence uploads when online
self.addEventListener('sync', (event) => {
  if (event.tag === 'evidence-upload') {
    event.waitUntil(syncEvidenceUploads());
  }
});

async function syncEvidenceUploads() {
  // Implementation for syncing pending evidence uploads
  // This would integrate with your evidence upload API
  console.log('Syncing evidence uploads...');
}