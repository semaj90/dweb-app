// Production Service Worker for Legal Processing
const CACHE_NAME = 'legal-processor-v1';
const STATIC_CACHE = 'static-v1';

// Cache strategies
const NETWORK_FIRST = ['api', 'rag-enhanced', 'recommendations'];
const CACHE_FIRST = ['static', 'assets', 'fonts'];

// Background sync queues
const ACTIVITY_QUEUE = 'activity-sync';
const TRAINING_QUEUE = 'model-training';

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then(cache =>
      cache.addAll([
        '/',
        '/app.html',
        '/service-worker.js',
        '/_app/immutable/start-*.js',
        '/_app/immutable/chunks/*.js'
      ])
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(names =>
      Promise.all(
        names.filter(name => 
          name !== CACHE_NAME && name !== STATIC_CACHE
        ).map(name => caches.delete(name))
      )
    ).then(() => self.clients.claim())
  );
});

// Network strategies
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  if (shouldUseNetworkFirst(url.pathname)) {
    event.respondWith(networkFirst(request));
  } else if (shouldUseCacheFirst(url.pathname)) {
    event.respondWith(cacheFirst(request));
  } else {
    event.respondWith(fetch(request));
  }
});

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    return cachedResponse || new Response('Offline', { status: 503 });
  }
}

async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) return cachedResponse;
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    return new Response('Resource unavailable', { status: 503 });
  }
}

// Background sync for user activities
self.addEventListener('sync', event => {
  if (event.tag === ACTIVITY_QUEUE) {
    event.waitUntil(syncUserActivities());
  } else if (event.tag === TRAINING_QUEUE) {
    event.waitUntil(syncModelTraining());
  }
});

// Message handling from main thread
self.addEventListener('message', event => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'RECORD_ACTIVITY':
      handleActivityRecord(data);
      break;
    case 'TRAIN_RECOMMENDATIONS':
      handleRecommendationTraining(data);
      break;
    case 'CACHE_WARM':
      warmCaches(data.urls);
      break;
    case 'QUEUE_STATUS':
      event.ports[0].postMessage(getQueueStatus());
      break;
  }
});

async function handleActivityRecord(data) {
  try {
    // Store activity locally first
    const db = await openActivityDB();
    await storeActivity(db, data);
    
    // Attempt immediate sync
    await syncActivity(data);
  } catch (error) {
    // Queue for background sync
    await queueForSync(ACTIVITY_QUEUE, data);
    console.log('Activity queued for background sync');
  }
}

async function handleRecommendationTraining(data) {
  try {
    await fetch('/api/legal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        endpoint: 'train-recommendations',
        ...data
      })
    });
  } catch (error) {
    await queueForSync(TRAINING_QUEUE, data);
  }
}

async function syncUserActivities() {
  const activities = await getQueuedActivities();
  
  for (const activity of activities) {
    try {
      await syncActivity(activity.data);
      await removeFromQueue(ACTIVITY_QUEUE, activity.id);
    } catch (error) {
      console.error('Activity sync failed:', error);
    }
  }
}

async function syncModelTraining() {
  const trainings = await getQueuedTraining();
  
  for (const training of trainings) {
    try {
      await fetch('/api/legal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          endpoint: 'train-recommendations',
          ...training.data
        })
      });
      await removeFromQueue(TRAINING_QUEUE, training.id);
    } catch (error) {
      console.error('Training sync failed:', error);
    }
  }
}

async function syncActivity(data) {
  return fetch('/api/legal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      endpoint: 'activity',
      ...data
    })
  });
}

// IndexedDB operations for offline storage
async function openActivityDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('LegalProcessorDB', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = () => {
      const db = request.result;
      
      if (!db.objectStoreNames.contains('activities')) {
        const store = db.createObjectStore('activities', { keyPath: 'id', autoIncrement: true });
        store.createIndex('timestamp', 'timestamp');
        store.createIndex('userId', 'userId');
      }
      
      if (!db.objectStoreNames.contains('syncQueue')) {
        const queue = db.createObjectStore('syncQueue', { keyPath: 'id', autoIncrement: true });
        queue.createIndex('type', 'type');
        queue.createIndex('timestamp', 'timestamp');
      }
    };
  });
}

async function storeActivity(db, data) {
  const transaction = db.transaction(['activities'], 'readwrite');
  const store = transaction.objectStore('activities');
  return store.add({ ...data, id: Date.now() });
}

async function queueForSync(type, data) {
  const db = await openActivityDB();
  const transaction = db.transaction(['syncQueue'], 'readwrite');
  const store = transaction.objectStore('syncQueue');
  
  return store.add({
    type,
    data,
    timestamp: Date.now(),
    retries: 0
  });
}

async function getQueuedActivities() {
  const db = await openActivityDB();
  const transaction = db.transaction(['syncQueue'], 'readonly');
  const store = transaction.objectStore('syncQueue');
  const index = store.index('type');
  
  return new Promise((resolve) => {
    const request = index.getAll(ACTIVITY_QUEUE);
    request.onsuccess = () => resolve(request.result);
  });
}

async function getQueuedTraining() {
  const db = await openActivityDB();
  const transaction = db.transaction(['syncQueue'], 'readonly');
  const store = transaction.objectStore('syncQueue');
  const index = store.index('type');
  
  return new Promise((resolve) => {
    const request = index.getAll(TRAINING_QUEUE);
    request.onsuccess = () => resolve(request.result);
  });
}

async function removeFromQueue(type, id) {
  const db = await openActivityDB();
  const transaction = db.transaction(['syncQueue'], 'readwrite');
  const store = transaction.objectStore('syncQueue');
  return store.delete(id);
}

async function warmCaches(urls) {
  const cache = await caches.open(CACHE_NAME);
  return Promise.all(
    urls.map(url => 
      fetch(url).then(response => {
        if (response.ok) cache.put(url, response);
      }).catch(() => {})
    )
  );
}

function getQueueStatus() {
  return {
    caches: CACHE_NAME,
    backgroundSync: 'registered',
    lastSync: Date.now()
  };
}

function shouldUseNetworkFirst(pathname) {
  return NETWORK_FIRST.some(pattern => pathname.includes(pattern));
}

function shouldUseCacheFirst(pathname) {
  return CACHE_FIRST.some(pattern => pathname.includes(pattern));
}

// Performance monitoring
self.addEventListener('notificationclick', event => {
  event.notification.close();
  
  if (event.action === 'view_results') {
    clients.openWindow('/results');
  }
});

// Web Push for real-time updates
self.addEventListener('push', event => {
  const data = event.data?.json() || {};
  
  const options = {
    body: data.message || 'Processing complete',
    icon: '/icon-192.png',
    badge: '/badge-72.png',
    actions: [
      { action: 'view_results', title: 'View Results' },
      { action: 'dismiss', title: 'Dismiss' }
    ],
    tag: 'legal-processor',
    renotify: true
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'Legal Processor', options)
  );
});