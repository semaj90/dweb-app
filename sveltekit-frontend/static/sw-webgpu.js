// ================================================================================
// WEBGPU + WEBASSEMBLY SERVICE WORKER
// ================================================================================
// Advanced caching strategy for WebGPU shaders, WASM modules, 3D assets,
// with aggressive optimization, cache preloading, and performance monitoring
// ================================================================================

const CACHE_VERSION = 'v1.2.0';
const CACHE_NAMES = {
	webgpu: `webgpu-shaders-${CACHE_VERSION}`,
	wasm: `wasm-modules-${CACHE_VERSION}`,
	assets3d: `3d-assets-${CACHE_VERSION}`,
	api: `api-cache-${CACHE_VERSION}`,
	static: `static-resources-${CACHE_VERSION}`
};

// ============================================================================
// CACHE STRATEGIES & CONFIGURATION
// ============================================================================

const CACHE_STRATEGIES = {
	'cache-first': 'CACHE_FIRST',
	'network-first': 'NETWORK_FIRST',
	'stale-while-revalidate': 'STALE_WHILE_REVALIDATE',
	'network-only': 'NETWORK_ONLY',
	'cache-only': 'CACHE_ONLY'
};

const RESOURCE_PATTERNS = {
	webgpu: [
		/\.wgsl$/i,           // WebGPU Shading Language
		/webgpu.*\.js$/i,     // WebGPU modules
		/compute.*\.js$/i     // Compute shaders
	],
	wasm: [
		/\.wasm$/i,           // WebAssembly binaries
		/\.wat$/i,            // WebAssembly text
		/wasm.*\.js$/i        // WASM loaders
	],
	assets3d: [
		/\.(gltf|glb|obj|fbx|dae)$/i,  // 3D models
		/\.(jpg|jpeg|png|webp|hdr|exr)$/i, // Textures
		/\.(mp4|webm|ogv)$/i,          // Video textures
		/three.*\.js$/i                 // Three.js modules
	],
	api: [
		/\/api\/v1\/gpu\//,
		/\/api\/v1\/wasm\//,
		/\/api\/v1\/cache\//,
		/\/api\/v1\/performance\//
	]
};

const CACHE_CONFIG = {
	maxAge: {
		webgpu: 7 * 24 * 60 * 60 * 1000,    // 7 days
		wasm: 30 * 24 * 60 * 60 * 1000,     // 30 days
		assets3d: 7 * 24 * 60 * 60 * 1000,  // 7 days
		api: 5 * 60 * 1000,                 // 5 minutes
		static: 24 * 60 * 60 * 1000         // 24 hours
	},
	maxEntries: {
		webgpu: 100,
		wasm: 50,
		assets3d: 200,
		api: 500,
		static: 1000
	},
	compression: {
		enabled: true,
		threshold: 1024,  // Compress files > 1KB
		types: ['text/javascript', 'application/json', 'text/css']
	}
};

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

let performanceMetrics = {
	cacheHits: 0,
	cacheMisses: 0,
	networkRequests: 0,
	totalLatency: 0,
	averageLatency: 0,
	memoryUsage: 0,
	compressionRatio: 0,
	lastCleanup: Date.now()
};

let clientPorts = new Map();

// ============================================================================
// INSTALLATION & ACTIVATION
// ============================================================================

self.addEventListener('install', (event) => {
	console.log('🔧 Service Worker installing...');
	
	event.waitUntil(
		Promise.all([
			preloadCriticalResources(),
			self.skipWaiting()
		])
	);
});

self.addEventListener('activate', (event) => {
	console.log('✅ Service Worker activating...');
	
	event.waitUntil(
		Promise.all([
			cleanupOldCaches(),
			self.clients.claim(),
			initializePerformanceMonitoring()
		])
	);
});

// ============================================================================
// CRITICAL RESOURCE PRELOADING
// ============================================================================

async function preloadCriticalResources() {
	const cache = await caches.open(CACHE_NAMES.webgpu);
	
	const criticalResources = [
		'/webgpu/shaders/matrix-multiply.wgsl',
		'/webgpu/shaders/vector-embedding.wgsl',
		'/wasm/simd-math.wasm',
		'/wasm/gpu-bridge.wasm',
		'/three/loaders/GLTFLoader.js'
	];

	return Promise.allSettled(
		criticalResources.map(async (url) => {
			try {
				const response = await fetch(url);
				if (response.ok) {
					await cache.put(url, response.clone());
					console.log(`📦 Preloaded: ${url}`);
				}
			} catch (error) {
				console.warn(`Failed to preload ${url}:`, error);
			}
		})
	);
}

// ============================================================================
// CACHE CLEANUP
// ============================================================================

async function cleanupOldCaches() {
	const allCacheNames = await caches.keys();
	const currentCacheNames = Object.values(CACHE_NAMES);
	
	return Promise.all(
		allCacheNames.map(async (cacheName) => {
			if (!currentCacheNames.includes(cacheName)) {
				console.log(`🗑️ Deleting old cache: ${cacheName}`);
				return caches.delete(cacheName);
			}
		})
	);
}

async function performCacheCleanup() {
	const now = Date.now();
	
	for (const [cacheType, cacheName] of Object.entries(CACHE_NAMES)) {
		const cache = await caches.open(cacheName);
		const keys = await cache.keys();
		
		let deletedCount = 0;
		const maxAge = CACHE_CONFIG.maxAge[cacheType];
		const maxEntries = CACHE_CONFIG.maxEntries[cacheType];
		
		// Age-based cleanup
		for (const request of keys) {
			const response = await cache.match(request);
			if (response) {
				const cachedTime = new Date(response.headers.get('sw-cached-time') || 0).getTime();
				if (now - cachedTime > maxAge) {
					await cache.delete(request);
					deletedCount++;
				}
			}
		}
		
		// Size-based cleanup (LRU)
		const remainingKeys = await cache.keys();
		if (remainingKeys.length > maxEntries) {
			const sortedKeys = await Promise.all(
				remainingKeys.map(async (key) => {
					const response = await cache.match(key);
					const lastAccessed = new Date(response.headers.get('sw-last-accessed') || 0).getTime();
					return { key, lastAccessed };
				})
			);
			
			sortedKeys.sort((a, b) => a.lastAccessed - b.lastAccessed);
			const keysToDelete = sortedKeys.slice(0, sortedKeys.length - maxEntries);
			
			for (const { key } of keysToDelete) {
				await cache.delete(key);
				deletedCount++;
			}
		}
		
		console.log(`🧹 Cleaned up ${deletedCount} entries from ${cacheType} cache`);
	}
	
	performanceMetrics.lastCleanup = now;
}

// ============================================================================
// REQUEST HANDLING
// ============================================================================

self.addEventListener('fetch', (event) => {
	const { request } = event;
	const url = new URL(request.url);
	
	// Skip non-HTTP requests
	if (!request.url.startsWith('http')) {
		return;
	}
	
	// Determine cache strategy
	const strategy = determineCacheStrategy(request);
	const cacheType = determineResourceType(request);
	
	event.respondWith(
		handleRequest(request, strategy, cacheType)
	);
});

function determineCacheStrategy(request) {
	const url = new URL(request.url);
	
	// API requests: network-first
	if (RESOURCE_PATTERNS.api.some(pattern => pattern.test(url.pathname))) {
		return CACHE_STRATEGIES['network-first'];
	}
	
	// WebGPU/WASM: cache-first
	if (RESOURCE_PATTERNS.webgpu.some(pattern => pattern.test(url.pathname)) ||
		RESOURCE_PATTERNS.wasm.some(pattern => pattern.test(url.pathname))) {
		return CACHE_STRATEGIES['cache-first'];
	}
	
	// 3D Assets: stale-while-revalidate
	if (RESOURCE_PATTERNS.assets3d.some(pattern => pattern.test(url.pathname))) {
		return CACHE_STRATEGIES['stale-while-revalidate'];
	}
	
	// Default: network-first
	return CACHE_STRATEGIES['network-first'];
}

function determineResourceType(request) {
	const url = new URL(request.url);
	
	for (const [type, patterns] of Object.entries(RESOURCE_PATTERNS)) {
		if (patterns.some(pattern => pattern.test(url.pathname))) {
			return type;
		}
	}
	
	return 'static';
}

async function handleRequest(request, strategy, cacheType) {
	const startTime = performance.now();
	const cacheName = CACHE_NAMES[cacheType] || CACHE_NAMES.static;
	const cache = await caches.open(cacheName);
	
	try {
		switch (strategy) {
			case CACHE_STRATEGIES['cache-first']:
				return await cacheFirst(request, cache);
			case CACHE_STRATEGIES['network-first']:
				return await networkFirst(request, cache);
			case CACHE_STRATEGIES['stale-while-revalidate']:
				return await staleWhileRevalidate(request, cache);
			case CACHE_STRATEGIES['network-only']:
				return await fetch(request);
			case CACHE_STRATEGIES['cache-only']:
				return await cache.match(request) || new Response('Not found', { status: 404 });
			default:
				return await networkFirst(request, cache);
		}
	} catch (error) {
		console.error('Request handling error:', error);
		return new Response('Service Worker Error', { status: 500 });
	} finally {
		const endTime = performance.now();
		updatePerformanceMetrics(endTime - startTime);
	}
}

// ============================================================================
// CACHE STRATEGIES IMPLEMENTATION
// ============================================================================

async function cacheFirst(request, cache) {
	const cachedResponse = await cache.match(request);
	
	if (cachedResponse) {
		performanceMetrics.cacheHits++;
		await updateResponseHeaders(cachedResponse, cache, request);
		return cachedResponse;
	}
	
	performanceMetrics.cacheMisses++;
	const networkResponse = await fetch(request);
	
	if (networkResponse.ok) {
		await cacheResponse(cache, request, networkResponse.clone());
	}
	
	return networkResponse;
}

async function networkFirst(request, cache) {
	try {
		performanceMetrics.networkRequests++;
		const networkResponse = await fetch(request);
		
		if (networkResponse.ok) {
			await cacheResponse(cache, request, networkResponse.clone());
		}
		
		return networkResponse;
	} catch (error) {
		const cachedResponse = await cache.match(request);
		
		if (cachedResponse) {
			performanceMetrics.cacheHits++;
			return cachedResponse;
		}
		
		throw error;
	}
}

async function staleWhileRevalidate(request, cache) {
	const cachedResponse = await cache.match(request);
	
	// Revalidate in background
	const networkResponsePromise = fetch(request).then(async (networkResponse) => {
		if (networkResponse.ok) {
			await cacheResponse(cache, request, networkResponse.clone());
		}
		return networkResponse;
	}).catch(() => {
		// Ignore network errors in background
	});
	
	if (cachedResponse) {
		performanceMetrics.cacheHits++;
		
		// Don't await the network response
		networkResponsePromise;
		
		return cachedResponse;
	}
	
	// If no cache, wait for network
	performanceMetrics.cacheMisses++;
	return await networkResponsePromise;
}

// ============================================================================
// RESPONSE CACHING & COMPRESSION
// ============================================================================

async function cacheResponse(cache, request, response) {
	const clonedResponse = response.clone();
	
	// Add caching headers
	const headers = new Headers(clonedResponse.headers);
	headers.set('sw-cached-time', new Date().toISOString());
	headers.set('sw-last-accessed', new Date().toISOString());
	headers.set('sw-cache-version', CACHE_VERSION);
	
	// Compress if applicable
	let body = await clonedResponse.arrayBuffer();
	let compressed = false;
	
	if (shouldCompress(clonedResponse, body.byteLength)) {
		try {
			const compressionStream = new CompressionStream('gzip');
			const stream = new Response(body).body.pipeThrough(compressionStream);
			body = await new Response(stream).arrayBuffer();
			headers.set('sw-compressed', 'true');
			headers.set('content-encoding', 'gzip');
			compressed = true;
			
			// Update compression metrics
			const originalSize = clonedResponse.headers.get('content-length') || body.byteLength;
			performanceMetrics.compressionRatio = body.byteLength / originalSize;
		} catch (error) {
			console.warn('Compression failed:', error);
		}
	}
	
	const responseToCache = new Response(body, {
		status: clonedResponse.status,
		statusText: clonedResponse.statusText,
		headers
	});
	
	await cache.put(request, responseToCache);
	console.log(`💾 Cached: ${request.url} (${compressed ? 'compressed' : 'uncompressed'})`);
}

async function updateResponseHeaders(response, cache, request) {
	const headers = new Headers(response.headers);
	headers.set('sw-last-accessed', new Date().toISOString());
	
	const updatedResponse = new Response(response.body, {
		status: response.status,
		statusText: response.statusText,
		headers
	});
	
	await cache.put(request, updatedResponse);
}

function shouldCompress(response, size) {
	if (!CACHE_CONFIG.compression.enabled) return false;
	if (size < CACHE_CONFIG.compression.threshold) return false;
	
	const contentType = response.headers.get('content-type') || '';
	return CACHE_CONFIG.compression.types.some(type => contentType.includes(type));
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

function initializePerformanceMonitoring() {
	// Periodic cleanup every 30 minutes
	setInterval(performCacheCleanup, 30 * 60 * 1000);
	
	// Performance reporting every 5 minutes
	setInterval(reportPerformanceMetrics, 5 * 60 * 1000);
	
	console.log('📊 Performance monitoring initialized');
}

function updatePerformanceMetrics(latency) {
	performanceMetrics.totalLatency += latency;
	const totalRequests = performanceMetrics.cacheHits + performanceMetrics.cacheMisses + performanceMetrics.networkRequests;
	
	if (totalRequests > 0) {
		performanceMetrics.averageLatency = performanceMetrics.totalLatency / totalRequests;
	}
	
	// Estimate memory usage
	if ('storage' in navigator && 'estimate' in navigator.storage) {
		navigator.storage.estimate().then(estimate => {
			performanceMetrics.memoryUsage = estimate.usage || 0;
		});
	}
}

function reportPerformanceMetrics() {
	const hitRate = performanceMetrics.cacheHits / 
		(performanceMetrics.cacheHits + performanceMetrics.cacheMisses) || 0;
	
	const report = {
		type: 'PERFORMANCE_UPDATE',
		metrics: {
			...performanceMetrics,
			hitRate: hitRate * 100,
			totalRequests: performanceMetrics.cacheHits + performanceMetrics.cacheMisses + performanceMetrics.networkRequests
		}
	};
	
	// Send to all connected clients
	clientPorts.forEach(port => {
		try {
			port.postMessage(report);
		} catch (error) {
			console.warn('Failed to send performance report:', error);
		}
	});
	
	console.log('📊 Performance Report:', report.metrics);
}

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

self.addEventListener('message', (event) => {
	const { data, ports } = event;
	
	switch (data.type) {
		case 'INIT_PORT':
			if (ports && ports[0]) {
				clientPorts.set(event.source?.id || Date.now(), ports[0]);
				console.log('📡 Client port registered');
			}
			break;
			
		case 'CACHE_INVALIDATE':
			handleCacheInvalidation(data.pattern);
			break;
			
		case 'PERFORMANCE_REQUEST':
			reportPerformanceMetrics();
			break;
			
		case 'CACHE_PRELOAD':
			preloadResources(data.urls);
			break;
			
		default:
			console.log('Unknown message type:', data.type);
	}
});

async function handleCacheInvalidation(pattern) {
	console.log(`🗑️ Cache invalidation requested: ${pattern}`);
	
	for (const cacheName of Object.values(CACHE_NAMES)) {
		const cache = await caches.open(cacheName);
		const keys = await cache.keys();
		
		for (const request of keys) {
			if (new RegExp(pattern).test(request.url)) {
				await cache.delete(request);
				console.log(`🗑️ Invalidated: ${request.url}`);
			}
		}
	}
}

async function preloadResources(urls) {
	console.log(`📦 Preloading ${urls.length} resources...`);
	
	const results = await Promise.allSettled(
		urls.map(async (url) => {
			try {
				const response = await fetch(url);
				if (response.ok) {
					const cacheType = determineResourceType({ url });
					const cache = await caches.open(CACHE_NAMES[cacheType] || CACHE_NAMES.static);
					await cacheResponse(cache, new Request(url), response.clone());
					return { url, success: true };
				}
			} catch (error) {
				return { url, success: false, error: error.message };
			}
		})
	);
	
	const successful = results.filter(r => r.value?.success).length;
	console.log(`📦 Preloaded ${successful}/${urls.length} resources`);
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

self.addEventListener('error', (event) => {
	console.error('Service Worker error:', event.error);
});

self.addEventListener('unhandledrejection', (event) => {
	console.error('Service Worker unhandled rejection:', event.reason);
});

console.log('🚀 WebGPU + WebAssembly Service Worker loaded');