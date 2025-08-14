// ================================================================================
// SERVICE WORKER FOR GPU THREADING & BACKGROUND PROCESSING
// ================================================================================
// WebGL • IndexedDB • WebAssembly • GPU Compute • Real-time Processing
// ================================================================================

// Service Worker Registration
const CACHE_NAME = 'legal-ai-v1';
const STATIC_CACHE = 'legal-ai-static-v1';
const DYNAMIC_CACHE = 'legal-ai-dynamic-v1';

// GPU Compute Worker
class GPUComputeWorker {
    constructor() {
        this.gl = null;
        this.programs = new Map();
        this.buffers = new Map();
        this.textures = new Map();
        this.indexedDB = null;
        this.isInitialized = false;
    }

    async initialize() {
        try {
            // Initialize WebGL2 context
            const canvas = new OffscreenCanvas(1, 1);
            this.gl = canvas.getContext('webgl2', {
                antialias: false,
                depth: false,
                stencil: false,
                alpha: false,
                preserveDrawingBuffer: false,
                powerPreference: 'high-performance',
                desynchronized: true
            });

            if (!this.gl) {
                throw new Error('WebGL2 not supported');
            }

            // Initialize IndexedDB
            await this.initIndexedDB();

            // Initialize GPU programs
            await this.initializeGPUPrograms();

            this.isInitialized = true;
            console.log('✅ GPU Compute Worker initialized');
            
            return true;
        } catch (error) {
            console.error('❌ GPU Worker initialization failed:', error);
            return false;
        }
    }

    async initIndexedDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('LegalAIGPUCache', 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.indexedDB = request.result;
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Vertex buffer store
                if (!db.objectStoreNames.contains('vertexBuffers')) {
                    const vertexStore = db.createObjectStore('vertexBuffers', { keyPath: 'id' });
                    vertexStore.createIndex('timestamp', 'timestamp');
                    vertexStore.createIndex('size', 'size');
                }
                
                // Compute results store
                if (!db.objectStoreNames.contains('computeResults')) {
                    const computeStore = db.createObjectStore('computeResults', { keyPath: 'id' });
                    computeStore.createIndex('timestamp', 'timestamp');
                    computeStore.createIndex('type', 'type');
                }
                
                // Embedding cache
                if (!db.objectStoreNames.contains('embeddings')) {
                    const embeddingStore = db.createObjectStore('embeddings', { keyPath: 'hash' });
                    embeddingStore.createIndex('timestamp', 'timestamp');
                    embeddingStore.createIndex('model', 'model');
                }
            };
        });
    }

    async initializeGPUPrograms() {
        // Vector similarity compute shader
        const vectorSimilarityVertex = `#version 300 es
            in vec4 position;
            in vec4 vector_a;
            in vec4 vector_b;
            
            out vec4 v_vector_a;
            out vec4 v_vector_b;
            
            void main() {
                gl_Position = position;
                v_vector_a = vector_a;
                v_vector_b = vector_b;
            }
        `;
        
        const vectorSimilarityFragment = `#version 300 es
            precision highp float;
            
            in vec4 v_vector_a;
            in vec4 v_vector_b;
            
            out vec4 fragColor;
            
            void main() {
                // Compute cosine similarity
                float dot_product = dot(v_vector_a, v_vector_b);
                float norm_a = length(v_vector_a);
                float norm_b = length(v_vector_b);
                
                float similarity = dot_product / (norm_a * norm_b);
                fragColor = vec4(similarity, similarity, similarity, 1.0);
            }
        `;
        
        this.programs.set('vectorSimilarity', this.createProgram(vectorSimilarityVertex, vectorSimilarityFragment));
        
        // K-means clustering compute shader
        const kmeansVertex = `#version 300 es
            in vec4 position;
            in vec4 dataPoint;
            in vec4 centroid;
            
            out vec4 v_dataPoint;
            out vec4 v_centroid;
            
            void main() {
                gl_Position = position;
                v_dataPoint = dataPoint;
                v_centroid = centroid;
            }
        `;
        
        const kmeansFragment = `#version 300 es
            precision highp float;
            
            in vec4 v_dataPoint;
            in vec4 v_centroid;
            
            out vec4 fragColor;
            
            void main() {
                // Compute distance to centroid
                vec4 diff = v_dataPoint - v_centroid;
                float distance = dot(diff, diff);
                fragColor = vec4(distance, distance, distance, 1.0);
            }
        `;
        
        this.programs.set('kmeans', this.createProgram(kmeansVertex, kmeansFragment));
        
        // Document embedding compute shader
        const embeddingVertex = `#version 300 es
            in vec4 position;
            in vec4 tokenData;
            
            out vec4 v_tokenData;
            
            void main() {
                gl_Position = position;
                v_tokenData = tokenData;
            }
        `;
        
        const embeddingFragment = `#version 300 es
            precision highp float;
            
            in vec4 v_tokenData;
            
            out vec4 fragColor;
            
            void main() {
                // Simplified embedding computation
                float embedding = sin(v_tokenData.x) * cos(v_tokenData.y) + 
                                sin(v_tokenData.z) * cos(v_tokenData.w);
                fragColor = vec4(embedding, embedding, embedding, 1.0);
            }
        `;
        
        this.programs.set('embedding', this.createProgram(embeddingVertex, embeddingFragment));
    }

    createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);
        
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(program));
            return null;
        }
        
        return program;
    }

    createShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
            return null;
        }
        
        return shader;
    }

    async createVertexBuffer(id, data, usage = 'STATIC_DRAW') {
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(data), this.gl[usage]);
        
        const bufferInfo = {
            id,
            buffer,
            size: data.length,
            usage,
            timestamp: Date.now()
        };
        
        this.buffers.set(id, bufferInfo);
        
        // Cache in IndexedDB
        await this.cacheVertexBuffer(bufferInfo, data);
        
        return bufferInfo;
    }

    async cacheVertexBuffer(bufferInfo, data) {
        if (!this.indexedDB) return;
        
        const transaction = this.indexedDB.transaction(['vertexBuffers'], 'readwrite');
        const store = transaction.objectStore('vertexBuffers');
        
        await store.put({
            id: bufferInfo.id,
            data: Array.from(data),
            size: bufferInfo.size,
            usage: bufferInfo.usage,
            timestamp: bufferInfo.timestamp
        });
    }

    async executeVectorSimilarity(vectorA, vectorB) {
        if (!this.isInitialized) {
            throw new Error('GPU Worker not initialized');
        }
        
        const program = this.programs.get('vectorSimilarity');
        if (!program) {
            throw new Error('Vector similarity program not found');
        }
        
        // Create buffers
        const bufferA = await this.createVertexBuffer('vector_a', vectorA);
        const bufferB = await this.createVertexBuffer('vector_b', vectorB);
        
        // Setup program
        this.gl.useProgram(program);
        
        // Bind buffers and execute
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, bufferA.buffer);
        const positionLocation = this.gl.getAttribLocation(program, 'vector_a');
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.vertexAttribPointer(positionLocation, 4, this.gl.FLOAT, false, 0, 0);
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, bufferB.buffer);
        const vectorBLocation = this.gl.getAttribLocation(program, 'vector_b');
        this.gl.enableVertexAttribArray(vectorBLocation);
        this.gl.vertexAttribPointer(vectorBLocation, 4, this.gl.FLOAT, false, 0, 0);
        
        // Render to compute
        this.gl.drawArrays(this.gl.POINTS, 0, Math.min(vectorA.length, vectorB.length) / 4);
        
        // Read results
        const results = new Float32Array(4);
        this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.FLOAT, results);
        
        return results[0]; // Similarity score
    }

    async executeKMeans(dataPoints, k, maxIterations = 100) {
        if (!this.isInitialized) {
            throw new Error('GPU Worker not initialized');
        }
        
        const program = this.programs.get('kmeans');
        if (!program) {
            throw new Error('K-means program not found');
        }
        
        // Initialize centroids randomly
        const centroids = [];
        for (let i = 0; i < k; i++) {
            const centroid = [];
            for (let j = 0; j < dataPoints[0].length; j++) {
                centroid.push(Math.random() * 2 - 1);
            }
            centroids.push(centroid);
        }
        
        let clusters = [];
        
        for (let iteration = 0; iteration < maxIterations; iteration++) {
            clusters = await this.assignClusters(dataPoints, centroids, program);
            const newCentroids = this.updateCentroids(dataPoints, clusters, k);
            
            // Check convergence
            if (this.centroidsConverged(centroids, newCentroids)) {
                break;
            }
            
            centroids.splice(0, centroids.length, ...newCentroids);
        }
        
        return { clusters, centroids };
    }

    async assignClusters(dataPoints, centroids, program) {
        this.gl.useProgram(program);
        
        const clusters = [];
        
        for (let i = 0; i < dataPoints.length; i++) {
            let minDistance = Infinity;
            let assignedCluster = 0;
            
            for (let j = 0; j < centroids.length; j++) {
                // Create buffers for this computation
                const pointBuffer = await this.createVertexBuffer(`point_${i}`, dataPoints[i]);
                const centroidBuffer = await this.createVertexBuffer(`centroid_${j}`, centroids[j]);
                
                // Bind and compute distance
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, pointBuffer.buffer);
                const pointLocation = this.gl.getAttribLocation(program, 'dataPoint');
                this.gl.enableVertexAttribArray(pointLocation);
                this.gl.vertexAttribPointer(pointLocation, 4, this.gl.FLOAT, false, 0, 0);
                
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, centroidBuffer.buffer);
                const centroidLocation = this.gl.getAttribLocation(program, 'centroid');
                this.gl.enableVertexAttribArray(centroidLocation);
                this.gl.vertexAttribPointer(centroidLocation, 4, this.gl.FLOAT, false, 0, 0);
                
                this.gl.drawArrays(this.gl.POINTS, 0, 1);
                
                // Read distance result
                const result = new Float32Array(4);
                this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.FLOAT, result);
                
                const distance = result[0];
                if (distance < minDistance) {
                    minDistance = distance;
                    assignedCluster = j;
                }
            }
            
            clusters.push(assignedCluster);
        }
        
        return clusters;
    }

    updateCentroids(dataPoints, clusters, k) {
        const newCentroids = [];
        
        for (let i = 0; i < k; i++) {
            const clusterPoints = dataPoints.filter((_, index) => clusters[index] === i);
            
            if (clusterPoints.length === 0) {
                // Keep old centroid if no points assigned
                newCentroids.push(Array(dataPoints[0].length).fill(0));
                continue;
            }
            
            const centroid = Array(dataPoints[0].length).fill(0);
            for (const point of clusterPoints) {
                for (let j = 0; j < point.length; j++) {
                    centroid[j] += point[j];
                }
            }
            
            for (let j = 0; j < centroid.length; j++) {
                centroid[j] /= clusterPoints.length;
            }
            
            newCentroids.push(centroid);
        }
        
        return newCentroids;
    }

    centroidsConverged(oldCentroids, newCentroids, threshold = 0.001) {
        for (let i = 0; i < oldCentroids.length; i++) {
            for (let j = 0; j < oldCentroids[i].length; j++) {
                if (Math.abs(oldCentroids[i][j] - newCentroids[i][j]) > threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    async computeEmbedding(tokens) {
        if (!this.isInitialized) {
            throw new Error('GPU Worker not initialized');
        }
        
        // Check cache first
        const hash = this.hashTokens(tokens);
        const cached = await this.getCachedEmbedding(hash);
        if (cached) {
            return cached.embedding;
        }
        
        const program = this.programs.get('embedding');
        if (!program) {
            throw new Error('Embedding program not found');
        }
        
        this.gl.useProgram(program);
        
        // Process tokens in batches
        const embedding = [];
        const batchSize = 256;
        
        for (let i = 0; i < tokens.length; i += batchSize) {
            const batch = tokens.slice(i, i + batchSize);
            const tokenData = this.tokenizeForGPU(batch);
            
            const tokenBuffer = await this.createVertexBuffer(`tokens_${i}`, tokenData);
            
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, tokenBuffer.buffer);
            const tokenLocation = this.gl.getAttribLocation(program, 'tokenData');
            this.gl.enableVertexAttribArray(tokenLocation);
            this.gl.vertexAttribPointer(tokenLocation, 4, this.gl.FLOAT, false, 0, 0);
            
            this.gl.drawArrays(this.gl.POINTS, 0, batch.length);
            
            // Read embedding results
            const batchEmbedding = new Float32Array(batch.length * 4);
            this.gl.readPixels(0, 0, batch.length, 1, this.gl.RGBA, this.gl.FLOAT, batchEmbedding);
            
            embedding.push(...batchEmbedding);
        }
        
        // Cache result
        await this.cacheEmbedding(hash, embedding);
        
        return embedding;
    }

    tokenizeForGPU(tokens) {
        const tokenData = [];
        for (const token of tokens) {
            // Simple tokenization for GPU processing
            const hash = this.simpleHash(token);
            tokenData.push(
                (hash & 0xFF) / 255,
                ((hash >> 8) & 0xFF) / 255,
                ((hash >> 16) & 0xFF) / 255,
                ((hash >> 24) & 0xFF) / 255
            );
        }
        return tokenData;
    }

    simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash;
    }

    hashTokens(tokens) {
        return btoa(JSON.stringify(tokens));
    }

    async getCachedEmbedding(hash) {
        if (!this.indexedDB) return null;
        
        const transaction = this.indexedDB.transaction(['embeddings'], 'readonly');
        const store = transaction.objectStore('embeddings');
        
        return new Promise((resolve) => {
            const request = store.get(hash);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(null);
        });
    }

    async cacheEmbedding(hash, embedding) {
        if (!this.indexedDB) return;
        
        const transaction = this.indexedDB.transaction(['embeddings'], 'readwrite');
        const store = transaction.objectStore('embeddings');
        
        await store.put({
            hash,
            embedding: Array.from(embedding),
            model: 'gpu-compute-v1',
            timestamp: Date.now()
        });
    }

    async cleanup() {
        // Clean up GPU resources
        this.buffers.forEach(({ buffer }) => {
            this.gl.deleteBuffer(buffer);
        });
        
        this.programs.forEach((program) => {
            this.gl.deleteProgram(program);
        });
        
        if (this.indexedDB) {
            this.indexedDB.close();
        }
        
        this.buffers.clear();
        this.programs.clear();
        this.textures.clear();
    }
}

// Service Worker Event Handlers
const gpuWorker = new GPUComputeWorker();
let isGPUInitialized = false;

self.addEventListener('install', (event) => {
    console.log('🔧 Service Worker installing...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => {
            return cache.addAll([
                '/',
                '/manifest.json',
                '/static/js/gpu-worker.js',
                '/static/css/main.css'
            ]);
        })
    );
    
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    console.log('✅ Service Worker activated');
    
    event.waitUntil(
        Promise.all([
            // Clean old caches
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames.filter((cacheName) => {
                        return cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE;
                    }).map((cacheName) => {
                        return caches.delete(cacheName);
                    })
                );
            }),
            
            // Initialize GPU worker
            gpuWorker.initialize().then((success) => {
                isGPUInitialized = success;
                console.log(`🎮 GPU Worker: ${success ? 'Initialized' : 'Failed'}`);
            })
        ])
    );
    
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Handle API requests
    if (url.pathname.startsWith('/api/gpu/')) {
        event.respondWith(handleGPURequest(request));
        return;
    }
    
    // Handle static assets
    event.respondWith(
        caches.match(request).then((response) => {
            return response || fetch(request).then((fetchResponse) => {
                return caches.open(DYNAMIC_CACHE).then((cache) => {
                    cache.put(request, fetchResponse.clone());
                    return fetchResponse;
                });
            });
        })
    );
});

self.addEventListener('message', async (event) => {
    const { type, data, id } = event.data;
    
    try {
        let result;
        
        switch (type) {
            case 'GPU_VECTOR_SIMILARITY':
                result = await gpuWorker.executeVectorSimilarity(data.vectorA, data.vectorB);
                break;
                
            case 'GPU_KMEANS':
                result = await gpuWorker.executeKMeans(data.dataPoints, data.k, data.maxIterations);
                break;
                
            case 'GPU_EMBEDDING':
                result = await gpuWorker.computeEmbedding(data.tokens);
                break;
                
            case 'GPU_CREATE_BUFFER':
                result = await gpuWorker.createVertexBuffer(data.id, data.data, data.usage);
                break;
                
            case 'GPU_STATUS':
                result = {
                    initialized: isGPUInitialized,
                    buffers: gpuWorker.buffers.size,
                    programs: gpuWorker.programs.size
                };
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
        
        event.ports[0].postMessage({
            id,
            success: true,
            result
        });
        
    } catch (error) {
        event.ports[0].postMessage({
            id,
            success: false,
            error: error.message
        });
    }
});

async function handleGPURequest(request) {
    if (!isGPUInitialized) {
        return new Response(JSON.stringify({
            error: 'GPU Worker not initialized'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
    
    const url = new URL(request.url);
    const endpoint = url.pathname.split('/').pop();
    
    try {
        const requestData = await request.json();
        let result;
        
        switch (endpoint) {
            case 'similarity':
                result = await gpuWorker.executeVectorSimilarity(
                    requestData.vectorA, 
                    requestData.vectorB
                );
                break;
                
            case 'clustering':
                result = await gpuWorker.executeKMeans(
                    requestData.dataPoints, 
                    requestData.k, 
                    requestData.maxIterations
                );
                break;
                
            case 'embedding':
                result = await gpuWorker.computeEmbedding(requestData.tokens);
                break;
                
            default:
                throw new Error(`Unknown endpoint: ${endpoint}`);
        }
        
        return new Response(JSON.stringify({
            success: true,
            result,
            timestamp: Date.now(),
            gpu: true
        }), {
            headers: { 'Content-Type': 'application/json' }
        });
        
    } catch (error) {
        return new Response(JSON.stringify({
            success: false,
            error: error.message,
            timestamp: Date.now()
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Background sync for offline capabilities
self.addEventListener('sync', (event) => {
    if (event.tag === 'gpu-compute-sync') {
        event.waitUntil(syncGPUComputations());
    }
});

async function syncGPUComputations() {
    // Sync pending GPU computations when back online
    console.log('🔄 Syncing GPU computations...');
    
    if (!isGPUInitialized) return;
    
    // Process any queued computations from IndexedDB
    const transaction = gpuWorker.indexedDB.transaction(['computeResults'], 'readonly');
    const store = transaction.objectStore('computeResults');
    
    return new Promise((resolve) => {
        const request = store.getAll();
        request.onsuccess = async () => {
            const pendingComputations = request.result.filter(
                item => item.status === 'pending'
            );
            
            for (const computation of pendingComputations) {
                try {
                    // Re-process computation
                    await processComputation(computation);
                } catch (error) {
                    console.error('Sync computation error:', error);
                }
            }
            
            resolve();
        };
    });
}

async function processComputation(computation) {
    // Process individual computation
    console.log('Processing computation:', computation.id);
}

console.log('🚀 GPU Service Worker loaded and ready');
