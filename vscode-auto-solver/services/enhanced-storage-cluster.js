/**
 * 🗄️ Enhanced Multi-Storage Cluster Service
 * Integrates pgvector, Redis, Qdrant, LangChain.js, Loki.js, IndexedDB, Fuse.js
 * Service Workers with concurrency and multi-cluster support
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import cluster from 'cluster';
import os from 'os';
import { performance } from 'perf_hooks';

// Storage Engines
class MultiStorageCluster {
    constructor() {
        this.storageEngines = new Map();
        this.workers = new Map();
        this.serviceWorkers = new Set();
        this.concurrencyLevel = os.cpus().length;
        this.loadBalancer = new LoadBalancer();
        
        this.initializeStorageEngines();
    }

    async initializeStorageEngines() {
        console.log('🗄️ Initializing Multi-Storage Cluster...');
        
        // PostgreSQL with pgvector
        this.storageEngines.set('postgres', new PostgreSQLVectorStore());
        
        // Redis Native Windows
        this.storageEngines.set('redis', new RedisNativeStore());
        
        // Qdrant Vector Database
        this.storageEngines.set('qdrant', new QdrantVectorStore());
        
        // LangChain.js Integration
        this.storageEngines.set('langchain', new LangChainJSStore());
        
        // Loki.js In-Memory Database
        this.storageEngines.set('loki', new LokiJSStore());
        
        // IndexedDB Browser Storage
        this.storageEngines.set('indexeddb', new IndexedDBStore());
        
        // Fuse.js Fuzzy Search
        this.storageEngines.set('fuse', new FuseJSSearchEngine());
        
        // Initialize all engines in parallel
        const initPromises = Array.from(this.storageEngines.values()).map(engine => engine.initialize());
        await Promise.allSettled(initPromises);
        
        console.log('✅ Multi-Storage Cluster initialized');
    }

    async startServiceWorkers() {
        for (let i = 0; i < this.concurrencyLevel; i++) {
            const worker = new Worker(new URL(import.meta.url), {
                workerData: { 
                    workerId: i,
                    type: 'storage-service-worker'
                }
            });
            
            this.serviceWorkers.add(worker);
            
            worker.on('message', (message) => {
                this.handleServiceWorkerMessage(i, message);
            });
        }
    }
}

// PostgreSQL with pgvector
class PostgreSQLVectorStore {
    constructor() {
        this.pool = null;
        this.vectorDimension = 768;
    }

    async initialize() {
        const { Pool } = await import('pg');
        this.pool = new Pool({
            connectionString: process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db',
            max: 20,
            idleTimeoutMillis: 30000,
        });

        await this.createVectorTables();
        console.log('✅ PostgreSQL Vector Store initialized');
    }

    async createVectorTables() {
        const sql = `
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Enhanced semantic embeddings table
        CREATE TABLE IF NOT EXISTS semantic_embeddings (
            id SERIAL PRIMARY KEY,
            content_hash TEXT UNIQUE NOT NULL,
            content_text TEXT NOT NULL,
            content_type TEXT DEFAULT 'code',
            language TEXT,
            file_path TEXT,
            
            -- Vector embeddings (768 dimensions for sentence-transformers)
            embeddings vector(768),
            
            -- Metadata as JSONB
            metadata JSONB DEFAULT '{}'::jsonb,
            semantic_features JSONB DEFAULT '{}'::jsonb,
            
            -- Performance fields
            processing_time_ms INTEGER,
            confidence_score FLOAT DEFAULT 0,
            
            -- Clustering fields
            cluster_id TEXT,
            similarity_group INTEGER,
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );

        -- Vector similarity search index (HNSW for better performance)
        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
        ON semantic_embeddings USING hnsw (embeddings vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

        -- JSONB GIN indexes for fast metadata queries
        CREATE INDEX IF NOT EXISTS idx_metadata_gin ON semantic_embeddings USING GIN (metadata);
        CREATE INDEX IF NOT EXISTS idx_semantic_features_gin ON semantic_embeddings USING GIN (semantic_features);
        
        -- Multi-column indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_content_type_lang ON semantic_embeddings (content_type, language);
        CREATE INDEX IF NOT EXISTS idx_file_path_hash ON semantic_embeddings (file_path, content_hash);
        CREATE INDEX IF NOT EXISTS idx_cluster_similarity ON semantic_embeddings (cluster_id, similarity_group);
        `;

        await this.pool.query(sql);
    }

    async storeEmbedding(data) {
        const query = `
        INSERT INTO semantic_embeddings (
            content_hash, content_text, content_type, language, file_path,
            embeddings, metadata, semantic_features, processing_time_ms, 
            confidence_score, cluster_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (content_hash) 
        DO UPDATE SET 
            content_text = EXCLUDED.content_text,
            embeddings = EXCLUDED.embeddings,
            metadata = EXCLUDED.metadata,
            semantic_features = EXCLUDED.semantic_features,
            updated_at = NOW()
        RETURNING id;
        `;

        return await this.pool.query(query, [
            data.contentHash,
            data.contentText,
            data.contentType || 'code',
            data.language,
            data.filePath,
            `[${data.embeddings.join(',')}]`, // Convert array to vector format
            JSON.stringify(data.metadata || {}),
            JSON.stringify(data.semanticFeatures || {}),
            data.processingTimeMs || 0,
            data.confidenceScore || 0,
            data.clusterId
        ]);
    }

    async similaritySearch(queryEmbedding, options = {}) {
        const {
            limit = 10,
            threshold = 0.7,
            contentType = null,
            language = null,
            clusterId = null
        } = options;

        let whereClause = '';
        const params = [`[${queryEmbedding.join(',')}]`, limit];
        let paramIndex = 3;

        if (contentType) {
            whereClause += ` AND content_type = $${paramIndex++}`;
            params.push(contentType);
        }

        if (language) {
            whereClause += ` AND language = $${paramIndex++}`;
            params.push(language);
        }

        if (clusterId) {
            whereClause += ` AND cluster_id = $${paramIndex++}`;
            params.push(clusterId);
        }

        const query = `
        SELECT 
            id, content_hash, content_text, content_type, language, file_path,
            metadata, semantic_features, confidence_score, cluster_id,
            (embeddings <=> $1::vector) as distance,
            (1 - (embeddings <=> $1::vector)) as similarity
        FROM semantic_embeddings 
        WHERE (1 - (embeddings <=> $1::vector)) > ${threshold} ${whereClause}
        ORDER BY embeddings <=> $1::vector
        LIMIT $2;
        `;

        return await this.pool.query(query, params);
    }

    async clusterEmbeddings(options = {}) {
        const { method = 'kmeans', clusters = 10 } = options;
        
        // Use PostgreSQL for basic clustering, or delegate to Python/R for advanced clustering
        const query = `
        WITH clustered AS (
            SELECT id, embeddings,
                   kmeans(embeddings::vector, ${clusters}) OVER () as cluster_id
            FROM semantic_embeddings
        )
        UPDATE semantic_embeddings 
        SET cluster_id = clustered.cluster_id::text,
            similarity_group = clustered.cluster_id
        FROM clustered
        WHERE semantic_embeddings.id = clustered.id;
        `;

        return await this.pool.query(query);
    }
}

// Redis Native Windows Store
class RedisNativeStore {
    constructor() {
        this.client = null;
        this.subscriber = null;
        this.publisher = null;
    }

    async initialize() {
        // Use ioredis for better Windows compatibility
        const Redis = await import('ioredis');
        
        this.client = new Redis.default({
            host: 'localhost',
            port: 6379,
            maxRetriesPerRequest: 3,
            retryDelayOnFailover: 100,
            lazyConnect: true,
        });

        this.subscriber = this.client.duplicate();
        this.publisher = this.client.duplicate();

        await this.client.connect();
        await this.subscriber.connect();
        await this.publisher.connect();

        // Set up Redis Streams for real-time processing
        await this.setupStreams();

        console.log('✅ Redis Native Store initialized');
    }

    async setupStreams() {
        // Create streams for different data types
        const streams = [
            'semantic-processing',
            'error-analysis', 
            'solution-patterns',
            'performance-metrics'
        ];

        for (const stream of streams) {
            try {
                await this.client.xgroup('CREATE', stream, 'processors', '$', 'MKSTREAM');
            } catch (error) {
                // Group might already exist
            }
        }
    }

    async storeCache(key, data, ttl = 3600) {
        const serialized = JSON.stringify(data);
        if (ttl > 0) {
            return await this.client.setex(key, ttl, serialized);
        } else {
            return await this.client.set(key, serialized);
        }
    }

    async getCache(key) {
        const data = await this.client.get(key);
        return data ? JSON.parse(data) : null;
    }

    async pushToStream(streamName, data) {
        return await this.client.xadd(streamName, '*', 'data', JSON.stringify(data));
    }

    async readFromStream(streamName, group, consumer) {
        const results = await this.client.xreadgroup('GROUP', group, consumer, 'COUNT', 10, 'STREAMS', streamName, '>');
        
        return results?.map(([stream, messages]) => 
            messages.map(([id, fields]) => ({
                id,
                stream,
                data: JSON.parse(fields[1]) // Assuming 'data' field
            }))
        ).flat() || [];
    }

    async storeVector(key, vector, metadata = {}) {
        // Store vector as hash with metadata
        const pipe = this.client.pipeline();
        pipe.hset(key, 'vector', JSON.stringify(vector));
        pipe.hset(key, 'metadata', JSON.stringify(metadata));
        pipe.hset(key, 'timestamp', Date.now());
        return await pipe.exec();
    }

    async searchSimilarVectors(queryVector, options = {}) {
        // Redis doesn't have native vector similarity search
        // This would typically delegate to Redis modules like RedisAI or RediSearch
        // For now, we'll implement a simple approach
        const { limit = 10, keyPattern = 'vector:*' } = options;
        
        const keys = await this.client.keys(keyPattern);
        const similarities = [];
        
        for (const key of keys.slice(0, 100)) { // Limit for performance
            const vectorStr = await this.client.hget(key, 'vector');
            if (vectorStr) {
                const vector = JSON.parse(vectorStr);
                const similarity = this.cosineSimilarity(queryVector, vector);
                similarities.push({ key, similarity, vector });
            }
        }
        
        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }
}

// Qdrant Vector Database
class QdrantVectorStore {
    constructor() {
        this.client = null;
        this.collectionName = 'vscode-semantic';
    }

    async initialize() {
        const { QdrantClient } = await import('@qdrant/js-client-rest');
        
        this.client = new QdrantClient({
            url: process.env.QDRANT_URL || 'http://localhost:6333'
        });

        await this.setupCollection();
        console.log('✅ Qdrant Vector Store initialized');
    }

    async setupCollection() {
        try {
            // Create collection with optimized settings
            await this.client.createCollection(this.collectionName, {
                vectors: {
                    size: 768,
                    distance: 'Cosine'
                },
                optimizers_config: {
                    default_segment_number: 2
                },
                hnsw_config: {
                    m: 16,
                    ef_construct: 200,
                    full_scan_threshold: 10000
                }
            });
        } catch (error) {
            // Collection might already exist
            console.log('Qdrant collection exists or creation failed:', error.message);
        }
    }

    async upsertPoints(points) {
        const formattedPoints = points.map(point => ({
            id: point.id || this.generateId(),
            vector: point.vector,
            payload: point.metadata || {}
        }));

        return await this.client.upsert(this.collectionName, {
            wait: true,
            points: formattedPoints
        });
    }

    async search(queryVector, options = {}) {
        const {
            limit = 10,
            scoreThreshold = 0.7,
            filter = null
        } = options;

        return await this.client.search(this.collectionName, {
            vector: queryVector,
            limit,
            score_threshold: scoreThreshold,
            with_payload: true,
            filter
        });
    }

    async createIndex(field) {
        return await this.client.createFieldIndex(this.collectionName, field, {
            field_type: 'keyword'
        });
    }

    generateId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
}

// LangChain.js Integration
class LangChainJSStore {
    constructor() {
        this.vectorStore = null;
        this.embeddings = null;
        this.retriever = null;
    }

    async initialize() {
        // Import LangChain modules
        const { MemoryVectorStore } = await import('langchain/vectorstores/memory');
        const { OpenAIEmbeddings } = await import('langchain/embeddings/openai');
        const { RecursiveCharacterTextSplitter } = await import('langchain/text_splitter');
        const { Document } = await import('langchain/document');

        // Use local embeddings if available, otherwise OpenAI
        if (process.env.USE_LOCAL_EMBEDDINGS === 'true') {
            // Use a local embedding service (like sentence-transformers via API)
            this.embeddings = new LocalEmbeddings();
        } else {
            this.embeddings = new OpenAIEmbeddings({
                openAIApiKey: process.env.OPENAI_API_KEY
            });
        }

        this.vectorStore = new MemoryVectorStore(this.embeddings);
        this.textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        });

        this.retriever = this.vectorStore.asRetriever({
            searchType: 'similarity',
            searchKwargs: { k: 10 }
        });

        console.log('✅ LangChain.js Store initialized');
    }

    async addDocuments(documents) {
        const langchainDocs = documents.map(doc => new Document({
            pageContent: doc.content,
            metadata: doc.metadata || {}
        }));

        // Split documents into chunks
        const splitDocs = await this.textSplitter.splitDocuments(langchainDocs);
        
        // Add to vector store
        await this.vectorStore.addDocuments(splitDocs);
        
        return splitDocs.length;
    }

    async similaritySearch(query, options = {}) {
        const { k = 10 } = options;
        return await this.vectorStore.similaritySearch(query, k);
    }

    async retrieveRelevant(query) {
        return await this.retriever.getRelevantDocuments(query);
    }

    async createChain() {
        const { RetrievalQAChain } = await import('langchain/chains');
        const { ChatOpenAI } = await import('langchain/chat_models/openai');

        const model = new ChatOpenAI({
            temperature: 0,
            modelName: 'gpt-3.5-turbo'
        });

        return RetrievalQAChain.fromLLM(model, this.retriever);
    }
}

// Local Embeddings Implementation
class LocalEmbeddings {
    constructor() {
        this.apiUrl = process.env.LOCAL_EMBEDDINGS_URL || 'http://localhost:8000/embeddings';
    }

    async embedDocuments(texts) {
        const response = await fetch(this.apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts })
        });

        if (!response.ok) {
            throw new Error(`Embedding API error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.embeddings;
    }

    async embedQuery(text) {
        const embeddings = await this.embedDocuments([text]);
        return embeddings[0];
    }
}

// Loki.js In-Memory Database
class LokiJSStore {
    constructor() {
        this.db = null;
        this.collections = new Map();
    }

    async initialize() {
        const loki = await import('lokijs');
        
        this.db = new loki.default('vscode-auto-solver.db', {
            autoload: false,
            autoloadCallback: this.databaseInitialize.bind(this),
            autosave: true,
            autosaveInterval: 4000
        });

        // Create collections
        this.createCollections();
        
        console.log('✅ Loki.js Store initialized');
    }

    createCollections() {
        // Semantic analysis collection
        const semanticCollection = this.db.addCollection('semantic_analysis', {
            indices: ['contentHash', 'filePath', 'language', 'timestamp']
        });
        
        // Problem patterns collection
        const patternsCollection = this.db.addCollection('problem_patterns', {
            indices: ['problemType', 'language', 'frequency']
        });

        // Solution cache collection
        const solutionsCollection = this.db.addCollection('solution_cache', {
            indices: ['problemHash', 'confidence', 'timestamp']
        });

        this.collections.set('semantic', semanticCollection);
        this.collections.set('patterns', patternsCollection);
        this.collections.set('solutions', solutionsCollection);
    }

    databaseInitialize() {
        // Database loaded callback
        console.log('Loki.js database initialized');
    }

    insert(collectionName, document) {
        const collection = this.collections.get(collectionName);
        if (collection) {
            return collection.insert({
                ...document,
                timestamp: Date.now()
            });
        }
        throw new Error(`Collection ${collectionName} not found`);
    }

    find(collectionName, query = {}) {
        const collection = this.collections.get(collectionName);
        if (collection) {
            return collection.find(query);
        }
        return [];
    }

    update(collectionName, document) {
        const collection = this.collections.get(collectionName);
        if (collection) {
            return collection.update(document);
        }
        throw new Error(`Collection ${collectionName} not found`);
    }

    remove(collectionName, document) {
        const collection = this.collections.get(collectionName);
        if (collection) {
            return collection.remove(document);
        }
        throw new Error(`Collection ${collectionName} not found`);
    }

    // Advanced queries
    chainQuery(collectionName) {
        const collection = this.collections.get(collectionName);
        if (collection) {
            return collection.chain();
        }
        throw new Error(`Collection ${collectionName} not found`);
    }
}

// IndexedDB Store (for browser-side caching)
class IndexedDBStore {
    constructor() {
        this.dbName = 'VSCodeAutoSolver';
        this.dbVersion = 1;
        this.db = null;
    }

    async initialize() {
        if (typeof window === 'undefined') {
            console.log('⚠️ IndexedDB not available in Node.js environment');
            return;
        }

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                // Create object stores
                if (!db.objectStoreNames.contains('semantic_cache')) {
                    const semanticStore = db.createObjectStore('semantic_cache', { keyPath: 'id', autoIncrement: true });
                    semanticStore.createIndex('contentHash', 'contentHash', { unique: false });
                    semanticStore.createIndex('filePath', 'filePath', { unique: false });
                    semanticStore.createIndex('timestamp', 'timestamp', { unique: false });
                }

                if (!db.objectStoreNames.contains('solution_cache')) {
                    const solutionStore = db.createObjectStore('solution_cache', { keyPath: 'id', autoIncrement: true });
                    solutionStore.createIndex('problemHash', 'problemHash', { unique: false });
                    solutionStore.createIndex('confidence', 'confidence', { unique: false });
                }
            };
        });
    }

    async store(storeName, data) {
        if (!this.db) return null;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readwrite');
            const store = transaction.objectStore(storeName);
            const request = store.add({ ...data, timestamp: Date.now() });

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async get(storeName, key) {
        if (!this.db) return null;

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            const request = store.get(key);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async query(storeName, indexName, value) {
        if (!this.db) return [];

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            const index = store.index(indexName);
            const request = index.getAll(value);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
}

// Fuse.js Fuzzy Search Engine
class FuseJSSearchEngine {
    constructor() {
        this.indices = new Map();
        this.defaultOptions = {
            includeScore: true,
            threshold: 0.3,
            location: 0,
            distance: 100,
            maxPatternLength: 32,
            minMatchCharLength: 2,
            keys: ['content', 'filePath', 'language']
        };
    }

    async initialize() {
        const Fuse = await import('fuse.js');
        this.Fuse = Fuse.default;
        console.log('✅ Fuse.js Search Engine initialized');
    }

    createIndex(name, documents, options = {}) {
        const fuseOptions = { ...this.defaultOptions, ...options };
        const fuse = new this.Fuse(documents, fuseOptions);
        this.indices.set(name, {
            fuse,
            documents,
            options: fuseOptions
        });
        return fuse;
    }

    search(indexName, query, options = {}) {
        const index = this.indices.get(indexName);
        if (!index) {
            throw new Error(`Index ${indexName} not found`);
        }

        const results = index.fuse.search(query, options);
        return results.map(result => ({
            item: result.item,
            score: result.score,
            matches: result.matches
        }));
    }

    updateIndex(indexName, documents) {
        const index = this.indices.get(indexName);
        if (index) {
            index.documents = documents;
            index.fuse = new this.Fuse(documents, index.options);
        }
    }

    removeFromIndex(indexName, predicate) {
        const index = this.indices.get(indexName);
        if (index) {
            index.documents = index.documents.filter(doc => !predicate(doc));
            index.fuse = new this.Fuse(index.documents, index.options);
        }
    }

    // Advanced search with multiple indices
    multiIndexSearch(query, indexNames = null) {
        const indicesToSearch = indexNames || Array.from(this.indices.keys());
        const allResults = [];

        for (const indexName of indicesToSearch) {
            const results = this.search(indexName, query);
            allResults.push(...results.map(r => ({ ...r, index: indexName })));
        }

        // Combine and sort by score
        return allResults.sort((a, b) => a.score - b.score);
    }
}

// Load Balancer for Multi-Storage Access
class LoadBalancer {
    constructor() {
        this.strategies = {
            'round-robin': new RoundRobinStrategy(),
            'least-connections': new LeastConnectionsStrategy(),
            'weighted': new WeightedStrategy(),
            'performance-based': new PerformanceBasedStrategy()
        };
        this.currentStrategy = this.strategies['performance-based'];
        this.healthCheck = new HealthCheck();
    }

    setStrategy(strategyName) {
        if (this.strategies[strategyName]) {
            this.currentStrategy = this.strategies[strategyName];
        }
    }

    async route(operation, storageEngines, options = {}) {
        const availableEngines = await this.healthCheck.getHealthyEngines(storageEngines);
        const selectedEngine = this.currentStrategy.select(availableEngines, operation, options);
        
        try {
            const result = await selectedEngine.execute(operation, options);
            this.currentStrategy.recordSuccess(selectedEngine, performance.now() - options.startTime);
            return result;
        } catch (error) {
            this.currentStrategy.recordFailure(selectedEngine, error);
            throw error;
        }
    }
}

// Load Balancing Strategies
class RoundRobinStrategy {
    constructor() {
        this.currentIndex = 0;
    }

    select(engines) {
        const engine = engines[this.currentIndex % engines.length];
        this.currentIndex++;
        return engine;
    }

    recordSuccess() { /* No-op for round robin */ }
    recordFailure() { /* No-op for round robin */ }
}

class LeastConnectionsStrategy {
    constructor() {
        this.connections = new Map();
    }

    select(engines) {
        let minConnections = Infinity;
        let selectedEngine = engines[0];

        for (const engine of engines) {
            const connections = this.connections.get(engine) || 0;
            if (connections < minConnections) {
                minConnections = connections;
                selectedEngine = engine;
            }
        }

        this.connections.set(selectedEngine, minConnections + 1);
        return selectedEngine;
    }

    recordSuccess(engine) {
        const current = this.connections.get(engine) || 1;
        this.connections.set(engine, Math.max(0, current - 1));
    }

    recordFailure(engine) {
        this.recordSuccess(engine); // Same logic for connection counting
    }
}

class WeightedStrategy {
    constructor() {
        this.weights = new Map();
        this.defaults = {
            'postgres': 0.3,
            'redis': 0.2,
            'qdrant': 0.2,
            'langchain': 0.1,
            'loki': 0.1,
            'fuse': 0.1
        };
    }

    select(engines, operation) {
        const weights = this.getWeights(engines, operation);
        const totalWeight = weights.reduce((sum, w) => sum + w.weight, 0);
        const random = Math.random() * totalWeight;

        let current = 0;
        for (const { engine, weight } of weights) {
            current += weight;
            if (random <= current) {
                return engine;
            }
        }

        return engines[0]; // Fallback
    }

    getWeights(engines, operation) {
        return engines.map(engine => ({
            engine,
            weight: this.weights.get(engine) || this.defaults[engine.name] || 0.1
        }));
    }

    recordSuccess() { /* Could adjust weights based on performance */ }
    recordFailure() { /* Could adjust weights based on failures */ }
}

class PerformanceBasedStrategy {
    constructor() {
        this.metrics = new Map();
        this.windowSize = 100; // Keep last 100 measurements
    }

    select(engines) {
        let bestEngine = engines[0];
        let bestScore = -1;

        for (const engine of engines) {
            const score = this.calculatePerformanceScore(engine);
            if (score > bestScore) {
                bestScore = score;
                bestEngine = engine;
            }
        }

        return bestEngine;
    }

    calculatePerformanceScore(engine) {
        const metrics = this.metrics.get(engine);
        if (!metrics || metrics.length === 0) {
            return 1.0; // Default score for new engines
        }

        const avgResponseTime = metrics.reduce((sum, m) => sum + m.responseTime, 0) / metrics.length;
        const errorRate = metrics.filter(m => m.error).length / metrics.length;
        
        // Lower response time and error rate = higher score
        return 1.0 / (avgResponseTime / 1000 + 1) * (1 - errorRate);
    }

    recordSuccess(engine, responseTime) {
        this.addMetric(engine, { responseTime, error: false, timestamp: Date.now() });
    }

    recordFailure(engine, error) {
        this.addMetric(engine, { responseTime: 0, error: true, timestamp: Date.now() });
    }

    addMetric(engine, metric) {
        if (!this.metrics.has(engine)) {
            this.metrics.set(engine, []);
        }

        const engineMetrics = this.metrics.get(engine);
        engineMetrics.push(metric);

        // Keep only recent measurements
        if (engineMetrics.length > this.windowSize) {
            engineMetrics.shift();
        }
    }
}

// Health Check System
class HealthCheck {
    constructor() {
        this.healthStatus = new Map();
        this.checkInterval = 30000; // 30 seconds
        this.startHealthChecks();
    }

    startHealthChecks() {
        setInterval(() => {
            this.performHealthChecks();
        }, this.checkInterval);
    }

    async performHealthChecks() {
        // Implementation would ping each storage engine
        console.log('🏥 Performing health checks...');
    }

    async getHealthyEngines(engines) {
        // For now, return all engines - in production, filter by health status
        return engines;
    }
}

// Service Worker Message Handling
if (!isMainThread && workerData?.type === 'storage-service-worker') {
    const { workerId } = workerData;
    const storageCluster = new MultiStorageCluster();

    parentPort.on('message', async (message) => {
        try {
            const { operation, data, options } = message;
            let result;

            switch (operation) {
                case 'store-embedding':
                    result = await handleStoreEmbedding(data, options);
                    break;
                case 'similarity-search':
                    result = await handleSimilaritySearch(data, options);
                    break;
                case 'fuzzy-search':
                    result = await handleFuzzySearch(data, options);
                    break;
                case 'cache-operation':
                    result = await handleCacheOperation(data, options);
                    break;
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }

            parentPort.postMessage({
                success: true,
                result,
                workerId
            });
        } catch (error) {
            parentPort.postMessage({
                success: false,
                error: error.message,
                workerId
            });
        }
    });

    async function handleStoreEmbedding(data, options) {
        // Determine best storage engine based on data characteristics
        const storageEngine = options.preferredEngine || 'postgres';
        
        switch (storageEngine) {
            case 'postgres':
                return await storageCluster.storageEngines.get('postgres').storeEmbedding(data);
            case 'qdrant':
                return await storageCluster.storageEngines.get('qdrant').upsertPoints([data]);
            default:
                throw new Error(`Unsupported storage engine: ${storageEngine}`);
        }
    }

    async function handleSimilaritySearch(data, options) {
        const { queryVector, storageEngine = 'postgres' } = data;
        
        switch (storageEngine) {
            case 'postgres':
                return await storageCluster.storageEngines.get('postgres').similaritySearch(queryVector, options);
            case 'qdrant':
                return await storageCluster.storageEngines.get('qdrant').search(queryVector, options);
            case 'redis':
                return await storageCluster.storageEngines.get('redis').searchSimilarVectors(queryVector, options);
            default:
                throw new Error(`Unsupported storage engine: ${storageEngine}`);
        }
    }

    async function handleFuzzySearch(data, options) {
        const { query, indexName = 'default' } = data;
        const fuseEngine = storageCluster.storageEngines.get('fuse');
        return await fuseEngine.search(indexName, query, options);
    }

    async function handleCacheOperation(data, options) {
        const { operation: cacheOp, key, value, ttl } = data;
        const redisEngine = storageCluster.storageEngines.get('redis');
        
        switch (cacheOp) {
            case 'set':
                return await redisEngine.storeCache(key, value, ttl);
            case 'get':
                return await redisEngine.getCache(key);
            case 'delete':
                return await redisEngine.client.del(key);
            default:
                throw new Error(`Unknown cache operation: ${cacheOp}`);
        }
    }

    console.log(`🔧 Storage Service Worker ${workerId} initialized`);
}

export { 
    MultiStorageCluster, 
    PostgreSQLVectorStore, 
    RedisNativeStore, 
    QdrantVectorStore, 
    LangChainJSStore, 
    LokiJSStore, 
    IndexedDBStore, 
    FuseJSSearchEngine,
    LoadBalancer 
};