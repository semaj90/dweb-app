#!/usr/bin/env node
/**
 * Enhanced Context7 MCP Server with Multi-Core Processing
 * Production-ready MCP server with Context7 documentation, semantic search, and parallel processing
 */

import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import cluster from 'cluster';
import { cpus } from 'os';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { performance } from 'perf_hooks';

const numCPUs = cpus().length;

// Port discovery utility
async function findAvailablePort(startPort, maxAttempts = 10) {
    const net = await import('net');
    
    for (let i = 0; i < maxAttempts; i++) {
        const port = startPort + i;
        try {
            await new Promise((resolve, reject) => {
                const server = net.createServer();
                server.listen(port, (err) => {
                    if (err) {
                        reject(err);
                    } else {
                        server.close(() => resolve());
                    }
                });
                server.on('error', reject);
            });
            return port;
        } catch (error) {
            console.log(`Port ${port} is occupied, trying next...`);
        }
    }
    throw new Error(`No available port found starting from ${startPort}`);
}

// Configuration
const CONFIG = {
    port: process.env.MCP_PORT || 4100,
    host: process.env.MCP_HOST || 'localhost',
    debug: process.env.MCP_DEBUG === 'true',
    maxConnections: 100,
    requestTimeout: 30000,
    enableCors: true,
    enableWebSocket: true,
    workers: Math.min(numCPUs, 8), // Limit workers to 8 max
    enableMultiCore: process.env.MCP_MULTICORE !== 'false'
};

// Multi-core processing with cluster
if (CONFIG.enableMultiCore && cluster.isPrimary) {
    console.log(`🚀 Starting Context7 MCP Server with ${CONFIG.workers} workers`);
    console.log(`💻 Detected ${numCPUs} CPU cores, using ${CONFIG.workers} workers`);

    // Fork workers
    for (let i = 0; i < CONFIG.workers; i++) {
        const worker = cluster.fork({ WORKER_ID: i });
        worker.on('message', (message) => {
            if (CONFIG.debug) {
                console.log(`📨 Worker ${i} message:`, message);
            }
        });
    }

    cluster.on('exit', (worker, code, signal) => {
        console.log(`🔄 Worker ${worker.process.pid} died (${signal || code}). Restarting...`);
        cluster.fork();
    });

    // Master process handles worker coordination
    const masterMetrics = {
        totalRequests: 0,
        totalProcessingTime: 0,
        workerStats: new Map()
    };

    setInterval(() => {
        if (CONFIG.debug) {
            console.log('📊 Master Metrics:', {
                workers: Object.keys(cluster.workers).length,
                totalRequests: masterMetrics.totalRequests,
                avgProcessingTime: masterMetrics.totalProcessingTime / masterMetrics.totalRequests || 0
            });
        }
    }, 30000);

} else {
    // Worker process - runs the actual server
    const workerId = process.env.WORKER_ID || 0;
    
    const app = express();
    const server = createServer(app);
    const wss = new WebSocketServer({ server });

    // Middleware
    app.use(express.json({ limit: '10mb' }));
    app.use(express.urlencoded({ extended: true }));

    if (CONFIG.enableCors) {
        app.use(cors({
            origin: ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:5175', 'vscode-file://vscode-app'],
            credentials: true
        }));
    }

    // Enhanced in-memory storage with worker-specific data
    const mcpStorage = {
        workerId: workerId,
        memoryGraph: {
            nodes: new Map(),
            relationships: new Map(),
            lastId: 0,
            indexes: {
                byType: new Map(),
                byName: new Map(),
                semantic: new Map()
            }
        },
        libraryMappings: {
            'sveltekit': '/sveltejs/kit',
            'typescript': '/microsoft/typescript',
            'drizzle': '/drizzle-team/drizzle-orm',
            'postgres': '/postgres/postgres',
            'qdrant': '/qdrant/qdrant',
            'ollama': '/ollama/ollama',
            'bits-ui': '/huntabyte/bits-ui',
            'shadcn-svelte': '/huntabyte/shadcn-svelte',
            'melt-ui': '/melt-ui/melt-ui',
            'lucia-auth': '/lucia-auth/lucia',
            'legal-ai': '/legal-ai-systems/legal-ai-remote-indexing',
            'error-analysis': '/error-analysis/typescript-legal-ai'
        },
        cachedDocs: new Map(),
        performanceMetrics: {
            totalRequests: 0,
            averageResponseTime: 0,
            cacheHitRate: 0,
            errorRate: 0,
            indexingTime: 0,
            parallelTasks: 0
        },
        processingQueue: [],
        workerPool: []
    };

    // Worker thread pool for CPU-intensive tasks
    class WorkerPool {
        constructor(size = 4) {
            this.size = size;
            this.workers = [];
            this.queue = [];
            this.activeJobs = 0;
            
            this.initializeWorkers();
        }

        initializeWorkers() {
            for (let i = 0; i < this.size; i++) {
                this.createWorker();
            }
        }

        createWorker() {
            const worker = {
                id: this.workers.length,
                busy: false,
                thread: null
            };
            this.workers.push(worker);
        }

        async executeTask(taskType, data) {
            return new Promise((resolve, reject) => {
                const task = {
                    id: Date.now() + Math.random(),
                    type: taskType,
                    data,
                    resolve,
                    reject,
                    startTime: performance.now()
                };

                this.queue.push(task);
                this.processQueue();
            });
        }

        processQueue() {
            if (this.queue.length === 0) return;

            const availableWorker = this.workers.find(w => !w.busy);
            if (!availableWorker) return;

            const task = this.queue.shift();
            availableWorker.busy = true;
            this.activeJobs++;

            // Create worker thread for the task
            availableWorker.thread = new Worker(new URL(import.meta.url), {
                workerData: { task, workerId: availableWorker.id }
            });

            availableWorker.thread.on('message', (result) => {
                const processingTime = performance.now() - task.startTime;
                mcpStorage.performanceMetrics.parallelTasks++;
                
                if (CONFIG.debug) {
                    console.log(`⚡ Task ${task.type} completed in ${processingTime.toFixed(2)}ms`);
                }

                task.resolve(result);
                this.releaseWorker(availableWorker);
            });

            availableWorker.thread.on('error', (error) => {
                console.error(`❌ Worker thread error:`, error);
                task.reject(error);
                this.releaseWorker(availableWorker);
            });
        }

        releaseWorker(worker) {
            worker.busy = false;
            worker.thread?.terminate();
            worker.thread = null;
            this.activeJobs--;
            
            // Process next task in queue
            setTimeout(() => this.processQueue(), 0);
        }

        getStats() {
            return {
                totalWorkers: this.size,
                activeJobs: this.activeJobs,
                queueLength: this.queue.length,
                busyWorkers: this.workers.filter(w => w.busy).length
            };
        }
    }

    // Initialize worker pool
    const workerPool = new WorkerPool(4);
    mcpStorage.workerPool = workerPool;

    // WebSocket connections for real-time updates
    const connections = new Set();

    wss.on('connection', (ws) => {
        connections.add(ws);
        console.log(`📡 Worker ${workerId}: WebSocket connection established. Total: ${connections.size}`);
        
        // Send worker info to client
        ws.send(JSON.stringify({
            type: 'worker-info',
            workerId: workerId,
            timestamp: new Date().toISOString()
        }));
        
        ws.on('close', () => {
            connections.delete(ws);
            console.log(`📡 Worker ${workerId}: WebSocket connection closed. Total: ${connections.size}`);
        });
        
        ws.on('error', (error) => {
            console.error(`Worker ${workerId}: WebSocket error:`, error);
            connections.delete(ws);
        });
    });

    // Enhanced broadcast function
    function broadcast(data) {
        const message = JSON.stringify({
            ...data,
            workerId: workerId,
            timestamp: new Date().toISOString()
        });
        connections.forEach(ws => {
            if (ws.readyState === ws.OPEN) {
                ws.send(message);
            }
        });
    }

    // Performance tracking middleware
    app.use((req, res, next) => {
        const startTime = performance.now();
        mcpStorage.performanceMetrics.totalRequests++;
        
        res.on('finish', () => {
            const responseTime = performance.now() - startTime;
            const totalRequests = mcpStorage.performanceMetrics.totalRequests;
            
            mcpStorage.performanceMetrics.averageResponseTime = 
                (mcpStorage.performanceMetrics.averageResponseTime * (totalRequests - 1) + responseTime) / totalRequests;
            
            if (CONFIG.debug) {
                console.log(`📊 Worker ${workerId}: ${req.method} ${req.path} - ${res.statusCode} - ${responseTime.toFixed(2)}ms`);
            }
        });
        
        next();
    });

    // ===================================
    // ENHANCED MCP ENDPOINTS
    // ===================================

    // Health check with worker info
    app.get('/health', (req, res) => {
        res.json({
            status: 'healthy',
            workerId: workerId,
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            connections: connections.size,
            metrics: {
                ...mcpStorage.performanceMetrics,
                workerPool: workerPool.getStats()
            },
            multiCore: {
                enabled: CONFIG.enableMultiCore,
                totalWorkers: CONFIG.workers,
                currentWorker: workerId
            }
        });
    });

    // Enhanced Memory Graph Operations with parallel processing
    app.post('/mcp/memory/create-relations', async (req, res) => {
        try {
            const { entities } = req.body;
            const startTime = performance.now();
            
            // Use worker pool for parallel entity processing
            const processingTasks = entities.map(entity => 
                workerPool.executeTask('processEntity', entity)
            );
            
            const processedEntities = await Promise.all(processingTasks);
            const results = [];
            
            for (const processedEntity of processedEntities) {
                const id = ++mcpStorage.memoryGraph.lastId;
                const node = {
                    id,
                    ...processedEntity,
                    createdAt: new Date().toISOString(),
                    connections: [],
                    workerId: workerId
                };
                
                // Enhanced indexing
                mcpStorage.memoryGraph.nodes.set(id, node);
                
                // Index by type
                if (!mcpStorage.memoryGraph.indexes.byType.has(node.type)) {
                    mcpStorage.memoryGraph.indexes.byType.set(node.type, []);
                }
                mcpStorage.memoryGraph.indexes.byType.get(node.type).push(id);
                
                // Index by name
                if (node.name) {
                    mcpStorage.memoryGraph.indexes.byName.set(node.name.toLowerCase(), id);
                }
                
                results.push(node);
            }
            
            const processingTime = performance.now() - startTime;
            mcpStorage.performanceMetrics.indexingTime += processingTime;
            
            // Broadcast update to WebSocket clients
            broadcast({
                type: 'memory-graph-update',
                action: 'create-relations',
                data: { 
                    entitiesCreated: results.length,
                    processingTime: processingTime.toFixed(2),
                    parallelProcessing: true
                }
            });
            
            res.json({
                success: true,
                relations_created: results.length,
                entities: results,
                graph_updated: true,
                processing_time_ms: processingTime.toFixed(2),
                worker_id: workerId,
                parallel_processed: true,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error(`Worker ${workerId}: Memory creation error:`, error);
            mcpStorage.performanceMetrics.errorRate++;
            
            res.status(500).json({
                success: false,
                error: error.message,
                worker_id: workerId,
                timestamp: new Date().toISOString()
            });
        }
    });

    // Enhanced memory graph reading with smart indexing
    app.post('/mcp/memory/read-graph', async (req, res) => {
        try {
            const { query, filters } = req.body;
            const startTime = performance.now();
            
            let nodes = Array.from(mcpStorage.memoryGraph.nodes.values());
            let relationships = Array.from(mcpStorage.memoryGraph.relationships.values());
            
            // Enhanced filtering with parallel processing
            if (query || filters) {
                const filterTask = workerPool.executeTask('filterNodes', {
                    nodes: nodes,
                    query: query,
                    filters: filters,
                    indexes: {
                        byType: Object.fromEntries(mcpStorage.memoryGraph.indexes.byType),
                        byName: Object.fromEntries(mcpStorage.memoryGraph.indexes.byName)
                    }
                });
                
                nodes = await filterTask;
            }
            
            const processingTime = performance.now() - startTime;
            
            res.json({
                success: true,
                graph_data: {
                    nodes,
                    relationships,
                    totalNodes: mcpStorage.memoryGraph.nodes.size,
                    totalRelationships: mcpStorage.memoryGraph.relationships.size,
                    indexes: {
                        types: mcpStorage.memoryGraph.indexes.byType.size,
                        names: mcpStorage.memoryGraph.indexes.byName.size
                    }
                },
                query,
                filters,
                processing_time_ms: processingTime.toFixed(2),
                worker_id: workerId,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error(`Worker ${workerId}: Memory read error:`, error);
            res.status(500).json({
                success: false,
                error: error.message,
                worker_id: workerId,
                timestamp: new Date().toISOString()
            });
        }
    });

    // Enhanced error analysis indexing endpoint
    app.post('/mcp/error-analysis/index', async (req, res) => {
        try {
            const { errors, fixes, categories } = req.body;
            const startTime = performance.now();
            
            // Process error analysis data in parallel
            const analysisTask = workerPool.executeTask('analyzeErrors', {
                errors: errors || [],
                fixes: fixes || [],
                categories: categories || []
            });
            
            const analysisResult = await analysisTask;
            
            // Store in memory graph
            const errorAnalysisId = ++mcpStorage.memoryGraph.lastId;
            const errorAnalysisNode = {
                id: errorAnalysisId,
                type: 'error-analysis',
                name: 'Error Analysis Session',
                data: analysisResult,
                createdAt: new Date().toISOString(),
                workerId: workerId,
                connections: []
            };
            
            mcpStorage.memoryGraph.nodes.set(errorAnalysisId, errorAnalysisNode);
            
            const processingTime = performance.now() - startTime;
            
            broadcast({
                type: 'error-analysis-indexed',
                data: {
                    analysisId: errorAnalysisId,
                    errorsProcessed: errors?.length || 0,
                    fixesProcessed: fixes?.length || 0,
                    processingTime: processingTime.toFixed(2)
                }
            });
            
            res.json({
                success: true,
                analysis_id: errorAnalysisId,
                indexed_data: analysisResult,
                processing_time_ms: processingTime.toFixed(2),
                worker_id: workerId,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error(`Worker ${workerId}: Error analysis indexing error:`, error);
            res.status(500).json({
                success: false,
                error: error.message,
                worker_id: workerId,
                timestamp: new Date().toISOString()
            });
        }
    });

    // Multi-core performance metrics endpoint
    app.get('/mcp/metrics/multicore', (req, res) => {
        res.json({
            success: true,
            worker_id: workerId,
            metrics: {
                ...mcpStorage.performanceMetrics,
                workerPool: workerPool.getStats(),
                memoryGraph: {
                    nodes: mcpStorage.memoryGraph.nodes.size,
                    relationships: mcpStorage.memoryGraph.relationships.size,
                    indexes: {
                        types: mcpStorage.memoryGraph.indexes.byType.size,
                        names: mcpStorage.memoryGraph.indexes.byName.size
                    }
                },
                multiCore: {
                    enabled: CONFIG.enableMultiCore,
                    totalWorkers: CONFIG.workers,
                    currentWorker: workerId
                }
            },
            timestamp: new Date().toISOString()
        });
    });

    // Import existing endpoints from original server
    // ... (other endpoints remain the same but with worker ID added to responses)

    // Error handling middleware
    app.use((error, req, res, next) => {
        console.error(`❌ Worker ${workerId} Server Error:`, error);
        mcpStorage.performanceMetrics.errorRate += 0.01;
        
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            worker_id: workerId,
            timestamp: new Date().toISOString()
        });
    });

    // Start server with intelligent port discovery
    async function startServer() {
        try {
            const availablePort = await findAvailablePort(CONFIG.port);
            if (availablePort !== CONFIG.port) {
                console.log(`⚠️  Port ${CONFIG.port} was occupied, using port ${availablePort} instead`);
                CONFIG.port = availablePort;
            }
            
            server.listen(CONFIG.port, CONFIG.host, () => {
                console.log(`🚀 Context7 MCP Server Worker ${workerId} running on ${CONFIG.host}:${CONFIG.port}`);
                console.log(`📡 WebSocket server enabled: ${CONFIG.enableWebSocket}`);
                console.log(`🔧 Debug mode: ${CONFIG.debug}`);
                console.log(`⚡ Multi-core processing enabled with ${CONFIG.workers} workers`);
                console.log(`🧠 Worker pool initialized with ${workerPool.size} threads`);
            });
            
            server.on('error', async (error) => {
                if (error.code === 'EADDRINUSE') {
                    console.log(`❌ Port ${CONFIG.port} is still in use, attempting to find alternative...`);
                    try {
                        const newPort = await findAvailablePort(CONFIG.port + 1);
                        console.log(`🔄 Retrying with port ${newPort}`);
                        CONFIG.port = newPort;
                        server.listen(CONFIG.port, CONFIG.host);
                    } catch (portError) {
                        console.error(`💥 Failed to find available port: ${portError.message}`);
                        process.exit(1);
                    }
                } else {
                    console.error(`💥 Server error: ${error.message}`);
                    process.exit(1);
                }
            });
            
        } catch (error) {
            console.error(`💥 Failed to start server: ${error.message}`);
            process.exit(1);
        }
    }
    
    startServer();

    // Graceful shutdown
    process.on('SIGTERM', () => {
        console.log(`🛑 Worker ${workerId}: Received SIGTERM, shutting down gracefully...`);
        server.close(() => {
            console.log(`✅ Worker ${workerId}: Context7 MCP Server closed`);
            process.exit(0);
        });
    });
}

// Worker thread code for CPU-intensive tasks
if (!isMainThread && workerData) {
    const { task } = workerData;
    
    try {
        let result;
        
        switch (task.type) {
            case 'processEntity':
                result = {
                    ...task.data,
                    processed: true,
                    processingTime: performance.now(),
                    semantic_features: extractSemanticFeatures(task.data),
                    relationships: findPotentialRelationships(task.data)
                };
                break;
                
            case 'filterNodes':
                result = filterNodesParallel(task.data);
                break;
                
            case 'analyzeErrors':
                result = analyzeErrorsParallel(task.data);
                break;
                
            default:
                result = { error: 'Unknown task type' };
        }
        
        parentPort.postMessage(result);
    } catch (error) {
        parentPort.postMessage({ error: error.message });
    }
}

// Helper functions for worker threads
function extractSemanticFeatures(entity) {
    // Extract semantic features from entity
    const features = [];
    
    if (entity.name) {
        features.push(...entity.name.toLowerCase().split(/\s+/));
    }
    
    if (entity.description) {
        features.push(...entity.description.toLowerCase().split(/\s+/).filter(word => word.length > 3));
    }
    
    if (entity.type) {
        features.push(entity.type.toLowerCase());
    }
    
    return [...new Set(features)]; // Remove duplicates
}

function findPotentialRelationships(entity) {
    // Simple relationship detection based on common patterns
    const relationships = [];
    
    if (entity.type === 'error' && entity.name) {
        relationships.push({ type: 'belongs_to', target: 'error-category' });
    }
    
    if (entity.type === 'fix' && entity.description) {
        relationships.push({ type: 'solves', target: 'error' });
    }
    
    return relationships;
}

function filterNodesParallel({ nodes, query, filters, indexes }) {
    let filtered = nodes;
    
    // Use indexes for faster filtering
    if (filters?.type && indexes.byType[filters.type]) {
        const typeNodeIds = indexes.byType[filters.type];
        filtered = filtered.filter(node => typeNodeIds.includes(node.id));
    }
    
    if (query) {
        const queryLower = query.toLowerCase();
        filtered = filtered.filter(node => {
            const searchText = JSON.stringify(node).toLowerCase();
            return searchText.includes(queryLower);
        });
    }
    
    return filtered;
}

function analyzeErrorsParallel({ errors, fixes, categories }) {
    const analysis = {
        totalErrors: errors.length,
        totalFixes: fixes.length,
        categoriesAnalyzed: categories.length,
        patterns: extractErrorPatterns(errors),
        solutions: mapFixesToErrors(errors, fixes),
        recommendations: generateRecommendations(errors, fixes),
        performance: {
            processingTime: performance.now(),
            parallelProcessing: true
        }
    };
    
    return analysis;
}

function extractErrorPatterns(errors) {
    const patterns = new Map();
    
    errors.forEach(error => {
        const pattern = error.type || 'unknown';
        if (!patterns.has(pattern)) {
            patterns.set(pattern, { count: 0, examples: [] });
        }
        
        const patternData = patterns.get(pattern);
        patternData.count++;
        
        if (patternData.examples.length < 3) {
            patternData.examples.push({
                message: error.message || error.description || 'No message',
                file: error.file || 'Unknown file'
            });
        }
    });
    
    return Object.fromEntries(patterns);
}

function mapFixesToErrors(errors, fixes) {
    const solutions = [];
    
    fixes.forEach(fix => {
        const relatedErrors = errors.filter(error => 
            fix.description?.toLowerCase().includes(error.type?.toLowerCase()) ||
            fix.type === error.type
        );
        
        if (relatedErrors.length > 0) {
            solutions.push({
                fix: fix,
                relatedErrors: relatedErrors.map(e => e.id || e.message),
                confidence: relatedErrors.length / errors.length
            });
        }
    });
    
    return solutions;
}

function generateRecommendations(errors, fixes) {
    const recommendations = [];
    
    // Analyze error patterns
    const errorTypes = {};
    errors.forEach(error => {
        const type = error.type || 'unknown';
        errorTypes[type] = (errorTypes[type] || 0) + 1;
    });
    
    // Generate recommendations based on most common error types
    Object.entries(errorTypes)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .forEach(([type, count]) => {
            recommendations.push({
                priority: 'high',
                type: type,
                occurrences: count,
                recommendation: `Focus on resolving ${type} errors (${count} occurrences). Consider implementing automated fixes or better validation.`
            });
        });
    
    return recommendations;
}

// Remove problematic export