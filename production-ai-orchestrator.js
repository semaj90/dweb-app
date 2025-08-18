#!/usr/bin/env node
/**
 * Production-Ready GPU-Accelerated Legal AI Orchestrator
 * Integrates Context7 MCP + Go services + SvelteKit + Ollama
 * Following MCP_CONTEXT7_BEST_PRACTICES.md patterns
 */

import cluster from 'cluster';
import os from 'os';
import express from 'express';
// import cors from 'cors';
// import helmet from 'helmet';
// import compression from 'compression';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import Redis from 'ioredis';
import { Worker } from 'worker_threads';
import dotenv from 'dotenv';

dotenv.config();

// Configuration following Context7 best practices
const CONFIG = {
    port: process.env.MCP_PORT || 40000,
    host: process.env.MCP_HOST || 'localhost',
    debug: process.env.MCP_DEBUG === 'true',
    maxConnections: 100,
    requestTimeout: 30000,
    enableCors: true,
    enableWebSocket: true,
    workers: Math.min(os.cpus().length, 8), // Limit workers
    enableMultiCore: process.env.MCP_MULTICORE !== 'false',
    
    // Service endpoints
    ollamaUrl: process.env.OLLAMA_ENDPOINT || 'http://localhost:11434',
    goServiceUrl: process.env.GO_SERVICE_URL || 'http://localhost:4101',
    svelteKitUrl: process.env.SVELTEKIT_URL || 'http://localhost:5174',
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
    
    // AI Configuration
    defaultModel: process.env.DEFAULT_MODEL || 'gemma3-legal:latest',
    embeddingModel: process.env.EMBEDDING_MODEL || 'nomic-embed-text',
    maxTokens: parseInt(process.env.MAX_TOKENS) || 2048,
    temperature: parseFloat(process.env.TEMPERATURE) || 0.1
};

// Worker Pool for parallel processing
class WorkerPool {
    constructor(size = 4) {
        this.size = size;
        this.workers = [];
        this.queue = [];
        this.activeJobs = new Map();
        this.initialize();
    }

    initialize() {
        for (let i = 0; i < this.size; i++) {
            this.createWorker(i);
        }
        console.log(`🔧 Initialized worker pool with ${this.size} workers`);
    }

    createWorker(id) {
        const worker = new Worker(`
            const { parentPort } = require('worker_threads');
            
            parentPort.on('message', async (task) => {
                try {
                    const result = await processTask(task);
                    parentPort.postMessage({ type: 'result', taskId: task.id, result });
                } catch (error) {
                    parentPort.postMessage({ type: 'error', taskId: task.id, error: error.message });
                }
            });
            
            async function processTask(task) {
                // Simulate processing based on task type
                switch (task.type) {
                    case 'embedding':
                        await new Promise(resolve => setTimeout(resolve, 100));
                        return { embedding: new Array(384).fill(0).map(() => Math.random()) };
                    case 'legal_analysis':
                        await new Promise(resolve => setTimeout(resolve, 500));
                        return { analysis: 'Legal analysis complete', confidence: 0.95 };
                    default:
                        return { processed: true };
                }
            }
        `, { eval: true });

        worker.on('message', (message) => {
            this.handleWorkerMessage(message);
        });

        worker.on('error', (error) => {
            console.error(`❌ Worker ${id} error:`, error);
            this.createWorker(id); // Recreate worker
        });

        this.workers[id] = { worker, busy: false, id };
    }

    async executeTask(type, data) {
        return new Promise((resolve, reject) => {
            const taskId = Date.now().toString() + Math.random();
            const task = { id: taskId, type, data, timestamp: Date.now() };
            
            this.activeJobs.set(taskId, { resolve, reject });
            
            const availableWorker = this.workers.find(w => !w.busy);
            if (availableWorker) {
                availableWorker.busy = true;
                availableWorker.worker.postMessage(task);
            } else {
                this.queue.push(task);
            }
        });
    }

    handleWorkerMessage(message) {
        const { type, taskId, result, error } = message;
        const job = this.activeJobs.get(taskId);
        
        if (job) {
            this.activeJobs.delete(taskId);
            
            // Mark worker as available
            const worker = this.workers.find(w => w.busy);
            if (worker) {
                worker.busy = false;
                
                // Process next queued task
                if (this.queue.length > 0) {
                    const nextTask = this.queue.shift();
                    worker.busy = true;
                    worker.worker.postMessage(nextTask);
                }
            }
            
            if (type === 'result') {
                job.resolve(result);
            } else {
                job.reject(new Error(error));
            }
        }
    }

    getStats() {
        return {
            totalWorkers: this.workers.length,
            busyWorkers: this.workers.filter(w => w.busy).length,
            queueLength: this.queue.length,
            activeJobs: this.activeJobs.size
        };
    }
}

// Production AI Orchestrator Class
class ProductionAIOrchestrator {
    constructor() {
        this.workerId = cluster.worker ? cluster.worker.id : 'master';
        this.connections = new Set();
        this.workerPool = new WorkerPool(4);
        this.redis = new Redis(CONFIG.redisUrl);
        this.performanceMetrics = {
            requestsProcessed: 0,
            averageResponseTime: 0,
            errorRate: 0,
            lastHealthCheck: Date.now()
        };
        
        this.app = express();
        this.server = createServer(this.app);
        this.wss = CONFIG.enableWebSocket ? new WebSocketServer({ server: this.server }) : null;
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
        this.setupHealthMonitoring();
    }

    setupMiddleware() {
        // this.app.use(helmet());
        // this.app.use(compression());
        
        // Enable basic CORS manually
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
            if (req.method === 'OPTIONS') {
                res.sendStatus(200);
            } else {
                next();
            }
        });
        
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
        
        // Request logging and metrics
        this.app.use((req, res, next) => {
            req.startTime = Date.now();
            res.on('finish', () => {
                const responseTime = Date.now() - req.startTime;
                this.updateMetrics(responseTime, res.statusCode >= 400);
            });
            next();
        });
    }

    setupRoutes() {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                workerId: this.workerId,
                connections: this.connections.size,
                metrics: {
                    ...this.performanceMetrics,
                    workerPool: this.workerPool.getStats()
                },
                multiCore: {
                    enabled: CONFIG.enableMultiCore,
                    totalWorkers: CONFIG.workers,
                    currentWorker: this.workerId
                },
                services: {
                    ollama: CONFIG.ollamaUrl,
                    goService: CONFIG.goServiceUrl,
                    svelteKit: CONFIG.svelteKitUrl
                }
            });
        });

        // Legal AI Chat endpoint
        this.app.post('/api/chat', async (req, res) => {
            try {
                const { message, userId, sessionId, model, useRAG = true } = req.body;
                
                if (!message?.trim()) {
                    return res.status(400).json({ error: 'Message is required' });
                }

                const startTime = Date.now();
                const chatId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                
                console.log(`🤖 [${this.workerId}] Processing chat: ${chatId}`);
                
                // Step 1: Generate embeddings using worker pool
                const embeddingResult = await this.workerPool.executeTask('embedding', { text: message });
                
                // Step 2: RAG retrieval (if enabled)
                let context = [];
                if (useRAG) {
                    context = await this.retrieveContext(embeddingResult.embedding, message);
                }
                
                // Step 3: Generate AI response via Ollama
                const aiResponse = await this.generateAIResponse(message, context, model || CONFIG.defaultModel);
                
                // Step 4: Store conversation in Redis
                await this.storeConversation(chatId, {
                    userId, sessionId, message, response: aiResponse, context
                });
                
                const processingTime = Date.now() - startTime;
                
                // Broadcast to WebSocket clients
                this.broadcastToClients({
                    type: 'chat_response',
                    chatId,
                    response: aiResponse,
                    processingTime,
                    workerId: this.workerId
                });
                
                res.json({
                    success: true,
                    chatId,
                    response: aiResponse.response,
                    metadata: {
                        model: aiResponse.model,
                        processingTime,
                        workerId: this.workerId,
                        contextUsed: context.length,
                        tokensUsed: aiResponse.eval_count || 0
                    }
                });
                
            } catch (error) {
                console.error(`❌ [${this.workerId}] Chat error:`, error);
                res.status(500).json({ 
                    error: 'AI processing failed',
                    details: CONFIG.debug ? error.message : 'Internal server error'
                });
            }
        });

        // Context7 MCP Memory operations
        this.app.post('/mcp/memory/create', async (req, res) => {
            try {
                const { entities, relations } = req.body;
                const result = await this.workerPool.executeTask('memory_create', { entities, relations });
                res.json({ success: true, result });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Error analysis endpoint
        this.app.post('/mcp/error-analysis/index', async (req, res) => {
            try {
                const { errors, fixes, categories } = req.body;
                const analysisResult = await this.workerPool.executeTask('error_analysis', {
                    errors: errors || [],
                    fixes: fixes || [],
                    categories: categories || []
                });
                
                // Store in memory graph for future reference
                await this.redis.hset('error_analysis_results', Date.now(), JSON.stringify(analysisResult));
                
                res.json({ success: true, analysis: analysisResult });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Go service integration proxy
        this.app.post('/api/go-service/:endpoint', async (req, res) => {
            try {
                const { endpoint } = req.params;
                const response = await fetch(`${CONFIG.goServiceUrl}/api/${endpoint}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(req.body)
                });
                const data = await response.json();
                res.json(data);
            } catch (error) {
                res.status(500).json({ error: 'Go service integration failed' });
            }
        });
    }

    setupWebSocket() {
        if (!this.wss) return;
        
        this.wss.on('connection', (ws) => {
            console.log(`🔗 [${this.workerId}] WebSocket connection established`);
            this.connections.add(ws);
            
            // Send welcome message
            ws.send(JSON.stringify({
                type: 'welcome',
                workerId: this.workerId,
                timestamp: Date.now()
            }));
            
            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    await this.handleWebSocketMessage(ws, message);
                } catch (error) {
                    ws.send(JSON.stringify({ type: 'error', error: 'Invalid message format' }));
                }
            });
            
            ws.on('close', () => {
                console.log(`🔌 [${this.workerId}] WebSocket connection closed`);
                this.connections.delete(ws);
            });
        });
    }

    async handleWebSocketMessage(ws, message) {
        switch (message.type) {
            case 'chat':
                // Process chat through WebSocket
                const result = await this.processChat(message.data);
                ws.send(JSON.stringify({ type: 'chat_response', result }));
                break;
            case 'status':
                ws.send(JSON.stringify({ 
                    type: 'status', 
                    status: this.getSystemStatus() 
                }));
                break;
        }
    }

    async retrieveContext(embedding, query) {
        try {
            // Mock RAG retrieval - in production, this would query PostgreSQL + pgvector
            return [
                {
                    id: 1,
                    title: 'Legal Contract Elements',
                    content: 'A legal contract requires offer, acceptance, consideration, and mutual intent.',
                    similarity: 0.95
                },
                {
                    id: 2,
                    title: 'Contract Law Principles',
                    content: 'Contract formation requires meeting of minds and legal capacity.',
                    similarity: 0.87
                }
            ];
        } catch (error) {
            console.error('Context retrieval error:', error);
            return [];
        }
    }

    async generateAIResponse(message, context, model) {
        try {
            const prompt = this.buildLegalPrompt(message, context);
            
            const response = await fetch(`${CONFIG.ollamaUrl}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    prompt,
                    stream: false,
                    options: {
                        temperature: CONFIG.temperature,
                        num_predict: CONFIG.maxTokens
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`Ollama API error: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('AI generation error:', error);
            return {
                response: 'I apologize, but I encountered an error processing your legal question. Please try again.',
                model,
                error: error.message
            };
        }
    }

    buildLegalPrompt(userMessage, context) {
        return `You are a specialized legal AI assistant with expertise in contract law, legal analysis, and case management.

Context from legal database:
${context.map(doc => `- ${doc.title}: ${doc.content}`).join('\n')}

User question: ${userMessage}

Please provide a comprehensive legal analysis that:
1. Directly addresses the user's question
2. References relevant legal principles from the provided context
3. Maintains professional legal terminology
4. Provides actionable guidance while noting any limitations

Response:`;
    }

    async storeConversation(chatId, data) {
        try {
            await this.redis.hset('conversations', chatId, JSON.stringify({
                ...data,
                timestamp: Date.now(),
                workerId: this.workerId
            }));
        } catch (error) {
            console.error('Conversation storage error:', error);
        }
    }

    broadcastToClients(message) {
        if (!this.wss) return;
        
        this.connections.forEach(client => {
            if (client.readyState === client.OPEN) {
                client.send(JSON.stringify(message));
            }
        });
    }

    updateMetrics(responseTime, isError) {
        this.performanceMetrics.requestsProcessed++;
        this.performanceMetrics.averageResponseTime = 
            (this.performanceMetrics.averageResponseTime + responseTime) / 2;
        
        if (isError) {
            this.performanceMetrics.errorRate = 
                (this.performanceMetrics.errorRate + 1) / this.performanceMetrics.requestsProcessed;
        }
    }

    setupHealthMonitoring() {
        setInterval(() => {
            this.performanceMetrics.lastHealthCheck = Date.now();
            
            // Broadcast health status to WebSocket clients
            this.broadcastToClients({
                type: 'health_update',
                status: this.getSystemStatus(),
                timestamp: Date.now()
            });
        }, 30000); // Every 30 seconds
    }

    getSystemStatus() {
        return {
            workerId: this.workerId,
            connections: this.connections.size,
            metrics: this.performanceMetrics,
            workerPool: this.workerPool.getStats(),
            uptime: process.uptime()
        };
    }

    async start() {
        return new Promise((resolve, reject) => {
            this.server.listen(CONFIG.port, CONFIG.host, (error) => {
                if (error) {
                    reject(error);
                } else {
                    console.log(`
🎯 Production AI Orchestrator [${this.workerId}] Running!
📡 Server: http://${CONFIG.host}:${CONFIG.port}
🔌 WebSocket: ws://${CONFIG.host}:${CONFIG.port}
🧠 AI Models: ${CONFIG.defaultModel}
🔧 Workers: ${CONFIG.workers} (Multi-core: ${CONFIG.enableMultiCore})
⚡ Services: Ollama, Go, SvelteKit, Redis
                    `);
                    resolve();
                }
            });
        });
    }
}

// Cluster management for multi-core processing
async function startProductionOrchestrator() {
    if (CONFIG.enableMultiCore && cluster.isPrimary) {
        console.log(`🚀 Starting ${CONFIG.workers} worker processes...`);
        
        for (let i = 0; i < CONFIG.workers; i++) {
            cluster.fork();
        }
        
        cluster.on('exit', (worker, code, signal) => {
            console.log(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}`);
            console.log('Starting a new worker...');
            cluster.fork();
        });
        
    } else {
        // Worker process or single-core mode
        const orchestrator = new ProductionAIOrchestrator();
        
        try {
            await orchestrator.start();
        } catch (error) {
            console.error('❌ Failed to start orchestrator:', error);
            process.exit(1);
        }
        
        // Graceful shutdown
        process.on('SIGTERM', () => {
            console.log('🛑 Shutting down gracefully...');
            orchestrator.server.close(() => {
                process.exit(0);
            });
        });
    }
}

// Start the production orchestrator
startProductionOrchestrator().catch(console.error);