/**
 * Production Server with Multi-Node Cluster, GPU Acceleration, and Context7 Integration
 * Supports high-concurrency PDF processing with advanced AI features
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const Redis = require('ioredis');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cluster = require('cluster');
const os = require('os');
const path = require('path');
const fs = require('fs').promises;
const WebSocket = require('ws');
const http = require('http');

// Configuration
const CONFIG = {
    port: process.env.PORT || 3000,
    svelteKitPort: process.env.SVELTEKIT_PORT || 5173,
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
    ollamaUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
    context7Url: process.env.CONTEXT7_URL || 'http://localhost:40000',
    maxFileSize: 50 * 1024 * 1024, // 50MB
    maxConcurrentJobs: process.env.MAX_CONCURRENT_JOBS || 10,
    gpuEnabled: process.env.GPU_ENABLED === 'true',
    environment: process.env.NODE_ENV || 'development'
};

class ProductionServer {
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.wss = new WebSocket.Server({ server: this.server });
        this.redis = new Redis(CONFIG.redisUrl);
        this.activeJobs = new Map();
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSockets();
    }

    setupMiddleware() {
        // Security and CORS
        this.app.use(cors({
            origin: process.env.FRONTEND_URL || 'http://localhost:5173',
            credentials: true
        }));

        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

        // File upload
        const storage = multer.memoryStorage();
        this.upload = multer({ 
            storage,
            limits: { fileSize: CONFIG.maxFileSize },
            fileFilter: (req, file, cb) => {
                const allowedTypes = [
                    'application/pdf',
                    'image/jpeg',
                    'image/png',
                    'image/gif',
                    'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                ];
                cb(null, allowedTypes.includes(file.mimetype));
            }
        });

        // Request logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
            next();
        });
    }

    setupRoutes() {
        // Health check
        this.app.get('/api/health', async (req, res) => {
            try {
                const health = {
                    status: 'healthy',
                    timestamp: new Date().toISOString(),
                    services: {},
                    cluster: {
                        pid: process.pid,
                        isMaster: cluster.isMaster,
                        workers: cluster.isMaster ? Object.keys(cluster.workers).length : 1
                    }
                };

                // Check Ollama
                try {
                    const ollamaResponse = await fetch(`${CONFIG.ollamaUrl}/api/tags`);
                    health.services.ollama = ollamaResponse.ok ? 'healthy' : 'degraded';
                } catch {
                    health.services.ollama = 'unhealthy';
                }

                // Check Redis
                try {
                    await this.redis.ping();
                    health.services.redis = 'healthy';
                } catch {
                    health.services.redis = 'unhealthy';
                }

                // Check Context7
                try {
                    const context7Response = await fetch(`${CONFIG.context7Url}/health`);
                    health.services.context7 = context7Response.ok ? 'healthy' : 'degraded';
                } catch {
                    health.services.context7 = 'degraded';
                }

                res.json(health);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Multi-PDF processing endpoint
        this.app.post('/api/process-pdfs', this.upload.array('files', 20), async (req, res) => {
            try {
                const { files } = req;
                const options = {
                    jurisdiction: req.body.jurisdiction || 'federal',
                    enhanceRag: req.body.enhanceRag === 'true',
                    context7Analysis: req.body.context7Analysis === 'true',
                    extractEntities: req.body.extractEntities === 'true'
                };

                if (!files || files.length === 0) {
                    return res.status(400).json({ error: 'No files provided' });
                }

                const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                console.log(`📄 Processing ${files.length} files (Job: ${jobId})`);

                // Queue processing job
                const jobData = {
                    id: jobId,
                    files: files.map(file => ({
                        originalname: file.originalname,
                        mimetype: file.mimetype,
                        size: file.size,
                        buffer: file.buffer.toString('base64')
                    })),
                    options,
                    timestamp: new Date(),
                    status: 'queued'
                };

                await this.redis.lpush('processing_queue', JSON.stringify(jobData));
                this.activeJobs.set(jobId, { status: 'queued', progress: 0 });

                // Start processing
                this.processJobAsync(jobId);

                res.json({
                    success: true,
                    jobId,
                    message: `Queued ${files.length} files for processing`,
                    estimatedTime: files.length * 15 // seconds
                });

            } catch (error) {
                console.error('❌ PDF processing error:', error);
                res.status(500).json({
                    success: false,
                    error: error.message
                });
            }
        });

        // Job status endpoint
        this.app.get('/api/job/:jobId/status', async (req, res) => {
            const { jobId } = req.params;
            
            try {
                const jobStatus = this.activeJobs.get(jobId);
                if (!jobStatus) {
                    return res.status(404).json({ error: 'Job not found' });
                }

                res.json(jobStatus);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Enhanced RAG query endpoint
        this.app.post('/api/enhanced-rag/query', async (req, res) => {
            try {
                const { query, context, maxResults = 5 } = req.body;

                if (!query) {
                    return res.status(400).json({ error: 'Query is required' });
                }

                // Generate embedding for query
                const embedding = await this.generateEmbedding(query);

                // Search vector database
                const searchResults = await this.performVectorSearch(embedding, maxResults);

                // Enhance with Context7 if available
                let context7Results = [];
                if (req.body.includeContext7) {
                    context7Results = await this.queryContext7(query);
                }

                // Generate RAG response
                const ragResponse = await this.generateRAGResponse(query, searchResults, context7Results);

                res.json({
                    success: true,
                    query,
                    results: searchResults,
                    context7Results,
                    response: ragResponse,
                    processingTime: ragResponse.processingTime
                });

            } catch (error) {
                console.error('❌ Enhanced RAG error:', error);
                res.status(500).json({
                    success: false,
                    error: error.message
                });
            }
        });

        // Context7 integration endpoints
        this.app.post('/api/context7/analyze', async (req, res) => {
            try {
                const { content, type, options = {} } = req.body;

                const response = await fetch(`${CONFIG.context7Url}/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content, type, options })
                });

                if (response.ok) {
                    const analysis = await response.json();
                    res.json(analysis);
                } else {
                    throw new Error('Context7 analysis failed');
                }

            } catch (error) {
                console.error('❌ Context7 error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // AI Chat endpoint
        this.app.post('/api/ai/chat', async (req, res) => {
            try {
                const { message, context, chatHistory = [] } = req.body;

                // Build enhanced prompt with context
                const enhancedPrompt = this.buildEnhancedPrompt(message, context, chatHistory);

                // Query local LLM
                const response = await fetch(`${CONFIG.ollamaUrl}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'gemma3-legal',
                        prompt: enhancedPrompt,
                        stream: false,
                        options: {
                            temperature: 0.7,
                            max_tokens: 1000
                        }
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    // Generate contextual suggestions
                    const suggestions = this.generateSuggestions(message, context);

                    res.json({
                        response: data.response,
                        suggestions,
                        metadata: {
                            model: 'gemma3-legal',
                            processingTime: data.total_duration,
                            tokensGenerated: data.eval_count
                        }
                    });
                } else {
                    throw new Error('LLM request failed');
                }

            } catch (error) {
                console.error('❌ AI Chat error:', error);
                res.status(500).json({
                    error: 'Chat service temporarily unavailable',
                    fallback: 'Please try again or rephrase your question.'
                });
            }
        });

        // Proxy to SvelteKit in development
        if (CONFIG.environment === 'development') {
            this.app.use('/', createProxyMiddleware({
                target: `http://localhost:${CONFIG.svelteKitPort}`,
                changeOrigin: true,
                ws: true
            }));
        }
    }

    setupWebSockets() {
        this.wss.on('connection', (ws) => {
            console.log('📡 WebSocket client connected');

            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    
                    if (data.type === 'subscribe_job') {
                        ws.jobId = data.jobId;
                    }
                } catch (error) {
                    console.error('WebSocket message error:', error);
                }
            });

            ws.on('close', () => {
                console.log('📡 WebSocket client disconnected');
            });
        });
    }

    async processJobAsync(jobId) {
        try {
            const jobData = JSON.parse(await this.redis.rpop('processing_queue'));
            if (!jobData || jobData.id !== jobId) return;

            this.activeJobs.set(jobId, { status: 'processing', progress: 0 });
            this.broadcastJobUpdate(jobId, { status: 'processing', progress: 0 });

            const results = [];
            const totalFiles = jobData.files.length;

            for (let i = 0; i < totalFiles; i++) {
                const file = jobData.files[i];
                const progress = Math.round(((i + 1) / totalFiles) * 100);

                try {
                    // Process individual file
                    const result = await this.processFile(file, jobData.options);
                    results.push(result);

                    // Update progress
                    this.activeJobs.set(jobId, { 
                        status: 'processing', 
                        progress,
                        completed: i + 1,
                        total: totalFiles
                    });
                    this.broadcastJobUpdate(jobId, { progress, status: 'processing' });

                } catch (error) {
                    console.error(`❌ File processing failed: ${file.originalname}`, error);
                    results.push({
                        filename: file.originalname,
                        error: error.message,
                        success: false
                    });
                }
            }

            // Job completed
            const finalResult = {
                status: 'completed',
                progress: 100,
                results,
                summary: {
                    totalFiles,
                    successful: results.filter(r => r.success).length,
                    failed: results.filter(r => !r.success).length,
                    processingTime: Date.now() - new Date(jobData.timestamp).getTime()
                }
            };

            this.activeJobs.set(jobId, finalResult);
            this.broadcastJobUpdate(jobId, finalResult);

            // Cache results
            await this.redis.setex(`job_result_${jobId}`, 3600, JSON.stringify(finalResult));

        } catch (error) {
            console.error(`❌ Job ${jobId} failed:`, error);
            
            const errorResult = {
                status: 'error',
                error: error.message,
                progress: 0
            };

            this.activeJobs.set(jobId, errorResult);
            this.broadcastJobUpdate(jobId, errorResult);
        }
    }

    async processFile(file, options) {
        const startTime = Date.now();

        try {
            // Simulate file processing (replace with actual implementation)
            const mockResult = {
                filename: file.originalname,
                success: true,
                extractedText: `Extracted text from ${file.originalname}`,
                summary: `AI-generated summary for ${file.originalname}`,
                entities: ['Entity1', 'Entity2', 'Entity3'],
                prosecutionScore: Math.random() * 0.4 + 0.6,
                processingTime: Date.now() - startTime,
                metadata: {
                    fileSize: file.size,
                    mimeType: file.mimetype,
                    jurisdiction: options.jurisdiction
                }
            };

            return mockResult;

        } catch (error) {
            return {
                filename: file.originalname,
                success: false,
                error: error.message,
                processingTime: Date.now() - startTime
            };
        }
    }

    broadcastJobUpdate(jobId, update) {
        this.wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN && client.jobId === jobId) {
                client.send(JSON.stringify({
                    type: 'job_update',
                    jobId,
                    data: update
                }));
            }
        });
    }

    async generateEmbedding(text) {
        // Mock embedding generation (replace with actual implementation)
        return Array.from({ length: 384 }, () => Math.random());
    }

    async performVectorSearch(embedding, maxResults) {
        // Mock vector search (replace with actual pgvector implementation)
        return Array.from({ length: maxResults }, (_, i) => ({
            id: `doc_${i}`,
            content: `Mock search result ${i + 1}`,
            similarity: Math.random() * 0.5 + 0.5,
            metadata: { source: `document_${i + 1}.pdf` }
        }));
    }

    async queryContext7(query) {
        try {
            const response = await fetch(`${CONFIG.context7Url}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Context7 query failed:', error);
        }

        return [];
    }

    async generateRAGResponse(query, searchResults, context7Results) {
        const startTime = Date.now();

        try {
            const contextText = searchResults.map(r => r.content).join('\n\n');
            const prompt = `Based on the following context, answer the question: "${query}"\n\nContext:\n${contextText}`;

            const response = await fetch(`${CONFIG.ollamaUrl}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'gemma3-legal',
                    prompt,
                    stream: false
                })
            });

            if (response.ok) {
                const data = await response.json();
                return {
                    text: data.response,
                    processingTime: Date.now() - startTime,
                    model: 'gemma3-legal'
                };
            }
        } catch (error) {
            console.error('RAG generation failed:', error);
        }

        return {
            text: 'I apologize, but I cannot generate a response at this time.',
            processingTime: Date.now() - startTime,
            error: true
        };
    }

    buildEnhancedPrompt(message, context, chatHistory) {
        let prompt = `You are a legal AI assistant. Answer the following question: "${message}"\n\n`;

        if (context) {
            if (context.recentReports?.length > 0) {
                prompt += `Recent Reports:\n${context.recentReports.map(r => `- ${r.title}: ${r.summary}`).join('\n')}\n\n`;
            }
            
            if (context.topCitations?.length > 0) {
                prompt += `Relevant Citations:\n${context.topCitations.map(c => `- ${c.title} (${c.source})`).join('\n')}\n\n`;
            }
        }

        if (chatHistory.length > 0) {
            prompt += `Previous conversation:\n${chatHistory.slice(-5).map(msg => 
                `${msg.type === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
            ).join('\n')}\n\n`;
        }

        prompt += 'Please provide a helpful and accurate response.';
        return prompt;
    }

    generateSuggestions(message, context) {
        const suggestions = [
            'Analyze related case law',
            'Generate a case summary',
            'Find similar precedents',
            'Extract key legal issues',
            'Review contract terms'
        ];

        // Return 3 relevant suggestions
        return suggestions.slice(0, 3);
    }

    start() {
        this.server.listen(CONFIG.port, () => {
            console.log(`🚀 Production server running on port ${CONFIG.port}`);
            console.log(`📊 Redis connected: ${CONFIG.redisUrl}`);
            console.log(`🤖 Ollama URL: ${CONFIG.ollamaUrl}`);
            console.log(`🧠 Context7 URL: ${CONFIG.context7Url}`);
            console.log(`⚡ GPU Enabled: ${CONFIG.gpuEnabled}`);
        });
    }
}

// Start server
if (cluster.isMaster && process.env.CLUSTER_MODE === 'true') {
    // Cluster mode
    const numWorkers = os.cpus().length;
    console.log(`🎯 Starting ${numWorkers} worker processes...`);

    for (let i = 0; i < numWorkers; i++) {
        cluster.fork();
    }

    cluster.on('exit', (worker, code, signal) => {
        console.log(`⚠️ Worker ${worker.process.pid} died. Restarting...`);
        cluster.fork();
    });
} else {
    // Single process mode
    const server = new ProductionServer();
    server.start();
}

module.exports = ProductionServer;