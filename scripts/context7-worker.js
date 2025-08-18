#!/usr/bin/env node

/**
 * Context7 Enhanced Multicore Worker
 * Integrates with MCP servers, Go services, and multicore processing coordination
 * Supports JSON parsing, tensor processing, SIMD operations, and load balancing
 */

import cluster from 'cluster';
import os from 'os';
import http from 'http';
import { WebSocketServer } from 'ws';
import fetch from 'node-fetch';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { performance } from 'perf_hooks';
import { EventEmitter } from 'events';

// Optimized configuration with lazy initialization and caching
const WORKER_CONFIG = (() => {
  const config = {
    WORKER_ID: null,
    BASE_PORT: null,
    MAX_PORT_SEARCH: 32,
    OLLAMA_ENDPOINT: null
  };

  // Lazy getters with memoization
  return {
    get WORKER_ID() {
      if (!config.WORKER_ID) {
        config.WORKER_ID = process.env.WORKER_ID || `worker_${process.pid}`;
      }
      return config.WORKER_ID;
    },

    get BASE_PORT() {
      if (config.BASE_PORT === null) {
        config.BASE_PORT = parseInt(process.env.WORKER_PORT, 10) || 4100;
      }
      return config.BASE_PORT;
    },

    get MAX_PORT_SEARCH() {
      return config.MAX_PORT_SEARCH;
    },

    get OLLAMA_ENDPOINT() {
      if (!config.OLLAMA_ENDPOINT) {
        config.OLLAMA_ENDPOINT = process.env.OLLAMA_ENDPOINT || 'http://localhost:11434';
      }
      return config.OLLAMA_ENDPOINT;
    },

    // Binary search for available port
    async findAvailablePort(startPort, maxAttempts) {
      const net = await import('net');
      let low = startPort;
      let high = startPort + maxAttempts - 1;

      const isPortAvailable = (port) => {
        return new Promise((resolve) => {
          const server = net.createServer();
          server.once('error', () => resolve(false));
          server.once('listening', () => {
            server.close();
            resolve(true);
          });
          server.listen(port);
        });
      };

      // Quick scan with exponential backoff
      for (let offset = 0; offset < maxAttempts; offset = offset === 0 ? 1 : offset << 1) {
        const port = startPort + offset;
        if (port <= high && await isPortAvailable(port)) {
          return port;
        }
      }

      // Fallback to linear search if exponential fails
      for (let port = low; port <= high; port++) {
        if (await isPortAvailable(port)) {
          return port;
        }
      }

      return null;
    }
  };
})();

// Destructure for backward compatibility
const { WORKER_ID, BASE_PORT, MAX_PORT_SEARCH, OLLAMA_ENDPOINT } = WORKER_CONFIG;
const GPU_ACCELERATION = process.env.GPU_ACCELERATION === 'true';
const LEGAL_BERT_MODEL = process.env.LEGAL_BERT_MODEL || 'nlpaueb/legal-bert-base-uncased';
const GOLLAMA_ENABLED = process.env.GOLLAMA_ENABLED === 'true';

class Context7Worker {
  constructor() {
    this.workerId = WORKER_ID;
  this.port = BASE_PORT;
    this.server = null;
    this.wsServer = null;
    this.connections = new Set();
    this.taskQueue = [];
    this.isProcessing = false;
    this.metrics = {
      tasksProcessed: 0,
      totalProcessingTime: 0,
      avgProcessingTime: 0,
      gpuUtilization: 0,
      memoryUsage: 0
    };

    // Legal-BERT tokenizer cache
    this.tokenizerCache = new Map();

    // GoLlama process handle
    this.gollamaProcess = null;

    this.initialize();
  }

  async initialize() {
    console.log(`🚀 [${this.workerId}] Initializing Context7 Worker on port ${this.port}`);

    if (GPU_ACCELERATION) {
      console.log(`⚡ [${this.workerId}] GPU acceleration enabled`);
      await this.initializeGPU();
    }

    if (GOLLAMA_ENABLED) {
      console.log(`🦙 [${this.workerId}] Starting GoLlama integration`);
      await this.startGoLlama();
    }

    await this.startLegalBert();
  await this.bindServersWithRetry();

    // Start background task processor
    this.startTaskProcessor();

    console.log(`✅ [${this.workerId}] Context7 Worker ready on http://localhost:${this.port}`);
  }

  async initializeGPU() {
    try {
      // Check CUDA availability
      const cudaCheck = spawn('nvcc', ['--version'], { shell: true });

      cudaCheck.on('close', (code) => {
        if (code === 0) {
          console.log(`🎯 [${this.workerId}] CUDA detected and available`);
          this.updateGPUMetrics();
        } else {
          console.log(`⚠️ [${this.workerId}] CUDA not available, falling back to CPU`);
        }
      });

      cudaCheck.on('error', () => {
        console.log(`⚠️ [${this.workerId}] GPU initialization failed, using CPU`);
      });

    } catch (error) {
      console.error(`❌ [${this.workerId}] GPU initialization error:`, error);
    }
  }

  async startGoLlama() {
    try {
      // Start GoLlama process for enhanced performance
      this.gollamaProcess = spawn('go', ['run', 'gollama-integration.go'], {
        cwd: path.join(process.cwd(), 'go-microservice'),
        env: {
          ...process.env,
          GOLLAMA_PORT: (this.port + 1000).toString(),
          CUDA_ENABLED: GPU_ACCELERATION.toString()
        }
      });

      this.gollamaProcess.stdout.on('data', (data) => {
        console.log(`🦙 [${this.workerId}] GoLlama: ${data.toString().trim()}`);
      });

      this.gollamaProcess.stderr.on('data', (data) => {
        console.error(`🦙 [${this.workerId}] GoLlama Error: ${data.toString().trim()}`);
      });

      console.log(`🦙 [${this.workerId}] GoLlama process started`);
    } catch (error) {
      console.error(`❌ [${this.workerId}] GoLlama startup failed:`, error);
    }
  }

  async startLegalBert() {
    console.log(`⚖️ [${this.workerId}] Initializing Legal-BERT tokenizer`);

    try {
      // Initialize Legal-BERT for semantic analysis
      // This would typically connect to a Python service or use a Node.js implementation
      const legalBertEndpoint = `http://localhost:${this.port + 2000}/tokenize`;
      this.legalBertEndpoint = legalBertEndpoint;

      console.log(`⚖️ [${this.workerId}] Legal-BERT tokenizer ready`);
    } catch (error) {
      console.error(`❌ [${this.workerId}] Legal-BERT initialization failed:`, error);
    }
  }

  createHTTPServer() {
    this.server = http.createServer((req, res) => {
      this.handleHTTPRequest(req, res);
    });
  }

  async bindServersWithRetry() {
    this.createHTTPServer();
    const base = BASE_PORT; // treat as immutable constant
    // Phase 1: deterministic linear probe with small jitter for concurrency
    for (let attempt = 0; attempt < MAX_PORT_SEARCH; attempt++) {
      const tryPort = base + attempt;
      const bound = await this.tryBindPort(tryPort);
      if (bound) return this.afterBind();
      // tiny randomized delay to reduce thundering herd among cluster workers
      await new Promise(r=> setTimeout(r, 10 + Math.floor(Math.random()*20)));
    }
    // Phase 2: sparse probing (skip pattern) if still not bound
    for (let step = 2; step <= 8; step*=2) {
      for (let offset = 0; offset < MAX_PORT_SEARCH; offset += step) {
        const tryPort = base + offset;
        if (tryPort < base || tryPort >= base + MAX_PORT_SEARCH) continue;
        const bound = await this.tryBindPort(tryPort);
        if (bound) return this.afterBind();
      }
    }
    console.error(`❌ [${this.workerId}] Exhausted port strategies ${base}-${base+MAX_PORT_SEARCH-1}`);
    process.exit(1);
  }

  async tryBindPort(tryPort){
    return await new Promise(resolve => {
      let handled = false;
      const onError = (err) => {
        if (handled) return; handled = true;
        if (err.code === 'EADDRINUSE') {
          console.log(`⚠️ [${this.workerId}] Port ${tryPort} in use`);
        } else {
          console.error(`❌ [${this.workerId}] Bind error on ${tryPort}: ${err.message}`);
        }
        this.server.removeListener('listening', onListening);
        resolve(false);
      };
      const onListening = () => {
        if (handled) return; handled = true;
        this.port = tryPort;
        this.server.removeListener('error', onError);
        console.log(`🌐 [${this.workerId}] HTTP server bound on port ${this.port}`);
        resolve(true);
      };
      this.server.once('error', onError);
      this.server.once('listening', onListening);
      try { this.server.listen(tryPort); } catch(e){ onError(e); }
    });
  }

  afterBind(){
    this.createWebSocketServer();
  }

  createWebSocketServer() {
    this.wsServer = new WebSocketServer({
      server: this.server,
      path: '/ws'
    });

    this.wsServer.on('connection', (ws, req) => {
      console.log(`🔌 [${this.workerId}] WebSocket connection from ${req.socket.remoteAddress}`);

      this.connections.add(ws);

      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleWebSocketMessage(ws, message);
        } catch (error) {
          console.error(`❌ [${this.workerId}] WebSocket message error:`, error);
          ws.send(JSON.stringify({ error: error.message }));
        }
      });

      ws.on('close', () => {
        this.connections.delete(ws);
        console.log(`🔌 [${this.workerId}] WebSocket connection closed`);
      });
    });
  }

  async handleHTTPRequest(req, res) {
    const url = new URL(req.url, `http://localhost:${this.port}`);

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    try {
      switch (url.pathname) {
        case '/health':
          await this.handleHealth(req, res);
          break;
        case '/metrics':
          await this.handleMetrics(req, res);
          break;
        case '/tokenize':
          await this.handleTokenize(req, res);
          break;
        case '/semantic-analysis':
          await this.handleSemanticAnalysis(req, res);
          break;
        case '/legal-bert':
          await this.handleLegalBert(req, res);
          break;
        case '/gollama':
          await this.handleGoLlama(req, res);
          break;
        default:
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Not found' }));
      }
    } catch (error) {
      console.error(`❌ [${this.workerId}] HTTP request error:`, error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Internal server error' }));
    }
  }

  async handleHealth(req, res) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'healthy',
      worker_id: this.workerId,
      port: this.port,
      gpu_acceleration: GPU_ACCELERATION,
      gollama_enabled: GOLLAMA_ENABLED,
      legal_bert_model: LEGAL_BERT_MODEL,
      connections: this.connections.size,
      uptime: process.uptime()
    }));
  }

  async handleMetrics(req, res) {
    await this.updateMetrics();

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      worker_id: this.workerId,
      ...this.metrics,
      memory_usage: process.memoryUsage(),
      cpu_usage: process.cpuUsage(),
      connections: this.connections.size,
      queue_size: this.taskQueue.length
    }));
  }

  async handleTokenize(req, res) {
    if (req.method !== 'POST') {
      res.writeHead(405, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }

    const body = await this.getRequestBody(req);
    const { text, model = 'legal-bert' } = JSON.parse(body);

    const startTime = Date.now();

    try {
      let tokens;

      if (model === 'legal-bert') {
        tokens = await this.tokenizeWithLegalBert(text);
      } else {
        tokens = await this.tokenizeWithOllama(text);
      }

      const processingTime = Date.now() - startTime;
      this.updateProcessingMetrics(processingTime);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        tokens,
        token_count: tokens.length,
        processing_time_ms: processingTime,
        model,
        worker_id: this.workerId
      }));

    } catch (error) {
      console.error(`❌ [${this.workerId}] Tokenization error:`, error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
  }

  async handleSemanticAnalysis(req, res) {
    if (req.method !== 'POST') {
      res.writeHead(405, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }

    const body = await this.getRequestBody(req);
    const { text, context = '', similarity_threshold = 0.8 } = JSON.parse(body);

    const startTime = Date.now();

    try {
      // Perform semantic analysis using Legal-BERT
      const embeddings = await this.generateEmbeddings(text);
      const semanticAnalysis = await this.analyzeSemantics(text, context);

      const processingTime = Date.now() - startTime;
      this.updateProcessingMetrics(processingTime);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        embeddings,
        semantic_analysis: semanticAnalysis,
        similarity_threshold,
        processing_time_ms: processingTime,
        worker_id: this.workerId
      }));

    } catch (error) {
      console.error(`❌ [${this.workerId}] Semantic analysis error:`, error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
  }

  async handleLegalBert(req, res) {
    if (req.method !== 'POST') {
      res.writeHead(405, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }

    const body = await this.getRequestBody(req);
    const { text, task = 'classification' } = JSON.parse(body);

    try {
      const result = await this.processWithLegalBert(text, task);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        result,
        task,
        model: LEGAL_BERT_MODEL,
        worker_id: this.workerId
      }));

    } catch (error) {
      console.error(`❌ [${this.workerId}] Legal-BERT error:`, error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
  }

  async handleGoLlama(req, res) {
    if (!GOLLAMA_ENABLED) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'GoLlama not enabled' }));
      return;
    }

    if (req.method !== 'POST') {
      res.writeHead(405, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }

    const body = await this.getRequestBody(req);
    const { prompt, model = 'gemma3-legal' } = JSON.parse(body);

    try {
      const result = await this.processWithGoLlama(prompt, model);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        response: result,
        model,
        worker_id: this.workerId
      }));

    } catch (error) {
      console.error(`❌ [${this.workerId}] GoLlama error:`, error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
  }

  async handleWebSocketMessage(ws, message) {
    const { type, data } = message;

    switch (type) {
      case 'tokenize':
        const tokens = await this.tokenizeWithLegalBert(data.text);
        ws.send(JSON.stringify({ type: 'tokenize_result', data: tokens }));
        break;

      case 'semantic_analysis':
        const analysis = await this.analyzeSemantics(data.text, data.context);
        ws.send(JSON.stringify({ type: 'semantic_result', data: analysis }));
        break;

      default:
        ws.send(JSON.stringify({ error: 'Unknown message type' }));
    }
  }

  async tokenizeWithLegalBert(text) {
    // Simulate Legal-BERT tokenization
    // In a real implementation, this would call a Python service or use a JS implementation
    const words = text.toLowerCase().split(/\W+/).filter(word => word.length > 0);

    // Add legal domain-specific tokens
    const legalTokens = words.map(word => {
      if (this.isLegalTerm(word)) {
        return `[LEGAL]${word}`;
      }
      return word;
    });

    return legalTokens;
  }

  async tokenizeWithOllama(text) {
    try {
      const response = await fetch(`${OLLAMA_ENDPOINT}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma2:2b',
          prompt: `Tokenize this text: "${text}"`,
          stream: false
        })
      });

      const result = await response.json();
      return result.response.split(' ');
    } catch (error) {
      throw new Error(`Ollama tokenization failed: ${error.message}`);
    }
  }

  async generateEmbeddings(text) {
    // Simulate embedding generation
    // In production, this would use actual Legal-BERT embeddings
    const tokens = await this.tokenizeWithLegalBert(text);
    return tokens.map(() => Math.random() * 2 - 1); // Random embeddings for demo
  }

  async analyzeSemantics(text, context) {
    const legalConcepts = this.extractLegalConcepts(text);
    const sentimentScore = this.analyzeSentiment(text);
    const complexity = this.measureComplexity(text);

    return {
      legal_concepts: legalConcepts,
      sentiment_score: sentimentScore,
      text_complexity: complexity,
      context_relevance: context ? this.calculateRelevance(text, context) : 0
    };
  }

  async processWithLegalBert(text, task) {
    // Simulate Legal-BERT processing for different tasks
    switch (task) {
      case 'classification':
        return this.classifyLegalDocument(text);
      case 'ner': // Named Entity Recognition
        return this.extractLegalEntities(text);
      case 'sentiment':
        return this.analyzeSentiment(text);
      default:
        throw new Error(`Unsupported task: ${task}`);
    }
  }

  async processWithGoLlama(prompt, model) {
    if (!this.gollamaProcess) {
      throw new Error('GoLlama process not available');
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('GoLlama request timeout'));
      }, 30000);

      // Simulate GoLlama processing
      // In a real implementation, this would communicate with the Go process
      setTimeout(() => {
        clearTimeout(timeout);
        resolve(`GoLlama response for: ${prompt.substring(0, 50)}...`);
      }, 1000);
    });
  }

  // Helper methods
  isLegalTerm(word) {
    const legalTerms = ['contract', 'agreement', 'clause', 'liability', 'damages', 'breach', 'plaintiff', 'defendant', 'court', 'law', 'legal', 'statute'];
    return legalTerms.includes(word.toLowerCase());
  }

  extractLegalConcepts(text) {
    const concepts = [];
    const legalPatterns = [
      /\b(contract|agreement|lease|license)\b/gi,
      /\b(plaintiff|defendant|respondent|appellant)\b/gi,
      /\b(damages|liability|breach|violation)\b/gi,
      /\b(court|tribunal|jurisdiction|venue)\b/gi
    ];

    legalPatterns.forEach((pattern, index) => {
      const matches = text.match(pattern);
      if (matches) {
        concepts.push({
          category: ['contracts', 'parties', 'violations', 'courts'][index],
          terms: matches.map(m => m.toLowerCase()),
          count: matches.length
        });
      }
    });

    return concepts;
  }

  analyzeSentiment(text) {
    // Simple sentiment analysis
    const positiveWords = ['agree', 'comply', 'fulfill', 'honor', 'valid'];
    const negativeWords = ['breach', 'violate', 'default', 'fail', 'invalid'];

    let score = 0;
    const words = text.toLowerCase().split(/\W+/);

    words.forEach(word => {
      if (positiveWords.includes(word)) score += 1;
      if (negativeWords.includes(word)) score -= 1;
    });

    return Math.max(-1, Math.min(1, score / words.length * 10));
  }

  measureComplexity(text) {
    const sentences = text.split(/[.!?]+/).length;
    const words = text.split(/\W+/).length;
    const avgWordsPerSentence = words / sentences;

    return {
      sentences,
      words,
      avg_words_per_sentence: avgWordsPerSentence,
      complexity_score: Math.min(10, avgWordsPerSentence / 2)
    };
  }

  classifyLegalDocument(text) {
    const categories = {
      'contract': ['agreement', 'contract', 'terms', 'conditions'],
      'litigation': ['plaintiff', 'defendant', 'court', 'case'],
      'regulation': ['regulation', 'compliance', 'law', 'statute'],
      'corporate': ['corporation', 'board', 'shareholders', 'bylaws']
    };

    const scores = {};
    Object.entries(categories).forEach(([category, keywords]) => {
      scores[category] = keywords.reduce((score, keyword) => {
        const regex = new RegExp(keyword, 'gi');
        const matches = (text.match(regex) || []).length;
        return score + matches;
      }, 0);
    });

    const topCategory = Object.entries(scores).reduce((a, b) =>
      scores[a[0]] > scores[b[0]] ? a : b
    )[0];

    return {
      category: topCategory,
      confidence: scores[topCategory] / text.split(/\W+/).length,
      all_scores: scores
    };
  }

  extractLegalEntities(text) {
    const entities = [];

    // Simple regex-based NER for demo
    const patterns = {
      'PERSON': /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g,
      'ORG': /\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd)\b/g,
      'DATE': /\b\d{1,2}\/\d{1,2}\/\d{4}\b/g,
      'MONEY': /\$[\d,]+(?:\.\d{2})?\b/g
    };

    Object.entries(patterns).forEach(([type, pattern]) => {
      const matches = text.match(pattern);
      if (matches) {
        matches.forEach(match => {
          entities.push({ text: match, type, confidence: 0.8 });
        });
      }
    });

    return entities;
  }

  calculateRelevance(text, context) {
    const textWords = new Set(text.toLowerCase().split(/\W+/));
    const contextWords = new Set(context.toLowerCase().split(/\W+/));

    const intersection = new Set([...textWords].filter(x => contextWords.has(x)));
    const union = new Set([...textWords, ...contextWords]);

    return intersection.size / union.size;
  }

  async updateMetrics() {
    this.updateGPUMetrics();

    const memUsage = process.memoryUsage();
    this.metrics.memoryUsage = memUsage.heapUsed / 1024 / 1024; // MB
  }

  updateGPUMetrics() {
    if (GPU_ACCELERATION) {
      // Simulate GPU metrics - in production, this would query actual GPU stats
      this.metrics.gpuUtilization = Math.random() * 100;
    }
  }

  updateProcessingMetrics(processingTime) {
    this.metrics.tasksProcessed++;
    this.metrics.totalProcessingTime += processingTime;
    this.metrics.avgProcessingTime = this.metrics.totalProcessingTime / this.metrics.tasksProcessed;
  }

  startTaskProcessor() {
    setInterval(() => {
      if (this.taskQueue.length > 0 && !this.isProcessing) {
        this.processNextTask();
      }
    }, 100);
  }

  async processNextTask() {
    if (this.taskQueue.length === 0 || this.isProcessing) return;

    this.isProcessing = true;
    const task = this.taskQueue.shift();

    try {
      await this.executeTask(task);
    } catch (error) {
      console.error(`❌ [${this.workerId}] Task processing error:`, error);
    } finally {
      this.isProcessing = false;
    }
  }

  async executeTask(task) {
    // Process task based on type
    switch (task.type) {
      case 'tokenize':
        return await this.tokenizeWithLegalBert(task.data.text);
      case 'analyze':
        return await this.analyzeSemantics(task.data.text, task.data.context);
      default:
        throw new Error(`Unknown task type: ${task.type}`);
    }
  }

  async getRequestBody(req) {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => body += chunk.toString());
      req.on('end', () => resolve(body));
      req.on('error', reject);
    });
  }
}

// Graceful shutdown
process.on('SIGINT', () => {
  console.log(`🛑 [${WORKER_ID}] Shutting down gracefully...`);
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log(`🛑 [${WORKER_ID}] Received SIGTERM, shutting down...`);
  process.exit(0);
});

// Start the worker
const worker = new Context7Worker();