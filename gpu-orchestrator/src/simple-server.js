/**
 * Simplified GPU-Accelerated Legal AI Chat Server
 * Direct REST/WebSocket without complex XState machine
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import Redis from 'ioredis';
import dotenv from 'dotenv';

import { DatabaseService } from './services/database.js';
import { OllamaService } from './services/ollama.js';
import { CudaWorkerPool } from './services/cuda-worker-pool.js';
import { EmbeddingService } from './services/embedding.js';
import { TTSService } from './services/tts.js';

dotenv.config();

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Configuration
const CONFIG = {
  PORT: process.env.GPU_ORCHESTRATOR_PORT || 4001,
  REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379',
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  OLLAMA_ENDPOINT: process.env.OLLAMA_ENDPOINT || 'http://localhost:11434',
  CUDA_ENABLED: process.env.CUDA_ENABLED === 'true',
  GPU_MEMORY_LIMIT: process.env.GPU_MEMORY_LIMIT || '6GB',
  MAX_CONCURRENT_REQUESTS: parseInt(process.env.MAX_CONCURRENT_REQUESTS) || 10
};

// Initialize services
let redis, dbService, ollamaService, cudaWorkerPool, embeddingService, ttsService;
let activeRequests = 0;
let systemHealth = 'healthy';

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:3000'],
  credentials: true
}));
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Chat processing pipeline
async function processLegalChatRequest(message, options = {}) {
  try {
    activeRequests++;
    console.log(`🔹 Processing: "${message.substring(0, 50)}..."`);

    // Step 1: Generate embedding
    console.log('🔹 Generating embedding...');
    const embedding = await embeddingService.generateEmbedding(message, {
      model: 'nomic-embed-text',
      useCuda: CONFIG.CUDA_ENABLED
    });

    // Step 2: Retrieve legal context
    console.log('🔹 Retrieving legal context...');
    const context = await dbService.semanticSearch(embedding, {
      limit: 5,
      threshold: 0.7,
      documentTypes: ['case_law', 'contract', 'regulation']
    });

    // Step 3: Generate response with Gemma3
    console.log('🔹 Generating AI response with Gemma3...');
    const prompt = buildLegalPrompt(message, context);
    const response = await ollamaService.generateResponse({
      model: 'gemma3-legal:latest',
      prompt,
      stream: false,
      options: {
        num_gpu: 35,
        temperature: 0.1,
        top_p: 0.9
      }
    });

    // Step 4: Generate TTS (optional)
    let audioResponse = null;
    if (options.enableTTS) {
      console.log('🔹 Generating speech audio...');
      audioResponse = await ttsService.generateSpeech(response.response, {
        voice: 'legal-assistant-voice',
        speed: 1.0,
        format: 'mp3'
      });
    }

    activeRequests = Math.max(0, activeRequests - 1);

    return {
      success: true,
      response: response.response,
      audio: audioResponse,
      context: context.length,
      metadata: {
        model: 'gemma3-legal',
        processingTime: response.total_duration || 0,
        gpuAccelerated: CONFIG.CUDA_ENABLED,
        tokens: response.eval_count || 0
      }
    };

  } catch (error) {
    activeRequests = Math.max(0, activeRequests - 1);
    console.error('❌ Chat processing error:', error);
    throw error;
  }
}

function buildLegalPrompt(userMessage, context) {
  return `You are a highly specialized legal AI assistant with expertise in case law, contracts, and legal analysis.

Context from legal database:
${context.map(doc => `- ${doc.title}: ${doc.summary}`).join('\n')}

User question: ${userMessage}

Please provide a comprehensive legal analysis that:
1. Directly addresses the user's question
2. References relevant legal principles from the provided context
3. Includes appropriate citations where applicable
4. Maintains professional legal terminology
5. Provides actionable guidance while noting any limitations

Response:`;
}

// Routes
app.post('/chat', async (req, res) => {
  try {
    const { message, userId, sessionId, enableTTS = false } = req.body;
    
    if (!message?.trim()) {
      return res.status(400).json({ error: 'Message is required' });
    }

    if (activeRequests >= CONFIG.MAX_CONCURRENT_REQUESTS) {
      return res.status(429).json({ error: 'Server busy, please try again' });
    }

    const result = await processLegalChatRequest(message.trim(), { enableTTS });
    
    res.json({
      ...result,
      sessionId,
      userId,
      timestamp: Date.now()
    });
    
  } catch (error) {
    console.error('Chat endpoint error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      details: error.message 
    });
  }
});

app.get('/health', async (req, res) => {
  try {
    const healthChecks = await Promise.allSettled([
      redis.ping(),
      dbService.healthCheck(),
      ollamaService.healthCheck(),
      CONFIG.CUDA_ENABLED ? cudaWorkerPool.healthCheck() : Promise.resolve('disabled')
    ]);

    const services = {
      redis: healthChecks[0].status === 'fulfilled' && healthChecks[0].value === 'PONG',
      database: healthChecks[1].status === 'fulfilled' && healthChecks[1].value,
      ollama: healthChecks[2].status === 'fulfilled' && healthChecks[2].value,
      cuda: healthChecks[3].status === 'fulfilled' ? healthChecks[3].value : 'disabled'
    };

    const allHealthy = Object.values(services).every(status => status === true || status === 'disabled');
    systemHealth = allHealthy ? 'healthy' : 'degraded';

    res.json({
      status: systemHealth,
      activeRequests,
      gpuEnabled: CONFIG.CUDA_ENABLED,
      uptime: process.uptime(),
      services
    });
  } catch (error) {
    console.error('Health check error:', error);
    res.status(500).json({ status: 'error', error: error.message });
  }
});

app.get('/metrics', async (req, res) => {
  try {
    const metrics = await redis.hgetall('legal_ai_metrics');
    
    res.json({
      ...metrics,
      activeRequests,
      systemHealth,
      cudaStatus: CONFIG.CUDA_ENABLED ? await cudaWorkerPool.getStatus() : 'disabled',
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('Metrics error:', error);
    res.status(500).json({ error: error.message });
  }
});

// WebSocket handling
wss.on('connection', (ws) => {
  console.log('🔗 New WebSocket connection established');
  
  ws.on('message', async (data) => {
    try {
      const message = JSON.parse(data.toString());
      
      if (message.type === 'chat') {
        const result = await processLegalChatRequest(message.content, {
          enableTTS: message.enableTTS || false
        });
        
        ws.send(JSON.stringify({
          type: 'response',
          id: message.id,
          ...result,
          timestamp: Date.now()
        }));
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
      ws.send(JSON.stringify({ 
        type: 'error',
        error: 'Invalid message format or processing error',
        details: error.message
      }));
    }
  });
  
  ws.on('close', () => {
    console.log('🔌 WebSocket connection closed');
  });
});

// Initialize services and start server
async function startServer() {
  try {
    console.log('🚀 Initializing Simplified GPU-Accelerated Legal AI...');
    
    // Initialize Redis
    redis = new Redis(CONFIG.REDIS_URL);
    console.log('✅ Redis connected');
    
    // Initialize database
    dbService = new DatabaseService(CONFIG.DATABASE_URL);
    await dbService.initialize();
    console.log('✅ Database service initialized');
    
    // Initialize Ollama
    ollamaService = new OllamaService(CONFIG.OLLAMA_ENDPOINT);
    await ollamaService.initialize();
    console.log('✅ Ollama service connected');
    
    // Initialize CUDA workers if enabled
    cudaWorkerPool = new CudaWorkerPool({
      enabled: CONFIG.CUDA_ENABLED,
      maxWorkers: 4,
      memoryLimit: CONFIG.GPU_MEMORY_LIMIT
    });
    
    if (CONFIG.CUDA_ENABLED) {
      await cudaWorkerPool.initialize();
      console.log('✅ CUDA worker pool initialized');
    }
    
    // Initialize embedding service
    embeddingService = new EmbeddingService(ollamaService, cudaWorkerPool);
    console.log('✅ Embedding service ready');
    
    // Initialize TTS service
    ttsService = new TTSService();
    console.log('✅ TTS service ready');
    
    // Start server
    server.listen(CONFIG.PORT, () => {
      console.log(`
🎯 Simplified GPU-Accelerated Legal AI Running!
📡 Server: http://localhost:${CONFIG.PORT}
🔌 WebSocket: ws://localhost:${CONFIG.PORT}
🖥️  GPU Acceleration: ${CONFIG.CUDA_ENABLED ? '✅ Enabled' : '❌ Disabled'}
🧠 AI Models: Gemma3-Legal, NoMic-Embed-Text
💾 Database: PostgreSQL with pgvector
⚡ Redis: Caching and metrics
🎤 TTS: Ready for voice responses

Ready for legal AI requests! 🚀
      `);
    });
    
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down gracefully...');
  await redis.quit();
  await dbService.close();
  if (CONFIG.CUDA_ENABLED) {
    await cudaWorkerPool.destroy();
  }
  server.close();
  process.exit(0);
});

// Start the server
startServer().catch(console.error);