/**
 * GPU-Accelerated Legal AI Orchestrator
 * Integrates Ollama/Gemma3, CUDA workers, PostgreSQL pgvector, and XState
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { createMachine, interpret, assign } from 'xstate';
import Redis from 'ioredis';
import dotenv from 'dotenv';

// Import our custom modules
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
const redis = new Redis(CONFIG.REDIS_URL);
const dbService = new DatabaseService(CONFIG.DATABASE_URL);
const ollamaService = new OllamaService(CONFIG.OLLAMA_ENDPOINT);
const cudaWorkerPool = new CudaWorkerPool({
  enabled: CONFIG.CUDA_ENABLED,
  maxWorkers: 4,
  memoryLimit: CONFIG.GPU_MEMORY_LIMIT
});
const embeddingService = new EmbeddingService(ollamaService, cudaWorkerPool);
const ttsService = new TTSService();

// XState machine for legal AI chat orchestration
const legalAIChatMachine = createMachine({
  id: 'legalAIChat',
  initial: 'idle',
  context: {
    activeRequests: 0,
    gpuUtilization: 0,
    systemHealth: 'healthy',
    lastActivity: Date.now(),
    chatSessions: new Map(),
    processingQueue: []
  },
  states: {
    idle: {
      on: {
        CHAT_REQUEST: {
          target: 'processing',
          guard: 'canAcceptRequest'
        },
        HEALTH_CHECK: {
          target: 'healthCheck'
        },
        AUTO_OPTIMIZE: {
          target: 'optimizing'
        }
      },
      after: {
        60000: 'optimizing' // Auto-optimize every minute
      }
    },
    processing: {
      initial: 'embedding',
      states: {
        embedding: {
          invoke: {
            src: 'generateEmbedding',
            onDone: {
              target: 'retrieval',
              actions: assign({
                currentEmbedding: (_, event) => event.data
              })
            },
            onError: {
              target: '#legalAIChat.error',
              actions: assign({
                lastError: (_, event) => event.data
              })
            }
          }
        },
        retrieval: {
          invoke: {
            src: 'retrieveContext',
            onDone: {
              target: 'inference',
              actions: assign({
                retrievedContext: (_, event) => event.data
              })
            },
            onError: {
              target: '#legalAIChat.error'
            }
          }
        },
        inference: {
          invoke: {
            src: 'generateResponse',
            onDone: {
              target: 'tts',
              actions: assign({
                generatedResponse: (_, event) => event.data
              })
            },
            onError: {
              target: '#legalAIChat.error'
            }
          }
        },
        tts: {
          invoke: {
            src: 'generateSpeech',
            onDone: {
              target: '#legalAIChat.completed',
              actions: assign({
                audioResponse: (_, event) => event.data
              })
            },
            onError: {
              // TTS is optional, continue without audio
              target: '#legalAIChat.completed'
            }
          }
        }
      }
    },
    completed: {
      entry: ['sendResponse', 'updateMetrics'],
      always: {
        target: 'idle',
        actions: assign({
          activeRequests: (context) => Math.max(0, context.activeRequests - 1)
        })
      }
    },
    healthCheck: {
      invoke: {
        src: 'performHealthCheck',
        onDone: {
          target: 'idle',
          actions: assign({
            systemHealth: (_, event) => event.data.status
          })
        }
      }
    },
    optimizing: {
      invoke: {
        src: 'optimizeSystem',
        onDone: {
          target: 'idle'
        }
      }
    },
    error: {
      entry: 'logError',
      after: {
        5000: 'idle' // Return to idle after 5 seconds
      }
    }
  }
}, {
  guards: {
    canAcceptRequest: (context) => {
      return context.activeRequests < CONFIG.MAX_CONCURRENT_REQUESTS && 
             context.systemHealth === 'healthy';
    }
  },
  services: {
    generateEmbedding: async (context, event) => {
      console.log('🔹 Generating embedding for:', event.message);
      return await embeddingService.generateEmbedding(event.message, {
        model: 'nomic-embed-text',
        useCuda: CONFIG.CUDA_ENABLED
      });
    },
    
    retrieveContext: async (context, event) => {
      console.log('🔹 Retrieving legal context...');
      return await dbService.semanticSearch(context.currentEmbedding, {
        limit: 5,
        threshold: 0.7,
        documentTypes: ['case_law', 'contract', 'regulation']
      });
    },
    
    generateResponse: async (context, event) => {
      console.log('🔹 Generating AI response with Gemma3...');
      const prompt = await buildLegalPrompt(event.message, context.retrievedContext);
      return await ollamaService.generateResponse({
        model: 'gemma3-legal:latest',
        prompt,
        stream: false,
        options: {
          num_gpu: 35, // Use all GPU layers
          temperature: 0.1, // Low temperature for legal accuracy
          top_p: 0.9
        }
      });
    },
    
    generateSpeech: async (context, event) => {
      console.log('🔹 Generating speech audio...');
      return await ttsService.generateSpeech(context.generatedResponse.response, {
        voice: 'legal-assistant-voice',
        speed: 1.0,
        format: 'mp3'
      });
    },
    
    performHealthCheck: async () => {
      console.log('🔹 Performing system health check...');
      const checks = await Promise.allSettled([
        redis.ping(),
        dbService.healthCheck(),
        ollamaService.healthCheck(),
        cudaWorkerPool.healthCheck()
      ]);
      
      const healthy = checks.every(check => check.status === 'fulfilled');
      return { status: healthy ? 'healthy' : 'degraded' };
    },
    
    optimizeSystem: async () => {
      console.log('🔹 Optimizing system performance...');
      await cudaWorkerPool.optimize();
      await dbService.optimizeConnections();
      return true;
    }
  },
  actions: {
    sendResponse: (context, event) => {
      // Send response via WebSocket to connected clients
      const response = {
        id: event.requestId,
        text: context.generatedResponse?.response || 'Sorry, I encountered an error.',
        audio: context.audioResponse || null,
        metadata: {
          processingTime: Date.now() - context.lastActivity,
          gpuAccelerated: CONFIG.CUDA_ENABLED,
          model: 'gemma3-legal'
        }
      };
      
      broadcastToClients(response);
    },
    
    updateMetrics: (context) => {
      // Update Redis metrics
      redis.hset('legal_ai_metrics', {
        'requests_processed': Date.now(),
        'gpu_utilization': context.gpuUtilization,
        'active_requests': context.activeRequests
      });
    },
    
    logError: (context, event) => {
      console.error('❌ Legal AI Chat Error:', event.data);
      redis.lpush('legal_ai_errors', JSON.stringify({
        error: event.data,
        timestamp: Date.now(),
        context: context
      }));
    }
  }
});

// Start XState service
const chatService = interpret(legalAIChatMachine);
chatService.start();

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000'],
  credentials: true
}));
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Routes
app.post('/chat', async (req, res) => {
  try {
    const { message, userId, sessionId } = req.body;
    
    if (!message?.trim()) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Send event to XState machine
    chatService.send({
      type: 'CHAT_REQUEST',
      message: message.trim(),
      userId,
      sessionId,
      requestId
    });
    
    res.json({
      requestId,
      status: 'processing',
      message: 'Request queued for processing'
    });
    
  } catch (error) {
    console.error('Chat endpoint error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/health', async (req, res) => {
  chatService.send('HEALTH_CHECK');
  
  const state = chatService.getSnapshot();
  res.json({
    status: state.context.systemHealth,
    activeRequests: state.context.activeRequests,
    gpuEnabled: CONFIG.CUDA_ENABLED,
    uptime: process.uptime(),
    services: {
      redis: await redis.ping() === 'PONG',
      database: await dbService.healthCheck(),
      ollama: await ollamaService.healthCheck(),
      cuda: CONFIG.CUDA_ENABLED ? await cudaWorkerPool.healthCheck() : 'disabled'
    }
  });
});

app.get('/metrics', async (req, res) => {
  const metrics = await redis.hgetall('legal_ai_metrics');
  const state = chatService.getSnapshot();
  
  res.json({
    ...metrics,
    currentState: state.value,
    activeRequests: state.context.activeRequests,
    systemHealth: state.context.systemHealth
  });
});

// WebSocket handling
wss.on('connection', (ws) => {
  console.log('🔗 New WebSocket connection established');
  
  ws.on('message', async (data) => {
    try {
      const message = JSON.parse(data.toString());
      
      if (message.type === 'chat') {
        const requestId = `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        chatService.send({
          type: 'CHAT_REQUEST',
          message: message.content,
          userId: message.userId,
          sessionId: message.sessionId,
          requestId,
          ws // Store WebSocket for direct response
        });
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
      ws.send(JSON.stringify({ error: 'Invalid message format' }));
    }
  });
  
  ws.on('close', () => {
    console.log('🔌 WebSocket connection closed');
  });
});

// Helper functions
function broadcastToClients(message) {
  wss.clients.forEach(client => {
    if (client.readyState === client.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

async function buildLegalPrompt(userMessage, context) {
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

// Initialize services and start server
async function startServer() {
  try {
    console.log('🚀 Initializing GPU-Accelerated Legal AI Orchestrator...');
    
    // Initialize database connection
    await dbService.initialize();
    console.log('✅ Database service initialized');
    
    // Initialize CUDA workers if enabled
    if (CONFIG.CUDA_ENABLED) {
      await cudaWorkerPool.initialize();
      console.log('✅ CUDA worker pool initialized');
    }
    
    // Test Ollama connection
    await ollamaService.initialize();
    console.log('✅ Ollama service connected');
    
    // Start server
    server.listen(CONFIG.PORT, () => {
      console.log(`
🎯 GPU-Accelerated Legal AI Orchestrator Running!
📡 Server: http://localhost:${CONFIG.PORT}
🔌 WebSocket: ws://localhost:${CONFIG.PORT}
🖥️  GPU Acceleration: ${CONFIG.CUDA_ENABLED ? '✅ Enabled' : '❌ Disabled'}
🧠 AI Models: Gemma3-Legal, NoMic-Embed-Text
💾 Database: PostgreSQL with pgvector
⚡ Redis: Caching and pub/sub
🎤 TTS: Ready for voice responses
      `);
    });
    
  } catch (error) {
    console.error('❌ Failed to start orchestrator:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down gracefully...');
  chatService.stop();
  await redis.quit();
  await dbService.close();
  server.close();
  process.exit(0);
});

// Start the server
startServer().catch(console.error);