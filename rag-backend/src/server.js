/**
 * Enhanced RAG Backend Server
 * Production-ready server with PostgreSQL + pgvector, Redis caching, 
 * Ollama integration, multi-agent orchestration, and WebSocket support
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import dotenv from 'dotenv';
import winston from 'winston';

// Import services
import { DatabaseService } from './services/database.js';
import { VectorService } from './services/vector.js';
import { CacheService } from './services/cache.js';
import { OllamaService } from './services/ollama.js';
import { DocumentProcessor } from './services/document-processor.js';
import { AgentOrchestrator } from './services/agent-orchestrator.js';
import { HealthService } from './services/health.js';

// Import routes
import { createRAGRoutes } from './routes/rag.js';
import { createDocumentRoutes } from './routes/documents.js';
import { createAgentRoutes } from './routes/agents.js';
import { createHealthRoutes } from './routes/health.js';

dotenv.config();

const app = express();
const httpServer = createServer(app);
const io = new SocketIOServer(httpServer, {
  cors: {
    origin: process.env.CORS_ORIGIN || "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

// Configure Winston logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors({
  origin: process.env.CORS_ORIGIN || "http://localhost:5173",
  credentials: true
}));
app.use(limiter);
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Services initialization
let services = {};

async function initializeServices() {
  try {
    logger.info('🚀 Initializing Enhanced RAG Backend Services...');
    
    // Initialize database connection
    services.database = new DatabaseService({
      host: process.env.DB_HOST || 'localhost',
      port: process.env.DB_PORT || 5432,
      database: process.env.DB_NAME || 'deeds_web_db',
      username: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || 'password'
    });
    await services.database.initialize();
    logger.info('✅ Database service initialized');

    // Initialize vector service with pgvector
    services.vector = new VectorService(services.database);
    await services.vector.initialize();
    logger.info('✅ Vector service initialized');

    // Initialize Redis cache
    services.cache = new CacheService({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });
    await services.cache.initialize();
    logger.info('✅ Cache service initialized');

    // Initialize Ollama service
    services.ollama = new OllamaService({
      baseUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
      defaultModel: process.env.OLLAMA_MODEL || 'gemma2:9b',
      embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text'
    });
    await services.ollama.initialize();
    logger.info('✅ Ollama service initialized');

    // Initialize document processor
    services.documentProcessor = new DocumentProcessor({
      ollama: services.ollama,
      vector: services.vector,
      cache: services.cache
    });
    logger.info('✅ Document processor initialized');

    // Initialize agent orchestrator
    services.agentOrchestrator = new AgentOrchestrator({
      ollama: services.ollama,
      database: services.database,
      cache: services.cache
    });
    logger.info('✅ Agent orchestrator initialized');

    // Initialize health service
    services.health = new HealthService(services);
    logger.info('✅ Health service initialized');

    logger.info('🎉 All services initialized successfully!');
    return true;
  } catch (error) {
    logger.error('❌ Failed to initialize services:', error);
    return false;
  }
}

// Socket.IO for real-time updates
io.on('connection', (socket) => {
  logger.info(`Client connected: ${socket.id}`);
  
  socket.on('join-room', (room) => {
    socket.join(room);
    logger.info(`Client ${socket.id} joined room: ${room}`);
  });

  socket.on('disconnect', () => {
    logger.info(`Client disconnected: ${socket.id}`);
  });
});

// Routes
app.use('/api/v1/rag', createRAGRoutes(services, io));
app.use('/api/v1/documents', createDocumentRoutes(services, io));
app.use('/api/v1/agents', createAgentRoutes(services, io));
app.use('/health', createHealthRoutes(services));

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Start server
const PORT = process.env.PORT || 8000;

async function startServer() {
  const initialized = await initializeServices();
  
  if (!initialized) {
    logger.error('❌ Failed to initialize services. Exiting...');
    process.exit(1);
  }

  httpServer.listen(PORT, () => {
    logger.info(`🚀 Enhanced RAG Backend running on port ${PORT}`);
    logger.info(`📊 Health check: http://localhost:${PORT}/health`);
    logger.info(`🔍 RAG API: http://localhost:${PORT}/api/v1/rag`);
    logger.info(`📄 Documents API: http://localhost:${PORT}/api/v1/documents`);
    logger.info(`🤖 Agents API: http://localhost:${PORT}/api/v1/agents`);
  });
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully...');
  
  // Close all services
  if (services.database) await services.database.close();
  if (services.cache) await services.cache.close();
  
  httpServer.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully...');
  
  // Close all services
  if (services.database) await services.database.close();
  if (services.cache) await services.cache.close();
  
  httpServer.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

// Start the server
startServer().catch((error) => {
  logger.error('Failed to start server:', error);
  process.exit(1);
});

export { app, services, io };