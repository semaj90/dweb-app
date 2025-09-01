/// <reference types="vite/client" />
import { Server } from 'socket.io';
import { dev } from "$app/environment";
import Redis from 'ioredis';
import type { RequestHandler } from './$types';


// WebSocket server for real-time updates
let io: Server | null = null;
let redis: Redis | null = null;

// Initialize WebSocket server and Redis subscriber
function initializeWebSocket() {
  if (io) return io;

  // Create Socket.IO server
  io = new Server({
    cors: {
      origin: dev ? 'http://localhost:5173' : false,
      methods: ['GET', 'POST'],
    },
    transports: ['websocket', 'polling'],
  });

  // Initialize Redis subscriber for job progress
  redis = new Redis({
    host: import.meta.env.REDIS_HOST || 'localhost',
    port: parseInt(import.meta.env.REDIS_PORT || '6379'),
    password: import.meta.env.REDIS_PASSWORD,
    lazyConnect: true,
  });

  // Handle WebSocket connections
  io.on('connection', (socket) => {
    console.log(`🔌 Client connected: ${socket.id}`);

    // Join case-specific rooms for targeted updates
    socket.on('join-case', (caseId: string) => {
      socket.join(`case-${caseId}`);
      console.log(`📂 Client ${socket.id} joined case room: ${caseId}`);
    });

    // Join upload-specific rooms for progress tracking
    socket.on('join-upload', (uploadId: string) => {
      socket.join(`upload-${uploadId}`);
      console.log(`📤 Client ${socket.id} joined upload room: ${uploadId}`);

      // Send current progress if available
      getCurrentProgress(uploadId).then(progress => {
        if (progress) {
          socket.emit('upload-progress', progress);
        }
      });
    });

    // Handle tensor processing subscription
    socket.on('subscribe-tensor', (jobId: string) => {
      socket.join(`tensor-${jobId}`);
      console.log(`🧮 Client ${socket.id} subscribed to tensor job: ${jobId}`);
    });

    // Handle search result streaming
    socket.on('subscribe-search', (searchId: string) => {
      socket.join(`search-${searchId}`);
      console.log(`🔍 Client ${socket.id} subscribed to search: ${searchId}`);
    });

    // Handle attention tracking
    socket.on('user-attention', (data: {
      type: 'focus' | 'blur' | 'scroll' | 'click' | 'typing';
      timestamp: string;
      metadata?: unknown;
    }) => {
      // Track user attention for AI context switching
      trackUserAttention(socket.id, data);
    });

    // Handle real-time collaboration
    socket.on('document-edit', (data: {
      documentId: string;
      change: any;
      userId: string;
    }) => {
      // Broadcast document changes to other collaborators
      socket.to(`doc-${data.documentId}`).emit('document-change', {
        change: data.change,
        userId: data.userId,
        timestamp: new Date().toISOString(),
      });
    });

    socket.on('disconnect', () => {
      console.log(`🔌 Client disconnected: ${socket.id}`);
    });
  });

  // Subscribe to Redis channels for job progress
  setupRedisSubscriptions();

  return io;
}

// Setup Redis subscriptions for job progress updates
function setupRedisSubscriptions() {
  if (!redis || !io) return;

  // Subscribe to job progress updates
  redis.psubscribe('progress:*', 'result:*', 'error:*');

  redis.on('pmessage', (pattern, channel, message) => {
    try {
      const data = JSON.parse(message);

      if (channel.startsWith('progress:')) {
        const uploadId = channel.split(':')[1];

        // Emit progress to specific upload room
        io?.to(`upload-${uploadId}`).emit('upload-progress', {
          uploadId,
          ...data,
          timestamp: new Date().toISOString(),
        });

        // Also emit to case room if caseId is available
        if (data.caseId) {
          io?.to(`case-${data.caseId}`).emit('case-progress', {
            uploadId,
            ...data,
            timestamp: new Date().toISOString(),
          });
        }
      }

      if (channel.startsWith('result:')) {
        const jobId = channel.split(':')[1];

        // Emit results to tensor processing rooms
        io?.to(`tensor-${jobId}`).emit('tensor-result', {
          jobId,
          result: data,
          timestamp: new Date().toISOString(),
        });
      }

      if (channel.startsWith('error:')) {
        const uploadId = channel.split(':')[1];

        // Emit errors to relevant rooms
        io?.to(`upload-${uploadId}`).emit('upload-error', {
          uploadId,
          error: data,
          timestamp: new Date().toISOString(),
        });
      }

    } catch (error: any) {
      console.error('❌ Failed to parse Redis message:', error);
    }
  });
}

// Track user attention for AI context switching
async function trackUserAttention(socketId: string, data: any): Promise<void> {
  if (!redis) return;

  const attentionEvent = {
    socketId,
    ...data,
    serverTimestamp: new Date().toISOString(),
  };

  // Store in Redis with expiration (1 hour)
  await redis.setex(
    `attention:${socketId}:${Date.now()}`,
    3600,
    JSON.stringify(attentionEvent)
  );

  // Trigger AI context switching if needed
  if (data.type === 'typing' && data.metadata?.query) {
    await triggerAIContextSwitching(socketId, data.metadata.query);
  }
}

// Trigger AI context switching based on user attention
async function triggerAIContextSwitching(socketId: string, query: string): Promise<void> {
  try {
    // Analyze query for legal context
    const contextResponse = await fetch('http://localhost:8080/api/context/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        socketId,
        timestamp: new Date().toISOString(),
      }),
    });

    if (contextResponse.ok) {
      const context = await contextResponse.json();

      // Emit context suggestions to client
      io?.to(socketId).emit('ai-context-suggestion', {
        query,
        suggestions: context.suggestions,
        relevantDocuments: context.documents,
        confidence: context.confidence,
      });
    }
  } catch (error: any) {
    console.error('❌ AI context switching failed:', error);
  }
}

// Get current progress for an upload
async function getCurrentProgress(uploadId: string): Promise<any> {
  if (!redis) return null;

  try {
    const progressData = await redis.get(`progress:${uploadId}`);
    return progressData ? JSON.parse(progressData) : null;
  } catch (error: any) {
    console.error('❌ Failed to get current progress:', error);
    return null;
  }
}

// Broadcast progress update to specific rooms
export function broadcastProgress(
  uploadId: string,
  caseId: string,
  progress: any
) {
  if (!io) return;

  const progressData = {
    uploadId,
    caseId,
    ...progress,
    timestamp: new Date().toISOString(),
  };

  // Emit to upload-specific room
  io.to(`upload-${uploadId}`).emit('upload-progress', progressData);

  // Emit to case-specific room
  io.to(`case-${caseId}`).emit('case-progress', progressData);
}

// Broadcast tensor processing results
export function broadcastTensorResult(jobId: string, result: any) {
  if (!io) return;

  io.to(`tensor-${jobId}`).emit('tensor-result', {
    jobId,
    result,
    timestamp: new Date().toISOString(),
  });
}

// Broadcast search results in real-time
export function broadcastSearchResults(searchId: string, results: any) {
  if (!io) return;

  io.to(`search-${searchId}`).emit('search-results', {
    searchId,
    results,
    timestamp: new Date().toISOString(),
  });
}

// HTTP handler for WebSocket endpoint
export const GET: RequestHandler = async ({ url }) => {
  const server = initializeWebSocket();

  // Return WebSocket connection info
  return new Response(JSON.stringify({
    status: 'WebSocket server running',
    endpoint: '/api/ws',
    features: [
      'Real-time upload progress',
      'Tensor processing updates',
      'AI context switching',
      'Document collaboration',
      'Search result streaming',
    ],
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
};

// Cleanup function
export function closeWebSocket() {
  if (io) {
    io.close();
    io = null;
  }
  if (redis) {
    redis.disconnect();
    redis = null;
  }
}