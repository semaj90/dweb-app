import type { ProgressMsg } from '$lib/types/progress';
import Redis from 'ioredis';

// In-memory WebSocket session registry
const sessions = new Map<string, Set<WebSocket>>();

// Redis client for pub/sub (optional but recommended for scaling)
let redis: Redis | null = null;
let redisPub: Redis | null = null;
let redisSub: Redis | null = null;

// Initialize Redis if available
try {
  const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
  redis = new Redis(redisUrl);
  redisPub = new Redis(redisUrl);
  redisSub = new Redis(redisUrl);

  // Subscribe to Redis channels for distributed WebSocket messaging
  redisSub.subscribe('ws:progress');
  redisSub.on('message', (channel, message) => {
    if (channel === 'ws:progress') {
      try {
        const { sessionId, msg } = JSON.parse(message);
        broadcastToSession(sessionId, msg);
      } catch (err) {
        console.error('Redis message parse error:', err);
      }
    }
  });

  console.log('WebSocket broker connected to Redis');
} catch (err) {
  console.warn('Redis not available for WebSocket broker, using in-memory only:', err);
}

// Register a new WebSocket connection for a session
export function registerWsConnection(sessionId: string, ws: WebSocket) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, new Set());
  }
  sessions.get(sessionId)!.add(ws);

  // Handle cleanup when connection closes
  ws.addEventListener('close', () => {
    const set = sessions.get(sessionId);
    if (!set) return;
    
    set.delete(ws);
    if (set.size === 0) {
      sessions.delete(sessionId);
    }
  });

  // Handle connection errors
  ws.addEventListener('error', (error) => {
    console.error('WebSocket error for session', sessionId, ':', error);
    const set = sessions.get(sessionId);
    if (set) {
      set.delete(ws);
      if (set.size === 0) {
        sessions.delete(sessionId);
      }
    }
  });

  console.log(`WebSocket registered for session ${sessionId}, total connections: ${sessions.get(sessionId)?.size || 0}`);
}

// Send message to all connections in a session (local only)
function broadcastToSession(sessionId: string, msg: ProgressMsg) {
  const set = sessions.get(sessionId);
  if (!set || set.size === 0) {
    console.debug('No local WebSocket sessions for', sessionId);
    return;
  }

  const msgString = JSON.stringify(msg);
  const deadConnections: WebSocket[] = [];

  for (const ws of set) {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(msgString);
      } else {
        deadConnections.push(ws);
      }
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      deadConnections.push(ws);
    }
  }

  // Clean up dead connections
  for (const deadWs of deadConnections) {
    set.delete(deadWs);
  }

  if (set.size === 0) {
    sessions.delete(sessionId);
  }
}

// Send message to session (distributed via Redis if available)
export function sendWsMessageToSession(sessionId: string, msg: ProgressMsg) {
  // Always try local first
  broadcastToSession(sessionId, msg);

  // If Redis is available, publish for other server instances
  if (redisPub) {
    try {
      redisPub.publish('ws:progress', JSON.stringify({ sessionId, msg }));
    } catch (error) {
      console.error('Failed to publish WebSocket message to Redis:', error);
    }
  }
}

// Get session statistics
export function getSessionStats() {
  const stats = {
    totalSessions: sessions.size,
    totalConnections: 0,
    sessions: {} as Record<string, number>
  };

  for (const [sessionId, connections] of sessions.entries()) {
    const count = connections.size;
    stats.totalConnections += count;
    stats.sessions[sessionId] = count;
  }

  return stats;
}

// Cleanup function for graceful shutdown
export async function closeWsBroker() {
  // Close all WebSocket connections
  for (const [sessionId, connections] of sessions.entries()) {
    for (const ws of connections) {
      try {
        ws.close(1001, 'Server shutting down');
      } catch (error) {
        console.error('Error closing WebSocket:', error);
      }
    }
  }
  sessions.clear();

  // Close Redis connections
  if (redis) await redis.quit();
  if (redisPub) await redisPub.quit();
  if (redisSub) await redisSub.quit();
}

// Heartbeat to keep connections alive
export function startHeartbeat() {
  const interval = setInterval(() => {
    for (const [sessionId, connections] of sessions.entries()) {
      const deadConnections: WebSocket[] = [];
      
      for (const ws of connections) {
        try {
          if (ws.readyState === WebSocket.OPEN) {
            ws.ping();
          } else {
            deadConnections.push(ws);
          }
        } catch (error) {
          deadConnections.push(ws);
        }
      }

      // Remove dead connections
      for (const deadWs of deadConnections) {
        connections.delete(deadWs);
      }

      if (connections.size === 0) {
        sessions.delete(sessionId);
      }
    }
  }, 30000); // 30 seconds

  return () => clearInterval(interval);
}