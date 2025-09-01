// src/lib/server/wsBroker.ts
import WebSocket from 'ws';
import { Redis } from 'ioredis';
import type { ProgressMsg } from '$lib/types/progress';

// In-memory session registry
const sessions = new Map<string, Set<WebSocket>>();

// Redis client for pub/sub across instances
let redis: Redis | null = null;
let subscriber: Redis | null = null;

export async function initializeWsBroker(): Promise<void> {
  try {
    const redisUrl = import.meta.env.REDIS_URL || 'redis://localhost:6379';
    
    // Publisher redis connection
    redis = new Redis(redisUrl);
    
    // Subscriber redis connection (separate connection required for pub/sub)
    subscriber = new Redis(redisUrl);
    
    // ioredis connects automatically, no need to call connect()
    
    // Subscribe to progress messages channel
    await subscriber.subscribe('evidence:progress');
    
    subscriber.on('message', (channel, message) => {
      if (channel === 'evidence:progress') {
        try {
          const data = JSON.parse(message);
          const { sessionId, ...msg } = data;
          
          // Send to local WebSocket connections
          sendWsMessageToSessionLocal(sessionId, msg);
          
        } catch (error: any) {
          console.error('❌ Error parsing Redis pub/sub message:', error);
        }
      }
    });
    
    redis.on('error', (err) => {
      console.error('❌ Redis publisher error:', err);
    });
    
    subscriber.on('error', (err) => {
      console.error('❌ Redis subscriber error:', err);
    });
    
    console.log('✅ WebSocket broker initialized with Redis pub/sub');
    
  } catch (error: any) {
    console.error('❌ Failed to initialize WebSocket broker:', error);
    // Continue without Redis - local only mode
  }
}

// Register a WebSocket connection for a session
export function registerWsConnection(sessionId: string, ws: WebSocket): void {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, new Set());
  }
  
  sessions.get(sessionId)!.add(ws);
  
  console.log(`🔌 WebSocket connected for session: ${sessionId} (${sessions.get(sessionId)!.size} total)`);
  
  // Setup cleanup on close
  ws.on('close', () => {
    const sessionSet = sessions.get(sessionId);
    if (!sessionSet) return;
    
    sessionSet.delete(ws);
    
    if (sessionSet.size === 0) {
      sessions.delete(sessionId);
      console.log(`🔌 Session ${sessionId} cleaned up - no active connections`);
    } else {
      console.log(`🔌 WebSocket disconnected for session: ${sessionId} (${sessionSet.size} remaining)`);
    }
  });
  
  ws.on('error', (error) => {
    console.error(`❌ WebSocket error for session ${sessionId}:`, error);
  });
  
  // Send initial connection confirmation
  try {
    ws.send(JSON.stringify({
      type: 'connection-established',
      sessionId,
      timestamp: new Date().toISOString()
    }));
  } catch (error: any) {
    console.error('❌ Error sending connection confirmation:', error);
  }
}

// Send message to local WebSocket connections only
function sendWsMessageToSessionLocal(sessionId: string, msg: ProgressMsg): void {
  const sessionSet = sessions.get(sessionId);
  if (!sessionSet || sessionSet.size === 0) {
    console.debug(`📭 No local WebSocket connections for session: ${sessionId}`);
    return;
  }
  
  const messageStr = JSON.stringify({
    ...msg,
    timestamp: new Date().toISOString(),
    sessionId
  });
  
  // Send to all connections for this session
  for (const ws of sessionSet) {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(messageStr);
      } else {
        sessionSet.delete(ws); // Clean up dead connections
      }
    } catch (error: any) {
      console.error(`❌ Error sending WebSocket message to session ${sessionId}:`, error);
      sessionSet.delete(ws); // Remove failed connection
    }
  }
  
  console.log(`📤 Sent message to ${sessionSet.size} WebSocket connections for session: ${sessionId}`);
}

// Send message to session (with Redis pub/sub for horizontal scaling)
export function sendWsMessageToSession(sessionId: string, msg: ProgressMsg): void {
  // Send to local connections immediately
  sendWsMessageToSessionLocal(sessionId, msg);
  
  // Publish to Redis for other instances
  if (redis && redis.status === 'ready') {
    try {
      redis.publish('evidence:progress', JSON.stringify({
        sessionId,
        ...msg
      }));
    } catch (error: any) {
      console.error('❌ Error publishing to Redis:', error);
    }
  }
  
  // Store message in Redis for offline clients (with TTL)
  if (redis && redis.status === 'ready') {
    try {
      const key = `session:${sessionId}:messages`;
      const messageData = JSON.stringify({
        ...msg,
        timestamp: new Date().toISOString()
      });
      
      // Store last 50 messages with 1 hour TTL
      redis.pipeline()
        .lpush(key, messageData)
        .ltrim(key, 0, 49)
        .expire(key, 3600)
        .exec();
        
    } catch (error: any) {
      console.error('❌ Error storing message in Redis:', error);
    }
  }
}

// Get missed messages for a session (when client reconnects)
export async function getMissedMessages(sessionId: string, since?: string): Promise<ProgressMsg[]> {
  if (!redis || redis.status !== 'ready') {
    return [];
  }
  
  try {
    const key = `session:${sessionId}:messages`;
    const messages = await redis.lrange(key, 0, -1);
    
    return messages
      .map(msg => {
        try {
          return JSON.parse(msg);
        } catch {
          return null;
        }
      })
      .filter(Boolean)
      .filter(msg => !since || new Date(msg.timestamp) > new Date(since))
      .reverse(); // Return in chronological order
      
  } catch (error: any) {
    console.error('❌ Error getting missed messages:', error);
    return [];
  }
}

// Get session connection count
export function getSessionConnectionCount(sessionId: string): number {
  return sessions.get(sessionId)?.size || 0;
}

// Get all active sessions
export function getActiveSessions(): string[] {
  return Array.from(sessions.keys());
}

// Broadcast to all sessions (admin functionality)
export function broadcastToAllSessions(msg: ProgressMsg): void {
  for (const sessionId of sessions.keys()) {
    sendWsMessageToSession(sessionId, msg);
  }
}

// Health check
export function wsHealthCheck(): { local: number; redis: boolean } {
  const localConnections = Array.from(sessions.values())
    .reduce((total, set) => total + set.size, 0);
    
  return {
    local: localConnections,
    redis: redis?.status === 'ready'
  };
}

// Graceful shutdown
export async function closeWsBroker(): Promise<void> {
  try {
    // Close all WebSocket connections
    for (const sessionSet of sessions.values()) {
      for (const ws of sessionSet) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(1001, 'Server shutting down');
        }
      }
    }
    
    sessions.clear();
    
    // Close Redis connections
    if (subscriber) {
      await subscriber.quit();
      subscriber = null;
    }
    
    if (redis) {
      await redis.quit();
      redis = null;
    }
    
    console.log('✅ WebSocket broker closed gracefully');
    
  } catch (error: any) {
    console.error('❌ Error closing WebSocket broker:', error);
  }
}
