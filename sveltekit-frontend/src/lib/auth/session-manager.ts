// Enhanced Session Management with Redis Integration
// Handles secure session storage, management, and cleanup

import { dev } from '$app/environment';
import { redis } from '$lib/server/cache/redis-service';
import { randomBytes, createHash } from 'crypto';
import type { AuthUser, AuthSession } from './auth-store.js';
import type { UserRole } from './roles.js';
import type { Redis as IORedisClient } from 'ioredis';

export interface SessionData {
  id: string;
  userId: string;
  email: string;
  role: UserRole;
  isActive: boolean;
  createdAt: Date;
  expiresAt: Date;
  lastActivity: Date;
  ipAddress?: string;
  userAgent?: string;
  deviceFingerprint?: string;
  metadata: Record<string, any>;
}

export interface SessionConfig {
  maxAge: number; // Session duration in milliseconds
  maxInactivity: number; // Max inactivity before session expires
  renewalThreshold: number; // Renew session if less than this time remains
  maxSessionsPerUser: number; // Maximum concurrent sessions per user
  cleanupInterval: number; // Cleanup expired sessions interval
}

const DEFAULT_CONFIG: SessionConfig = {
  maxAge: 24 * 60 * 60 * 1000, // 24 hours
  maxInactivity: 30 * 60 * 1000, // 30 minutes
  renewalThreshold: 2 * 60 * 60 * 1000, // 2 hours
  maxSessionsPerUser: 5,
  cleanupInterval: 60 * 60 * 1000, // 1 hour
};

export class SessionManager {
  private static instance: SessionManager | null = null;
  private redisClient: any = null;
  private config: SessionConfig;
  private cleanupTimer: NodeJS.Timeout | null = null;
  private isInitialized = false;

  private constructor(config: Partial<SessionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  static getInstance(config?: Partial<SessionConfig>): SessionManager {
    if (!SessionManager.instance) {
      SessionManager.instance = new SessionManager(config);
    }
    return SessionManager.instance;
  }

  /**
   * Initialize Redis connection and session management
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Use centralized Redis service
      this.redisClient = redis;
      await this.redisClient.connect();

      // Setup error handling
      this.redisClient.on('error', (err) => {
        console.error('Redis session store error:', err);
      });

      this.redisClient.on('connect', () => {
        console.log('Redis session store connected');
      });

      // Connect to Redis
      await this.redisClient.connect();

      // Start cleanup timer
      this.startCleanupTimer();

      this.isInitialized = true;
      console.log('Session manager initialized successfully');
    } catch (error: any) {
      console.error('Failed to initialize session manager:', error);
      throw error;
    }
  }

  /**
   * Create a new session
   */
  async createSession(
    user: AuthUser, 
    request: {
      ipAddress?: string;
      userAgent?: string;
      deviceFingerprint?: string;
      metadata?: Record<string, any>;
    } = {}
  ): Promise<SessionData> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    // Generate session ID
    const sessionId = this.generateSessionId();
    
    // Create session data
    const now = new Date();
    const expiresAt = new Date(now.getTime() + this.config.maxAge);
    
    const sessionData: SessionData = {
      id: sessionId,
      userId: user.id,
      email: user.email,
      role: user.role,
      isActive: user.isActive,
      createdAt: now,
      expiresAt,
      lastActivity: now,
      ipAddress: request.ipAddress,
      userAgent: request.userAgent,
      deviceFingerprint: request.deviceFingerprint,
      metadata: request.metadata || {},
    };

    // Enforce max sessions per user
    await this.enforceSessionLimits(user.id);

    // Store session in Redis
    const sessionKey = this.getSessionKey(sessionId);
    const userSessionsKey = this.getUserSessionsKey(user.id);
    
    // Use pipeline for atomic operations
    const pipeline = this.redisClient.multi();
    
    // Store session data
    pipeline.set(sessionKey, JSON.stringify(sessionData));
    pipeline.expire(sessionKey, Math.ceil(this.config.maxAge / 1000));
    
    // Add to user's session list
    pipeline.sAdd(userSessionsKey, sessionId);
    pipeline.expire(userSessionsKey, Math.ceil(this.config.maxAge / 1000));
    
    // Store session activity index
    const activityKey = this.getActivityKey(sessionId);
    pipeline.set(activityKey, now.getTime().toString());
    pipeline.expire(activityKey, Math.ceil(this.config.maxAge / 1000));
    
    await pipeline.exec();

    console.log(`Session created for user ${user.id}: ${sessionId}`);
    return sessionData;
  }

  /**
   * Get session data
   */
  async getSession(sessionId: string): Promise<SessionData | null> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      const sessionKey = this.getSessionKey(sessionId);
      const data = await this.redisClient.get(sessionKey);
      
      if (!data || typeof data !== 'string') {
        return null;
      }

      const sessionData: SessionData = JSON.parse(data);
      
      // Convert date strings back to Date objects
      sessionData.createdAt = new Date(sessionData.createdAt);
      sessionData.expiresAt = new Date(sessionData.expiresAt);
      sessionData.lastActivity = new Date(sessionData.lastActivity);

      // Check if session is expired
      if (this.isSessionExpired(sessionData)) {
        await this.destroySession(sessionId);
        return null;
      }

      return sessionData;
    } catch (error: any) {
      console.error('Error getting session:', error);
      return null;
    }
  }

  /**
   * Update session activity
   */
  async updateSessionActivity(
    sessionId: string, 
    metadata?: Record<string, any>
  ): Promise<boolean> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      const sessionData = await this.getSession(sessionId);
      if (!sessionData) {
        return false;
      }

      // Update last activity and metadata
      const now = new Date();
      sessionData.lastActivity = now;
      
      if (metadata) {
        sessionData.metadata = { ...sessionData.metadata, ...metadata };
      }

      // Check if session should be renewed
      const timeUntilExpiry = sessionData.expiresAt.getTime() - now.getTime();
      if (timeUntilExpiry < this.config.renewalThreshold) {
        sessionData.expiresAt = new Date(now.getTime() + this.config.maxAge);
      }

      // Update in Redis
      const sessionKey = this.getSessionKey(sessionId);
      const activityKey = this.getActivityKey(sessionId);
      
      const pipeline = this.redisClient.multi();
      pipeline.set(sessionKey, JSON.stringify(sessionData));
      pipeline.expire(sessionKey, Math.ceil(this.config.maxAge / 1000));
      pipeline.set(activityKey, now.getTime().toString());
      pipeline.expire(activityKey, Math.ceil(this.config.maxAge / 1000));
      
      await pipeline.exec();
      return true;
    } catch (error: any) {
      console.error('Error updating session activity:', error);
      return false;
    }
  }

  /**
   * Destroy a session
   */
  async destroySession(sessionId: string): Promise<boolean> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      const sessionData = await this.getSession(sessionId);
      if (!sessionData) {
        return false;
      }

      const sessionKey = this.getSessionKey(sessionId);
      const userSessionsKey = this.getUserSessionsKey(sessionData.userId);
      const activityKey = this.getActivityKey(sessionId);

      // Remove from Redis
      const pipeline = this.redisClient.multi();
      pipeline.del(sessionKey);
      pipeline.sRem(userSessionsKey, sessionId);
      pipeline.del(activityKey);
      
      await pipeline.exec();

      console.log(`Session destroyed: ${sessionId}`);
      return true;
    } catch (error: any) {
      console.error('Error destroying session:', error);
      return false;
    }
  }

  /**
   * Destroy all sessions for a user
   */
  async destroyUserSessions(userId: string, exceptSessionId?: string): Promise<number> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      const userSessionsKey = this.getUserSessionsKey(userId);
      const sessionIds = await this.redisClient.sMembers(userSessionsKey);
      
      if (sessionIds.length === 0) {
        return 0;
      }

      const pipeline = this.redisClient.multi();
      let destroyedCount = 0;

      for (const sessionId of sessionIds) {
        if (exceptSessionId && sessionId === exceptSessionId) {
          continue; // Skip the exception session
        }

        const sessionKey = this.getSessionKey(sessionId);
        const activityKey = this.getActivityKey(sessionId);
        
        pipeline.del(sessionKey);
        pipeline.del(activityKey);
        pipeline.sRem(userSessionsKey, sessionId);
        destroyedCount++;
      }

      if (destroyedCount > 0) {
        await pipeline.exec();
        console.log(`Destroyed ${destroyedCount} sessions for user ${userId}`);
      }

      return destroyedCount;
    } catch (error: any) {
      console.error('Error destroying user sessions:', error);
      return 0;
    }
  }

  /**
   * Get all active sessions for a user
   */
  async getUserSessions(userId: string): Promise<SessionData[]> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      const userSessionsKey = this.getUserSessionsKey(userId);
      const sessionIds = await this.redisClient.sMembers(userSessionsKey);
      
      const sessions: SessionData[] = [];
      for (const sessionId of sessionIds) {
        const session = await this.getSession(sessionId);
        if (session) {
          sessions.push(session);
        }
      }

      // Sort by last activity (most recent first)
      sessions.sort((a, b) => b.lastActivity.getTime() - a.lastActivity.getTime());
      
      return sessions;
    } catch (error: any) {
      console.error('Error getting user sessions:', error);
      return [];
    }
  }

  /**
   * Get session statistics
   */
  async getSessionStats(): Promise<{
    totalSessions: number;
    activeSessions: number;
    expiredSessions: number;
    userSessionCounts: Record<string, number>;
  }> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      // Get all session keys
      const sessionKeys = await this.redisClient.keys(this.getSessionKey('*'));
      const totalSessions = sessionKeys.length;
      
      let activeSessions = 0;
      let expiredSessions = 0;
      const userSessionCounts: Record<string, number> = {};

      for (const key of sessionKeys) {
        const sessionId = key.replace(this.getSessionKey(''), '');
        const session = await this.getSession(sessionId);
        
        if (session) {
          activeSessions++;
          userSessionCounts[session.userId] = (userSessionCounts[session.userId] || 0) + 1;
        } else {
          expiredSessions++;
        }
      }

      return {
        totalSessions,
        activeSessions,
        expiredSessions,
        userSessionCounts,
      };
    } catch (error: any) {
      console.error('Error getting session stats:', error);
      return {
        totalSessions: 0,
        activeSessions: 0,
        expiredSessions: 0,
        userSessionCounts: {},
      };
    }
  }

  /**
   * Cleanup expired sessions
   */
  async cleanupExpiredSessions(): Promise<number> {
    if (!this.redisClient) {
      throw new Error('Session manager not initialized');
    }

    try {
      const sessionKeys = await this.redisClient.keys(this.getSessionKey('*'));
      let cleanedCount = 0;

      for (const key of sessionKeys) {
        const sessionId = key.replace(this.getSessionKey(''), '');
        const session = await this.getSession(sessionId);
        
        if (!session || this.isSessionExpired(session)) {
          await this.destroySession(sessionId);
          cleanedCount++;
        }
      }

      if (cleanedCount > 0) {
        console.log(`Cleaned up ${cleanedCount} expired sessions`);
      }

      return cleanedCount;
    } catch (error: any) {
      console.error('Error cleaning up expired sessions:', error);
      return 0;
    }
  }

  /**
   * Shutdown session manager
   */
  async shutdown(): Promise<void> {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }

    if (this.redisClient) {
      await this.redisClient.quit();
      this.redisClient = null;
    }

    this.isInitialized = false;
    console.log('Session manager shut down');
  }

  // Private helper methods

  private generateSessionId(): string {
    const bytes = randomBytes(32);
    return createHash('sha256').update(bytes).digest('hex');
  }

  private getSessionKey(sessionId: string): string {
    return `session:${sessionId}`;
  }

  private getUserSessionsKey(userId: string): string {
    return `user_sessions:${userId}`;
  }

  private getActivityKey(sessionId: string): string {
    return `activity:${sessionId}`;
  }

  private isSessionExpired(session: SessionData): boolean {
    const now = new Date();
    
    // Check absolute expiration
    if (session.expiresAt < now) {
      return true;
    }

    // Check inactivity expiration
    const inactivityTime = now.getTime() - session.lastActivity.getTime();
    if (inactivityTime > this.config.maxInactivity) {
      return true;
    }

    return false;
  }

  private async enforceSessionLimits(userId: string): Promise<void> {
    const userSessions = await this.getUserSessions(userId);
    
    if (userSessions.length >= this.config.maxSessionsPerUser) {
      // Sort by last activity (oldest first)
      const sortedSessions = userSessions.sort(
        (a, b) => a.lastActivity.getTime() - b.lastActivity.getTime()
      );
      
      // Remove oldest sessions to make room for new one
      const sessionsToRemove = sortedSessions.slice(0, userSessions.length - this.config.maxSessionsPerUser + 1);
      
      for (const session of sessionsToRemove) {
        await this.destroySession(session.id);
      }
      
      console.log(`Enforced session limit for user ${userId}: removed ${sessionsToRemove.length} old sessions`);
    }
  }

  private startCleanupTimer(): void {
    this.cleanupTimer = setInterval(async () => {
      try {
        await this.cleanupExpiredSessions();
      } catch (error: any) {
        console.error('Error during scheduled cleanup:', error);
      }
    }, this.config.cleanupInterval);
  }
}

// Export singleton instance
export const sessionManager = SessionManager.getInstance({
  maxAge: parseInt(import.meta.env.SESSION_MAX_AGE || '86400000'), // 24 hours default
  maxInactivity: parseInt(import.meta.env.SESSION_MAX_INACTIVITY || '1800000'), // 30 minutes default
  renewalThreshold: parseInt(import.meta.env.SESSION_RENEWAL_THRESHOLD || '7200000'), // 2 hours default
  maxSessionsPerUser: parseInt(import.meta.env.MAX_SESSIONS_PER_USER || '5'),
  cleanupInterval: parseInt(import.meta.env.SESSION_CLEANUP_INTERVAL || '3600000'), // 1 hour default
});

// Export types and utilities - avoid duplicate exports