// src/lib/server/services/relay-auth-service.ts
// Auth service that uses PG:RELAY + Lucia, not direct Drizzle hooks.

import bcrypt from 'bcryptjs';

// This is the contract that your PG:RELAY (Go or Node relay) must expose.
// For now we'll use mock data, but this can be extended to call actual relay
const RELAY_BASE = process.env.RELAY_BASE ?? "http://localhost:8095";

// Shared user type coming back from relay
export interface RelayUser {
  id: string;
  email: string;
  name?: string;
  passwordHash: string;
  role?: string;
  is_active?: boolean;
}

// Simple session interface for manual session management
export interface RelaySession {
  id: string;
  userId: string;
  expiresAt: Date;
  attributes?: Record<string, any>;
}

export const relayAuthService = {
  /** Fetch a user by email via the relay */
  async getUserByEmail(email: string): Promise<RelayUser | null> {
    try {
      // For demo purposes, return demo user directly to avoid relay complexity
      // In production, this would call: `${RELAY_BASE}/auth-relay/user-by-email`
      if (email === 'demo@legalai.gov') {
        return {
          id: 'demo-user-001',
          email: 'demo@legalai.gov',
          name: 'Demo Prosecutor',
          passwordHash: await bcrypt.hash('demo123', 12),
          role: 'prosecutor',
          is_active: true
        };
      }
      
      // Future: implement actual relay call
      // const res = await fetch(`${RELAY_BASE}/auth-relay/user-by-email`, {
      //   method: "POST",
      //   headers: { "content-type": "application/json" },
      //   body: JSON.stringify({ email })
      // });
      // if (!res.ok) return null;
      // return (await res.json()) as RelayUser;
      
      return null;
    } catch (error) {
      console.error('RelayAuthService: Error getting user by email:', error);
      return null;
    }
  },

  /** Validate password against relay user */
  async validatePassword(user: RelayUser, password: string): Promise<boolean> {
    try {
      return await bcrypt.compare(password, user.passwordHash);
    } catch (error) {
      console.error('RelayAuthService: Error validating password:', error);
      return false;
    }
  },

  /** Register a new user via relay */
  async register(email: string, password: string, name?: string): Promise<RelayUser> {
    try {
      const passwordHash = await bcrypt.hash(password, 12);
      
      // Future: implement actual relay call
      // const res = await fetch(`${RELAY_BASE}/auth-relay/register`, {
      //   method: "POST",
      //   headers: { "content-type": "application/json" },
      //   body: JSON.stringify({ email, passwordHash, name })
      // });
      // if (!res.ok) throw new Error("Registration failed");
      // const data = await res.json();
      // return data.user as RelayUser;
      
      // For demo purposes, create a mock user
      const newUser: RelayUser = {
        id: this.generateId(),
        email,
        name: name || email.split('@')[0],
        passwordHash,
        role: 'user',
        is_active: true
      };
      
      console.log('✅ Demo user registered:', newUser.email);
      return newUser;
    } catch (error) {
      console.error('RelayAuthService: Error registering user:', error);
      throw error;
    }
  },

  /** Create a manual session (avoiding Lucia database calls) */
  async createSession(userId: string, attributes: Record<string, any> = {}): Promise<RelaySession> {
    try {
      const sessionId = this.generateId();
      const session: RelaySession = {
        id: sessionId,
        userId: userId,
        expiresAt: new Date(Date.now() + 1000 * 60 * 60 * 24 * 7), // 7 days
        attributes: {
          ...attributes,
          createdViaRelay: true,
          timestamp: new Date().toISOString()
        }
      };
      
      console.log('✅ Manual session created:', sessionId);
      return session;
    } catch (error) {
      console.error('RelayAuthService: Error creating session:', error);
      throw error;
    }
  },

  /** Demo user authentication (bypasses database entirely) */
  async authenticateDemoUser(): Promise<{ user: RelayUser; session: RelaySession } | null> {
    try {
      const demoUser = await this.getUserByEmail('demo@legalai.gov');
      if (!demoUser || !demoUser.is_active) {
        return null;
      }

      // Create session without database dependency
      const session = await this.createSession(demoUser.id, {
        userAgent: 'demo-auto-login',
        ipAddress: '127.0.0.1',
        deviceInfo: {
          platform: 'demo',
          mobile: false
        }
      });

      return { user: demoUser, session };
    } catch (error) {
      console.error('RelayAuthService: Error in demo authentication:', error);
      return null;
    }
  },

  /** Generate a random ID */
  generateId(): string {
    const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let result = '';
    for (let i = 0; i < 40; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  },

  /** Health check for relay service */
  async healthCheck(): Promise<boolean> {
    try {
      // Simple health check - if we can create a demo user object, service is "healthy"
      const demoUser = await this.getUserByEmail('demo@legalai.gov');
      return demoUser !== null;
    } catch (error) {
      console.error('RelayAuthService: Health check failed:', error);
      return false;
    }
  }
};