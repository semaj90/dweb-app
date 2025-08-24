/**
 * Enhanced Authentication Service
 * Production-ready auth with Lucia v3, PostgreSQL, and vector embeddings
 */

import { Lucia } from "lucia";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";
import { dev } from "$app/environment";
import { db } from "./db/index";
import { sessions, users } from "./db/schema-unified";
import { eq } from "drizzle-orm";
import { Argon2id } from "oslo/password";
import { generateId } from "lucia";
import type { RequestEvent } from "@sveltejs/kit";

// Initialize Lucia with Drizzle adapter
const adapter = new DrizzlePostgreSQLAdapter(db, sessions, users);

export const lucia = new Lucia(adapter, {
  sessionCookie: {
    attributes: {
      secure: !dev, // HTTPS in production
      sameSite: "strict",
      httpOnly: true
    }
  },
  getUserAttributes: (attributes) => {
    return {
      id: attributes.id,
      email: attributes.email,
      displayName: attributes.displayName,
      firstName: attributes.firstName,
      lastName: attributes.lastName,
      role: attributes.role,
      emailVerified: attributes.emailVerified,
      isActive: attributes.isActive,
      legalSpecialties: attributes.legalSpecialties,
      preferences: attributes.preferences,
      createdAt: attributes.createdAt
    };
  }
});

declare module "lucia" {
  interface Register {
    Lucia: typeof lucia;
    DatabaseUserAttributes: DatabaseUserAttributes;
  }
}

interface DatabaseUserAttributes {
  id: string;
  email: string;
  displayName: string | null;
  firstName: string | null;
  lastName: string | null;
  role: string;
  emailVerified: Date | null;
  isActive: boolean;
  legalSpecialties: unknown;
  preferences: unknown;
  createdAt: Date;
}

// Authentication utilities
export class AuthService {
  private argon2id = new Argon2id();

  /**
   * Register a new user with enhanced profile data
   */
  async register(data: {
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
    displayName?: string;
    legalSpecialties?: string[];
  }) {
    // Check if user already exists
    const existingUser = await db.select().from(users).where(eq(users.email, data.email)).limit(1);
    
    if (existingUser.length > 0) {
      throw new Error("User already exists");
    }

    // Hash password
    const passwordHash = await this.argon2id.hash(data.password);
    
    // Generate user ID
    const userId = generateId(15);

    // Create user
    const [newUser] = await db.insert(users).values({
      id: userId,
      email: data.email,
      passwordHash,
      firstName: data.firstName || null,
      lastName: data.lastName || null,
      displayName: data.displayName || `${data.firstName || ''} ${data.lastName || ''}`.trim() || null,
      legalSpecialties: data.legalSpecialties || [],
      preferences: {
        theme: 'system',
        language: 'en',
        notifications: {
          email: true,
          push: false,
          caseAlerts: true
        }
      },
      isActive: true
    }).returning();

    return newUser;
  }

  /**
   * Login user with email and password
   */
  async login(email: string, password: string) {
    // Find user by email
    const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
    
    if (!user || !user.passwordHash) {
      throw new Error("Invalid email or password");
    }

    // Check if user is active
    if (!user.isActive || user.isSuspended) {
      throw new Error("Account is deactivated or suspended");
    }

    // Check if account is locked
    if (user.lockedUntil && user.lockedUntil > new Date()) {
      throw new Error("Account is temporarily locked");
    }

    // Verify password
    const validPassword = await this.argon2id.verify(user.passwordHash, password);
    
    if (!validPassword) {
      // Increment login attempts
      await this.handleFailedLogin(user.id);
      throw new Error("Invalid email or password");
    }

    // Reset login attempts on successful login
    await db.update(users)
      .set({ 
        loginAttempts: 0, 
        lockedUntil: null,
        lastLoginAt: new Date()
      })
      .where(eq(users.id, user.id));

    return user;
  }

  /**
   * Handle failed login attempts with account locking
   */
  private async handleFailedLogin(userId: string) {
    const [user] = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    
    if (!user) return;

    const newAttempts = (user.loginAttempts || 0) + 1;
    const updateData: any = { loginAttempts: newAttempts };

    // Lock account after 5 failed attempts
    if (newAttempts >= 5) {
      updateData.lockedUntil = new Date(Date.now() + 30 * 60 * 1000); // Lock for 30 minutes
    }

    await db.update(users).set(updateData).where(eq(users.id, userId));
  }

  /**
   * Create session for user
   */
  async createSession(userId: string) {
    const session = await lucia.createSession(userId, {});
    return session;
  }

  /**
   * Validate session
   */
  async validateSession(sessionId: string) {
    const result = await lucia.validateSession(sessionId);
    return result;
  }

  /**
   * Invalidate session (logout)
   */
  async invalidateSession(sessionId: string) {
    await lucia.invalidateSession(sessionId);
  }

  /**
   * Invalidate all user sessions
   */
  async invalidateUserSessions(userId: string) {
    await lucia.invalidateUserSessions(userId);
  }

  /**
   * Logout user by invalidating session
   */
  async logout(sessionId?: string) {
    if (sessionId) {
      await this.invalidateSession(sessionId);
    }
  }

  /**
   * Request password reset (placeholder for email integration)
   */
  async requestPasswordReset(email: string) {
    // Find user by email
    const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
    
    if (!user) {
      // Don't reveal if email exists or not for security
      return { success: true };
    }

    // TODO: Implement email sending service
    // For now, just log the reset request
    console.log(`Password reset requested for user: ${email}`);
    
    return { success: true };
  }

  /**
   * Update user profile
   */
  async updateProfile(userId: string, data: Partial<{
    firstName: string;
    lastName: string;
    displayName: string;
    bio: string;
    timezone: string;
    locale: string;
    legalSpecialties: string[];
    preferences: Record<string, any>;
  }>) {
    const [updatedUser] = await db.update(users)
      .set({
        ...data,
        updatedAt: new Date()
      })
      .where(eq(users.id, userId))
      .returning();

    return updatedUser;
  }

  /**
   * Change user password
   */
  async changePassword(userId: string, currentPassword: string, newPassword: string) {
    const [user] = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    
    if (!user || !user.passwordHash) {
      throw new Error("User not found");
    }

    // Verify current password
    const validPassword = await this.argon2id.verify(user.passwordHash, currentPassword);
    
    if (!validPassword) {
      throw new Error("Current password is incorrect");
    }

    // Hash new password
    const newPasswordHash = await this.argon2id.hash(newPassword);

    // Update password
    await db.update(users)
      .set({ 
        passwordHash: newPasswordHash,
        updatedAt: new Date()
      })
      .where(eq(users.id, userId));

    // Invalidate all existing sessions to force re-login
    await this.invalidateUserSessions(userId);
  }
}

export const authService = new AuthService();

/**
 * Helper function to get user from request event
 */
export async function getUser(event: RequestEvent) {
  const sessionId = event.cookies.get(lucia.sessionCookieName);
  
  if (!sessionId) {
    return { user: null, session: null };
  }

  const result = await lucia.validateSession(sessionId);
  
  if (result.session && result.session.fresh) {
    const sessionCookie = lucia.createSessionCookie(result.session.id);
    event.cookies.set(sessionCookie.name, sessionCookie.value, sessionCookie.attributes);
  }
  
  if (!result.session) {
    const sessionCookie = lucia.createBlankSessionCookie();
    event.cookies.set(sessionCookie.name, sessionCookie.value, sessionCookie.attributes);
  }

  return result;
}

/**
 * Require authenticated user middleware
 */
export async function requireAuth(event: RequestEvent) {
  const { user, session } = await getUser(event);
  
  if (!user || !session) {
    throw new Error("Authentication required");
  }

  return { user, session };
}
