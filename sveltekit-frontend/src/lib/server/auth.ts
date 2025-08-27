/**
 * Enhanced Authentication Service
 * Production-ready auth with Lucia v3, PostgreSQL, and vector embeddings
 */

import { Lucia } from "lucia";
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle";
import { dev } from "$app/environment";
import { db } from "./db/index";
import { sessions, users } from "./db/schema-postgres";
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
      // httpOnly is set by Lucia by default
    }
  },
  getUserAttributes: (attributes) => {
    return {
      id: attributes.id,
      email: attributes.email,
      username: attributes.username,
      first_name: attributes.first_name,
      last_name: attributes.last_name,
      role: attributes.role,
      department: attributes.department,
      jurisdiction: attributes.jurisdiction,
      avatar_url: attributes.avatar_url,
      email_verified: attributes.email_verified,
      last_login_at: attributes.last_login_at,
      is_active: attributes.is_active,
      practice_areas: attributes.practice_areas,
      bar_number: attributes.bar_number,
      firm_name: attributes.firm_name,
      permissions: attributes.permissions,
      metadata: attributes.metadata,
      created_at: attributes.created_at,
      updated_at: attributes.updated_at
    };
  }
});

declare module "lucia" {
  interface Register {
    Lucia: typeof lucia;
    DatabaseUserAttributes: DatabaseUserAttributes;
  }
}

export interface DatabaseUserAttributes {
  id: string;
  email: string;
  username: string | null;
  first_name: string | null;
  last_name: string | null;
  role: string;
  department: string | null;
  jurisdiction: string | null;
  avatar_url: string | null;
  email_verified: boolean;
  last_login_at: Date | null;
  is_active: boolean;
  practice_areas: unknown;
  bar_number: string | null;
  firm_name: string | null;
  permissions: unknown;
  metadata: unknown;
  created_at: Date;
  updated_at: Date;
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
      hashed_password: passwordHash,
      first_name: data.firstName || null,
      last_name: data.lastName || null,
      username: data.displayName || `${data.firstName || ''} ${data.lastName || ''}`.trim() || null,
      practice_areas: data.legalSpecialties || [],
      is_active: true
    }).returning();

    return newUser;
  }

  /**
   * Login user with email and password
   */
  async login(email: string, password: string) {
    // Find user by email
    const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
    
    if (!user || !user.hashed_password) {
      throw new Error("Invalid email or password");
    }

    // Check if user is active
    if (!user.is_active) {
      throw new Error("Account is deactivated");
    }

    // Verify password
    const validPassword = await this.argon2id.verify(user.hashed_password, password);
    
    if (!validPassword) {
      throw new Error("Invalid email or password");
    }

    // Update last login time
    await db.update(users)
      .set({ 
        last_login_at: new Date()
      })
      .where(eq(users.id, user.id));

    return user;
  }

  /**
   * Handle failed login attempts (simplified - no account locking)
   */
  private async handleFailedLogin(userId: string) {
    console.log(`Failed login attempt for user: ${userId}`);
    // TODO: Implement proper failed login tracking when schema supports it
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
    avatarUrl: string;
    timezone: string;
    locale: string;
    legalSpecialties: string[];
    preferences: Record<string, any>;
  }>) {
    // Map camelCase input to snake_case database columns
    const updateData: any = {};
    
    if (data.firstName !== undefined) updateData.first_name = data.firstName;
    if (data.lastName !== undefined) updateData.last_name = data.lastName;
    if (data.displayName !== undefined) updateData.username = data.displayName;
    if (data.avatarUrl !== undefined) updateData.avatar_url = data.avatarUrl;
    if (data.legalSpecialties !== undefined) updateData.practice_areas = data.legalSpecialties;
    if (data.preferences !== undefined) updateData.metadata = data.preferences;
    
    // Add timestamp
    updateData.updated_at = new Date();

    const [updatedUser] = await db.update(users)
      .set(updateData)
      .where(eq(users.id, userId))
      .returning();

    return updatedUser;
  }

  /**
   * Change user password
   */
  async changePassword(userId: string, currentPassword: string, newPassword: string) {
    const [user] = await db.select().from(users).where(eq(users.id, userId)).limit(1);
    
    if (!user || !user.hashed_password) {
      throw new Error("User not found");
    }

    // Verify current password
    const validPassword = await this.argon2id.verify(user.hashed_password, currentPassword);
    
    if (!validPassword) {
      throw new Error("Current password is incorrect");
    }

    // Hash new password
    const newPasswordHash = await this.argon2id.hash(newPassword);

    // Update password
    await db.update(users)
      .set({ 
        hashed_password: newPasswordHash,
        updated_at: new Date()
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
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      ...sessionCookie.attributes,
      path: '/'
    });
  }
  
  if (!result.session) {
    const sessionCookie = lucia.createBlankSessionCookie();
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      ...sessionCookie.attributes,
      path: '/'
    });
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
