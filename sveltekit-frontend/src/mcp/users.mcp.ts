/**
 * MCP Tool: Users Management
 * Clean abstraction layer for user operations using Drizzle ORM + pgvector
 * Following the suggested architecture pattern for Legal AI Platform
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { eq, desc, and, sql, like, or } from 'drizzle-orm';
import { cosineDistance } from 'drizzle-orm';
import { users, sessions } from '$lib/server/db/schema';
import type { User } from '$lib/types';
import bcrypt from 'bcryptjs';

// Database connection (based on MCP pgvector docs)
const connectionString = process.env.DATABASE_URL!;
const client = postgres(connectionString);
const db = drizzle(client);

export interface MCPToolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: Record<string, any>;
}

export interface UserCreateParams {
  email: string;
  password?: string;
  username?: string;
  firstName?: string;
  lastName?: string;
  role?: 'admin' | 'prosecutor' | 'detective' | 'user';
  department?: string;
  jurisdiction?: string;
  permissions?: string[];
  practiceAreas?: string[];
  barNumber?: string;
  firmName?: string;
  profileEmbedding?: number[]; // For AI-powered user matching
  metadata?: Record<string, any>;
}

export interface UserUpdateParams {
  userId: string;
  updates: Partial<UserCreateParams>;
}

export interface UserSearchParams {
  query?: string;
  role?: string;
  department?: string;
  jurisdiction?: string;
  isActive?: boolean;
  limit?: number;
  offset?: number;
}

export interface UserProfileMatchParams {
  profileEmbedding: number[];
  excludeUserId?: string;
  practiceAreas?: string[];
  threshold?: number;
  limit?: number;
}

/**
 * MCP Tool: Users Management
 * Thin adapter wrapping Drizzle ORM operations for user management
 */
export class UsersMCPTool {
  
  /**
   * Create new user
   */
  async createUser(params: UserCreateParams): Promise<MCPToolResponse<User>> {
    try {
      const userData: any = {
        id: crypto.randomUUID(),
        email: params.email,
        username: params.username,
        first_name: params.firstName,
        last_name: params.lastName,
        role: params.role || 'user',
        department: params.department,
        jurisdiction: params.jurisdiction,
        permissions: params.permissions ? JSON.stringify(params.permissions) : JSON.stringify([]),
        practice_areas: params.practiceAreas ? JSON.stringify(params.practiceAreas) : JSON.stringify([]),
        bar_number: params.barNumber,
        firm_name: params.firmName,
        profile_embedding: params.profileEmbedding,
        metadata: params.metadata ? JSON.stringify(params.metadata) : JSON.stringify({}),
        is_active: true,
        email_verified: false,
        created_at: new Date(),
        updated_at: new Date()
      };

      // Hash password if provided
      if (params.password) {
        userData.hashed_password = await bcrypt.hash(params.password, 12);
      }

      const newUser = await db.insert(users).values(userData).returning();

      // Remove sensitive data before returning
      const { hashed_password, ...userWithoutPassword } = newUser[0];

      return {
        success: true,
        data: userWithoutPassword as User,
        metadata: {
          tool: 'users.createUser',
          timestamp: Date.now(),
          hasProfileEmbedding: !!params.profileEmbedding
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.createUser',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Load users with optional filtering
   */
  async loadUsers(params: UserSearchParams): Promise<MCPToolResponse<User[]>> {
    try {
      let query = db.select({
        id: users.id,
        email: users.email,
        username: users.username,
        first_name: users.first_name,
        last_name: users.last_name,
        role: users.role,
        department: users.department,
        jurisdiction: users.jurisdiction,
        permissions: users.permissions,
        is_active: users.is_active,
        email_verified: users.email_verified,
        avatar_url: users.avatar_url,
        last_login_at: users.last_login_at,
        practice_areas: users.practice_areas,
        bar_number: users.bar_number,
        firm_name: users.firm_name,
        metadata: users.metadata,
        created_at: users.created_at,
        updated_at: users.updated_at
      }).from(users);
      
      // Add filters based on params
      const conditions = [];
      if (params.role) {
        conditions.push(eq(users.role, params.role));
      }
      if (params.department) {
        conditions.push(eq(users.department, params.department));
      }
      if (params.jurisdiction) {
        conditions.push(eq(users.jurisdiction, params.jurisdiction));
      }
      if (params.isActive !== undefined) {
        conditions.push(eq(users.is_active, params.isActive));
      }
      if (params.query) {
        conditions.push(
          or(
            sql`${users.first_name} ILIKE ${'%' + params.query + '%'}`,
            sql`${users.last_name} ILIKE ${'%' + params.query + '%'}`,
            sql`${users.email} ILIKE ${'%' + params.query + '%'}`,
            sql`${users.username} ILIKE ${'%' + params.query + '%'}`
          )
        );
      }
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }
      
      query = query.orderBy(desc(users.updated_at));
      
      if (params.limit) {
        query = query.limit(params.limit);
      }
      
      if (params.offset) {
        query = query.offset(params.offset);
      }

      const results = await query;

      return {
        success: true,
        data: results as User[],
        metadata: {
          tool: 'users.loadUsers',
          count: results.length,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.loadUsers',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Update existing user
   */
  async updateUser(params: UserUpdateParams): Promise<MCPToolResponse<User>> {
    try {
      const updateData: any = {
        updated_at: new Date()
      };
      
      // Map camelCase to snake_case for database columns
      if (params.updates.firstName) updateData.first_name = params.updates.firstName;
      if (params.updates.lastName) updateData.last_name = params.updates.lastName;
      if (params.updates.username) updateData.username = params.updates.username;
      if (params.updates.role) updateData.role = params.updates.role;
      if (params.updates.department) updateData.department = params.updates.department;
      if (params.updates.jurisdiction) updateData.jurisdiction = params.updates.jurisdiction;
      if (params.updates.permissions) updateData.permissions = JSON.stringify(params.updates.permissions);
      if (params.updates.practiceAreas) updateData.practice_areas = JSON.stringify(params.updates.practiceAreas);
      if (params.updates.barNumber) updateData.bar_number = params.updates.barNumber;
      if (params.updates.firmName) updateData.firm_name = params.updates.firmName;
      if (params.updates.profileEmbedding) updateData.profile_embedding = params.updates.profileEmbedding;
      if (params.updates.metadata) updateData.metadata = JSON.stringify(params.updates.metadata);

      // Hash password if provided
      if (params.updates.password) {
        updateData.hashed_password = await bcrypt.hash(params.updates.password, 12);
      }

      const updatedUser = await db.update(users)
        .set(updateData)
        .where(eq(users.id, params.userId))
        .returning({
          id: users.id,
          email: users.email,
          username: users.username,
          first_name: users.first_name,
          last_name: users.last_name,
          role: users.role,
          department: users.department,
          jurisdiction: users.jurisdiction,
          permissions: users.permissions,
          is_active: users.is_active,
          email_verified: users.email_verified,
          avatar_url: users.avatar_url,
          last_login_at: users.last_login_at,
          practice_areas: users.practice_areas,
          bar_number: users.bar_number,
          firm_name: users.firm_name,
          metadata: users.metadata,
          created_at: users.created_at,
          updated_at: users.updated_at
        });

      if (updatedUser.length === 0) {
        return {
          success: false,
          error: 'User not found',
          metadata: {
            tool: 'users.updateUser',
            userId: params.userId,
            timestamp: Date.now()
          }
        };
      }

      return {
        success: true,
        data: updatedUser[0] as User,
        metadata: {
          tool: 'users.updateUser',
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.updateUser',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Find similar users by profile embedding (AI-powered matching)
   */
  async findSimilarUsers(params: UserProfileMatchParams): Promise<MCPToolResponse<User[]>> {
    try {
      let query = db.select({
        user: {
          id: users.id,
          email: users.email,
          username: users.username,
          first_name: users.first_name,
          last_name: users.last_name,
          role: users.role,
          department: users.department,
          jurisdiction: users.jurisdiction,
          practice_areas: users.practice_areas,
          bar_number: users.bar_number,
          firm_name: users.firm_name,
          created_at: users.created_at
        },
        similarity: sql<number>`1 - (${users.profile_embedding} <=> ${params.profileEmbedding}::vector) as similarity`
      })
        .from(users)
        .where(
          and(
            sql`${users.profile_embedding} IS NOT NULL`,
            eq(users.is_active, true)
          )
        );

      // Add filters
      const conditions = [];
      if (params.excludeUserId) {
        conditions.push(sql`${users.id} != ${params.excludeUserId}`);
      }
      if (params.threshold) {
        conditions.push(sql`1 - (${users.profile_embedding} <=> ${params.profileEmbedding}::vector) > ${params.threshold}`);
      }

      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }

      query = query.orderBy(sql`similarity DESC`).limit(params.limit || 10);

      const results = await query;

      return {
        success: true,
        data: results.map(row => row.user) as User[],
        metadata: {
          tool: 'users.findSimilarUsers',
          vectorDimensions: params.profileEmbedding.length,
          similarityMethod: 'cosineDistance',
          threshold: params.threshold || 0.7,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.findSimilarUsers',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Get user analytics and statistics
   */
  async getUserAnalytics(): Promise<MCPToolResponse<any>> {
    try {
      const totalUsers = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(users);
        
      const activeUsers = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(users)
        .where(eq(users.is_active, true));

      const usersByRole = await db.select({
        role: users.role,
        count: sql`count(*)`.mapWith(Number)
      })
        .from(users)
        .groupBy(users.role);

      const verifiedUsers = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(users)
        .where(eq(users.email_verified, true));

      const recentUsers = await db.select({ count: sql`count(*)`.mapWith(Number) })
        .from(users)
        .where(sql`${users.created_at} >= NOW() - INTERVAL '30 days'`);

      return {
        success: true,
        data: {
          totalUsers: totalUsers[0].count,
          activeUsers: activeUsers[0].count,
          verifiedUsers: verifiedUsers[0].count,
          recentUsers: recentUsers[0].count,
          usersByRole: usersByRole.reduce((acc, item) => {
            acc[item.role] = item.count;
            return acc;
          }, {} as Record<string, number>),
          lastUpdated: new Date()
        },
        metadata: {
          tool: 'users.getUserAnalytics',
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.getUserAnalytics',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Authenticate user by email and password
   */
  async authenticateUser(email: string, password: string): Promise<MCPToolResponse<User>> {
    try {
      const userResult = await db.select()
        .from(users)
        .where(and(eq(users.email, email), eq(users.is_active, true)))
        .limit(1);

      if (userResult.length === 0) {
        return {
          success: false,
          error: 'Invalid credentials',
          metadata: {
            tool: 'users.authenticateUser',
            timestamp: Date.now()
          }
        };
      }

      const user = userResult[0];
      
      if (!user.hashed_password) {
        return {
          success: false,
          error: 'Account has no password set',
          metadata: {
            tool: 'users.authenticateUser',
            timestamp: Date.now()
          }
        };
      }

      const passwordValid = await bcrypt.compare(password, user.hashed_password);
      
      if (!passwordValid) {
        return {
          success: false,
          error: 'Invalid credentials',
          metadata: {
            tool: 'users.authenticateUser',
            timestamp: Date.now()
          }
        };
      }

      // Update last login timestamp
      await db.update(users)
        .set({ last_login_at: new Date() })
        .where(eq(users.id, user.id));

      // Remove sensitive data before returning
      const { hashed_password, ...userWithoutPassword } = user;

      return {
        success: true,
        data: userWithoutPassword as User,
        metadata: {
          tool: 'users.authenticateUser',
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.authenticateUser',
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Get user by ID (without password)
   */
  async getUserById(userId: string): Promise<MCPToolResponse<User>> {
    try {
      const userResult = await db.select({
        id: users.id,
        email: users.email,
        username: users.username,
        first_name: users.first_name,
        last_name: users.last_name,
        role: users.role,
        department: users.department,
        jurisdiction: users.jurisdiction,
        permissions: users.permissions,
        is_active: users.is_active,
        email_verified: users.email_verified,
        avatar_url: users.avatar_url,
        last_login_at: users.last_login_at,
        practice_areas: users.practice_areas,
        bar_number: users.bar_number,
        firm_name: users.firm_name,
        metadata: users.metadata,
        created_at: users.created_at,
        updated_at: users.updated_at
      })
        .from(users)
        .where(eq(users.id, userId))
        .limit(1);

      if (userResult.length === 0) {
        return {
          success: false,
          error: 'User not found',
          metadata: {
            tool: 'users.getUserById',
            userId,
            timestamp: Date.now()
          }
        };
      }

      return {
        success: true,
        data: userResult[0] as User,
        metadata: {
          tool: 'users.getUserById',
          timestamp: Date.now()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: {
          tool: 'users.getUserById',
          timestamp: Date.now()
        }
      };
    }
  }
}

// Export singleton instance
export const usersMCPTool = new UsersMCPTool();