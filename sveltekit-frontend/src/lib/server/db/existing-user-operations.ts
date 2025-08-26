/**
 * User Management Operations for Existing Database Schema
 * Works with the current database structure (users, sessions, user_profiles, user_activities)
 */

import postgres from 'postgres';
import bcrypt from 'bcryptjs';
import { nanoid } from 'nanoid';

// Database connection with pgvector support
const connectionString = process.env.DATABASE_URL || 
  `postgresql://${process.env.DATABASE_USER || 'legal_admin'}:${process.env.DATABASE_PASSWORD || '123456'}@${process.env.DATABASE_HOST || 'localhost'}:${process.env.DATABASE_PORT || '5432'}/${process.env.DATABASE_NAME || 'legal_ai_db'}`;

const sql = postgres(connectionString, {
  types: {
    vector: {
      to: 1184,
      from: [1184],
      serialize: (x: number[]) => `[${x.join(',')}]`,
      parse: (x: string) => x.slice(1, -1).split(',').map(Number),
    },
  },
});

// Types based on existing database schema
export interface ExistingUser {
  id: string;
  email: string;
  name?: string;
  role: string;
  created_at?: Date;
  updated_at?: Date;
  hashed_password?: string;
  is_active?: boolean;
  first_name?: string;
  last_name?: string;
  email_verified?: Date;
  avatar_url?: string;
}

export interface ExistingSession {
  id: string;
  user_id: string;
  expires_at: Date;
  created_at?: Date;
}

export interface ExistingUserProfile {
  id: string;
  user_id: string;
  bio?: string;
  phone?: string;
  address?: any;
  preferences?: any;
  created_at?: Date;
  updated_at?: Date;
}

// Service Results
interface ServiceResult<T = any> {
  success: boolean;
  data?: T;
  user?: ExistingUser;
  session?: ExistingSession;
  profile?: ExistingUserProfile;
  error?: string;
}

// ============================================================================
// USER AUTHENTICATION SERVICE - Using Existing Schema
// ============================================================================

export class ExistingUserAuthService {
  /**
   * Register a new user with the existing schema
   */
  static async registerUser(userData: {
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
    role?: string;
    profileData?: any;
  }): Promise<ServiceResult> {
    try {
      // Check if user already exists
      const existingUser = await sql`
        SELECT id, email FROM users WHERE email = ${userData.email}
      `;

      if (existingUser.length > 0) {
        return {
          success: false,
          error: 'User with this email already exists'
        };
      }

      // Hash password
      const passwordHash = await bcrypt.hash(userData.password, 12);

      // Create user with existing schema
      const newUser = await sql`
        INSERT INTO users (
          email, 
          hashed_password, 
          first_name, 
          last_name, 
          role,
          is_active
        ) VALUES (
          ${userData.email},
          ${passwordHash},
          ${userData.firstName || null},
          ${userData.lastName || null},
          ${userData.role || 'user'},
          true
        )
        RETURNING id, email, first_name, last_name, role, created_at, is_active
      `;

      const user = newUser[0] as ExistingUser;

      // Create user profile if data provided
      let profile = null;
      if (userData.profileData) {
        const profileResult = await sql`
          INSERT INTO user_profiles (
            user_id,
            bio,
            phone,
            preferences
          ) VALUES (
            ${user.id},
            ${userData.profileData.bio || null},
            ${userData.profileData.phoneNumber || null},
            ${JSON.stringify(userData.profileData.preferences || {})}
          )
          RETURNING *
        `;
        profile = profileResult[0];
      }

      return {
        success: true,
        user,
        profile,
      };

    } catch (error) {
      console.error('User registration error:', error);
      return {
        success: false,
        error: 'Registration failed: ' + error.message
      };
    }
  }

  /**
   * Login user and create session
   */
  static async loginUser(credentials: {
    email: string;
    password: string;
    ipAddress?: string;
    userAgent?: string;
    rememberMe?: boolean;
  }): Promise<ServiceResult> {
    try {
      // Get user with password
      const userResult = await sql`
        SELECT id, email, hashed_password, first_name, last_name, role, is_active
        FROM users 
        WHERE email = ${credentials.email}
      `;

      if (userResult.length === 0) {
        return {
          success: false,
          error: 'Invalid credentials'
        };
      }

      const user = userResult[0] as ExistingUser;

      if (!user.is_active) {
        return {
          success: false,
          error: 'Account is deactivated'
        };
      }

      // Verify password
      const validPassword = await bcrypt.compare(credentials.password, user.hashed_password!);
      if (!validPassword) {
        return {
          success: false,
          error: 'Invalid credentials'
        };
      }

      // Create session
      const sessionId = nanoid(32);
      const expiresAt = new Date();
      expiresAt.setHours(expiresAt.getHours() + (credentials.rememberMe ? 24 * 30 : 24)); // 30 days or 1 day

      const sessionResult = await sql`
        INSERT INTO sessions (id, user_id, expires_at)
        VALUES (${sessionId}, ${user.id}, ${expiresAt})
        RETURNING *
      `;

      const session = sessionResult[0] as ExistingSession;

      // Get user profile
      const profileResult = await sql`
        SELECT * FROM user_profiles WHERE user_id = ${user.id}
      `;
      const profile = profileResult[0] || null;

      // Remove password from response
      delete user.hashed_password;

      return {
        success: true,
        user,
        session,
        profile,
      };

    } catch (error) {
      console.error('User login error:', error);
      return {
        success: false,
        error: 'Login failed: ' + error.message
      };
    }
  }

  /**
   * Validate session
   */
  static async validateSession(sessionId: string): Promise<ServiceResult> {
    try {
      const result = await sql`
        SELECT 
          s.id as session_id,
          s.expires_at,
          u.id, u.email, u.first_name, u.last_name, u.role, u.is_active
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.id = ${sessionId} AND s.expires_at > NOW()
      `;

      if (result.length === 0) {
        return {
          success: false,
          error: 'Invalid or expired session'
        };
      }

      const data = result[0];
      const user: ExistingUser = {
        id: data.id,
        email: data.email,
        first_name: data.first_name,
        last_name: data.last_name,
        role: data.role,
        is_active: data.is_active,
      };

      const session: ExistingSession = {
        id: data.session_id,
        user_id: data.id,
        expires_at: data.expires_at,
      };

      return {
        success: true,
        user,
        session,
      };

    } catch (error) {
      console.error('Session validation error:', error);
      return {
        success: false,
        error: 'Session validation failed: ' + error.message
      };
    }
  }

  /**
   * Logout user (invalidate session)
   */
  static async logoutUser(sessionId: string, ipAddress?: string): Promise<ServiceResult> {
    try {
      const result = await sql`
        DELETE FROM sessions WHERE id = ${sessionId}
      `;

      return {
        success: true,
        data: { deleted: result.count > 0 }
      };

    } catch (error) {
      console.error('User logout error:', error);
      return {
        success: false,
        error: 'Logout failed: ' + error.message
      };
    }
  }
}

// ============================================================================
// USER PROFILE SERVICE - Using Existing Schema
// ============================================================================

export class ExistingUserProfileService {
  /**
   * Get user profile
   */
  static async getUserProfile(userId: string): Promise<ServiceResult> {
    try {
      const result = await sql`
        SELECT 
          u.id, u.email, u.first_name, u.last_name, u.role, u.avatar_url,
          p.bio, p.phone, p.preferences
        FROM users u
        LEFT JOIN user_profiles p ON u.id = p.user_id
        WHERE u.id = ${userId}
      `;

      if (result.length === 0) {
        return {
          success: false,
          error: 'User not found'
        };
      }

      const data = result[0];
      const profile = {
        user: {
          id: data.id,
          email: data.email,
          first_name: data.first_name,
          last_name: data.last_name,
          role: data.role,
          avatar_url: data.avatar_url,
        },
        profile: {
          bio: data.bio,
          phone: data.phone,
          preferences: data.preferences,
        }
      };

      return {
        success: true,
        data: profile
      };

    } catch (error) {
      console.error('Get user profile error:', error);
      return {
        success: false,
        error: 'Failed to get profile: ' + error.message
      };
    }
  }

  /**
   * Update user profile
   */
  static async updateUserProfile(userId: string, updateData: any): Promise<ServiceResult> {
    try {
      // Update user table if needed
      if (updateData.firstName || updateData.lastName || updateData.avatarUrl) {
        await sql`
          UPDATE users 
          SET 
            first_name = COALESCE(${updateData.firstName}, first_name),
            last_name = COALESCE(${updateData.lastName}, last_name),
            avatar_url = COALESCE(${updateData.avatarUrl}, avatar_url),
            updated_at = NOW()
          WHERE id = ${userId}
        `;
      }

      // Update or create profile
      const profileData = {
        bio: updateData.bio,
        phone: updateData.phoneNumber,
        preferences: JSON.stringify(updateData.preferences || {}),
      };

      const result = await sql`
        INSERT INTO user_profiles (user_id, bio, phone, preferences)
        VALUES (${userId}, ${profileData.bio}, ${profileData.phone}, ${profileData.preferences})
        ON CONFLICT (user_id) DO UPDATE SET
          bio = EXCLUDED.bio,
          phone = EXCLUDED.phone,
          preferences = EXCLUDED.preferences,
          updated_at = NOW()
        RETURNING *
      `;

      return {
        success: true,
        profile: result[0]
      };

    } catch (error) {
      console.error('Update user profile error:', error);
      return {
        success: false,
        error: 'Failed to update profile: ' + error.message
      };
    }
  }

  /**
   * Delete user (soft delete)
   */
  static async deleteUser(userId: string): Promise<ServiceResult> {
    try {
      const result = await sql`
        UPDATE users 
        SET is_active = false, updated_at = NOW()
        WHERE id = ${userId}
      `;

      return {
        success: true,
        data: { deactivated: result.count > 0 }
      };

    } catch (error) {
      console.error('Delete user error:', error);
      return {
        success: false,
        error: 'Failed to delete user: ' + error.message
      };
    }
  }
}