/**
 * User Management Database Operations
 * Complete CRUD with PostgreSQL + pgvector + Drizzle ORM
 */

import { drizzle } from 'drizzle-orm/node-postgres';
import { eq, and, isNull, count, sql, desc } from './index.js';
import { cosineDistance } from 'drizzle-orm';
import postgres from 'postgres';
import bcrypt from 'bcryptjs';
import { nanoid } from 'nanoid';
import type { 
  User, 
  NewUser, 
  UserWithProfile, 
  UserProfile, 
  NewUserProfile,
  UserSession,
  NewUserSession,
  UserActivity,
  NewUserActivity,
  FullUserProfile
} from './schema/user-management.js';
import { 
  users, 
  userProfiles, 
  userSessions, 
  userActivityLog,
  insertUserSchema,
  updateUserSchema,
  insertProfileSchema,
  updateProfileSchema
} from './schema/user-management.js';

// ============================================================================
// DATABASE CONNECTION
// ============================================================================

const connectionString = import.meta.env.DATABASE_URL || 
  `postgresql://${import.meta.env.DATABASE_USER || 'legal_admin'}:${import.meta.env.DATABASE_PASSWORD || '123456'}@${import.meta.env.DATABASE_HOST || 'localhost'}:${import.meta.env.DATABASE_PORT || '5432'}/${import.meta.env.DATABASE_NAME || 'legal_ai_db'}`;

// Create connection with pgvector support
const queryClient = postgres(connectionString, {
  max: 20,
  idle_timeout: 20,
  connect_timeout: 10,
  prepare: false,
  types: {
    // Support for pgvector
    vector: {
      to: 1184,
      from: [1184],
      serialize: (x: number[]) => {
        if (Array.isArray(x)) {
          return `[${x.join(',')}]`;
        }
        return x || '[]';
      },
      parse: (x: string) => {
        if (typeof x === 'string' && x.startsWith('[') && x.endsWith(']')) {
          return x.slice(1, -1).split(',').map(Number);
        }
        return [];
      },
    },
  },
});

const userDb = drizzle(queryClient);

// ============================================================================
// USER AUTHENTICATION OPERATIONS
// ============================================================================

export class UserAuthService {
  /**
   * Register a new user with complete profile setup
   */
  static async registerUser(userData: {
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
    role?: string;
    jurisdiction?: string;
    practiceAreas?: string[];
    profileData?: Partial<NewUserProfile>;
  }): Promise<{ user: User; profile?: UserProfile; success: boolean; error?: string }> {
    try {
      // Validate input
      const validatedUser = insertUserSchema.parse({
        email: userData.email.toLowerCase(),
        firstName: userData.firstName,
        lastName: userData.lastName,
        role: userData.role || 'user',
        jurisdiction: userData.jurisdiction,
        practiceAreas: userData.practiceAreas,
        passwordHash: await bcrypt.hash(userData.password, 12),
      });

      // Check if user already exists
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, validatedUser.email))
        .limit(1);

      if (existingUser.length > 0) {
        return { user: existingUser[0], success: false, error: 'User already exists' };
      }

      // Create user with transaction
      const result = await userDb.transaction(async (tx) => {
        // Insert user
        const [newUser] = await tx.insert(users).values(validatedUser).returning();

        // Create profile if profile data provided
        let profile: UserProfile | undefined;
        if (userData.profileData) {
          const profileData = insertProfileSchema.parse({
            userId: newUser.id,
            ...userData.profileData,
          });
          [profile] = await tx.insert(userProfiles).values(profileData).returning();
        }

        // Log registration activity
        await tx.insert(userActivityLog).values({
          userId: newUser.id,
          action: 'user_registered',
          resource: 'user',
          resourceId: newUser.id.toString(),
          context: {
            registrationMethod: 'email',
            role: newUser.role,
            jurisdiction: newUser.jurisdiction,
          },
          success: true,
        });

        return { user: newUser, profile };
      });

      return { ...result, success: true };
    } catch (error: any) {
      console.error('User registration error:', error);
      return { 
        user: {} as User, 
        success: false, 
        error: error instanceof Error ? error.message : 'Registration failed' 
      };
    }
  }

  /**
   * Authenticate user login
   */
  static async authenticateUser(email: string, password: string, ipAddress?: string, userAgent?: string): Promise<{
    user?: User;
    profile?: UserProfile;
    session?: UserSession;
    success: boolean;
    error?: string;
  }> {
    try {
      // Find user with profile
      const userWithProfile = await db
        .select()
        .from(users)
        .leftJoin(userProfiles, eq(users.id, userProfiles.userId))
        .where(and(
          eq(users.email, email.toLowerCase()),
          eq(users.isActive, true),
          isNull(users.deletedAt)
        ))
        .limit(1);

      if (userWithProfile.length === 0) {
        return { success: false, error: 'Invalid credentials' };
      }

      const userData = userWithProfile[0];
      const user = userData.users;
      const profile = userData.user_profiles;

      // Verify password
      const passwordValid = await bcrypt.compare(password, user.passwordHash);
      if (!passwordValid) {
        // Log failed login attempt
        await userDb.insert(userActivityLog).values({
          userId: user.id,
          action: 'login_failed',
          resource: 'auth',
          context: { reason: 'invalid_password' },
          success: false,
          ipAddress,
          userAgent,
        });
        
        return { success: false, error: 'Invalid credentials' };
      }

      // Create session
      const sessionId = nanoid(32);
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours

      const [session] = await userDb.insert(userSessions).values({
        userId: user.id,
        sessionId,
        expiresAt,
        ipAddress,
        userAgent,
        sessionContext: {},
      }).returning();

      // Update last login time
      await db
        .update(users)
        .set({ lastLoginAt: new Date(), updatedAt: new Date() })
        .where(eq(users.id, user.id));

      // Log successful login
      await userDb.insert(userActivityLog).values({
        userId: user.id,
        action: 'login_success',
        resource: 'auth',
        context: { sessionId },
        success: true,
        ipAddress,
        userAgent,
      });

      return {
        user,
        profile: profile || undefined,
        session,
        success: true,
      };
    } catch (error: any) {
      console.error('Authentication error:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Authentication failed' 
      };
    }
  }

  /**
   * Validate session and get user data
   */
  static async validateSession(sessionId: string): Promise<{
    user?: User;
    profile?: UserProfile;
    session?: UserSession;
    valid: boolean;
  }> {
    try {
      const sessionData = await db
        .select()
        .from(userSessions)
        .innerJoin(users, eq(userSessions.userId, users.id))
        .leftJoin(userProfiles, eq(users.id, userProfiles.userId))
        .where(and(
          eq(userSessions.sessionId, sessionId),
          eq(userSessions.isActive, true),
          sql`${userSessions.expiresAt} > NOW()`
        ))
        .limit(1);

      if (sessionData.length === 0) {
        return { valid: false };
      }

      const data = sessionData[0];
      return {
        user: data.users,
        profile: data.user_profiles || undefined,
        session: data.user_sessions,
        valid: true,
      };
    } catch (error: any) {
      console.error('Session validation error:', error);
      return { valid: false };
    }
  }

  /**
   * Logout user by invalidating session
   */
  static async logoutUser(sessionId: string): Promise<{ success: boolean }> {
    try {
      await db
        .update(userSessions)
        .set({ isActive: false, updatedAt: new Date() })
        .where(eq(userSessions.sessionId, sessionId));

      return { success: true };
    } catch (error: any) {
      console.error('Logout error:', error);
      return { success: false };
    }
  }
}

// ============================================================================
// USER PROFILE OPERATIONS
// ============================================================================

export class UserProfileService {
  /**
   * Get complete user profile with all related data
   */
  static async getFullUserProfile(userId: number): Promise<FullUserProfile | null> {
    try {
      // Get user with profile
      const userData = await db
        .select()
        .from(users)
        .leftJoin(userProfiles, eq(users.id, userProfiles.userId))
        .where(and(
          eq(users.id, userId),
          eq(users.isActive, true),
          isNull(users.deletedAt)
        ))
        .limit(1);

      if (userData.length === 0) return null;

      const user = userData[0].users;
      const profile = userData[0].user_profiles;

      // Get active sessions
      const sessions = await db
        .select()
        .from(userSessions)
        .where(and(
          eq(userSessions.userId, userId),
          eq(userSessions.isActive, true),
          sql`${userSessions.expiresAt} > NOW()`
        ))
        .orderBy(desc(userSessions.createdAt));

      // Get recent activity
      const recentActivity = await db
        .select()
        .from(userActivityLog)
        .where(eq(userActivityLog.userId, userId))
        .orderBy(desc(userActivityLog.timestamp))
        .limit(20);

      return {
        ...user,
        profile: profile || undefined,
        sessions,
        recentActivity,
      };
    } catch (error: any) {
      console.error('Get full profile error:', error);
      return null;
    }
  }

  /**
   * Update user profile information
   */
  static async updateUserProfile(
    userId: number, 
    updates: Partial<NewUser & NewUserProfile>
  ): Promise<{ user?: User; profile?: UserProfile; success: boolean; error?: string }> {
    try {
      const result = await userDb.transaction(async (tx) => {
        let updatedUser: User | undefined;
        let updatedProfile: UserProfile | undefined;

        // Update user table fields
        const userFields = {
          firstName: updates.firstName,
          lastName: updates.lastName,
          jurisdiction: updates.jurisdiction,
          practiceAreas: updates.practiceAreas,
          barNumber: updates.barNumber,
          firmName: updates.firmName,
          updatedAt: new Date(),
        };

        // Filter out undefined values
        const userUpdates = Object.fromEntries(
          Object.entries(userFields).filter(([_, value]) => value !== undefined)
        );

        if (Object.keys(userUpdates).length > 0) {
          const validatedUpdates = updateUserSchema.parse(userUpdates);
          [updatedUser] = await tx
            .update(users)
            .set(validatedUpdates)
            .where(eq(users.id, userId))
            .returning();
        }

        // Update profile table fields
        const profileFields = {
          phoneNumber: updates.phoneNumber,
          address: updates.address,
          licenseNumber: updates.licenseNumber,
          yearsOfExperience: updates.yearsOfExperience,
          specializations: updates.specializations,
          education: updates.education,
          preferences: updates.preferences,
          avatarUrl: updates.avatarUrl,
          bio: updates.bio,
          updatedAt: new Date(),
        };

        const profileUpdates = Object.fromEntries(
          Object.entries(profileFields).filter(([_, value]) => value !== undefined)
        );

        if (Object.keys(profileUpdates).length > 0) {
          const validatedProfileUpdates = updateProfileSchema.parse(profileUpdates);
          
          // Check if profile exists
          const existingProfile = await tx
            .select()
            .from(userProfiles)
            .where(eq(userProfiles.userId, userId))
            .limit(1);

          if (existingProfile.length > 0) {
            // Update existing profile
            [updatedProfile] = await tx
              .update(userProfiles)
              .set(validatedProfileUpdates)
              .where(eq(userProfiles.userId, userId))
              .returning();
          } else {
            // Create new profile
            [updatedProfile] = await tx
              .insert(userProfiles)
              .values({ userId, ...validatedProfileUpdates })
              .returning();
          }
        }

        // Log update activity
        await tx.insert(userActivityLog).values({
          userId,
          action: 'profile_updated',
          resource: 'user_profile',
          resourceId: userId.toString(),
          context: {
            updatedFields: [...Object.keys(userUpdates), ...Object.keys(profileUpdates)],
          },
          success: true,
        });

        return { user: updatedUser, profile: updatedProfile };
      });

      return { ...result, success: true };
    } catch (error: any) {
      console.error('Update profile error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Profile update failed',
      };
    }
  }

  /**
   * Delete user account (soft delete)
   */
  static async deleteUser(userId: number): Promise<{ success: boolean; error?: string }> {
    try {
      await userDb.transaction(async (tx) => {
        // Soft delete user
        await tx
          .update(users)
          .set({ 
            isActive: false, 
            deletedAt: new Date(),
            updatedAt: new Date(),
          })
          .where(eq(users.id, userId));

        // Invalidate all sessions
        await tx
          .update(userSessions)
          .set({ isActive: false, updatedAt: new Date() })
          .where(eq(userSessions.userId, userId));

        // Log deletion activity
        await tx.insert(userActivityLog).values({
          userId,
          action: 'user_deleted',
          resource: 'user',
          resourceId: userId.toString(),
          context: { deletionType: 'soft_delete' },
          success: true,
        });
      });

      return { success: true };
    } catch (error: any) {
      console.error('Delete user error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'User deletion failed',
      };
    }
  }

  /**
   * Find similar users based on profile embedding (AI recommendations)
   */
  static async findSimilarUsers(userId: number, limit: number = 10): Promise<User[]> {
    try {
      const currentUser = await db
        .select({ embedding: users.profileEmbedding })
        .from(users)
        .where(eq(users.id, userId))
        .limit(1);

      if (currentUser.length === 0 || !currentUser[0].embedding) {
        return [];
      }

      const similarUsers = await db
        .select({
          user: users,
          similarity: sql<number>`1 - (${cosineDistance(users.profileEmbedding, currentUser[0].embedding)})`,
        })
        .from(users)
        .where(and(
          sql`${users.profileEmbedding} IS NOT NULL`,
          sql`${users.id} != ${userId}`,
          eq(users.isActive, true),
          isNull(users.deletedAt)
        ))
        .orderBy(sql`1 - (${cosineDistance(users.profileEmbedding, currentUser[0].embedding)}) DESC`)
        .limit(limit);

      return similarUsers.map(row => row.user);
    } catch (error: any) {
      console.error('Find similar users error:', error);
      return [];
    }
  }
}

// ============================================================================
// USER ACTIVITY TRACKING
// ============================================================================

export class UserActivityService {
  /**
   * Log user activity
   */
  static async logActivity(activity: NewUserActivity): Promise<void> {
    try {
      await userDb.insert(userActivityLog).values({
        ...activity,
        timestamp: new Date(),
      });
    } catch (error: any) {
      console.error('Log activity error:', error);
    }
  }

  /**
   * Get user activity history
   */
  static async getUserActivity(
    userId: number, 
    limit: number = 50, 
    offset: number = 0
  ): Promise<UserActivity[]> {
    try {
      return await db
        .select()
        .from(userActivityLog)
        .where(eq(userActivityLog.userId, userId))
        .orderBy(desc(userActivityLog.timestamp))
        .limit(limit)
        .offset(offset);
    } catch (error: any) {
      console.error('Get user activity error:', error);
      return [];
    }
  }

  /**
   * Get activity statistics for user
   */
  static async getActivityStats(userId: number, days: number = 30): Promise<{
    totalActions: number;
    uniqueActions: number;
    successRate: number;
    topActions: Array<{ action: string; count: number }>;
  }> {
    try {
      const dateThreshold = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

      const stats = await db
        .select({
          totalActions: count(),
          successRate: sql<number>`AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)`,
        })
        .from(userActivityLog)
        .where(and(
          eq(userActivityLog.userId, userId),
          sql`${userActivityLog.timestamp} >= ${dateThreshold}`
        ));

      const topActions = await db
        .select({
          action: userActivityLog.action,
          count: count(),
        })
        .from(userActivityLog)
        .where(and(
          eq(userActivityLog.userId, userId),
          sql`${userActivityLog.timestamp} >= ${dateThreshold}`
        ))
        .groupBy(userActivityLog.action)
        .orderBy(desc(count()))
        .limit(10);

      const uniqueActionsResult = await db
        .select({
          uniqueActions: sql<number>`COUNT(DISTINCT action)`,
        })
        .from(userActivityLog)
        .where(and(
          eq(userActivityLog.userId, userId),
          sql`${userActivityLog.timestamp} >= ${dateThreshold}`
        ));

      return {
        totalActions: stats[0]?.totalActions || 0,
        uniqueActions: uniqueActionsResult[0]?.uniqueActions || 0,
        successRate: stats[0]?.successRate || 0,
        topActions,
      };
    } catch (error: any) {
      console.error('Get activity stats error:', error);
      return {
        totalActions: 0,
        uniqueActions: 0,
        successRate: 0,
        topActions: [],
      };
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { userDb as db };
export default {
  UserAuthService,
  UserProfileService,
  UserActivityService,
};