/**
 * Current User API Endpoint
 * GET /api/auth/me
 * Enhanced with PostgreSQL + pgvector + Cognitive Cache integration
 */

import type { RequestHandler } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { users, cases, evidence } from '$lib/server/db/schema-postgres';
import { eq, sql, count } from 'drizzle-orm';
import { cognitiveCacheManager } from '$lib/services/cognitive-cache-integration';
import { dev } from '$app/environment';

export const GET: RequestHandler = async ({ locals }) => {
  try {
    const session = locals.session;
    const user = locals.user;

    if (!session || !user) {
      return json({
        success: false,
        message: 'Not authenticated',
        data: { user: null, authenticated: false }
      }, { status: 401 });
    }

    const userId = user.id;

    // Check cognitive cache for user profile data
    const cacheKey = `current_user_${userId}`;
    const cacheRequest = {
      key: cacheKey,
      type: 'legal-data' as const,
      context: {
        userId,
        workflowStep: 'auth-check',
        documentType: 'user-profile',
        priority: 'high' as const,
        semanticTags: ['auth', 'current-user', 'profile']
      }
    };

    // Try cognitive cache first
    const cachedUserData = await cognitiveCacheManager.get(cacheRequest);
    if (cachedUserData && cachedUserData.confidence > 0.8) {
      return json({
        success: true,
        data: {
          ...cachedUserData.data,
          authenticated: true,
          loadSource: 'cache'
        }
      });
    }

    // Get comprehensive user data from database
    const [userProfile, userStats] = await Promise.all([
      db.select({
        id: users.id,
        email: users.email,
        username: users.username,
        firstName: users.first_name,
        lastName: users.last_name,
        role: users.role,
        department: users.department,
        jurisdiction: users.jurisdiction,
        practiceAreas: users.practice_areas,
        barNumber: users.bar_number,
        firmName: users.firm_name,
        avatarUrl: users.avatar_url,
        lastLoginAt: users.last_login_at,
        permissions: users.permissions,
        isActive: users.is_active,
        emailVerified: users.email_verified,
        metadata: users.metadata,
        createdAt: users.created_at,
        updatedAt: users.updated_at
      }).from(users).where(eq(users.id, userId)).limit(1),

      db.select({
        totalCases: count(cases.id),
        activeCases: sql<number>`COUNT(CASE WHEN ${cases.status} IN ('open', 'active') THEN 1 END)`,
        totalEvidence: sql<number>`(SELECT COUNT(*) FROM ${evidence} WHERE ${evidence.created_by} = ${userId})`
      }).from(cases).where(eq(cases.created_by, userId))
    ]);

    const profile = userProfile[0];
    const stats = userStats[0];

    if (!profile) {
      return json({ user: null }, { status: 404 });
    }

    const userData = {
      user: {
        id: profile.id,
        email: profile.email,
        username: profile.username,
        firstName: profile.firstName,
        lastName: profile.lastName,
        displayName: profile.firstName ? `${profile.firstName} ${profile.lastName}` : profile.username || profile.email.split('@')[0],
        role: profile.role,
        department: profile.department,
        jurisdiction: profile.jurisdiction,
        practiceAreas: profile.practiceAreas || [],
        barNumber: profile.barNumber,
        firmName: profile.firmName,
        avatarUrl: profile.avatarUrl,
        lastLoginAt: profile.lastLoginAt,
        permissions: profile.permissions || [],
        isActive: profile.isActive,
        emailVerified: profile.emailVerified,
        metadata: profile.metadata || {},
        createdAt: profile.createdAt,
        updatedAt: profile.updatedAt
      },
      activity: {
        totalCases: stats?.totalCases || 0,
        activeCases: stats?.activeCases || 0,
        totalEvidence: stats?.totalEvidence || 0
      },
      authenticated: true,
      loadSource: 'database'
    };

    // Cache the user data for future requests
    await cognitiveCacheManager.set(cacheRequest, userData, {
      distributeAcrossCaches: true,
      cognitiveValue: 0.9
    });

    return json({
      success: true,
      data: userData
    });

  } catch (error: any) {
    console.error("Auth me error:", error);
    return json({ user: null }, { status: 401 });
  }
};
