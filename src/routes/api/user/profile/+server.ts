// GET /api/user/profile - Get user profile
// PUT /api/user/profile - Update user profile
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { db } from '$lib/yorha/db';
import { units, userActivity, userAchievements, userEquipment } from '$lib/yorha/db/schema';
import { eq } from 'drizzle-orm';
import { AuthService } from '$lib/yorha/services/auth.service';
import { VectorService } from '$lib/yorha/services/vector.service';
import { z } from 'zod';

// GET user profile
export const GET: RequestHandler = async ({ cookies, url }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    
    if (!sessionToken) {
      return json({
        success: false,
        error: 'Authentication required'
      }, { status: 401 });
    }
    
    const authService = new AuthService();
    const sessionData = await authService.validateSession(sessionToken);
    
    if (!sessionData) {
      return json({
        success: false,
        error: 'Invalid session'
      }, { status: 401 });
    }
    
    const { unit } = sessionData;
    
    // Get additional profile data if requested
    const includeActivity = url.searchParams.get('includeActivity') === 'true';
    const includeAchievements = url.searchParams.get('includeAchievements') === 'true';
    const includeEquipment = url.searchParams.get('includeEquipment') === 'true';
    
    const response: unknown = {
      success: true,
      data: {
        unit: {
          id: unit.id,
          unitId: unit.unitId,
          email: unit.email,
          name: unit.name,
          unitType: unit.unitType,
          level: unit.level,
          xp: unit.xp,
          rank: unit.rank,
          bio: unit.bio,
          avatarUrl: unit.avatarUrl,
          missionsCompleted: unit.missionsCompleted,
          combatRating: unit.combatRating,
          hoursActive: unit.hoursActive,
          achievementsUnlocked: unit.achievementsUnlocked,
          emailVerified: unit.emailVerified,
          twoFactorEnabled: unit.twoFactorEnabled,
          settings: unit.settings,
          createdAt: unit.createdAt,
          lastLoginAt: unit.lastLoginAt
        }
      }
    };
    
    // Include recent activity
    if (includeActivity) {
      const activities = await db.query.userActivity.findMany({
        where: eq(userActivity.userId, unit.id),
        orderBy: (activity, { desc }) => [desc(activity.createdAt)],
        limit: 20
      });
      response.data.activities = activities;
    }
    
    // Include achievements
    if (includeAchievements) {
      const achievements = await db.query.userAchievements.findMany({
        where: eq(userAchievements.userId, unit.id),
        with: {
          achievement: true
        }
      });
      response.data.achievements = achievements;
    }
    
    // Include equipment
    if (includeEquipment) {
      const equipment = await db.query.userEquipment.findMany({
        where: eq(userEquipment.userId, unit.id),
        with: {
          equipment: true
        }
      });
      response.data.equipment = equipment;
    }
    
    return json(response);
  } catch (error: unknown) {
    console.error('Get profile error:', error);
    return json({
      success: false,
      error: 'Failed to fetch profile'
    }, { status: 500 });
  }
};

// Update profile schema
const updateProfileSchema = z.object({
  name: z.string().min(2).max(100).optional(),
  bio: z.string().max(500).optional(),
  avatarUrl: z.string().url().optional(),
  settings: z.object({
    notifications: z.boolean().optional(),
    profileVisibility: z.enum(['public', 'squad', 'private']).optional(),
    showActivityStatus: z.boolean().optional(),
    dataCollection: z.boolean().optional(),
    theme: z.string().optional()
  }).optional()
});

// PUT update profile
export const PUT: RequestHandler = async ({ request, cookies }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    
    if (!sessionToken) {
      return json({
        success: false,
        error: 'Authentication required'
      }, { status: 401 });
    }
    
    const authService = new AuthService();
    const sessionData = await authService.validateSession(sessionToken);
    
    if (!sessionData) {
      return json({
        success: false,
        error: 'Invalid session'
      }, { status: 401 });
    }
    
    const body = await request.json();
    const validated = updateProfileSchema.parse(body);
    
    // Update user profile
    const updateData: unknown = {
      updatedAt: new Date()
    };
    
    if (validated.name) updateData.name = validated.name;
    if (validated.bio) updateData.bio = validated.bio;
    if (validated.avatarUrl) updateData.avatarUrl = validated.avatarUrl;
    if (validated.settings) {
      // Merge with existing settings
      const currentUnit = await db.query.units.findFirst({
        where: eq(units.id, sessionData.unit.id)
      });
      
      updateData.settings = {
        ...currentUnit?.settings,
        ...validated.settings
      };
    }
    
    const [updatedUnit] = await db.update(units)
      .set(updateData)
      .where(eq(units.id, sessionData.unit.id))
      .returning();
    
    // Update vector embedding if profile changed significantly
    if (validated.name || validated.bio) {
      const vectorService = new VectorService();
      await vectorService.generateUserEmbedding(sessionData.unit.id);
    }
    
    // Log activity
    await db.insert(userActivity).values({
      userId: sessionData.unit.id,
      activityType: 'profile_update',
      description: 'Profile information updated',
      metadata: { 
        fields: Object.keys(validated),
        sessionId: sessionData.session.id
      }
    });
    
    return json({
      success: true,
      data: {
        unit: {
          id: updatedUnit.id,
          unitId: updatedUnit.unitId,
          email: updatedUnit.email,
          name: updatedUnit.name,
          unitType: updatedUnit.unitType,
          level: updatedUnit.level,
          xp: updatedUnit.xp,
          rank: updatedUnit.rank,
          bio: updatedUnit.bio,
          avatarUrl: updatedUnit.avatarUrl,
          missionsCompleted: updatedUnit.missionsCompleted,
          combatRating: updatedUnit.combatRating,
          hoursActive: updatedUnit.hoursActive,
          achievementsUnlocked: updatedUnit.achievementsUnlocked,
          settings: updatedUnit.settings,
          updatedAt: updatedUnit.updatedAt
        }
      }
    });
  } catch (error: unknown) {
    console.error('Update profile error:', error);
    
    if (error.name === 'ZodError') {
      return json({
        success: false,
        error: 'Validation failed',
        details: error.errors
      }, { status: 400 });
    }
    
    return json({
      success: false,
      error: 'Failed to update profile'
    }, { status: 500 });
  }
};