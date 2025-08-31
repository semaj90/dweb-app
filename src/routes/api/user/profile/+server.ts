// GET /api/user/profile - Get user profile
// PUT /api/user/profile - Update user profile
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { db } from '$lib/yorha/db';
import { units, userActivity, userAchievements, userEquipment } from '$lib/yorha/db/schema';
import { eq, desc } from 'drizzle-orm';
import { AuthService } from '$lib/yorha/services/auth.service';
import type { Unit } from '$lib/yorha/db/schema';
import { VectorService } from '$lib/yorha/services/vector.service';
import { z } from 'zod';

// Helper to map unit safely (avoid TS errors for properties not present on Unit type)
const mapUnit = (u: any) => ({
  id: u.id,
  unitId: u.unitId,
  email: u.email,
  name: u.name,
  unitType: u.unitType,
  level: u.level ?? null,
  xp: u.xp ?? null,
  rank: u.rank ?? null,
  bio: u.bio ?? null,
  avatarUrl: u.avatarUrl ?? null,
  missionsCompleted: u.missionsCompleted ?? 0,
  combatRating: u.combatRating ?? null,
  hoursActive: u.hoursActive ?? null,
  achievementsUnlocked: u.achievementsUnlocked ?? null,
  emailVerified: u.emailVerified ?? false,
  twoFactorEnabled: u.twoFactorEnabled ?? false,
  settings: u.settings ?? {},
  createdAt: u.createdAt ?? null,
  updatedAt: u.updatedAt ?? null,
  lastLoginAt: u.lastLoginAt ?? null
});

// GET user profile
export const GET: RequestHandler = async ({ cookies, url }): Promise<any> => {
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

    // sessionData.unit is sourced from the DB schema; ensure correct typing
    const unit = sessionData.unit as Unit;

    // Get additional profile data if requested
    const includeActivity = url.searchParams.get('includeActivity') === 'true';
    const includeAchievements = url.searchParams.get('includeAchievements') === 'true';
    const includeEquipment = url.searchParams.get('includeEquipment') === 'true';

    const response: any = {
      success: true,
      data: {
        unit: mapUnit(unit),
        // Embedding / vector metadata (supports async embedding pipeline)
        embedding: {
          status: (unit as any).embeddingStatus ?? 'unknown',
          lastUpdatedAt: (unit as any).embeddingUpdatedAt ?? null,
          asyncGeneration: true
        }
      }
    };

    // Include recent activity
    if (includeActivity) {
      const activities = await db.query.userActivity.findMany({
        where: eq(userActivity.userId, unit.id),
        orderBy: (activity) => [desc(activity.createdAt)],
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
  } catch (error: any) {
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
export const PUT: RequestHandler = async ({ request, cookies }): Promise<any> => {
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
    const updateData: any = {
      updatedAt: new Date()
    };

    if (validated.name) updateData.name = validated.name;
    if (validated.bio) updateData.bio = validated.bio;
    if (validated.avatarUrl) updateData.avatarUrl = validated.avatarUrl;
    if (validated.settings) {
      // Merge with existing settings (ensure object)
      const currentUnit = await db.query.units.findFirst({
        where: eq(units.id, sessionData.unit.id)
      });

      const currentSettings = (currentUnit?.settings ?? {}) as Record<string, any>;

      updateData.settings = {
        ...currentSettings,
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
        unit: mapUnit(updatedUnit)
      }
    });
  } catch (error: any) {
    console.error('Update profile error:', error);

    if (error instanceof z.ZodError) {
      return json({
        success: false,
        error: 'Validation failed',
        details: error.issues
      }, { status: 400 });
    }

    return json({
      success: false,
      error: 'Failed to update profile'
    }, { status: 500 });
  }
};
