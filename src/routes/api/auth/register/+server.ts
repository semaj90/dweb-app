// POST /api/auth/register
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { AuthService } from '$lib/yorha/services/auth.service';
import { z } from 'zod';

const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  name: z.string().min(2).max(100),
  unitType: z.enum(['combat', 'scanner', 'support', 'operator', 'healer']).optional()
});

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const body = await request.json();
    
    // Validate input
    const validated = registerSchema.parse(body);
    
    // Create auth service
    const authService = new AuthService();
    
    // Register user
    const { unit, session } = await authService.register(validated);
    
    // Set session cookie
    cookies.set('yorha_session', session.token, {
      path: '/',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });
    
    // Return user data (without sensitive info)
    return json({
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
          createdAt: unit.createdAt
        },
        sessionToken: session.token
      }
    });
  } catch (error: any) {
    console.error('Registration error:', error);
    
    if (error.name === 'ZodError') {
      return json({
        success: false,
        error: 'Validation failed',
        details: error.errors
      }, { status: 400 });
    }
    
    return json({
      success: false,
      error: error.message || 'Registration failed'
    }, { status: 500 });
  }
};