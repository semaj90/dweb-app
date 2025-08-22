// POST /api/auth/login
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { AuthService } from '$lib/yorha/services/auth.service';
import { z } from 'zod';

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string()
});

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const body = await request.json();
    
    // Validate input
    const validated = loginSchema.parse(body);
    
    // Create auth service
    const authService = new AuthService();
    
    // Login user
    const { unit, session } = await authService.login(validated.email, validated.password);
    
    // Set session cookie
    cookies.set('yorha_session', session.token, {
      path: '/',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });
    
    // Return user data
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
          lastLoginAt: unit.lastLoginAt
        },
        sessionToken: session.token
      }
    });
  } catch (error: unknown) {
    console.error('Login error:', error);
    
    if (error.name === 'ZodError') {
      return json({
        success: false,
        error: 'Invalid email or password format'
      }, { status: 400 });
    }
    
    if (error.message === 'Invalid credentials') {
      return json({
        success: false,
        error: 'Invalid email or password'
      }, { status: 401 });
    }
    
    return json({
      success: false,
      error: 'Login failed'
    }, { status: 500 });
  }
};