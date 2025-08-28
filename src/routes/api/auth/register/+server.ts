/**
 * User Registration API Endpoint
 * POST /api/auth/register
 * Integrates PostgreSQL + pgvector + Drizzle + Cognitive Cache
 */

import type { RequestHandler } from './$types';
import { json, error } from '@sveltejs/kit';
import { ExistingUserAuthService as UserAuthService } from '$lib/server/db/existing-user-operations.js';
import { cognitiveCacheManager } from '$lib/services/cognitive-cache-integration';
import { z } from 'zod';
import { dev } from '$app/environment';

// Enhanced registration schema for legal professionals
const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 
           'Password must contain uppercase, lowercase, number, and special character'),
  firstName: z.string().min(2, 'First name must be at least 2 characters').max(100),
  lastName: z.string().min(2, 'Last name must be at least 2 characters').max(100),
  username: z.string().min(3, 'Username must be at least 3 characters').max(50).optional(),
  role: z.enum(['attorney', 'paralegal', 'investigator', 'admin', 'clerk']).default('attorney'),
  department: z.string().max(100).optional(),
  jurisdiction: z.string().max(100).optional(),
  practiceAreas: z.array(z.string()).default([]),
  barNumber: z.string().max(50).optional(),
  firmName: z.string().max(200).optional(),
  acceptTerms: z.boolean().refine(val => val === true, 'Must accept terms and conditions'),
  marketingConsent: z.boolean().default(false)
});

export const POST: RequestHandler = async ({ request, getClientAddress, cookies }) => {
  try {
    // Parse and validate request body
    const body = await request.json().catch(() => ({}));
    const validatedData = registerSchema.parse(body);

    // Get client information
    const ipAddress = getClientAddress();
    const userAgent = request.headers.get('user-agent') || undefined;

    // Check cognitive cache for registration attempts (prevent spam)
    const rateLimitKey = `registration_attempt_${ipAddress}`;
    const rateLimitCache = await cognitiveCacheManager.get({
      key: rateLimitKey,
      type: 'legal-data',
      context: { ipAddress, action: 'registration' }
    });

    if (rateLimitCache && rateLimitCache.data?.attempts >= 5) {
      throw error(429, {
        message: 'Too many registration attempts. Please try again later.',
        code: 'RATE_LIMIT_EXCEEDED'
      });
    }

    // Prepare user registration data
    const registrationData = {
      email: validatedData.email,
      password: validatedData.password,
      firstName: validatedData.firstName,
      lastName: validatedData.lastName,
      username: validatedData.username || validatedData.email.split('@')[0],
      role: validatedData.role,
      department: validatedData.department,
      jurisdiction: validatedData.jurisdiction,
      practiceAreas: validatedData.practiceAreas,
      barNumber: validatedData.barNumber,
      firmName: validatedData.firmName,
      ipAddress,
      userAgent,
      marketingConsent: validatedData.marketingConsent
    };

    // Register user with comprehensive database integration
    const result = await UserAuthService.registerUser(registrationData);

    if (!result.success) {
      // Update rate limit cache
      await cognitiveCacheManager.set({
        key: rateLimitKey,
        type: 'legal-data',
        context: { ipAddress, action: 'registration_failed' }
      }, {
        attempts: (rateLimitCache?.data?.attempts || 0) + 1,
        lastAttempt: Date.now()
      });

      throw error(400, {
        message: result.error || 'Registration failed',
        code: 'REGISTRATION_FAILED'
      });
    }

    // Set session cookie with enhanced security
    const cookieOptions = {
      path: '/',
      httpOnly: true,
      secure: !dev, // Only secure in production
      sameSite: 'strict' as const,
      maxAge: 60 * 60 * 24 * 7, // 7 days
    };

    cookies.set('session_id', result.session!.sessionId, cookieOptions);

    // Cache successful registration (clear rate limits)
    await cognitiveCacheManager.set({
      key: `new_user_${result.user!.id}`,
      type: 'legal-data',
      context: { 
        userId: result.user!.id, 
        action: 'registration_success',
        role: result.user!.role 
      }
    }, {
      registrationDate: Date.now(),
      ipAddress,
      userAgent,
      practiceAreas: validatedData.practiceAreas
    });

    // Remove sensitive information from response
    const { passwordHash, ...userResponse } = result.user!;

    // Return successful registration response with comprehensive data
    return json({
      success: true,
      message: 'Registration successful. Welcome to the legal AI platform!',
      data: {
        user: {
          ...userResponse,
          displayName: `${userResponse.firstName} ${userResponse.lastName}`,
          initials: `${userResponse.firstName[0]}${userResponse.lastName[0]}`.toUpperCase(),
          profileComplete: !!(userResponse.barNumber && userResponse.jurisdiction)
        },
        session: {
          id: result.session!.sessionId,
          expiresAt: result.session!.expiresAt,
        },
        onboarding: {
          profileComplete: !!(validatedData.barNumber && validatedData.jurisdiction),
          nextSteps: [
            !validatedData.barNumber && 'Add bar number',
            !validatedData.jurisdiction && 'Set jurisdiction',
            !validatedData.practiceAreas.length && 'Select practice areas',
            'Complete profile setup'
          ].filter(Boolean)
        }
      },
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        source: 'comprehensive-auth-system'
      }
    }, {
      status: 201,
      headers: {
        'Content-Type': 'application/json',
        'Location': `/api/users/${result.user!.id}`,
        ...(dev && { 'Access-Control-Allow-Origin': '*' }),
      }
    });

  } catch (err) {
    console.error('Registration API error:', err);

    // Handle validation errors
    if (err instanceof z.ZodError) {
      return json({
        success: false,
        message: 'Validation failed',
        errors: err.errors.map(e => ({
          field: e.path.join('.'),
          message: e.message,
          code: e.code,
        })),
        meta: {
          timestamp: new Date().toISOString(),
          version: '1.0.0',
        }
      }, { 
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Handle other errors
    const statusCode = err.status || 500;
    const message = err.body?.message || err.message || 'Registration failed';

    return json({
      success: false,
      message,
      code: err.body?.code || 'INTERNAL_SERVER_ERROR',
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, { 
      status: statusCode,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

// OPTIONS handler for CORS preflight requests
export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': dev ? '*' : 'https://yourdomain.com',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400', // 24 hours
    }
  });
};