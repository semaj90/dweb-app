/**
 * User Login API Endpoint
 * POST /api/auth/login
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { ExistingUserAuthService as UserAuthService } from '$lib/server/db/existing-user-operations.js';
import { z } from 'zod';
import { dev } from '$app/environment';

// Login request validation schema
const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
  rememberMe: z.boolean().optional().default(false),
});

export const POST: RequestHandler = async ({ request, getClientAddress, cookies }) => {
  try {
    // Parse and validate request body
    const body = await request.json().catch(() => ({}));
    const validatedData = loginSchema.parse(body);

    // Get client information for logging
    const ipAddress = getClientAddress();
    const userAgent = request.headers.get('user-agent') || undefined;

    // Authenticate user
    const result = await UserAuthService.loginUser({
      email: validatedData.email,
      password: validatedData.password,
      ipAddress,
      userAgent,
      rememberMe: validatedData.rememberMe,
    });

    if (!result.success) {
      // Don't reveal whether email exists or not (security best practice)
      throw error(401, {
        message: 'Invalid email or password',
        code: 'AUTHENTICATION_FAILED'
      });
    }

    // Set session cookie
    const cookieOptions = {
      path: '/',
      httpOnly: true,
      secure: !dev, // Only secure in production
      sameSite: 'strict' as const,
      maxAge: validatedData.rememberMe ? 60 * 60 * 24 * 30 : 60 * 60 * 24, // 30 days or 1 day
    };

    cookies.set('session_id', result.session!.sessionId, cookieOptions);

    // Remove sensitive information from response
    const { passwordHash, ...userResponse } = result.user!;
    
    // Return successful login response
    return json({
      success: true,
      message: 'Login successful',
      data: {
        user: userResponse,
        session: {
          id: result.session!.sessionId,
          expiresAt: result.session!.expiresAt,
        },
        profile: result.profile || null,
      },
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        ...(dev && { 'Access-Control-Allow-Origin': '*' }),
      }
    });

  } catch (err: any) {
    console.error('Login API error:', err);

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

    // Handle authentication errors
    const statusCode = err.status || 500;
    const message = err.body?.message || err.message || 'Login failed';

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