/**
 * User Registration API Endpoint
 * POST /api/auth/register
 */

import { json, error, type RequestHandler } from '@sveltejs/kit';
import { ExistingUserAuthService as UserAuthService } from '$lib/server/db/existing-user-operations.js';
import { z } from 'zod';
import { dev } from '$app/environment';

// Registration request validation schema
const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  firstName: z.string().min(1, 'First name is required').max(100),
  lastName: z.string().min(1, 'Last name is required').max(100),
  role: z.enum(['attorney', 'paralegal', 'investigator', 'user']).default('user'),
  jurisdiction: z.string().optional(),
  practiceAreas: z.array(z.string()).optional(),
  
  // Profile information
  profileData: z.object({
    phoneNumber: z.string().optional(),
    licenseNumber: z.string().optional(),
    yearsOfExperience: z.number().min(0).max(100).optional(),
    specializations: z.array(z.string()).optional(),
    firmName: z.string().optional(),
    bio: z.string().max(1000).optional(),
    preferences: z.object({
      theme: z.enum(['light', 'dark', 'auto']).default('light'),
      language: z.string().default('en'),
      timezone: z.string().default('UTC'),
      notifications: z.object({
        email: z.boolean().default(true),
        push: z.boolean().default(true),
        sms: z.boolean().default(false),
      }).default({}),
      aiAssistance: z.object({
        autoSummarize: z.boolean().default(true),
        suggestCitations: z.boolean().default(true),
        riskAnalysis: z.boolean().default(true),
      }).default({}),
    }).default({}),
  }).optional(),
});

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  try {
    // Parse and validate request body
    const body = await request.json().catch(() => ({}));
    const validatedData = registerSchema.parse(body);

    // Get client information for logging
    const ipAddress = getClientAddress();
    const userAgent = request.headers.get('user-agent') || undefined;

    // Register user with complete profile setup
    const result = await UserAuthService.registerUser({
      ...validatedData,
      profileData: validatedData.profileData,
    });

    if (!result.success) {
      throw error(400, {
        message: result.error || 'Registration failed',
        code: 'REGISTRATION_FAILED'
      });
    }

    // Remove sensitive information from response
    const { passwordHash, ...userResponse } = result.user;
    
    // Return successful registration response
    return json({
      success: true,
      message: 'User registered successfully',
      data: {
        user: userResponse,
        profile: result.profile,
        hasProfile: !!result.profile,
      },
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, {
      status: 201,
      headers: {
        'Content-Type': 'application/json',
        ...(dev && { 'Access-Control-Allow-Origin': '*' }),
      }
    });

  } catch (err: any) {
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
