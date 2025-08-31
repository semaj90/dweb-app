import type { RequestHandler } from './$types';
import { json, error } from '@sveltejs/kit';
import { ensureError } from '$lib/utils/ensure-error';
import { ExistingUserAuthService as UserAuthService } from '$lib/server/db/existing-user-operations.js';
import { z } from 'zod';
import { dev } from '$app/environment';

// Multi-protocol support
import { redis } from '$lib/server/cache/redis-service';

// Auto-tagging integration  
interface AutoTaggingContext {
  userId: string;
  action: 'login' | 'register';
  metadata: {
    ipAddress?: string;
    userAgent?: string;
    timestamp: string;
    protocol: 'rest' | 'grpc' | 'quic';
    registrationData?: any;
  };
}

// Registration request validation schema with multi-protocol support
const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  firstName: z.string().min(1, 'First name is required').max(100),
  lastName: z.string().min(1, 'Last name is required').max(100),
  role: z.enum(['attorney', 'paralegal', 'investigator', 'user']).default('user'),
  jurisdiction: z.string().optional(),
  practiceAreas: z.array(z.string()).optional(),
  protocol: z.enum(['rest', 'grpc', 'quic']).optional().default('rest'),
  enableAutoTagging: z.boolean().optional().default(true),
  
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

// Multi-protocol registration functions
async function attemptQuicRegister(userData: any): Promise<{ success: boolean; data?: any; protocol: string }> {
  try {
    const response = await fetch('http://localhost:8230/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData),
      signal: AbortSignal.timeout(3000) // 3 second timeout for QUIC
    });
    
    if (response.ok) {
      const data = await response.json();
      return { success: true, data, protocol: 'quic' };
    }
  } catch (error) {
    console.log('⚠️ QUIC registration failed, falling back to gRPC');
  }
  
  return { success: false, protocol: 'quic' };
}

async function attemptGrpcRegister(userData: any): Promise<{ success: boolean; data?: any; protocol: string }> {
  try {
    const response = await fetch('http://localhost:50051/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData),
      signal: AbortSignal.timeout(8000) // 8 second timeout for gRPC
    });
    
    if (response.ok) {
      const data = await response.json();
      return { success: true, data, protocol: 'grpc' };
    }
  } catch (error) {
    console.log('⚠️ gRPC registration failed, using native PostgreSQL');
  }
  
  return { success: false, protocol: 'grpc' };
}

// Auto-tagging worker trigger function for registration
async function triggerRegistrationAutoTagging(context: AutoTaggingContext): Promise<void> {
  try {
    // Ensure Redis connection
    await redis.connect();

    const eventData = {
      id: `register-${context.userId}-${Date.now()}`,
      type: 'user_registration',
      action: 'tag',
      userId: context.userId,
      metadata: JSON.stringify(context.metadata),
      timestamp: Date.now().toString()
    };

    // Send to Redis stream for worker processing
    const result = await redis.xAdd('autotag:requests', '*', eventData);
    if (result) {
      console.log(`🏷️ Auto-tagging triggered for user registration: ${context.userId} (${result})`);
    } else {
      console.warn(`⚠️ Auto-tagging may have failed for user registration: ${context.userId}`);
    }
  } catch (error) {
    console.error('❌ Registration auto-tagging trigger failed:', error);
    // Don't fail registration if auto-tagging fails
  }
}

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  const startTime = Date.now();
  let protocol = 'rest';
  let processingTime = 0;
  
  try {
    // Parse and validate request body
    const body = await request.json().catch(() => ({}));
    const validatedData = registerSchema.parse(body);

    // Get client information for logging
    const ipAddress = getClientAddress();
    const userAgent = request.headers.get('user-agent') || undefined;

    // Multi-protocol registration with intelligent fallback
    let regResult: { success: boolean; data?: any; protocol: string } = { success: false, protocol: 'rest' };
    let pgResult: any = null;

    // Try requested protocol first, then fallback hierarchy: QUIC -> gRPC -> REST(PostgreSQL)
    const preferredProtocol = validatedData.protocol || 'rest';
    
    switch (preferredProtocol) {
      case 'quic':
        protocol = 'quic';
        try {
          regResult = await attemptQuicRegister(validatedData);
          if (regResult.success) {
            processingTime = Date.now() - startTime;
            console.log(`⚡ QUIC registration successful (${processingTime}ms)`);
            break;
          }
        } catch (error) {
          console.log('⚠️ QUIC unavailable, falling back to gRPC');
        }
        // Fallthrough to gRPC
        
      case 'grpc':
        protocol = 'grpc';
        try {
          regResult = await attemptGrpcRegister(validatedData);
          if (regResult.success) {
            processingTime = Date.now() - startTime;
            console.log(`🚀 gRPC registration successful (${processingTime}ms)`);
            break;
          }
        } catch (error) {
          console.log('⚠️ gRPC unavailable, using PostgreSQL');
        }
        // Fallthrough to PostgreSQL
        
      case 'rest':
      default:
        protocol = 'rest';
        // Native PostgreSQL registration (most reliable)
        pgResult = await UserAuthService.registerUser({
          ...validatedData,
          profileData: validatedData.profileData,
        });
        
        processingTime = Date.now() - startTime;
        console.log(`🗃️ PostgreSQL registration completed (${processingTime}ms)`);
        
        if (pgResult.success) {
          regResult = { success: true, data: pgResult, protocol: 'rest' };
        }
        break;
    }

    // If all protocols failed, use the PostgreSQL result for error handling
    const result = regResult.success ? regResult.data : pgResult;

    if (!regResult.success || !result?.success) {
      throw error(400, ensureError({
        message: result?.error || 'Registration failed',
        code: 'REGISTRATION_FAILED'
      }));
    }

    // Trigger auto-tagging if enabled
    if (validatedData.enableAutoTagging && result.user?.id) {
      const autoTagContext: AutoTaggingContext = {
        userId: result.user.id,
        action: 'register',
        metadata: {
          ipAddress,
          userAgent,
          timestamp: new Date().toISOString(),
          protocol: protocol as 'rest' | 'grpc' | 'quic',
          registrationData: {
            role: validatedData.role,
            jurisdiction: validatedData.jurisdiction,
            practiceAreas: validatedData.practiceAreas,
            processingTime,
            hasProfile: !!validatedData.profileData,
          },
        },
      };

      // Trigger auto-tagging asynchronously
      triggerRegistrationAutoTagging(autoTagContext).catch(console.error);
    }

    // Remove sensitive information from response
    const { passwordHash, ...userResponse } = result.user;
    
    // Return enhanced registration response with protocol and performance info
    return json({
      success: true,
      message: 'User registered successfully',
      data: {
        user: userResponse,
        profile: result.profile,
        hasProfile: !!result.profile,
      },
      protocol: {
        used: protocol,
        processingTime: `${processingTime}ms`,
        autoTagging: validatedData.enableAutoTagging,
      },
      meta: {
        timestamp: new Date().toISOString(),
        version: '2.0.0', // Updated for multi-protocol support
      }
    }, {
      status: 201,
      headers: {
        'Content-Type': 'application/json',
        'X-Auth-Protocol': protocol,
        'X-Processing-Time': `${processingTime}ms`,
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
