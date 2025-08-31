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
    sessionData?: any;
  };
}

// Login request validation schema with multi-protocol support
const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
  rememberMe: z.boolean().optional().default(false),
  protocol: z.enum(['rest', 'grpc', 'quic']).optional().default('rest'),
  enableAutoTagging: z.boolean().optional().default(true),
});

// Multi-protocol authentication functions
async function attemptQuicAuth(email: string, password: string): Promise<{ success: boolean; data?: any; protocol: string }> {
  try {
    const response = await fetch('http://localhost:8230/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      signal: AbortSignal.timeout(2000) // 2 second timeout for QUIC
    });
    
    if (response.ok) {
      const data = await response.json();
      return { success: true, data, protocol: 'quic' };
    }
  } catch (error) {
    console.log('⚠️ QUIC authentication failed, falling back to gRPC');
  }
  
  return { success: false, protocol: 'quic' };
}

async function attemptGrpcAuth(email: string, password: string): Promise<{ success: boolean; data?: any; protocol: string }> {
  try {
    const response = await fetch('http://localhost:50051/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      signal: AbortSignal.timeout(5000) // 5 second timeout for gRPC
    });
    
    if (response.ok) {
      const data = await response.json();
      return { success: true, data, protocol: 'grpc' };
    }
  } catch (error) {
    console.log('⚠️ gRPC authentication failed, using native PostgreSQL');
  }
  
  return { success: false, protocol: 'grpc' };
}

// Auto-tagging worker trigger function
async function triggerAutoTagging(context: AutoTaggingContext): Promise<void> {
  try {
    // Ensure Redis connection
    await redis.connect();

    const eventData = {
      id: `login-${context.userId}-${Date.now()}`,
      type: 'user_login',
      action: 'tag',
      userId: context.userId,
      metadata: JSON.stringify(context.metadata),
      timestamp: Date.now().toString()
    };

    // Send to Redis stream for worker processing
    const result = await redis.xAdd('autotag:requests', '*', eventData);
    if (result) {
      console.log(`🏷️ Auto-tagging triggered for user login: ${context.userId} (${result})`);
    } else {
      console.warn(`⚠️ Auto-tagging may have failed for user login: ${context.userId}`);
    }
  } catch (error) {
    console.error('❌ Auto-tagging trigger failed:', error);
    // Don't fail login if auto-tagging fails
  }
}

export const POST: RequestHandler = async ({ request, getClientAddress, cookies }) => {
  const startTime = Date.now();
  let protocol = 'rest';
  let processingTime = 0;
  
  try {
    // Parse and validate request body
    const body = await request.json().catch(() => ({}));
    const validatedData = loginSchema.parse(body);

    // Get client information for logging
    const ipAddress = getClientAddress();
    const userAgent = request.headers.get('user-agent') || undefined;

    // Multi-protocol authentication with intelligent fallback
    let authResult: { success: boolean; data?: any; protocol: string } = { success: false, protocol: 'rest' };
    let pgResult: any = null;

    // Try requested protocol first, then fallback hierarchy: QUIC -> gRPC -> REST(PostgreSQL)
    const preferredProtocol = validatedData.protocol || 'rest';
    
    switch (preferredProtocol) {
      case 'quic':
        protocol = 'quic';
        try {
          authResult = await attemptQuicAuth(validatedData.email, validatedData.password);
          if (authResult.success) {
            processingTime = Date.now() - startTime;
            console.log(`⚡ QUIC authentication successful (${processingTime}ms)`);
            break;
          }
        } catch (error) {
          console.log('⚠️ QUIC unavailable, falling back to gRPC');
        }
        // Fallthrough to gRPC
        
      case 'grpc':
        protocol = 'grpc';
        try {
          authResult = await attemptGrpcAuth(validatedData.email, validatedData.password);
          if (authResult.success) {
            processingTime = Date.now() - startTime;
            console.log(`🚀 gRPC authentication successful (${processingTime}ms)`);
            break;
          }
        } catch (error) {
          console.log('⚠️ gRPC unavailable, using PostgreSQL');
        }
        // Fallthrough to PostgreSQL
        
      case 'rest':
      default:
        protocol = 'rest';
        // Native PostgreSQL authentication (most reliable)
        pgResult = await UserAuthService.loginUser({
          email: validatedData.email,
          password: validatedData.password,
          ipAddress,
          userAgent,
          rememberMe: validatedData.rememberMe,
        });
        
        processingTime = Date.now() - startTime;
        console.log(`🗃️ PostgreSQL authentication completed (${processingTime}ms)`);
        
        if (pgResult.success) {
          authResult = { success: true, data: pgResult, protocol: 'rest' };
        }
        break;
    }

    // If all protocols failed, use the PostgreSQL result for error handling
    const result = authResult.success ? authResult.data : pgResult;

    if (!authResult.success || !result?.success) {
      // Don't reveal whether email exists or not (security best practice)
      throw error(401, ensureError({
        message: 'Invalid email or password',
        code: 'AUTHENTICATION_FAILED'
      }));
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

    // Trigger auto-tagging if enabled
    if (validatedData.enableAutoTagging && result.user?.id) {
      const autoTagContext: AutoTaggingContext = {
        userId: result.user.id,
        action: 'login',
        metadata: {
          ipAddress,
          userAgent,
          timestamp: new Date().toISOString(),
          protocol: protocol as 'rest' | 'grpc' | 'quic',
          sessionData: {
            sessionId: result.session?.sessionId,
            rememberMe: validatedData.rememberMe,
            processingTime,
          },
        },
      };

      // Trigger auto-tagging asynchronously
      triggerAutoTagging(autoTagContext).catch(console.error);
    }

    // Remove sensitive information from response
    const { passwordHash, ...userResponse } = result.user!;
    
    // Return enhanced login response with protocol and performance info
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
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Auth-Protocol': protocol,
        'X-Processing-Time': `${processingTime}ms`,
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