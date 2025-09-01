import type { RequestHandler } from './$types.js';
import { json } from '@sveltejs/kit';

/**
 * User Logout API Endpoint
 * POST /api/auth/logout
 * Simple session-based logout for demo purposes
 */

export const POST: RequestHandler = async ({ cookies }) => {
  try {
    // Get session tokens from cookies
    const sessionToken = cookies.get('session_token');
    const authToken = cookies.get('auth_token');
    
    // Clear all authentication cookies
    const cookieOptions = {
      path: '/',
      httpOnly: true,
      secure: import.meta.env.NODE_ENV === 'production',
      sameSite: 'strict' as const
    };
    
    if (sessionToken) {
      cookies.delete('session_token', cookieOptions);
    }
    if (authToken) {
      cookies.delete('auth_token', cookieOptions);
    }
    
    // Clear any other auth-related cookies
    cookies.delete('user_preferences', { path: '/' });
    cookies.delete('remember_me', { path: '/' });
    
    // Return successful logout response
    return json({
      success: true,
      message: 'Logout successful',
      data: null,
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    });

  } catch (error: any) {
    console.error('Logout API error:', error);

    // Even if there's an error, try to clear cookies for security
    const cookieOptions = {
      path: '/',
      httpOnly: true,
      secure: import.meta.env.NODE_ENV === 'production',
      sameSite: 'strict' as const
    };
    
    cookies.delete('session_token', cookieOptions);
    cookies.delete('auth_token', cookieOptions);

    return json({
      success: true, // Still return success as cookies are cleared
      message: 'Logout completed',
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
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