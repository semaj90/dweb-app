/**
 * User Logout API Endpoint
 * POST /api/auth/logout
 */

import { json, type RequestHandler } from '@sveltejs/kit';
import { ExistingUserAuthService as UserAuthService } from '$lib/server/db/existing-user-operations.js';
import { dev } from '$app/environment';

export const POST: RequestHandler = async ({ cookies, getClientAddress }) => {
  try {
    // Get session ID from cookie
    const sessionId = cookies.get('session_id');
    
    if (sessionId) {
      // Get client information for logging
      const ipAddress = getClientAddress();
      
      // Logout user (invalidate session)
      const result = await UserAuthService.logoutUser(sessionId, ipAddress);
      
      if (!result.success) {
        console.warn('Session logout failed:', result.error);
        // Continue with cookie deletion even if session invalidation failed
      }
    }
    
    // Clear session cookie
    cookies.delete('session_id', {
      path: '/',
      httpOnly: true,
      secure: !dev,
      sameSite: 'strict',
    });
    
    // Return successful logout response
    return json({
      success: true,
      message: 'Logout successful',
      data: null,
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
    console.error('Logout API error:', err);

    // Even if there's an error, clear the cookie for security
    cookies.delete('session_id', {
      path: '/',
      httpOnly: true,
      secure: !dev,
      sameSite: 'strict',
    });

    return json({
      success: false,
      message: 'Logout completed with warnings',
      code: 'LOGOUT_WARNING',
      meta: {
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      }
    }, { 
      status: 200, // Still return success as cookie is cleared
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