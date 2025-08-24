import { relayAuthService } from '$lib/server/services/relay-auth-service';
import type { RequestHandler } from './$types';

/**
 * Auto-login endpoint for demo user
 * POST /auth/login/auto
 * Uses relay authentication service to avoid direct database timeouts
 */
export const POST: RequestHandler = async ({ cookies, getClientAddress, request }) => {
  const clientIP = getClientAddress();
  const userAgent = request.headers.get('user-agent') || '';

  try {
    // Use relay authentication service instead of direct database calls
    console.log('🔄 Using RelayAuthService for demo authentication...');
    
    const authResult = await relayAuthService.authenticateDemoUser();
    
    if (!authResult) {
      return new Response(JSON.stringify({ 
        error: 'Demo authentication failed through relay service.' 
      }), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const { user, session } = authResult;

    // Check if account is active
    if (!user.is_active) {
      return new Response(JSON.stringify({ 
        error: 'Demo account is disabled.' 
      }), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    console.log('✅ Demo user authenticated via relay service:', user.email);

    // Create session cookie manually to avoid Lucia's database connection
    const cookieName = 'auth-session';
    const cookieValue = session.id;
    const isSecure = false; // dev mode
    
    cookies.set(cookieName, cookieValue, {
      path: '/',
      httpOnly: true,
      secure: isSecure,
      sameSite: 'strict',
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });

    console.log('✅ Session cookie set manually:', cookieName, '=', cookieValue);

    console.log('✅ Demo user auto-login successful:', user.email);

    // Return success response instead of redirect for API endpoint
    return new Response(JSON.stringify({ 
      success: true, 
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role
      },
      redirectTo: '/dashboard'
    }), { 
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Demo auto-login error:', error);
    return new Response(JSON.stringify({ 
      error: 'Auto-login failed. Please try manual login.' 
    }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};