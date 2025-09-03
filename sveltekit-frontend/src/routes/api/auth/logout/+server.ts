import { json } from '@sveltejs/kit';
import { dev } from '$app/environment';
import { lucia } from '$lib/server/auth';
import type { RequestEvent } from '@sveltejs/kit';

/**
 * PostgreSQL + Drizzle + Lucia Logout Endpoint
 * Properly invalidates sessions in the database
 */

export const POST = async ({ cookies, locals }: RequestEvent) => {
  // Check if user has an active session
  if (!locals.session) {
    return json({ 
      success: false, 
      message: 'No active session to logout' 
    }, { status: 400 });
  }

  try {
    // Invalidate the session in PostgreSQL database via Lucia
    await lucia.invalidateSession(locals.session.id);
    
    // Create and set blank session cookie
    const sessionCookie = lucia.createBlankSessionCookie();
    cookies.set(sessionCookie.name, sessionCookie.value, {
      path: '.',
      ...sessionCookie.attributes
    });

    // Log successful logout
    console.log('User logged out successfully:', {
      sessionId: locals.session.id,
      userId: locals.user?.id,
      timestamp: new Date().toISOString()
    });

    return json({ 
      success: true, 
      message: 'Successfully logged out',
      sessionInvalidated: true 
    });

  } catch (error) {
    console.error('Logout error:', error);
    
    // Even if database logout fails, clear the cookie
    const sessionCookie = lucia.createBlankSessionCookie();
    cookies.set(sessionCookie.name, sessionCookie.value, {
      path: '.',
      ...sessionCookie.attributes
    });

    return json({ 
      success: true, 
      message: 'Logged out (with errors)',
      error: 'Session cleanup encountered issues'
    });
  }
};

export const GET = async (ctx: any) => {
  if (!dev) return json({ error: 'GET not allowed in production' }, { status: 405 });
  return POST(ctx as any);
};

export const OPTIONS = async () => new Response(null, {
  status: 200,
  headers: {
    'Access-Control-Allow-Origin': dev ? '*' : 'https://yourdomain.com',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400'
  }
});