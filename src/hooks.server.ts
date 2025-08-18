// SvelteKit Hooks for YoRHa Interface
import type { Handle } from '@sveltejs/kit';
import { AuthService } from '$lib/yorha/services/auth.service';
import { sequence } from '@sveltejs/kit/hooks';

// Authentication hook
const authentication: Handle = async ({ event, resolve }) => {
  const sessionToken = event.cookies.get('yorha_session');
  
  if (sessionToken) {
    const authService = new AuthService();
    const sessionData = await authService.validateSession(sessionToken);
    
    if (sessionData) {
      // Add user to locals
      event.locals.user = sessionData.unit;
      event.locals.session = sessionData.session;
    } else {
      // Invalid session, clear cookie
      event.cookies.delete('yorha_session', { path: '/' });
    }
  }
  
  return resolve(event);
};

// Route protection hook
const routeProtection: Handle = async ({ event, resolve }) => {
  const protectedRoutes = [
    '/profile',
    '/dashboard',
    '/missions',
    '/equipment',
    '/settings'
  ];
  
  const authRequiredApiRoutes = [
    '/api/user',
    '/api/missions',
    '/api/equipment',
    '/api/achievements'
  ];
  
  const isProtectedRoute = protectedRoutes.some(route => 
    event.url.pathname.startsWith(route)
  );
  
  const isProtectedApiRoute = authRequiredApiRoutes.some(route => 
    event.url.pathname.startsWith(route)
  );
  
  if ((isProtectedRoute || isProtectedApiRoute) && !event.locals.user) {
    if (isProtectedApiRoute) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Authentication required'
      }), {
        status: 401,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
    
    // Redirect to login for web routes
    return new Response(null, {
      status: 302,
      headers: {
        Location: '/login'
      }
    });
  }
  
  return resolve(event);
};

// Activity logging hook
const activityLogging: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);
  
  // Log API activity for authenticated users
  if (event.locals.user && event.url.pathname.startsWith('/api/')) {
    const { db } = await import('$lib/yorha/db');
    const { userActivity } = await import('$lib/yorha/db/schema');
    
    try {
      await db.insert(userActivity).values({
        userId: event.locals.user.id,
        activityType: 'system_sync',
        description: `API request: ${event.request.method} ${event.url.pathname}`,
        metadata: {
          method: event.request.method,
          path: event.url.pathname,
          status: response.status,
          userAgent: event.request.headers.get('user-agent'),
          sessionId: event.locals.session?.id
        },
        ipAddress: event.getClientAddress(),
        userAgent: event.request.headers.get('user-agent'),
        sessionId: event.locals.session?.id
      });
    } catch (error) {
      console.error('Activity logging error:', error);
    }
  }
  
  return response;
};

// Security headers hook
const securityHeaders: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);
  
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=()'
  );
  
  if (event.url.protocol === 'https:') {
    response.headers.set(
      'Strict-Transport-Security',
      'max-age=31536000; includeSubDomains; preload'
    );
  }
  
  return response;
};

// Export combined hooks
export const handle = sequence(
  authentication,
  routeProtection,
  activityLogging,
  securityHeaders
);