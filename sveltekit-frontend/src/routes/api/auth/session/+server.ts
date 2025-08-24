// Enhanced Session API Endpoint
// Handles session validation, renewal, and management with Redis integration

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { sessionManager } from '$lib/auth/session-manager';
import { lucia } from '$lib/auth/session';
import { db } from '$lib/server/db/pg';
import { users } from '$lib/server/db/schema-postgres';
import { eq } from 'drizzle-orm';

// Initialize session manager
let sessionManagerInitialized = false;

async function ensureSessionManager() {
  if (!sessionManagerInitialized) {
    try {
      await sessionManager.initialize();
      sessionManagerInitialized = true;
    } catch (error) {
      console.error('Failed to initialize session manager:', error);
      throw error;
    }
  }
}

export const GET: RequestHandler = async ({ cookies, request, getClientAddress }) => {
  try {
    await ensureSessionManager();

    // Get session ID from cookie
    const sessionId = cookies.get(lucia.sessionCookieName);
    
    if (!sessionId) {
      return json({ 
        user: null, 
        session: null, 
        error: 'No session cookie found' 
      }, { status: 401 });
    }

    // Validate session with Lucia
    const { session: luciaSession, user: luciaUser } = await lucia.validateSession(sessionId);
    
    if (!luciaSession || !luciaUser) {
      // Clear invalid session cookie
      cookies.set(lucia.sessionCookieName, '', {
        expires: new Date(0),
        path: '/',
        httpOnly: true,
        secure: !process.env.NODE_ENV || process.env.NODE_ENV === 'production',
        sameSite: 'lax'
      });

      return json({ 
        user: null, 
        session: null, 
        error: 'Invalid session' 
      }, { status: 401 });
    }

    // Get additional user data from database
    const [userWithProfile] = await db
      .select()
      .from(users)
      .where(eq(users.id, luciaUser.id))
      .limit(1);

    if (!userWithProfile) {
      return json({ 
        user: null, 
        session: null, 
        error: 'User not found' 
      }, { status: 401 });
    }

    // Check if user is active
    if (!userWithProfile.isActive) {
      await lucia.invalidateSession(sessionId);
      cookies.set(lucia.sessionCookieName, '', {
        expires: new Date(0),
        path: '/',
        httpOnly: true,
        secure: !process.env.NODE_ENV || process.env.NODE_ENV === 'production',
        sameSite: 'lax'
      });

      return json({ 
        user: null, 
        session: null, 
        error: 'Account is inactive' 
      }, { status: 401 });
    }

    // Get session from Redis for additional metadata
    const redisSession = await sessionManager.getSession(sessionId);
    
    // Update session activity in Redis
    if (redisSession) {
      const userAgent = request.headers.get('user-agent');
      const ipAddress = getClientAddress();
      
      await sessionManager.updateSessionActivity(sessionId, {
        lastUserAgent: userAgent,
        lastIpAddress: ipAddress,
        lastRequestTime: new Date().toISOString(),
      });
    }

    // Prepare user object for response
    const responseUser = {
      id: userWithProfile.id,
      email: userWithProfile.email,
      firstName: userWithProfile.firstName,
      lastName: userWithProfile.lastName,
      name: userWithProfile.name,
      role: userWithProfile.role,
      isActive: userWithProfile.isActive,
      avatarUrl: userWithProfile.avatarUrl,
      emailVerified: userWithProfile.emailVerified,
      createdAt: userWithProfile.createdAt,
      updatedAt: userWithProfile.updatedAt,
    };

    // Prepare session object for response
    const responseSession = {
      id: luciaSession.id,
      userId: luciaSession.userId,
      expiresAt: luciaSession.expiresAt,
      fresh: luciaSession.fresh,
      // Add Redis session metadata if available
      ...(redisSession && {
        createdAt: redisSession.createdAt,
        lastActivity: redisSession.lastActivity,
        metadata: redisSession.metadata,
      }),
    };

    return json({
      user: responseUser,
      session: responseSession,
      success: true
    });

  } catch (error) {
    console.error('Session validation error:', error);
    return json({ 
      user: null, 
      session: null, 
      error: 'Session validation failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ cookies, request }) => {
  try {
    await ensureSessionManager();

    const { action, sessionId } = await request.json();

    switch (action) {
      case 'refresh': {
        if (!sessionId) {
          return json({ error: 'Session ID required for refresh' }, { status: 400 });
        }

        // Validate and refresh session
        const { session: luciaSession, user: luciaUser } = await lucia.validateSession(sessionId);
        
        if (!luciaSession || !luciaUser) {
          return json({ error: 'Invalid session' }, { status: 401 });
        }

        // Update activity in Redis
        await sessionManager.updateSessionActivity(sessionId, {
          refreshedAt: new Date().toISOString(),
        });

        return json({ 
          success: true, 
          message: 'Session refreshed',
          expiresAt: luciaSession.expiresAt 
        });
      }

      case 'activity': {
        if (!sessionId) {
          return json({ error: 'Session ID required for activity update' }, { status: 400 });
        }

        // Update activity timestamp
        const updated = await sessionManager.updateSessionActivity(sessionId, {
          activityPing: new Date().toISOString(),
        });

        if (!updated) {
          return json({ error: 'Session not found' }, { status: 404 });
        }

        return json({ success: true, message: 'Activity updated' });
      }

      case 'metadata': {
        const { metadata } = await request.json();
        
        if (!sessionId) {
          return json({ error: 'Session ID required for metadata update' }, { status: 400 });
        }

        if (!metadata || typeof metadata !== 'object') {
          return json({ error: 'Valid metadata object required' }, { status: 400 });
        }

        // Update session metadata
        const updated = await sessionManager.updateSessionActivity(sessionId, metadata);

        if (!updated) {
          return json({ error: 'Session not found' }, { status: 404 });
        }

        return json({ success: true, message: 'Metadata updated' });
      }

      default:
        return json({ error: 'Invalid action' }, { status: 400 });
    }

  } catch (error) {
    console.error('Session management error:', error);
    return json({ 
      error: 'Session management failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ cookies, request }) => {
  try {
    await ensureSessionManager();

    const sessionId = cookies.get(lucia.sessionCookieName);
    
    if (!sessionId) {
      return json({ error: 'No session to logout' }, { status: 400 });
    }

    // Invalidate session in Lucia
    await lucia.invalidateSession(sessionId);

    // Remove session from Redis
    await sessionManager.destroySession(sessionId);

    // Clear session cookie
    cookies.set(lucia.sessionCookieName, '', {
      expires: new Date(0),
      path: '/',
      httpOnly: true,
      secure: !process.env.NODE_ENV || process.env.NODE_ENV === 'production',
      sameSite: 'lax'
    });

    return json({ 
      success: true, 
      message: 'Session terminated successfully' 
    });

  } catch (error) {
    console.error('Session termination error:', error);
    return json({ 
      error: 'Session termination failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};