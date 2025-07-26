import { redirect, type Handle } from '@sveltejs/kit';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import Database from 'better-sqlite3';
import { eq } from 'drizzle-orm';
import * as schema from '$lib/server/db/schema-postgres';

// Initialize database with error handling
let db: any;

try {
  const dbPath = process.env.DATABASE_URL || './dev.db';
  const sqlite = new Database(dbPath);
  db = drizzle(sqlite, { schema });
  console.log('✅ Database connected successfully');
} catch (error) {
  console.error('❌ Database connection failed:', error);
  // Fallback to in-memory database
  const sqlite = new Database(':memory:');
  db = drizzle(sqlite, { schema });
  console.log('⚠️ Using in-memory database as fallback');
}

export { db };

export const handle: Handle = async ({ event, resolve }) => {
  // Initialize locals
  event.locals.user = null;
  event.locals.session = null;

  const sessionId = event.cookies.get('session_id');

  if (sessionId && db) {
    try {
      // Simple session validation (adjust based on your schema)
      const sessionQuery = `SELECT * FROM sessions WHERE id = ? AND expires_at > datetime('now')`;
      const sessionRecord = db.prepare(sessionQuery).get(sessionId);

      if (sessionRecord) {
        // Fetch user data
        const userQuery = `SELECT * FROM users WHERE id = ?`;
        const userRecord = db.prepare(userQuery).get(sessionRecord.user_id);

        if (userRecord) {
          event.locals.user = {
            id: userRecord.id,
            email: userRecord.email,
            name: userRecord.first_name + ' ' + userRecord.last_name,
            firstName: userRecord.first_name,
            lastName: userRecord.last_name,
            role: userRecord.role || 'user',
            isActive: userRecord.is_active !== 0,
            emailVerified: userRecord.email_verified !== 0 ? new Date(userRecord.email_verified) : null,
            createdAt: new Date(userRecord.created_at),
            updatedAt: new Date(userRecord.updated_at)
          };
          
          event.locals.session = {
            id: sessionRecord.id,
            userId: sessionRecord.user_id,
            expiresAt: sessionRecord.expires_at,
          };
        }
      } else {
        // Session expired or invalid, clear cookie
        event.cookies.delete('session_id', { path: '/' });
      }
    } catch (error) {
      console.error('Session validation error:', error);
      // Clear invalid session
      event.cookies.delete('session_id', { path: '/' });
    }
  }

  // Basic route protection
  const protectedRoutes = ['/admin', '/dashboard', '/cases', '/evidence'];
  const isProtectedRoute = protectedRoutes.some(route => event.url.pathname.startsWith(route));
  
  if (isProtectedRoute && !event.locals.user) {
    // Allow access to login/register pages
    if (!event.url.pathname.startsWith('/login') && !event.url.pathname.startsWith('/register')) {
      throw redirect(303, '/login');
    }
  }

  // Admin route protection
  if (event.url.pathname.startsWith('/admin') && event.locals.user?.role !== 'admin') {
    throw redirect(303, '/dashboard');
  }

  return resolve(event);
};
