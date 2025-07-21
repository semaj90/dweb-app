import { redirect, type Handle } from '@sveltejs/kit';
import { db } from '$lib/server/db'; // Your Drizzle DB instance
import { users, sessions } from '$lib/server/db/schema-postgres'; // Your Drizzle schemas
import { eq } from 'drizzle-orm';

export const handle: Handle = async ({ event, resolve }) => {
  const sessionId = event.cookies.get('session_id');

  if (sessionId) {
    // 1. Validate session against Drizzle DB
    const [sessionRecord] = await db.select().from(sessions).where(eq(sessions.id, sessionId));

    if (sessionRecord && sessionRecord.expiresAt > new Date()) {
      // 2. Fetch user data
      const [userRecord] = await db.select().from(users).where(eq(users.id, sessionRecord.userId));

      if (userRecord) {
        // 3. Populate locals
        event.locals.user = {
          id: userRecord.id,
          email: userRecord.email,
          name: userRecord.name || userRecord.email,
          role: userRecord.role,
          firstName: userRecord.firstName,
          lastName: userRecord.lastName,
          avatarUrl: userRecord.avatarUrl,
          emailVerified: userRecord.emailVerified,
          createdAt: userRecord.createdAt,
          updatedAt: userRecord.updatedAt,
          isActive: userRecord.isActive,
        };
        event.locals.session = {
          id: sessionRecord.id,
          userId: sessionRecord.userId,
          expiresAt: sessionRecord.expiresAt,
        };
      }
    } else {
      // Session expired or invalid, clear cookie
      event.cookies.delete('session_id', { path: '/' });
    }
  }

  // Protect routes (example)
  if (event.url.pathname.startsWith('/admin') && !event.locals.user?.role.includes('admin')) {
    throw redirect(303, '/login');
  }

  return resolve(event);
};