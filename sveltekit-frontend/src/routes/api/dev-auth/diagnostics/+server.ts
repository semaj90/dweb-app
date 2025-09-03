import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
  try {
    const { db } = await import('$lib/server/db/drizzle');
    const { users, sessions } = await import('$lib/server/db/schema-postgres');

    // Probe a minimal user select and session join manually to surface any join SQL issues
    let userCount = 0; let sessionCount = 0; let sample: any = null; let joinOk = true; let joinError: string | undefined;
    try {
      const usersResult = await db.select({ id: users.id }).from(users).limit(1);
      userCount = usersResult.length;
      sample = usersResult[0] || null;
    } catch (e: any) { joinOk = false; joinError = 'users select failed: ' + e.message; }

    try {
      const sessionsResult = await db.select({ id: sessions.id }).from(sessions).limit(1);
      sessionCount = sessionsResult.length;
    } catch (e: any) { joinOk = false; joinError = (joinError || '') + ' | sessions select failed: ' + e.message; }

    return json({
      ok: true,
      users: { count: userCount, sample },
      sessions: { count: sessionCount },
      joinOk,
      joinError
    });
  } catch (e: any) {
    return json({ ok: false, error: e.message, stack: e.stack }, { status: 500 });
  }
};
