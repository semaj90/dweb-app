import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { lucia } from '$lib/server/auth';
import { db } from '$lib/server/db/drizzle';
import { users, sessions } from '$lib/server/db/schema-postgres';
import { sql, desc } from 'drizzle-orm';

interface HealthWarning { code: string; message: string; }

export const GET: RequestHandler = async () => {
  const started = Date.now();
  const warnings: HealthWarning[] = [];
  let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

  const g: any = globalThis as any;
  const usersSameRef = g.__users_ref ? g.__users_ref === users : true;
  const sessionsSameRef = g.__sessions_ref ? g.__sessions_ref === sessions : true;
  const luciaInstanceReused = !!g.__lucia_instance && g.__lucia_instance === lucia;

  if (!usersSameRef || !sessionsSameRef) {
    status = 'degraded';
    warnings.push({ code: 'SCHEMA_IDENTITY_MISMATCH', message: 'Detected different users/sessions table object identity – potential duplicate import path.' });
  }
  if (!luciaInstanceReused) {
    status = 'degraded';
    warnings.push({ code: 'LUCIA_INSTANCE_NOT_REUSED', message: 'Lucia instance not stored globally (HMR duplication?)' });
  }

  let userCount: number | null = null;
  let sessionCount: number | null = null;
  let recentSessions: any[] = [];
  let countsError: string | null = null;

  try {
    const [{ value: uCount }] = await db
      .select({ value: sql<number>`count(*)` })
      .from(users);
    userCount = uCount;
    const [{ value: sCount }] = await db
      .select({ value: sql<number>`count(*)` })
      .from(sessions);
    sessionCount = sCount;
    recentSessions = await db
      .select({ id: sessions.id, user_id: sessions.user_id, created_at: sessions.created_at, expires_at: sessions.expires_at })
      .from(sessions)
      .orderBy(desc(sessions.created_at))
      .limit(5);
  } catch (e: any) {
    status = status === 'healthy' ? 'degraded' : status;
    countsError = e.message;
    warnings.push({ code: 'COUNT_QUERY_FAILED', message: e.message });
  }

  const durationMs = Date.now() - started;

  return json({
    status,
    timestamp: new Date().toISOString(),
    durationMs,
    adapter: {
      sessionCookieName: lucia.sessionCookieName,
      luciaInstanceReused,
    },
    schemaIdentity: {
      usersSameRef,
      sessionsSameRef,
    },
    counts: {
      userCount,
      sessionCount,
      recentSessions,
      countsError,
    },
    environment: {
      nodeVersion: process.version,
      pid: process.pid,
      uptime: process.uptime(),
      platform: process.platform,
    },
    warnings,
  }, {
    status: status === 'healthy' ? 200 : status === 'degraded' ? 206 : 503,
    headers: {
      'Cache-Control': 'no-cache',
      'X-Auth-Health': status,
    }
  });
};
