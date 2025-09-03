// CRITICAL PROCESS.CWD RESTORATION (must be first)
import type { Handle } from '@sveltejs/kit';
import nodeProcess from 'node:process';
// Auth import (lucia instance). We attempt to import but keep it optional in case build order issues.
// Attempt to import lucia auth instance; in some builds the barrel may differ.
// @ts-ignore - dynamic export shape
import * as authModule from '$lib/server/auth';

// Comprehensive process.cwd restoration with breadcrumb logging
function restoreProcessCwd() {
  const originalCwd = nodeProcess.cwd();
  let patchApplied = false;

  // Check and restore global process.cwd
  if (typeof process.cwd !== 'function') {
    const currentValue = (process as any).cwd;
    console.warn(`[PATCH] process.cwd was mutated to:`, typeof currentValue, currentValue);
    process.cwd = () => originalCwd;
    patchApplied = true;
  }

  // Check and restore globalThis.process.cwd
  if (globalThis.process && typeof globalThis.process.cwd !== 'function') {
    const currentValue = (globalThis.process as any).cwd;
    console.warn(`[PATCH] globalThis.process.cwd was mutated to:`, typeof currentValue, currentValue);
    globalThis.process.cwd = () => originalCwd;
    patchApplied = true;
  }

  if (patchApplied) {
    console.info(`[PATCH] process.cwd restored to function returning: "${originalCwd}"`);
  }

  // Final verification
  try {
    const testResult = process.cwd();
    console.info(`[VERIFY] process.cwd() test: "${testResult}" (type: ${typeof testResult})`);
  } catch (e) {
    console.error(`[VERIFY] process.cwd() test failed:`, e);
    // Failsafe fallback
    process.cwd = () => '/';
    console.warn(`[FAILSAFE] Set process.cwd to return root directory`);
  }
}

// Apply the restoration immediately
restoreProcessCwd();

// Defer heavier imports until after patch to avoid early evaluation using broken process
interface LazyDeps {
  db: any;
  sessions: any;
  users: any;
  eq: (a: any, b: any) => any;
  logger: { info: Function; error: Function; warn: Function };
  incrementMetric: (k: string) => void;
}

let _lazy: Partial<LazyDeps> = {};

async function ensureDeps(): Promise<LazyDeps> {
  if (_lazy.db) return _lazy as LazyDeps;
  const [dbMod, schemaMod, drizzleMod, loggerMod] = await Promise.all([
    import('$lib/server/db/drizzle'),
    import('$lib/server/db/schema-postgres'),
    import('drizzle-orm'),
    import('$lib/server/logger')
  ]);
  _lazy = {
    db: dbMod.db,
    sessions: (schemaMod as any).sessions,
    users: (schemaMod as any).users,
    eq: (drizzleMod as any).eq,
    logger: (loggerMod as any).logger,
    incrementMetric: (loggerMod as any).incrementMetric
  };
  return _lazy as LazyDeps;
}

export const handle: Handle = async ({ event, resolve }) => {
  // ---- Request metadata ----
  const start = performance.now();
  const requestId = globalThis.crypto?.randomUUID?.() ?? `req_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  (event.locals as any).requestId = requestId;

  const devAuthAuto = process.env.DEV_AUTH_AUTO === 'true' || (import.meta as any).env?.DEV_AUTH_AUTO === 'true';
  const devBypass = process.env.DEV_BYPASS_AUTH === 'true' || (import.meta as any).env?.DEV_BYPASS_AUTH === 'true';

  // ---- Step 1: Attempt Lucia session validation ----
  try {
    // Access lucia instance (exported const lucia ...). If missing, skip.
    // @ts-ignore
    const lucia: any = (authModule as any).lucia;
    if (!lucia) throw new Error('lucia instance not found');
    const { incrementMetric } = await ensureDeps();
    const luciaCookie = lucia.sessionCookieName;
    const existing = event.cookies.get(luciaCookie);
    if (existing) {
      const result = await lucia.validateSession(existing);
    const { session, user } = result || { session: null, user: null };
      if (session?.fresh) {
        const fresh = lucia.createSessionCookie(session.id);
        event.cookies.set(fresh.name, fresh.value, { path: '/', ...fresh.attributes });
      } else if (!session) {
        const blank = lucia.createBlankSessionCookie();
        event.cookies.set(blank.name, blank.value, { path: '/', ...blank.attributes });
      }
      (event.locals as any).user = user;
      (event.locals as any).session = session;
      incrementMetric?.(user ? 'session_lookup_success' : 'session_lookup_missing');
    } else {
      (event.locals as any).user = null;
      (event.locals as any).session = null;
    }
  } catch (err) {
    const { logger, incrementMetric } = await ensureDeps();
    logger.error('lucia.validation.failed', err);
    incrementMetric?.('session_lookup_fail');
  }

  // ---- Step 2: Dev auto / bypass (API routes only) ----
  if (!(event.locals as any).user && event.url.pathname.startsWith('/api/')) {
    const { db, users, sessions, eq, logger, incrementMetric } = await ensureDeps();
    // @ts-ignore access lucia
    const lucia: any = (authModule as any).lucia;
    if (devAuthAuto) {
      try {
        // find or create dev user
        let devUser = await db.query.users.findFirst({ where: eq(users.email, 'dev@example.com') });
        if (!devUser) {
          devUser = (await db.insert(users).values({
            email: 'dev@example.com',
            hashed_password: 'dev-hash',
            role: 'admin',
            is_active: true
          }).returning())[0];
        }
        if (lucia) {
          const session = await lucia.createSession(devUser.id, {});
          const sessionCookie = lucia.createSessionCookie(session.id);
          event.cookies.set(sessionCookie.name, sessionCookie.value, { path: '/', ...sessionCookie.attributes });
          (event.locals as any).user = { id: devUser.id };
          (event.locals as any).session = session;
          try {
            await db.insert(sessions).values({ id: session.id, user_id: devUser.id, expires_at: session.expiresAt })
              .onConflictDoNothing();
          } catch {/* ignore duplicate */ }
        } else {
          const sid = globalThis.crypto?.randomUUID?.() ?? `dev_${Date.now()}`;
          const expiresAt = new Date(Date.now() + 1000 * 60 * 60 * 8);
          await db.insert(sessions).values({ id: sid, user_id: devUser.id, expires_at: expiresAt })
            .onConflictDoUpdate({ target: sessions.id, set: { expires_at: expiresAt } });
          event.cookies.set('session', sid, { path: '/', httpOnly: true, sameSite: 'lax', maxAge: 60 * 60 * 8 });
          (event.locals as any).user = { id: devUser.id };
        }
        incrementMetric?.('session_dev_autocreated');
        logger.info('dev.session.autocreated', { userId: devUser.id });
      } catch (e) {
        const { logger } = await ensureDeps();
        logger.warn('dev.auto_session.failed', { error: (e as Error).message });
      }
    } else if (devBypass) {
      (event.locals as any).user = { id: 'ephemeral-dev-user' };
    }
  }

  // ---- Step 3: API request logging & routing hints ----
  if (event.url.pathname.startsWith('/api/')) {
    try {
      const { logger, incrementMetric } = await ensureDeps();
      logger.info('api.request', {
        requestId,
        method: event.request.method,
        path: event.url.pathname,
        ip: event.getClientAddress(),
        userAgent: event.request.headers.get('user-agent')
      });
      const route = event.url.pathname;
      if (route.startsWith('/api/ai/')) (event.locals as any).serviceRoute = 'enhanced-rag';
      else if (route.startsWith('/api/upload/')) (event.locals as any).serviceRoute = 'upload-service';
      else if (route.startsWith('/api/gpu/')) (event.locals as any).serviceRoute = 'gpu-orchestrator';
      incrementMetric?.('api_request_logged');
    } catch {/* noop */ }
  }

  const response = await resolve(event);

  // ---- Step 4: Observability & security headers ----
  const dur = performance.now() - start;
  response.headers.set('X-Request-ID', requestId);
  response.headers.set('X-Response-Time', `${dur.toFixed(2)}ms`);
  response.headers.set('Server-Timing', `total;dur=${dur.toFixed(2)}`);
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  if (event.url.pathname.startsWith('/api/')) {
    response.headers.set('Access-Control-Allow-Origin', 'http://localhost:5184');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Request-ID');
    response.headers.set('Access-Control-Allow-Credentials', 'true');
  }

  // ---- Step 5: API response logging ----
  if (event.url.pathname.startsWith('/api/')) {
    try {
      const { logger, incrementMetric } = await ensureDeps();
      logger.info('api.response', { requestId, status: response.status, durationMs: +dur.toFixed(2) });
      incrementMetric?.('api_response_logged');
    } catch {/* noop */ }
  }

  return response;
};