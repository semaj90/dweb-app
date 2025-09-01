// EARLY PROCESS.CWD PATCH (must occur before other server imports with side-effects)
import type { Handle } from '@sveltejs/kit';
import nodeProcess from 'node:process';

if (globalThis.process && typeof globalThis.process.cwd !== 'function') {
  // Minimal, synchronous patch
  try {
    globalThis.process.cwd = nodeProcess.cwd;
    console.info('[startup] Patched missing process.cwd');
  } catch (e) {
    console.warn('[startup] Failed to patch process.cwd', e);
    (globalThis.process as any).cwd = () => '/';
  }
}

// Lightweight diagnostic (avoid logger usage before ensured patch)
try {
  console.info('[diagnostic] process.cwd type:', typeof process.cwd, 'node version:', (process as any).versions?.node);
} catch {/* ignore */ }

// Defer heavier imports until after patch to avoid early evaluation using broken process
let _lazy: {
  db?: any;
  sessions?: any;
  eq?: any;
  logger?: any;
  incrementMetric?: (k: string) => void;
} = {};

async function ensureDeps() {
  if (_lazy.db) return _lazy;
  const [dbMod, schemaMod, drizzleMod, loggerMod] = await Promise.all([
    import('$lib/database/connection'),
    import('$lib/database/schema'),
    import('drizzle-orm'),
    import('$lib/server/logger')
  ]);
  _lazy = {
    db: dbMod.db,
    sessions: schemaMod.sessions,
    eq: drizzleMod.eq,
    logger: loggerMod.logger,
    incrementMetric: loggerMod.incrementMetric
  };
  return _lazy;
}

export const handle: Handle = async ({ event, resolve }) => {
  // -------- Observability: request correlation + timing --------
  const requestStart = performance.now();
  const requestId = (globalThis.crypto?.randomUUID?.() || `req_${Date.now()}_${Math.random().toString(16).slice(2)}`);
  // expose to downstream route handlers / endpoints
  (event.locals as any).requestId = requestId;

  // Session lookup (lightweight); failures are non-fatal
  try {
    const { db, sessions, eq, incrementMetric, logger } = await ensureDeps();
    const sessionId = event.cookies.get('session');
    if (sessionId) {
      const [session] = await db.select().from(sessions).where(eq(sessions.id, sessionId));
      if (session && session.expiresAt > new Date()) {
        event.locals.user = { id: session.userId };
        incrementMetric?.('session_lookup_success');
      } else {
        incrementMetric?.('session_lookup_expired_or_missing');
      }
    }
  } catch (err) {
    try {
      const { logger, incrementMetric } = await ensureDeps();
      logger.error('session lookup failed', err);
      incrementMetric?.('session_lookup_failures');
    } catch {/* swallow secondary errors */ }
  }

  // --- Unified API Gateway Middleware ---
  // Handle routing to Go microservices and protocol switching
  if (event.url.pathname.startsWith('/api/')) {
    try {
      const { logger, incrementMetric } = await ensureDeps();
      logger.info('api.request', {
        requestId,
        method: event.request.method,
        path: event.url.pathname,
        userAgent: event.request.headers.get('user-agent'),
        ip: event.getClientAddress()
      });
      // (Cannot mutate event.request.headers; instead expose routing intent via locals if needed)
      if (event.url.pathname.startsWith('/api/ai/')) {
        (event.locals as any).serviceRoute = 'enhanced-rag';
      } else if (event.url.pathname.startsWith('/api/upload/')) {
        (event.locals as any).serviceRoute = 'upload-service';
      } else if (event.url.pathname.startsWith('/api/gpu/')) {
        (event.locals as any).serviceRoute = 'gpu-orchestrator';
      }
      incrementMetric?.('api_request_processed');
    } catch (err) {
      console.warn('API gateway middleware failed:', err);
    }
  }

  // Add security headers
  const response = await resolve(event);

  // Observability response headers
  const durationMs = performance.now() - requestStart;
  response.headers.set('X-Request-ID', requestId);
  response.headers.set('X-Response-Time', `${durationMs.toFixed(2)}ms`);
  response.headers.set('Server-Timing', `total;dur=${durationMs.toFixed(2)}`);

  // Security headers for all responses
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  // CORS headers for API routes
  if (event.url.pathname.startsWith('/api/')) {
    response.headers.set('Access-Control-Allow-Origin', 'http://localhost:5184');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Request-ID');
    response.headers.set('Access-Control-Allow-Credentials', 'true');
  }

  // (Optional) log completion with duration & status for API routes
  if (event.url.pathname.startsWith('/api/')) {
    try {
      const { logger, incrementMetric } = await ensureDeps();
      logger.info('api.response', { requestId, status: response.status, durationMs: +durationMs.toFixed(2) });
      incrementMetric?.('api_response_emitted');
    } catch {/* ignore logging failures */ }
  }

  return response;
};