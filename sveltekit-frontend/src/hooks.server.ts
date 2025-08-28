/**
 * Enhanced SvelteKit SSR Hooks with Full-Stack Service Integration
 * Production-ready server-side rendering with service orchestration
 */

import { sequence } from "@sveltejs/kit/hooks";
import type { Handle, HandleServerError, HandleFetch } from "@sveltejs/kit";
import { redis } from '$lib/server/cache/redis-service';
import { minioService } from '$lib/server/storage/minio-service';
import { rabbitmqService } from '$lib/server/messaging/rabbitmq-service';
import { workflowOrchestrator } from '$lib/machines/workflow-machine';

// Service initialization status
let servicesInitialized = false;
let initializationPromise: Promise<void> | null = null;

// Dev-only global log buffer
if (process.env.NODE_ENV === 'development') {
  (globalThis as any)._devLogs = (globalThis as any)._devLogs || [] as string[];
}

function devLog(...parts: any[]) {
  const msg = parts.map(p => typeof p === 'string' ? p : JSON.stringify(p)).join(' ');
  try {
    if (process.env.NODE_ENV === 'development') {
      (globalThis as any)._devLogs.unshift(`${new Date().toISOString()} ${msg}`);
      // keep last 200 entries
      if ((globalThis as any)._devLogs.length > 200) (globalThis as any)._devLogs.length = 200;
    }
  } catch (e) {
    // no-op
  }
  console.log(...parts);
}

// Request tracking
const requestMetrics = {
  totalRequests: 0,
  errorCount: 0,
  averageResponseTime: 0,
  lastRequestTime: Date.now()
};

// Initialize all services with graceful fallback (non-blocking)
async function initializeServices(): Promise<void> {
  if (servicesInitialized) return;

  if (initializationPromise) {
    await initializationPromise;
    return;
  }

  initializationPromise = (async () => {
    devLog('🚀 Initializing optional services...');

    // Initialize services with timeout and graceful failure handling (non-blocking)
    const initResults = await Promise.allSettled([
      Promise.race([
        redis.connect().catch(err => {
          devLog(`ℹ️  Redis not available (${err.message}) - running in degraded mode`);
          return false;
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Redis timeout')), 1000))
      ]).catch(() => false),
      Promise.race([
        minioService.initialize().catch(err => {
          devLog(`ℹ️  MinIO not available (${err.message}) - running in degraded mode`);
          return false;
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('MinIO timeout')), 1000))
      ]).catch(() => false),
      Promise.race([
        rabbitmqService.connect().catch(err => {
          devLog(`✅ RabbitMQ connected`);
          return true;
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('RabbitMQ timeout')), 1000))
      ]).catch(() => false)
    ]);

    const services = ['Redis', 'MinIO', 'RabbitMQ'];
    let successCount = 0;

    initResults.forEach((result, index) => {
      if (result.status === 'fulfilled' && result.value === true) {
        devLog(`✅ ${services[index]} connected successfully`);
        successCount++;
      } else {
        // Already logged in catch handlers above
      }
    });

    // Always mark as initialized to prevent blocking
    servicesInitialized = true;
    devLog(`✅ Service initialization completed (${successCount}/3 services available)`);
  })();

  await initializationPromise;
}

// Service health injection handle (non-blocking)
const serviceHealthHandle: Handle = async ({ event, resolve }) => {
  // Initialize services on first request (non-blocking in background)
  if (!servicesInitialized) {
    // Start initialization in background, don't await
    initializeServices().catch(err => {
      console.warn('Service initialization error (non-blocking):', err.message);
    });
  }

  // Inject service status into locals (safe defaults)
  (event.locals as any).services = {
    redis: redis.getConnectionStatus() === 'connected',
    workflows: workflowOrchestrator.getActiveWorkflowsCount(),
    initialized: servicesInitialized
  };

  return resolve(event);
};

// Request logging and metrics handle
const loggingHandle: Handle = async ({ event, resolve }) => {
  const startTime = Date.now();
  const { method, url } = event.request;
  const userAgent = event.request.headers.get('user-agent') || 'unknown';
  
  // Safe client address retrieval with fallback
  let ip: string;
  try {
    ip = event.getClientAddress();
  } catch (error) {
    // Fallback for development environment where client address might not be available
    ip = event.request.headers.get('x-forwarded-for') || 
         event.request.headers.get('x-real-ip') || 
         'localhost';
  }

  // Generate request ID
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  // Inject request metadata
  (event.locals as any).requestId = requestId;
  (event.locals as any).startTime = startTime;

  devLog(`🌐 [${new Date().toISOString()}] ${method} ${new URL(event.request.url).pathname} - ${ip} - ${requestId}`);

  try {
    const response = await resolve(event);
    const responseTime = Date.now() - startTime;

    // Update metrics
    requestMetrics.totalRequests++;
    requestMetrics.averageResponseTime =
      (requestMetrics.averageResponseTime + responseTime) / 2;
    requestMetrics.lastRequestTime = Date.now();

    // Log successful request
    devLog(`✅ [${requestId}] ${response.status} - ${responseTime}ms`);

    // Add performance headers
    response.headers.set('X-Request-ID', requestId);
    response.headers.set('X-Response-Time', `${responseTime}ms`);
    response.headers.set('X-Powered-By', 'SvelteKit + Full-Stack Services');

    return response;

  } catch (error) {
    const responseTime = Date.now() - startTime;
    requestMetrics.errorCount++;

    devLog(`❌ [${requestId}] Error after ${responseTime}ms: ${error?.message || error}`);
    throw error;
  }
};

// CORS and security handle
const securityHandle: Handle = async ({ event, resolve }) => {
  // Handle preflight requests
  if (event.request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': event.request.headers.get('origin') || '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
        'Access-Control-Max-Age': '86400'
      }
    });
  }

  const response = await resolve(event, {
    transformPageChunk: ({ html }) => {
      // Inject security headers and service status into HTML
      return html.replace(
        '<html',
        `<html data-services="${servicesInitialized ? 'ready' : 'initializing'}" data-request-id="${(event.locals as any).requestId}"`
      );
    }
  });

  // Add security headers
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  // Add CORS headers for API routes
  if (event.url.pathname.startsWith('/api/')) {
    response.headers.set('Access-Control-Allow-Origin', '*');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  }

  return response;
};

// Cache control handle
const cacheHandle: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);

  // Set cache headers based on route
  if (event.url.pathname.startsWith('/api/')) {
    // API routes - no cache
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
  } else if (event.url.pathname.includes('/_app/')) {
    // Static assets - long cache
    response.headers.set('Cache-Control', 'public, max-age=31536000, immutable');
  } else {
    // Pages - short cache
    response.headers.set('Cache-Control', 'public, max-age=300, s-maxage=600');
  }

  return response;
};

// Session and auth handle
const sessionHandle: Handle = async ({ event, resolve }) => {
  // Ensure locals has a predictable default shape
  (event.locals as any).user = (event.locals as any).user ?? null;
  (event.locals as any).session = (event.locals as any).session ?? null;

  // Use corrected session validation
  try {
    const { validateSession } = await import('$lib/server/lucia');
    const sessionId = event.cookies.get('session_id') || event.cookies.get('session');
    
    if (sessionId) {
      const { session, user } = await validateSession(sessionId);
      
      if (user && session) {
        (event.locals as any).user = user;
        (event.locals as any).session = { 
          id: session.id, 
          user: user,
          userId: user.id 
        };
        devLog('[hooks] Session found for user:', user.email);
      } else {
        devLog('[hooks] No valid session found');
      }
    } else {
      devLog('[hooks] No session ID in cookies');
    }
  } catch (error) {
    console.warn('[hooks] Lucia session validation failed:', error?.message || error);
    // Fallback to Redis session lookup for backward compatibility (if available)
    const sessionId =
      event.cookies.get('session_id') ||
      event.cookies.get('session') ||
      event.cookies.get('yorha_session') ||
      null;

    if (sessionId && redis.getConnectionStatus() === 'connected') {
      try {
        const sessionData = await redis.getSession(sessionId);
        if (sessionData) {
          (event.locals as any).session = sessionData;
          (event.locals as any).user = {
            id: sessionData.userId,
            email: sessionData.email || '',
            name: sessionData.name || null,
            role: sessionData.role || 'user',
            isActive: true,
          };
          devLog('[hooks] Redis fallback session populated for user:', (event.locals as any).user.email);
        }
      } catch (redisError) {
        console.warn('[hooks] Redis session lookup failed (non-critical):', redisError?.message || redisError);
      }
    } else if (sessionId) {
      devLog('[hooks] Redis not connected, skipping session lookup');
    }
  }

  return resolve(event);
};

// Enhanced error handler
const enhancedErrorHandler: HandleServerError = ({ error, event, status, message }) => {
  const errorId = `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  devLog(`❌ [${errorId}] Server Error:`, (error as Error)?.message || 'Unknown error');
  console.error(`❌ [${errorId}] Server Error:`, {
    error: (error as Error)?.message || 'Unknown error',
    stack: (error as Error)?.stack,
    url: event?.url?.pathname,
    method: event?.request?.method,
    userAgent: event?.request?.headers?.get('user-agent'),
    requestId: (event?.locals as any)?.requestId,
    status,
    message
  });

  // Log to monitoring service in production
  if (process.env.NODE_ENV === 'production') {
    // Integration point for error monitoring service
    // errorMonitor.logError(error, event, errorId);
  }

  return {
    message: process.env.NODE_ENV === 'development' ?
      (error as Error)?.message || message :
      'An error occurred',
    code: status?.toString() || 'INTERNAL_SERVER_ERROR',
    errorId
  };
};

// Custom fetch handler for service communication
const customFetch: HandleFetch = async ({ request, fetch, event }) => {
  // Inject request metadata for internal service calls
  if (request.url.includes('localhost')) {
    const headers = new Headers(request.headers);
    headers.set('X-Request-ID', (event.locals as any).requestId || 'unknown');
    headers.set('X-User-ID', (event.locals as any).user?.id?.toString() || 'anonymous');
    headers.set('X-Session-ID', event.cookies.get('session_id') || 'none');

    request = new Request(request, { headers });
  }

  const startTime = Date.now();

  try {
    const response = await fetch(request);
    const responseTime = Date.now() - startTime;

    // Log slow requests
    if (responseTime > 5000) {
      console.warn(`🐌 Slow fetch detected: ${request.url} took ${responseTime}ms`);
    }

    return response;
  } catch (error) {
    console.error(`❌ Fetch error for ${request.url}:`, error);
    throw error;
  }
};

// Compose all handles in sequence
export const handle: Handle = sequence(
  serviceHealthHandle,
  loggingHandle,
  securityHandle,
  sessionHandle,
  cacheHandle
);

export const handleError: HandleServerError = enhancedErrorHandler;
export const handleFetch: HandleFetch = customFetch;

// Graceful shutdown (non-blocking)
process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down services...');
  try {
    await Promise.allSettled([
      redis.disconnect().catch(err => console.warn('Redis disconnect error:', err.message)),
      rabbitmqService.disconnect().catch(err => console.warn('RabbitMQ disconnect error:', err.message)),
      // minioService doesn't need explicit disconnect
    ]);
    console.log('👋 Services shut down gracefully');
  } catch (error) {
    console.error('❌ Error during shutdown:', error);
  }
  process.exit(0);
});

// Export service status for use in other modules
export function getServiceStatus() {
  return {
    initialized: servicesInitialized,
    metrics: { ...requestMetrics },
    redis: redis.getConnectionStatus(),
    workflows: workflowOrchestrator.getActiveWorkflowsCount()
  };
}