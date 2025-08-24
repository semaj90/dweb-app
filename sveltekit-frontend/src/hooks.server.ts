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

// Request tracking
const requestMetrics = {
  totalRequests: 0,
  errorCount: 0,
  averageResponseTime: 0,
  lastRequestTime: Date.now()
};

// Initialize all services
async function initializeServices(): Promise<void> {
  if (servicesInitialized) return;
  
  if (initializationPromise) {
    await initializationPromise;
    return;
  }

  initializationPromise = (async () => {
    console.log('🚀 Initializing full-stack services...');
    
    try {
      // Initialize services concurrently
      const initResults = await Promise.allSettled([
        redis.connect(),
        minioService.initialize(),
        rabbitmqService.connect()
      ]);

      const services = ['Redis', 'MinIO', 'RabbitMQ'];
      initResults.forEach((result, index) => {
        if (result.status === 'fulfilled' && result.value) {
          console.log(`✅ ${services[index]} initialized successfully`);
        } else {
          console.warn(`⚠️ ${services[index]} initialization failed:`, 
            result.status === 'rejected' ? result.reason : 'Unknown error');
        }
      });

      servicesInitialized = true;
      console.log('✅ All services initialization completed');
      
    } catch (error) {
      console.error('❌ Service initialization error:', error);
      servicesInitialized = false;
    }
  })();

  await initializationPromise;
}

// Service health injection handle
const serviceHealthHandle: Handle = async ({ event, resolve }) => {
  // Initialize services on first request
  if (!servicesInitialized) {
    await initializeServices();
  }

  // Inject service status into locals
  event.locals.services = {
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
  const ip = event.getClientAddress();

  // Generate request ID
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  // Inject request metadata
  event.locals.requestId = requestId;
  event.locals.startTime = startTime;

  console.log(`🌐 [${new Date().toISOString()}] ${method} ${new URL(event.request.url).pathname} - ${ip} - ${requestId}`);

  try {
    const response = await resolve(event);
    const responseTime = Date.now() - startTime;
    
    // Update metrics
    requestMetrics.totalRequests++;
    requestMetrics.averageResponseTime = 
      (requestMetrics.averageResponseTime + responseTime) / 2;
    requestMetrics.lastRequestTime = Date.now();

    // Log successful request
    console.log(`✅ [${requestId}] ${response.status} - ${responseTime}ms`);

    // Add performance headers
    response.headers.set('X-Request-ID', requestId);
    response.headers.set('X-Response-Time', `${responseTime}ms`);
    response.headers.set('X-Powered-By', 'SvelteKit + Full-Stack Services');

    return response;

  } catch (error) {
    const responseTime = Date.now() - startTime;
    requestMetrics.errorCount++;
    
    console.error(`❌ [${requestId}] Error after ${responseTime}ms:`, error);
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
        `<html data-services="${servicesInitialized ? 'ready' : 'initializing'}" data-request-id="${event.locals.requestId}"`
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
  const sessionId = event.cookies.get('session_id');
  
  if (sessionId && servicesInitialized) {
    try {
      const sessionData = await redis.getSession(sessionId);
      if (sessionData) {
        event.locals.session = sessionData;
        event.locals.user = {
          id: sessionData.userId,
          email: sessionData.email || '',
          name: sessionData.name || null,
          role: sessionData.role || 'user',
          isActive: true
        };
      }
    } catch (error) {
      console.warn('Session lookup failed:', error);
    }
  }

  return resolve(event);
};

// Enhanced error handler
const enhancedErrorHandler: HandleServerError = ({ error, event, status, message }) => {
  const errorId = `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  console.error(`❌ [${errorId}] Server Error:`, {
    error: (error as Error)?.message || 'Unknown error',
    stack: (error as Error)?.stack,
    url: event?.url?.pathname,
    method: event?.request?.method,
    userAgent: event?.request?.headers?.get('user-agent'),
    requestId: event?.locals?.requestId,
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
    headers.set('X-Request-ID', event.locals.requestId || 'unknown');
    headers.set('X-User-ID', event.locals.user?.id?.toString() || 'anonymous');
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

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down services...');
  try {
    await Promise.all([
      redis.disconnect(),
      rabbitmqService.disconnect(),
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