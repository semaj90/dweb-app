// @ts-nocheck
import type { Handle, HandleServerError } from "@sveltejs/kit";
import { validateSession } from "$lib/server/lucia";
// import { enhancedRAGService } from "$lib/services/enhanced-rag-service.js"; // Disabled during debugging
import type { SessionUser } from "$lib/types/auth";

// Enhanced server hooks with proper Lucia v3 authentication
export const handle: Handle = async ({ event, resolve }) => {
  // Skip database-dependent initializations in development
  const isDevelopment = process.env.NODE_ENV === 'development' || process.env.SKIP_RAG_INITIALIZATION === 'true';
  
  // Auto-initialize enhanced RAG on API requests (skip in development)
  // Temporarily disabled during debugging
  /*
  if (event.url.pathname.startsWith("/api/") && !isDevelopment) {
    try {
      await enhancedRAGService.initialize();
    } catch (error) {
      console.warn("Enhanced RAG initialization failed:", error);
    }
  }
  */

  // Custom session validation (no Lucia dependency) - skip database calls in development
  if (!isDevelopment) {
    try {
      const sessionId = event.cookies.get('session_id');
      
      if (sessionId) {
        const user = await validateSession(sessionId);
        event.locals.user = user;
        event.locals.session = sessionId;
      } else {
        event.locals.user = null;
        event.locals.session = null;
      }
    } catch (error) {
      console.error('Session validation error in hooks:', error);
      event.locals.user = null;
      event.locals.session = null;
    }
  } else {
    // Development mode: skip database validation
    event.locals.user = null;
    event.locals.session = null;
  }

  // Add CORS headers for API routes
  if (event.url.pathname.startsWith("/api")) {
    const response = await resolve(event);
    response.headers.set("Access-Control-Allow-Origin", "*");
    response.headers.set(
      "Access-Control-Allow-Methods",
      "GET, POST, PUT, DELETE, OPTIONS"
    );
    response.headers.set(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization"
    );
    return response;
  }

  return resolve(event);
};

export const handleError: HandleServerError = async ({
  error,
  event,
  status,
  message,
}) => {
  const errorId = crypto.randomUUID();

  console.error("Server error:", {
    errorId,
    error: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined,
    status,
    url: event.url.pathname,
    method: event.request.method,
  });

  return {
    message: "Internal Server Error",
    errorId,
  };
};