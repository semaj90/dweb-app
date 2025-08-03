import type { Handle, HandleServerError } from "@sveltejs/kit";
import { lucia } from "$lib/server/auth";
import { enhancedRAGService } from "$lib/services/enhanced-rag-service.js";
import type { User } from "$lib/types/user";

// Enhanced server hooks with proper Lucia v3 authentication
export const handle: Handle = async ({ event, resolve }) => {
  // Auto-initialize enhanced RAG on API requests
  if (event.url.pathname.startsWith("/api/")) {
    try {
      await enhancedRAGService.initialize();
    } catch (error) {
      console.warn("Enhanced RAG initialization failed:", error);
    }
  }

  // Lucia v3 session validation
  const sessionId = event.cookies.get(lucia.sessionCookieName);
  
  if (!sessionId) {
    event.locals.user = null;
    event.locals.session = null;
  } else {
    const { session, user } = await lucia.validateSession(sessionId);
    
    if (session && session.fresh) {
      const sessionCookie = lucia.createSessionCookie(session.id);
      event.cookies.set(sessionCookie.name, sessionCookie.value, {
        path: ".",
        ...sessionCookie.attributes,
      });
    }
    
    if (!session) {
      const sessionCookie = lucia.createBlankSessionCookie();
      event.cookies.set(sessionCookie.name, sessionCookie.value, {
        path: ".",
        ...sessionCookie.attributes,
      });
    }

    // Transform lucia user to app User type
    if (user) {
      const dbUser = user as any; // Lucia user with database attributes
      event.locals.user = {
        id: dbUser.id,
        email: dbUser.email || "",
        name: dbUser.name || dbUser.email || "",
        firstName: dbUser.firstName || "",
        lastName: dbUser.lastName || "",
        avatarUrl: dbUser.avatarUrl || "/avatars/default.png",
        role: (dbUser.role as "prosecutor" | "investigator" | "admin" | "user") || "user",
        isActive: Boolean(dbUser.isActive ?? true),
        emailVerified: dbUser.emailVerified ? (typeof dbUser.emailVerified === 'boolean' ? new Date() : dbUser.emailVerified) : null,
        createdAt: dbUser.createdAt ? new Date(dbUser.createdAt) : new Date(),
        updatedAt: dbUser.updatedAt ? new Date(dbUser.updatedAt) : new Date(),
      } as User;
    } else {
      event.locals.user = null;
    }

    event.locals.session = session;
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