import type { Handle, HandleServerError } from "@sveltejs/kit";
import { enhancedRAGService } from '$lib/services/enhanced-rag-service.js';
import type { Database } from '$lib/types';

// Enhanced server hooks with proper error handling and type safety
export const handle: Handle = async ({ event, resolve }) => {
  // Auto-initialize enhanced RAG on API requests
  if (event.url.pathname.startsWith('/api/')) {
    try {
      await enhancedRAGService.initialize();
    } catch (error) {
      console.warn('Enhanced RAG initialization failed:', error);
    }
  }

  // Mock user session - replace with actual auth
  const mockUser: Database.User = {
    id: 'user_123',
    email: 'demo@example.com',
    name: 'Demo User',
    firstName: 'Demo',
    lastName: 'User',
    avatarUrl: '/avatars/default.png',
    role: 'user',
    isActive: true,
    emailVerified: new Date(),
    createdAt: new Date(),
    updatedAt: new Date()
  };

  // Add user to locals for type safety
  event.locals.user = mockUser;

  // Add CORS headers for API routes
  if (event.url.pathname.startsWith("/api")) {
    const response = await resolve(event);
    response.headers.set("Access-Control-Allow-Origin", "*");
    response.headers.set(
      "Access-Control-Allow-Methods",
      "GET, POST, PUT, DELETE, OPTIONS",
    );
    response.headers.set(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization",
    );
    return response;
  }

  return resolve(event);
};

export const handleError: HandleServerError = async ({ error, event, status, message }) => {
  const errorId = crypto.randomUUID();
  
  console.error('Server error:', {
    errorId,
    error: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined,
    status,
    url: event.url.pathname,
    method: event.request.method
  });

  return {
    message: 'Internal Server Error',
    errorId
  };
};
