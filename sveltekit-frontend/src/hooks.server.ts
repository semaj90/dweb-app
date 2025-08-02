import type { Handle } from "@sveltejs/kit";
import { enhancedRAGService } from '$lib/services/enhanced-rag-service.js';

export const handle: Handle = async ({ event, resolve }) => {
  // Auto-initialize enhanced RAG on API requests
  if (event.url.pathname.startsWith('/api/')) {
    try {
      await enhancedRAGService.initialize();
    } catch (error) {
      console.warn('Enhanced RAG initialization failed:', error);
    }
  }
  
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
