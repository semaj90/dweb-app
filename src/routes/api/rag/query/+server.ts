// POST /api/rag/query - Process RAG query
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { EnhancedRAGService } from '$lib/yorha/services/rag.service';
import { AuthService } from '$lib/yorha/services/auth.service';
import { z } from 'zod';

const querySchema = z.object({
  query: z.string().min(1).max(1000),
  conversationId: z.string().optional(),
  filters: z.object({
    source: z.string().optional(),
    category: z.string().optional(),
    tags: z.array(z.string()).optional()
  }).optional(),
  includeHistory: z.boolean().optional()
});

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    let userId: string | undefined;
    
    // Optional authentication - RAG can work for both authenticated and guest users
    if (sessionToken) {
      const authService = new AuthService();
      const sessionData = await authService.validateSession(sessionToken);
      userId = sessionData?.unit.id;
    }
    
    const body = await request.json();
    const validated = querySchema.parse(body);
    
    const ragService = new EnhancedRAGService();
    
    // Build context for RAG
    const context = {
      query: validated.query,
      userId,
      conversationId: validated.conversationId,
      filters: validated.filters,
      previousMessages: validated.includeHistory ? [] : undefined // You would fetch this from DB
    };
    
    // Process RAG query
    const response = await ragService.query(context);
    
    return json({
      success: true,
      data: response
    });
  } catch (error: any) {
    console.error('RAG query error:', error);
    
    if (error.name === 'ZodError') {
      return json({
        success: false,
        error: 'Invalid query format',
        details: error.errors
      }, { status: 400 });
    }
    
    return json({
      success: false,
      error: 'Failed to process query'
    }, { status: 500 });
  }
};