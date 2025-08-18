// POST /api/rag/documents - Upload and index document
// GET /api/rag/documents - Search documents
// DELETE /api/rag/documents - Delete document
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { EnhancedRAGService } from '$lib/yorha/services/rag.service';
import { AuthService } from '$lib/yorha/services/auth.service';
import { db } from '$lib/yorha/db';
import { documents } from '$lib/yorha/db/schema';
import { eq, and, ilike, desc } from 'drizzle-orm';
import { z } from 'zod';

// Upload document schema
const uploadSchema = z.object({
  title: z.string().min(1).max(255),
  content: z.string().min(10),
  source: z.enum(['manual', 'upload', 'url', 'api']),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional()
});

// POST - Upload and index document
export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    
    if (!sessionToken) {
      return json({
        success: false,
        error: 'Authentication required'
      }, { status: 401 });
    }
    
    const authService = new AuthService();
    const sessionData = await authService.validateSession(sessionToken);
    
    if (!sessionData) {
      return json({
        success: false,
        error: 'Invalid session'
      }, { status: 401 });
    }
    
    const body = await request.json();
    const validated = uploadSchema.parse(body);
    
    const ragService = new EnhancedRAGService();
    
    // Index document
    const documentId = await ragService.indexDocument(
      validated.title,
      validated.content,
      validated.source,
      {
        ...validated.metadata,
        userId: sessionData.unit.id,
        tags: validated.tags
      }
    );
    
    return json({
      success: true,
      data: {
        documentId,
        message: 'Document indexed successfully'
      }
    });
  } catch (error: any) {
    console.error('Document upload error:', error);
    
    if (error.name === 'ZodError') {
      return json({
        success: false,
        error: 'Invalid document format',
        details: error.errors
      }, { status: 400 });
    }
    
    return json({
      success: false,
      error: 'Failed to index document'
    }, { status: 500 });
  }
};

// GET - Search documents
export const GET: RequestHandler = async ({ url, cookies }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    let userId: string | undefined;
    
    if (sessionToken) {
      const authService = new AuthService();
      const sessionData = await authService.validateSession(sessionToken);
      userId = sessionData?.unit.id;
    }
    
    const query = url.searchParams.get('query');
    const source = url.searchParams.get('source');
    const limit = parseInt(url.searchParams.get('limit') || '10');
    const offset = parseInt(url.searchParams.get('offset') || '0');
    
    if (query) {
      // Semantic search using RAG
      const ragService = new EnhancedRAGService();
      const results = await ragService.searchDocuments(query, {
        limit,
        source: source || undefined,
        userId
      });
      
      return json({
        success: true,
        data: {
          documents: results,
          total: results.length
        }
      });
    } else {
      // Regular database query
      const conditions = [];
      
      if (userId) {
        conditions.push(eq(documents.userId, userId));
      }
      
      if (source) {
        conditions.push(eq(documents.source, source));
      }
      
      const docs = await db.query.documents.findMany({
        where: conditions.length > 0 ? and(...conditions) : undefined,
        orderBy: [desc(documents.createdAt)],
        limit,
        offset
      });
      
      return json({
        success: true,
        data: {
          documents: docs,
          total: docs.length
        }
      });
    }
  } catch (error: any) {
    console.error('Document search error:', error);
    
    return json({
      success: false,
      error: 'Failed to search documents'
    }, { status: 500 });
  }
};

// DELETE - Delete document
export const DELETE: RequestHandler = async ({ request, cookies }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    
    if (!sessionToken) {
      return json({
        success: false,
        error: 'Authentication required'
      }, { status: 401 });
    }
    
    const authService = new AuthService();
    const sessionData = await authService.validateSession(sessionToken);
    
    if (!sessionData) {
      return json({
        success: false,
        error: 'Invalid session'
      }, { status: 401 });
    }
    
    const body = await request.json();
    const { documentId } = body;
    
    if (!documentId) {
      return json({
        success: false,
        error: 'Document ID required'
      }, { status: 400 });
    }
    
    // Check if user owns the document
    const doc = await db.query.documents.findFirst({
      where: and(
        eq(documents.id, documentId),
        eq(documents.userId, sessionData.unit.id)
      )
    });
    
    if (!doc) {
      return json({
        success: false,
        error: 'Document not found or unauthorized'
      }, { status: 404 });
    }
    
    const ragService = new EnhancedRAGService();
    await ragService.deleteDocument(documentId);
    
    return json({
      success: true,
      data: {
        message: 'Document deleted successfully'
      }
    });
  } catch (error: any) {
    console.error('Document delete error:', error);
    
    return json({
      success: false,
      error: 'Failed to delete document'
    }, { status: 500 });
  }
};