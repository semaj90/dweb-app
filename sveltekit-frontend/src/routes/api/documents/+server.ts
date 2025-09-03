
/**
 * Documents API with pgvector integration
 * Handles document CRUD operations with vector embeddings
 */

import { json, type RequestHandler } from '@sveltejs/kit';
import { db } from '$lib/server/db/connection';
import { documents, document_chunks, cases } from '$lib/server/schema/documents';
import { eq, desc, and, like, sql } from 'drizzle-orm';
import { createEmbedding } from '$lib/services/embedding-service';
import { redis } from '$lib/server/redis';
import type { Document, NewDocument } from '$lib/server/schema/documents';

const CACHE_TTL = 300; // 5 minutes

export const GET: RequestHandler = async ({ url }) => {
  try {
    const searchParams = url.searchParams;
    const page = parseInt(searchParams.get('page') || '1');
    const limit = parseInt(searchParams.get('limit') || '20');
    const search = searchParams.get('search');
    const caseId = searchParams.get('case_id');
    const documentType = searchParams.get('document_type');
    const riskLevel = searchParams.get('risk_level');
    
    const offset = (page - 1) * limit;
    
    // Build cache key
    const cacheKey = `documents:${JSON.stringify({ page, limit, search, caseId, documentType, riskLevel })}`;
    
    // Try cache first
    const cached = await redis.get(cacheKey);
    if (cached) {
      return json(JSON.parse(cached));
    }
    
    // Build query conditions
    const conditions = [];
    
    if (search) {
      conditions.push(
        sql`(
          to_tsvector('english', ${documents.title}) @@ plainto_tsquery('english', ${search})
          OR to_tsvector('english', ${documents.content}) @@ plainto_tsquery('english', ${search})
        )`
      );
    }
    
    if (caseId) {
      conditions.push(eq(documents.case_id, caseId));
    }
    
    if (documentType) {
      conditions.push(eq(documents.document_type, documentType));
    }
    
    if (riskLevel) {
      conditions.push(eq(documents.risk_level, riskLevel));
    }
    
    conditions.push(eq(documents.is_active, true));
    
    // Execute query with relations
    const query = db
      .select({
        id: documents.id,
        title: documents.title,
        content: documents.content,
        document_type: documents.document_type,
        confidence_level: documents.confidence_level,
        risk_level: documents.risk_level,
        priority: documents.priority,
        ai_summary: documents.ai_summary,
        ai_tags: documents.ai_tags,
        key_entities: documents.key_entities,
        source_url: documents.source_url,
        file_path: documents.file_path,
        file_type: documents.file_type,
        file_size: documents.file_size,
        case_id: documents.case_id,
        jurisdiction: documents.jurisdiction,
        practice_area: documents.practice_area,
        processing_status: documents.processing_status,
        processed_at: documents.processed_at,
        created_at: documents.created_at,
        updated_at: documents.updated_at,
        created_by: documents.created_by,
        is_indexed: documents.is_indexed,
        case_title: cases.title
      })
      .from(documents)
      .leftJoin(cases, eq(documents.case_id, cases.id))
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(documents.created_at))
      .limit(limit)
      .offset(offset);
    
    const results = await query;
    
    // Get total count for pagination
    const totalQuery = db
      .select({ count: sql`count(*)` })
      .from(documents)
      .where(conditions.length > 0 ? and(...conditions) : undefined);
    
    const totalResult = await totalQuery;
    const total = Number(totalResult[0].count);
    
    const response = {
      documents: results,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
        hasNext: page * limit < total,
        hasPrev: page > 1
      }
    };
    
    // Cache results
    await redis.setex(cacheKey, CACHE_TTL, JSON.stringify(response));
    
    return json(response);
    
  } catch (error) {
    console.error('Error fetching documents:', error);
    return json(
      { error: 'Failed to fetch documents', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const data = await request.json() as Partial<NewDocument> & {
      auto_embed?: boolean;
      extract_entities?: boolean;
      generate_summary?: boolean;
    };
    
    // Validate required fields
    if (!data.title || !data.content) {
      return json(
        { error: 'Title and content are required' },
        { status: 400 }
      );
    }
    
    // Generate embeddings if requested
    let embedding: number[] | undefined;
    let titleEmbedding: number[] | undefined;
    let summaryEmbedding: number[] | undefined;
    
    if (data.auto_embed !== false) {
      try {
        // Generate content embedding
        embedding = await createEmbedding(data.content);
        
        // Generate title embedding
        titleEmbedding = await createEmbedding(data.title);
        
        // Generate summary embedding if we have AI summary
        if (data.ai_summary) {
          summaryEmbedding = await createEmbedding(data.ai_summary);
        }
      } catch (embeddingError) {
        console.warn('Failed to generate embeddings:', embeddingError);
        // Continue without embeddings rather than failing
      }
    }
    
    // Prepare document data
    const documentData: NewDocument = {
      title: data.title,
      content: data.content,
      document_type: data.document_type || 'general',
      confidence_level: data.confidence_level || 0,
      risk_level: data.risk_level || 'low',
      priority: data.priority || 100,
      ai_summary: data.ai_summary,
      ai_analysis: data.ai_analysis,
      ai_tags: data.ai_tags,
      key_entities: data.key_entities,
      source_url: data.source_url,
      file_path: data.file_path,
      file_type: data.file_type,
      file_size: data.file_size,
      case_id: data.case_id,
      jurisdiction: data.jurisdiction,
      practice_area: data.practice_area,
      processing_status: data.processing_status || 'pending',
      embedding_model: data.embedding_model || 'nomic-embed-text',
      created_by: data.created_by,
      is_public: data.is_public || false,
      is_indexed: data.is_indexed || false
    };
    
    // Insert document
    const [newDocument] = await db
      .insert(documents)
      .values(documentData)
      .returning();
    
    // Update with embeddings if generated
    if (embedding || titleEmbedding || summaryEmbedding) {
      const updates: Partial<Document> = {};
      
      if (embedding) {
        updates.embedding = sql`${JSON.stringify(embedding)}::vector`;
      }
      if (titleEmbedding) {
        updates.title_embedding = sql`${JSON.stringify(titleEmbedding)}::vector`;
      }
      if (summaryEmbedding) {
        updates.summary_embedding = sql`${JSON.stringify(summaryEmbedding)}::vector`;
      }
      
      if (Object.keys(updates).length > 0) {
        updates.is_indexed = true;
        updates.processed_at = new Date();
        
        await db
          .update(documents)
          .set(updates)
          .where(eq(documents.id, newDocument.id));
      }
    }
    
    // Clear relevant caches
    const cachePattern = 'documents:*';
    const keys = await redis.keys(cachePattern);
    if (keys.length > 0) {
      await redis.del(...keys);
    }
    
    return json({
      document: newDocument,
      embeddings_generated: !!(embedding || titleEmbedding || summaryEmbedding),
      message: 'Document created successfully'
    }, { status: 201 });
    
  } catch (error) {
    console.error('Error creating document:', error);
    return json(
      { error: 'Failed to create document', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
};

export const PUT: RequestHandler = async ({ request, url }) => {
  try {
    const documentId = url.searchParams.get('id');
    if (!documentId) {
      return json({ error: 'Document ID is required' }, { status: 400 });
    }
    
    const data = await request.json() as Partial<Document> & {
      auto_embed?: boolean;
    };
    
    // Check if document exists
    const existingDocument = await db
      .select()
      .from(documents)
      .where(eq(documents.id, documentId))
      .limit(1);
    
    if (existingDocument.length === 0) {
      return json({ error: 'Document not found' }, { status: 404 });
    }
    
    // Generate new embeddings if content changed and auto_embed is enabled
    let embedding: number[] | undefined;
    let titleEmbedding: number[] | undefined;
    let summaryEmbedding: number[] | undefined;
    
    if (data.auto_embed !== false) {
      if (data.content && data.content !== existingDocument[0].content) {
        embedding = await createEmbedding(data.content);
      }
      
      if (data.title && data.title !== existingDocument[0].title) {
        titleEmbedding = await createEmbedding(data.title);
      }
      
      if (data.ai_summary && data.ai_summary !== existingDocument[0].ai_summary) {
        summaryEmbedding = await createEmbedding(data.ai_summary);
      }
    }
    
    // Prepare update data
    const updateData: Partial<Document> = {
      ...data,
      updated_at: new Date()
    };
    
    // Add embeddings if generated
    if (embedding) {
      updateData.embedding = sql`${JSON.stringify(embedding)}::vector`;
      updateData.is_indexed = true;
      updateData.processed_at = new Date();
    }
    if (titleEmbedding) {
      updateData.title_embedding = sql`${JSON.stringify(titleEmbedding)}::vector`;
    }
    if (summaryEmbedding) {
      updateData.summary_embedding = sql`${JSON.stringify(summaryEmbedding)}::vector`;
    }
    
    // Remove fields that shouldn't be updated directly
    delete updateData.auto_embed;
    delete updateData.id;
    delete updateData.created_at;
    
    const [updatedDocument] = await db
      .update(documents)
      .set(updateData)
      .where(eq(documents.id, documentId))
      .returning();
    
    // Clear relevant caches
    const cachePattern = 'documents:*';
    const keys = await redis.keys(cachePattern);
    if (keys.length > 0) {
      await redis.del(...keys);
    }
    
    return json({
      document: updatedDocument,
      embeddings_updated: !!(embedding || titleEmbedding || summaryEmbedding),
      message: 'Document updated successfully'
    });
    
  } catch (error) {
    console.error('Error updating document:', error);
    return json(
      { error: 'Failed to update document', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
};

export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const documentId = url.searchParams.get('id');
    if (!documentId) {
      return json({ error: 'Document ID is required' }, { status: 400 });
    }
    
    // Soft delete - mark as inactive
    const [deletedDocument] = await db
      .update(documents)
      .set({ 
        is_active: false, 
        updated_at: new Date() 
      })
      .where(eq(documents.id, documentId))
      .returning();
    
    if (!deletedDocument) {
      return json({ error: 'Document not found' }, { status: 404 });
    }
    
    // Clear relevant caches
    const cachePattern = 'documents:*';
    const keys = await redis.keys(cachePattern);
    if (keys.length > 0) {
      await redis.del(...keys);
    }
    
    return json({
      message: 'Document deleted successfully',
      document_id: documentId
    });
    
  } catch (error) {
    console.error('Error deleting document:', error);
    return json(
      { error: 'Failed to delete document', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
};
