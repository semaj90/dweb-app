import { json } from '@sveltejs/kit';
import { z } from 'zod';
import { db } from '$lib/db/client';
import { cases, documents, insertCaseSchema, selectCaseSchema } from '$lib/db/schema/rag-integration';
import { eq, desc, and, sql } from 'drizzle-orm';
import { randomUUID } from 'crypto';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


const createCaseSchema = z.object({
  id: z.string().uuid(),
  title: z.string().min(1).max(255),
  description: z.string().optional(),
  status: z.enum(['active', 'archived', 'deleted']).default('active'),
  metadata: z.record(z.any()).optional()
});

const updateCaseSchema = z.object({
  title: z.string().min(1).max(255).optional(),
  description: z.string().optional(),
  status: z.enum(['active', 'archived', 'deleted']).optional(),
  metadata: z.record(z.any()).optional()
});

// Create new case
export async function POST({ request }): Promise<any> {
  try {
    const body = await request.json();
    const { id, title, description, status, metadata } = createCaseSchema.parse(body);

    const [newCase] = await db
      .insert(cases)
      .values({
        uuid: id,
        title,
        description,
        status,
        metadata: metadata || {}
      })
      .returning();

    return json({
      success: true,
      case: {
        id: newCase.uuid,
        title: newCase.title,
        description: newCase.description,
        status: newCase.status,
        metadata: newCase.metadata,
        createdAt: newCase.createdAt,
        updatedAt: newCase.updatedAt
      }
    }, { status: 201 });

  } catch (error: any) {
    console.error('Case creation error:', error);
    
    if (error instanceof z.ZodError) {
      return json({ error: 'Invalid request data', details: error.errors }, { status: 400 });
    }
    
    if (error.code === '23505') { // PostgreSQL unique violation
      return json({ error: 'Case with this ID already exists' }, { status: 409 });
    }
    
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// List cases with optional filtering
export async function GET({ url }): Promise<any> {
  try {
    const status = url.searchParams.get('status');
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 100);
    const offset = Math.max(parseInt(url.searchParams.get('offset') || '0'), 0);
    const search = url.searchParams.get('search');
    const includeStats = url.searchParams.get('includeStats') === 'true';

    // Build base query
    let query = db
      .select({
        id: cases.id,
        uuid: cases.uuid,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        metadata: cases.metadata,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt
      })
      .from(cases);

    // Add filters
    const conditions = [];
    
    if (status) {
      conditions.push(eq(cases.status, status));
    }

    if (search) {
      // Search in title and description
      conditions.push(
        sql`(${cases.title} ILIKE ${'%' + search + '%'} OR ${cases.description} ILIKE ${'%' + search + '%'})`
      );
    }

    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }

    // Execute query with pagination
    const results = await query
      .orderBy(desc(cases.createdAt))
      .limit(limit)
      .offset(offset);

    // Optionally include document counts
    let casesWithStats = results;
    
    if (includeStats) {
      casesWithStats = await Promise.all(
        results.map(async (caseItem) => {
          const [stats] = await db
            .select({
              documentCount: sql<number>`count(*)`,
              processedCount: sql<number>`count(*) FILTER (WHERE processing_status = 'completed')`,
              totalSize: sql<number>`coalesce(sum(file_size), 0)`
            })
            .from(documents)
            .where(eq(documents.caseId, caseItem.id));

          return {
            ...caseItem,
            stats: {
              documentCount: Number(stats.documentCount || 0),
              processedCount: Number(stats.processedCount || 0),
              totalSize: Number(stats.totalSize || 0)
            }
          };
        })
      );
    }

    // Get total count for pagination
    const [totalCount] = await db
      .select({ count: sql<number>`count(*)` })
      .from(cases)
      .where(conditions.length > 0 ? and(...conditions) : undefined);

    return json({
      cases: casesWithStats.map(c => ({
        id: c.uuid,
        title: c.title,
        description: c.description,
        status: c.status,
        metadata: c.metadata,
        createdAt: c.createdAt,
        updatedAt: c.updatedAt,
        stats: c.stats || undefined
      })),
      pagination: {
        total: Number(totalCount.count),
        limit,
        offset,
        hasMore: offset + limit < Number(totalCount.count)
      }
    });

  } catch (error: any) {
    console.error('Case listing error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Update existing case
export async function PATCH({ request, url }): Promise<any> {
  const caseId = url.searchParams.get('id');
  
  if (!caseId) {
    return json({ error: 'Case ID required' }, { status: 400 });
  }

  try {
    const body = await request.json();
    const updates = updateCaseSchema.parse(body);

    // Check if case exists
    const [existingCase] = await db
      .select()
      .from(cases)
      .where(eq(cases.uuid, caseId))
      .limit(1);

    if (!existingCase) {
      return json({ error: 'Case not found' }, { status: 404 });
    }

    // Update case
    const [updatedCase] = await db
      .update(cases)
      .set({
        ...updates,
        updatedAt: new Date()
      })
      .where(eq(cases.uuid, caseId))
      .returning();

    return json({
      success: true,
      case: {
        id: updatedCase.uuid,
        title: updatedCase.title,
        description: updatedCase.description,
        status: updatedCase.status,
        metadata: updatedCase.metadata,
        createdAt: updatedCase.createdAt,
        updatedAt: updatedCase.updatedAt
      }
    });

  } catch (error: any) {
    console.error('Case update error:', error);
    
    if (error instanceof z.ZodError) {
      return json({ error: 'Invalid request data', details: error.errors }, { status: 400 });
    }
    
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Delete case (soft delete by default)
export async function DELETE({ url }): Promise<any> {
  const caseId = url.searchParams.get('id');
  const hard = url.searchParams.get('hard') === 'true';
  
  if (!caseId) {
    return json({ error: 'Case ID required' }, { status: 400 });
  }

  try {
    // Check if case exists
    const [existingCase] = await db
      .select()
      .from(cases)
      .where(eq(cases.uuid, caseId))
      .limit(1);

    if (!existingCase) {
      return json({ error: 'Case not found' }, { status: 404 });
    }

    if (hard) {
      // Hard delete - remove from database
      // Note: This should cascade to documents and chunks
      await db
        .delete(cases)
        .where(eq(cases.uuid, caseId));
        
      return json({ 
        success: true, 
        message: 'Case permanently deleted' 
      });
    } else {
      // Soft delete - mark as deleted
      const [deletedCase] = await db
        .update(cases)
        .set({
          status: 'deleted',
          updatedAt: new Date()
        })
        .where(eq(cases.uuid, caseId))
        .returning();

      return json({
        success: true,
        message: 'Case marked as deleted',
        case: {
          id: deletedCase.uuid,
          status: deletedCase.status,
          updatedAt: deletedCase.updatedAt
        }
      });
    }

  } catch (error: any) {
    console.error('Case deletion error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}