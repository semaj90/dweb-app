import { json, error } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { cases, caseDocuments, caseActivities, caseTimeline, users } from '$lib/server/db/schema';
import { eq, and, or, desc, count, ilike } from 'drizzle-orm';
import { z } from 'zod';
import { generateEmbedding } from '$lib/server/embedding-service';
import { validateSession } from '$lib/server/lucia';
import type { RequestHandler } from './$types';


// Authentication helper
async function getAuthenticatedUser(locals: App.Locals) {
  const user = locals.user;
  const session = locals.session;
  
  if (!user || !session) {
    throw error(401, { 
      message: 'Authentication required',
      code: 'UNAUTHENTICATED' 
    });
  }
  
  return { user, session };
}

// Validation schema for case creation with enhanced fields
const createCaseSchema = z.object({
  caseNumber: z.string().min(1, 'Case number is required'),
  title: z.string().min(1, 'Title is required'),
  description: z.string().optional(),
  priority: z.enum(['low', 'medium', 'high', 'critical']),
  status: z.enum(['draft', 'open', 'in_progress', 'review', 'closed']).default('draft'),
  category: z.string().default('general'),
  assignedAttorney: z.string().optional(),
  tags: z.array(z.string()).default([]),
  metadata: z.record(z.any()).default({})
});

// Validation schema for case updates
const updateCaseSchema = z.object({
  title: z.string().optional(),
  description: z.string().optional(),
  priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
  status: z.enum(['draft', 'open', 'in_progress', 'review', 'closed']).optional(),
  category: z.string().optional(),
  assignedAttorney: z.string().optional(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional()
});

// GET: Retrieve cases (authenticated users only see their own cases or cases assigned to them)
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    // Authenticate user
    const { user } = await getAuthenticatedUser(locals);

    const caseId = url.searchParams.get('id');
    const status = url.searchParams.get('status');
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '10'), 100); // Cap at 100
    const offset = parseInt(url.searchParams.get('offset') || '0');
    const search = url.searchParams.get('search');

    if (caseId) {
      // Get specific case with related data (only if user owns it or is assigned)
      const caseData = await db
        .select({
          id: cases.id,
          caseNumber: cases.caseNumber,
          title: cases.title,
          description: cases.description,
          priority: cases.priority,
          status: cases.status,
          assignedAttorney: cases.assigned_attorney,
          createdBy: cases.createdBy,
          createdAt: cases.created_at,
          updatedAt: cases.updated_at,
          metadata: cases.metadata
        })
        .from(cases)
        .where(
          and(
            eq(cases.id, caseId),
            or(
              eq(cases.createdBy, user.id), // User created the case
              eq(cases.assigned_attorney, user.id) // User is assigned to the case
            )
          )
        )
        .limit(1);
      
      if (caseData.length === 0) {
        throw error(404, { 
          message: 'Case not found or access denied',
          code: 'CASE_NOT_FOUND' 
        });
      }

      // Get related documents
      const documents = await db
        .select()
        .from(caseDocuments)
        .where(eq(caseDocuments.caseId, caseId))
        .orderBy(desc(caseDocuments.created_at));
      
      // Get activities
      const activities = await db
        .select()
        .from(caseActivities)
        .where(eq(caseActivities.caseId, caseId))
        .orderBy(desc(caseActivities.timestamp))
        .limit(50); // Limit activity history
      
      // Get timeline events
      const timeline = await db
        .select()
        .from(caseTimeline)
        .where(eq(caseTimeline.caseId, caseId))
        .orderBy(desc(caseTimeline.timestamp));

      return json({
        success: true,
        data: {
          ...caseData[0],
          documents,
          activities,
          timeline
        }
      });
    }

    // Build query for cases the user has access to
    const whereConditions = [
      or(
        eq(cases.createdBy, user.id),
        eq(cases.assigned_attorney, user.id)
      )
    ];

    if (status) {
      whereConditions.push(eq(cases.status, status));
    }

    // If search term provided, search in title and description
    if (search) {
      whereConditions.push(
        or(
          ilike(cases.title, `%${search}%`),
          ilike(cases.description, `%${search}%`)
        )
      );
    }

    // Get cases with pagination
    const result = await db
      .select({
        id: cases.id,
        caseNumber: cases.caseNumber,
        title: cases.title,
        description: cases.description,
        priority: cases.priority,
        status: cases.status,
        assignedAttorney: cases.assigned_attorney,
        createdBy: cases.createdBy,
        createdAt: cases.created_at,
        updatedAt: cases.updated_at,
        metadata: cases.metadata
      })
      .from(cases)
      .where(and(...whereConditions))
      .orderBy(desc(cases.updated_at))
      .limit(limit)
      .offset(offset);

    // Get total count for pagination
    const [{ total }] = await db
      .select({ total: count() })
      .from(cases)
      .where(and(...whereConditions));

    return json({
      success: true,
      data: result,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
        currentPage: Math.floor(offset / limit) + 1,
        totalPages: Math.ceil(total / limit)
      },
      user: {
        id: user.id,
        email: user.email,
        role: user.role
      }
    });

  } catch (err: any) {
    if (err.status) {
      // Re-throw SvelteKit errors
      throw err;
    }
    
    console.error('Error fetching cases:', err);
    throw error(500, { 
      message: 'Failed to fetch cases',
      code: 'FETCH_ERROR'
    });
  }
};

// POST: Create a new case (authenticated)
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Authenticate user
    const { user, session } = await getAuthenticatedUser(locals);

    // Parse and validate request body
    const body = await request.json();
    const validatedData = createCaseSchema.parse(body);

    // Generate unique ID and timestamps
    const caseId = crypto.randomUUID();
    const now = new Date();

    // Generate embedding for case content (title + description) using pgvector
    const caseContent = `${validatedData.title} ${validatedData.description || ''}`;
    let caseEmbedding = null;
    
    try {
      // Generate semantic embedding for similarity search
      caseEmbedding = await generateEmbedding(caseContent);
    } catch (embeddingError) {
      console.warn('Failed to generate case embedding:', embeddingError);
      // Continue without embedding - not critical for case creation
    }

    // Check for duplicate case number for this user
    const existingCase = await db
      .select({ id: cases.id })
      .from(cases)
      .where(
        and(
          eq(cases.caseNumber, validatedData.caseNumber),
          eq(cases.createdBy, user.id)
        )
      )
      .limit(1);

    if (existingCase.length > 0) {
      throw error(400, { 
        message: 'Case number already exists',
        code: 'DUPLICATE_CASE_NUMBER' 
      });
    }

    // Insert case into database
    const newCase = await db.insert(cases).values({
      id: caseId,
      caseNumber: validatedData.caseNumber,
      title: validatedData.title,
      description: validatedData.description || '',
      priority: validatedData.priority,
      status: validatedData.status,
      assigned_attorney: validatedData.assignedAttorney || null,
      createdBy: user.id,
      userId: user.id, // For compatibility
      metadata: {
        ...validatedData.metadata,
        tags: validatedData.tags,
        category: validatedData.category,
        createdByName: `${user.firstName || ''} ${user.lastName || ''}`.trim() || user.email,
        embedding: caseEmbedding ? true : false // Flag if embedding was generated
      },
      created_at: now,
      updated_at: now
    }).returning();

    // Create initial activity log
    await db.insert(caseActivities).values({
      id: crypto.randomUUID(),
      caseId: caseId,
      type: 'case_created',
      description: `Case ${validatedData.caseNumber} "${validatedData.title}" created`,
      userId: user.id,
      timestamp: now,
      metadata: {
        title: validatedData.title,
        priority: validatedData.priority,
        category: validatedData.category,
        userEmail: user.email
      }
    });

    // Create initial timeline event
    await db.insert(caseTimeline).values({
      id: crypto.randomUUID(),
      caseId: caseId,
      event: 'Case Created',
      description: `Case ${validatedData.caseNumber} "${validatedData.title}" was created by ${user.email}`,
      timestamp: now,
      type: 'milestone',
      metadata: {
        createdBy: user.id,
        createdByEmail: user.email,
        priority: validatedData.priority
      }
    });

    // If embedding was generated, store it in the embedding cache for future use
    if (caseEmbedding && caseEmbedding.length > 0) {
      try {
        const { embeddingCache } = await import('$lib/server/db/schema');
        const contentHash = await hashContent(caseContent);
        
        await db.insert(embeddingCache).values({
          id: crypto.randomUUID(),
          content_hash: contentHash,
          embedding: caseEmbedding,
          model_name: 'nomic-embed-text',
          metadata: {
            entityType: 'case',
            entityId: caseId,
            content: caseContent.substring(0, 500) // Store first 500 chars for reference
          },
          created_at: now,
          expires_at: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days
        }).onConflictDoNothing(); // Don't error if hash already exists
      } catch (cacheError) {
        console.warn('Failed to cache embedding:', cacheError);
        // Non-critical, continue
      }
    }

    return json({
      success: true,
      message: 'Case created successfully',
      data: {
        ...newCase[0],
        hasEmbedding: !!caseEmbedding,
        createdBy: {
          id: user.id,
          email: user.email,
          name: `${user.firstName || ''} ${user.lastName || ''}`.trim() || user.email
        }
      }
    }, { status: 201 });

  } catch (err: any) {
    if (err.status) {
      // Re-throw SvelteKit errors
      throw err;
    }
    
    console.error('Error creating case:', err);
    
    if (err instanceof z.ZodError) {
      throw error(400, { 
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        details: err.errors 
      });
    }

    throw error(500, { 
      message: 'Failed to create case',
      code: 'CREATE_ERROR'
    });
  }
};

// Helper function to hash content for embedding cache
async function hashContent(content: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const hash = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(hash))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

// PUT: Update an existing case (authenticated, owner or assigned only)
export const PUT: RequestHandler = async ({ request, url, locals }) => {
  try {
    // Authenticate user
    const { user } = await getAuthenticatedUser(locals);

    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      throw error(400, { 
        message: 'Case ID is required',
        code: 'MISSING_CASE_ID' 
      });
    }

    // Parse and validate request body
    const body = await request.json();
    const validatedData = updateCaseSchema.parse(body);
    const now = new Date();

    // Check if user has permission to update this case
    const existingCase = await db
      .select({
        id: cases.id,
        title: cases.title,
        createdBy: cases.createdBy,
        assigned_attorney: cases.assigned_attorney
      })
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (existingCase.length === 0) {
      throw error(404, { 
        message: 'Case not found',
        code: 'CASE_NOT_FOUND' 
      });
    }

    const caseRecord = existingCase[0];
    const hasPermission = caseRecord.createdBy === user.id || 
                         caseRecord.assigned_attorney === user.id ||
                         user.role === 'admin';

    if (!hasPermission) {
      throw error(403, { 
        message: 'Access denied - you can only update cases you created or are assigned to',
        code: 'ACCESS_DENIED' 
      });
    }

    // Generate new embedding if title or description changed
    let newEmbedding = null;
    const titleChanged = validatedData.title && validatedData.title !== caseRecord.title;
    const descriptionChanged = validatedData.description !== undefined;
    
    if (titleChanged || descriptionChanged) {
      try {
        const newTitle = validatedData.title || caseRecord.title;
        const newDescription = validatedData.description || '';
        const newContent = `${newTitle} ${newDescription}`;
        newEmbedding = await generateEmbedding(newContent);
      } catch (embeddingError) {
        console.warn('Failed to generate updated embedding:', embeddingError);
      }
    }

    // Build update object with only provided fields
    const updateData: any = {
      updated_at: now
    };

    if (validatedData.title) updateData.title = validatedData.title;
    if (validatedData.description !== undefined) updateData.description = validatedData.description;
    if (validatedData.priority) updateData.priority = validatedData.priority;
    if (validatedData.status) updateData.status = validatedData.status;
    if (validatedData.category) updateData.category = validatedData.category;
    if (validatedData.assignedAttorney !== undefined) updateData.assigned_attorney = validatedData.assignedAttorney;

    // Update metadata
    if (validatedData.metadata || validatedData.tags || newEmbedding) {
      updateData.metadata = {
        ...(caseRecord as any).metadata,
        ...validatedData.metadata,
        ...(validatedData.tags && { tags: validatedData.tags }),
        ...(newEmbedding && { embedding: true }),
        updatedBy: user.id,
        updatedByEmail: user.email,
        lastUpdated: now.toISOString()
      };
    }

    // Update case in database
    const updatedCase = await db
      .update(cases)
      .set(updateData)
      .where(eq(cases.id, caseId))
      .returning();

    // Log the update activity with detailed changes
    const changedFields = Object.keys(validatedData).filter(key => 
      validatedData[key] !== undefined && validatedData[key] !== null
    );

    await db.insert(caseActivities).values({
      id: crypto.randomUUID(),
      caseId: caseId,
      type: 'case_updated',
      description: `Case updated by ${user.email}. Changed: ${changedFields.join(', ')}`,
      userId: user.id,
      timestamp: now,
      metadata: {
        changes: validatedData,
        changedFields,
        updatedBy: user.id,
        updatedByEmail: user.email,
        hasNewEmbedding: !!newEmbedding
      }
    });

    // Update embedding cache if embedding was regenerated
    if (newEmbedding && newEmbedding.length > 0) {
      try {
        const { embeddingCache } = await import('$lib/server/db/schema');
        const newTitle = validatedData.title || caseRecord.title;
        const newDescription = validatedData.description || '';
        const newContent = `${newTitle} ${newDescription}`;
        const contentHash = await hashContent(newContent);
        
        await db.insert(embeddingCache).values({
          id: crypto.randomUUID(),
          content_hash: contentHash,
          embedding: newEmbedding,
          model_name: 'nomic-embed-text',
          metadata: {
            entityType: 'case',
            entityId: caseId,
            content: newContent.substring(0, 500),
            action: 'updated'
          },
          created_at: now,
          expires_at: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days
        }).onConflictDoUpdate({
          target: [embeddingCache.content_hash],
          set: {
            embedding: newEmbedding,
            updated_at: now
          }
        });
      } catch (cacheError) {
        console.warn('Failed to update embedding cache:', cacheError);
      }
    }

    return json({
      success: true,
      message: 'Case updated successfully',
      data: {
        ...updatedCase[0],
        hasNewEmbedding: !!newEmbedding,
        updatedBy: {
          id: user.id,
          email: user.email,
          name: `${user.firstName || ''} ${user.lastName || ''}`.trim() || user.email
        },
        changedFields
      }
    });

  } catch (err: any) {
    if (err.status) {
      throw err;
    }
    
    console.error('Error updating case:', err);
    
    if (err instanceof z.ZodError) {
      throw error(400, { 
        message: 'Validation failed',
        code: 'VALIDATION_ERROR',
        details: err.errors 
      });
    }

    throw error(500, { 
      message: 'Failed to update case',
      code: 'UPDATE_ERROR'
    });
  }
};

// DELETE: Delete a case (authenticated, owner or admin only)
export const DELETE: RequestHandler = async ({ url, locals }) => {
  try {
    // Authenticate user
    const { user } = await getAuthenticatedUser(locals);

    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      throw error(400, { 
        message: 'Case ID is required',
        code: 'MISSING_CASE_ID' 
      });
    }

    // Check if user has permission to delete this case
    const existingCase = await db
      .select({
        id: cases.id,
        title: cases.title,
        caseNumber: cases.caseNumber,
        createdBy: cases.createdBy,
        assigned_attorney: cases.assigned_attorney,
        status: cases.status
      })
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (existingCase.length === 0) {
      throw error(404, { 
        message: 'Case not found',
        code: 'CASE_NOT_FOUND' 
      });
    }

    const caseRecord = existingCase[0];
    
    // Only case creator or admin can delete cases
    const hasPermission = caseRecord.createdBy === user.id || user.role === 'admin';

    if (!hasPermission) {
      throw error(403, { 
        message: 'Access denied - only case creators or administrators can delete cases',
        code: 'DELETE_ACCESS_DENIED' 
      });
    }

    // Prevent deletion of cases that are in progress (safety check)
    const protectedStatuses = ['in_progress', 'review'];
    if (protectedStatuses.includes(caseRecord.status) && user.role !== 'admin') {
      throw error(400, { 
        message: `Cannot delete case with status '${caseRecord.status}'. Please close the case first or contact an administrator.`,
        code: 'CASE_STATUS_PROTECTED' 
      });
    }

    const now = new Date();

    // Log deletion activity before actual deletion
    await db.insert(caseActivities).values({
      id: crypto.randomUUID(),
      caseId: caseId,
      type: 'case_deleted',
      description: `Case "${caseRecord.title}" (${caseRecord.caseNumber}) deleted by ${user.email}`,
      userId: user.id,
      timestamp: now,
      metadata: {
        deletedCase: {
          id: caseRecord.id,
          title: caseRecord.title,
          caseNumber: caseRecord.caseNumber,
          status: caseRecord.status
        },
        deletedBy: user.id,
        deletedByEmail: user.email,
        isAdmin: user.role === 'admin'
      }
    });

    // Delete related data in proper order (maintain referential integrity)
    const deleteResults = await Promise.allSettled([
      db.delete(caseTimeline).where(eq(caseTimeline.caseId, caseId)),
      db.delete(caseActivities).where(eq(caseActivities.caseId, caseId)),
      db.delete(caseDocuments).where(eq(caseDocuments.caseId, caseId))
    ]);

    // Log any failures in related data deletion
    deleteResults.forEach((result, index) => {
      if (result.status === 'rejected') {
        const tableName = ['timeline', 'activities', 'documents'][index];
        console.warn(`Failed to delete ${tableName} for case ${caseId}:`, result.reason);
      }
    });

    // Delete the case itself
    const deletedCase = await db
      .delete(cases)
      .where(eq(cases.id, caseId))
      .returning();

    if (deletedCase.length === 0) {
      throw error(500, { 
        message: 'Failed to delete case from database',
        code: 'DELETE_FAILED' 
      });
    }

    // Clean up embedding cache for this case
    try {
      const { embeddingCache } = await import('$lib/server/db/schema');
      await db
        .delete(embeddingCache)
        .where(eq(embeddingCache.metadata, { entityType: 'case', entityId: caseId }));
    } catch (cacheError) {
      console.warn('Failed to clean up embedding cache:', cacheError);
      // Non-critical, continue
    }

    return json({
      success: true,
      message: `Case "${caseRecord.title}" deleted successfully`,
      data: {
        id: caseId,
        title: caseRecord.title,
        caseNumber: caseRecord.caseNumber,
        deletedBy: {
          id: user.id,
          email: user.email,
          name: `${user.firstName || ''} ${user.lastName || ''}`.trim() || user.email
        },
        deletedAt: now.toISOString(),
        relatedDataDeleted: {
          timeline: deleteResults[0].status === 'fulfilled',
          activities: deleteResults[1].status === 'fulfilled', 
          documents: deleteResults[2].status === 'fulfilled'
        }
      }
    });

  } catch (err: any) {
    if (err.status) {
      throw err;
    }
    
    console.error('Error deleting case:', err);
    throw error(500, { 
      message: 'Failed to delete case',
      code: 'DELETE_ERROR'
    });
  }
};