
import { ensureError } from '$lib/utils/ensure-error';
import { json, error } from '@sveltejs/kit';
import { db } from '$lib/database/connection';
import { cases, users } from '$lib/db/schema';
import { eq, and, desc } from 'drizzle-orm';
import { nanoid } from 'nanoid';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// GET - Fetch cases
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const limit = parseInt(url.searchParams.get('limit') || '10');
    const offset = parseInt(url.searchParams.get('offset') || '0');
    const status = url.searchParams.get('status');
    const priority = url.searchParams.get('priority');

    // Build query conditions
    const conditions = [];
    if (status) conditions.push(eq(cases.status, status));
    if (priority) conditions.push(eq(cases.priority, priority));

    // Query cases with optional filters
    const casesList = await db
      .select({
        id: cases.id,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        priority: cases.priority,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt,
        createdBy: cases.createdBy
      })
      .from(cases)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(cases.updatedAt))
      .limit(limit)
      .offset(offset);

    return json({
      success: true,
      data: casesList,
      pagination: {
        limit,
        offset,
        total: casesList.length
      }
    });

  } catch (err: any) {
    console.error('Error fetching cases:', err);
    return error(500, ensureError({
      message: 'Failed to fetch cases'
    }));
  }
};

// POST - Create new case
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    
    // Validate required fields
    if (!body.title || !body.description) {
      return error(400, ensureError({
        message: 'Title and description are required'
      }));
    }

    // Get current user (from auth or default)
    const currentUserId = locals.user?.id || 'system';

    // Generate case ID
    const caseId = `CASE-${new Date().getFullYear()}-${nanoid(6).toUpperCase()}`;

    // Prepare case data
    const newCase = {
      id: caseId,
      title: body.title,
      description: body.description,
      status: body.status || 'active',
      priority: body.priority || 'medium',
      createdBy: currentUserId,
      tags: body.tags || [],
      metadata: body.metadata || {}
    };

    // Insert into database
    const insertedCase = await db
      .insert(cases)
      .values(newCase)
      .returning({
        id: cases.id,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        priority: cases.priority,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt
      });

    // Return created case
    return json({
      success: true,
      data: insertedCase[0],
      message: 'Case created successfully'
    }, { status: 201 });

  } catch (err: any) {
    console.error('Error creating case:', err);
    
    // Handle specific database errors
    if (err.code === '23505') { // Unique constraint violation
      return error(409, ensureError({
        message: 'Case with this ID already exists'
      }));
    }

    return error(500, ensureError({
      message: 'Failed to create case'
    }));
  }
};

// PUT - Update existing case
export const PUT: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    
    if (!body.id) {
      return error(400, ensureError({
        message: 'Case ID is required for updates'
      }));
    }

    // Check if case exists
    const existingCase = await db
      .select({ id: cases.id })
      .from(cases)
      .where(eq(cases.id, body.id))
      .limit(1);

    if (existingCase.length === 0) {
      return error(404, ensureError({
        message: 'Case not found'
      }));
    }

    // Prepare update data
    const updateData: any = {
      updatedAt: new Date()
    };

    if (body.title) updateData.title = body.title;
    if (body.description) updateData.description = body.description;
    if (body.status) updateData.status = body.status;
    if (body.priority) updateData.priority = body.priority;
    if (body.tags) updateData.tags = body.tags;
    if (body.metadata) updateData.metadata = body.metadata;

    // Update case
    const updatedCase = await db
      .update(cases)
      .set(updateData)
      .where(eq(cases.id, body.id))
      .returning({
        id: cases.id,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        priority: cases.priority,
        updatedAt: cases.updatedAt
      });

    return json({
      success: true,
      data: updatedCase[0],
      message: 'Case updated successfully'
    });

  } catch (err: any) {
    console.error('Error updating case:', err);
    return error(500, ensureError({
      message: 'Failed to update case'
    }));
  }
};

// DELETE - Delete case
export const DELETE: RequestHandler = async ({ url, locals }) => {
  try {
    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      return error(400, ensureError({
        message: 'Case ID is required'
      }));
    }

    // Check if case exists
    const existingCase = await db
      .select({ id: cases.id })
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (existingCase.length === 0) {
      return error(404, ensureError({
        message: 'Case not found'
      }));
    }

    // Soft delete by updating status
    await db
      .update(cases)
      .set({ 
        status: 'deleted',
        updatedAt: new Date()
      })
      .where(eq(cases.id, caseId));

    return json({
      success: true,
      message: 'Case deleted successfully'
    });

  } catch (err: any) {
    console.error('Error deleting case:', err);
    return error(500, ensureError({
      message: 'Failed to delete case'
    }));
  }
};