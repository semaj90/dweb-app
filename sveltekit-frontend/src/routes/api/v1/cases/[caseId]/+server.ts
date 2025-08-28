/**
 * Individual Case CRUD API - Clean PostgreSQL + Drizzle Implementation
 * GET, PUT, DELETE operations for specific cases
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/server/db/client';
import { cases, evidence, users } from '$lib/db/schema';
import { eq, and } from 'drizzle-orm';
import { redis } from '$lib/server/cache/redis-service';

// GET /api/v1/cases/[caseId] - Retrieve specific case
export const GET: RequestHandler = async ({ params, getClientAddress }) => {
  try {
    const { caseId } = params;

    if (!caseId) {
      return json({
        success: false,
        error: 'caseId parameter is required'
      }, { status: 400 });
    }

    console.log(`🔍 GET /api/v1/cases/${caseId}`);

    // Try cache first
    const cacheKey = `case:${caseId}`;
    const cachedCase = await redis.get(cacheKey);
    
    if (cachedCase) {
      console.log(`✅ Case retrieved from cache: ${caseId}`);
      return json({
        success: true,
        data: { case: JSON.parse(cachedCase) },
        metadata: {
          timestamp: Date.now(),
          clientAddress: getClientAddress(),
          cacheStatus: 'hit'
        }
      });
    }

    // Query database with joins
    const [caseResult] = await db
      .select({
        id: cases.id,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        priority: cases.priority,
        caseNumber: cases.caseNumber,
        assignedTo: cases.assignedTo,
        createdBy: cases.createdBy,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt,
        metadata: cases.metadata,
        // Join user info
        assignedUser: {
          id: users.id,
          name: users.name,
          email: users.email
        }
      })
      .from(cases)
      .leftJoin(users, eq(cases.assignedTo, users.id))
      .where(eq(cases.id, caseId));

    if (!caseResult) {
      return json({
        success: false,
        error: 'Case not found'
      }, { status: 404 });
    }

    // Cache the result
    await redis.setex(cacheKey, 3600, JSON.stringify(caseResult));

    const response = {
      success: true,
      data: { case: caseResult },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        cacheStatus: 'miss'
      }
    };

    console.log(`✅ Case retrieved: ${caseResult.title}`);
    return json(response);

  } catch (error) {
    console.error(`❌ GET /api/v1/cases/${params.caseId} error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to retrieve case',
      metadata: { timestamp: Date.now() }
    }, { status: 500 });
  }
};

// PUT /api/v1/cases/[caseId] - Update specific case
export const PUT: RequestHandler = async ({ params, request, getClientAddress }) => {
  try {
    const { caseId } = params;

    if (!caseId) {
      return json({
        success: false,
        error: 'caseId parameter is required'
      }, { status: 400 });
    }

    const updates = await request.json();

    // Remove fields that shouldn't be updated directly
    const { id, createdAt, caseNumber, ...validUpdates } = updates;
    validUpdates.updatedAt = new Date();

    console.log(`📝 PUT /api/v1/cases/${caseId} - Updating fields:`, Object.keys(validUpdates));

    // Update the case
    const [updatedCase] = await db
      .update(cases)
      .set(validUpdates)
      .where(eq(cases.id, caseId))
      .returning();

    if (!updatedCase) {
      return json({
        success: false,
        error: 'Case not found'
      }, { status: 404 });
    }

    // Invalidate cache
    const cacheKey = `case:${caseId}`;
    await redis.del(cacheKey);

    // Publish update event
    await redis.publish('case:updated', JSON.stringify({
      caseId: updatedCase.id,
      title: updatedCase.title,
      status: updatedCase.status,
      fieldsUpdated: Object.keys(validUpdates),
      timestamp: Date.now()
    }));

    const response = {
      success: true,
      data: { case: updatedCase },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        operation: 'update',
        fieldsUpdated: Object.keys(validUpdates)
      }
    };

    console.log(`✅ Case updated: ${caseId}`);
    return json(response);

  } catch (error) {
    console.error(`❌ PUT /api/v1/cases/${params.caseId} error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to update case',
      metadata: { timestamp: Date.now() }
    }, { status: 500 });
  }
};

// DELETE /api/v1/cases/[caseId] - Delete specific case (soft delete)
export const DELETE: RequestHandler = async ({ params, getClientAddress }) => {
  try {
    const { caseId } = params;

    if (!caseId) {
      return json({
        success: false,
        error: 'caseId parameter is required'
      }, { status: 400 });
    }

    console.log(`🗑️ DELETE /api/v1/cases/${caseId}`);

    // Soft delete by updating status to 'archived'
    const [archivedCase] = await db
      .update(cases)
      .set({ 
        status: 'archived',
        updatedAt: new Date()
      })
      .where(eq(cases.id, caseId))
      .returning();

    if (!archivedCase) {
      return json({
        success: false,
        error: 'Case not found'
      }, { status: 404 });
    }

    // Invalidate cache
    const cacheKey = `case:${caseId}`;
    await redis.del(cacheKey);

    // Publish archive event
    await redis.publish('case:archived', JSON.stringify({
      caseId: archivedCase.id,
      title: archivedCase.title,
      timestamp: Date.now()
    }));

    const response = {
      success: true,
      data: {
        message: 'Case archived successfully',
        caseId,
        title: archivedCase.title
      },
      metadata: {
        timestamp: Date.now(),
        clientAddress: getClientAddress(),
        operation: 'soft_delete'
      }
    };

    console.log(`✅ Case archived: ${caseId}`);
    return json(response);

  } catch (error) {
    console.error(`❌ DELETE /api/v1/cases/${params.caseId} error:`, error);
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to delete case',
      metadata: { timestamp: Date.now() }
    }, { status: 500 });
  }
};