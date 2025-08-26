import { json, error } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { cases } from '$lib/server/schema';
import { eq, desc } from 'drizzle-orm';
import type { RequestHandler } from './$types';

// GET /api/cases - List cases for authenticated user
export const GET: RequestHandler = async ({ locals, url }) => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  try {
    const limit = Number(url.searchParams.get('limit')) || 10;
    const offset = Number(url.searchParams.get('offset')) || 0;

    const userCases = await db
      .select()
      .from(cases)
      .where(eq(cases.userId, locals.user.id))
      .orderBy(desc(cases.updatedAt))
      .limit(limit)
      .offset(offset);

    return json({
      success: true,
      data: userCases,
      pagination: {
        limit,
        offset,
        total: userCases.length
      }
    });

  } catch (err) {
    console.error('Error fetching cases:', err);
    throw error(500, 'Failed to fetch cases');
  }
};

// POST /api/cases - Create new case
export const POST: RequestHandler = async ({ locals, request }) => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  try {
    const body = await request.json();
    const { title, description, caseNumber, status = 'open' } = body;

    if (!title || !caseNumber) {
      throw error(400, 'Title and case number are required');
    }

    const [newCase] = await db
      .insert(cases)
      .values({
        title,
        description: description || '',
        caseNumber,
        status,
        userId: locals.user.id,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      .returning();

    return json({
      success: true,
      data: newCase
    }, { status: 201 });

  } catch (err) {
    console.error('Error creating case:', err);
    throw error(500, 'Failed to create case');
  }
};