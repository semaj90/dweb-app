import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { cases } from '$lib/database/schema/legal-documents';
import { eq, desc } from 'drizzle-orm';
import { z } from 'zod';
import { randomUUID } from 'crypto';

// Validation schema
const caseSchema = z.object({
  caseNumber: z.string().min(1),
  title: z.string().min(1),
  description: z.string().optional(),
  clientName: z.string().min(1),
  opposingParty: z.string().optional(),
  jurisdiction: z.string().default('federal'),
  courtName: z.string().optional(),
  judgeAssigned: z.string().optional(),
  caseType: z.enum(['civil', 'criminal', 'administrative', 'appellate', 'arbitration']),
  practiceArea: z.string(),
  priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium'),
  status: z.enum(['active', 'pending', 'closed', 'archived', 'on_hold']).default('active')
});

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const user = locals.user || { id: 'system' };
    const caseId = url.searchParams.get('id');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    const offset = parseInt(url.searchParams.get('offset') || '0');

    if (caseId) {
      const result = await db
        .select()
        .from(cases)
        .where(eq(cases.id, caseId))
        .limit(1);

      if (result.length === 0) {
        return json({ error: 'Case not found' }, { status: 404 });
      }

      return json({ success: true, data: result[0] });
    }

    const results = await db
      .select()
      .from(cases)
      .orderBy(desc(cases.createdAt))
      .limit(limit)
      .offset(offset);

    return json({ success: true, data: results });
  } catch (error) {
    console.error('Error fetching cases:', error);
    return json({ error: 'Failed to fetch cases' }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const user = locals.user || { id: 'system' };
    const body = await request.json();
    const validated = caseSchema.parse(body);
    const newCase = await db
      .insert(cases)
      .values({
        ...validated,
        id: randomUUID(),
        createdBy: user.id,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      .returning();
      .returning();

return json({ success: true, data: newCase[0] }, { status: 201 });
  } catch (error) {
  console.error('Error creating case:', error);
  if (error instanceof z.ZodError) {
    return json({ error: 'Validation failed', details: error.errors }, { status: 400 });
  }
  return json({ error: 'Failed to create case' }, { status: 500 });
}
};

export const PUT: RequestHandler = async ({ request, url }) => {
  try {
    const caseId = url.searchParams.get('id');
    if (!caseId) {
      return json({ error: 'Case ID required' }, { status: 400 });
    }

    const body = await request.json();
    const updated = await db
      .update(cases)
      .set({ ...body, updatedAt: new Date() })
      .where(eq(cases.id, caseId))
      .returning();

    if (updated.length === 0) {
      return json({ error: 'Case not found' }, { status: 404 });
    }

    return json({ success: true, data: updated[0] });
  } catch (error) {
    console.error('Error updating case:', error);
    return json({ error: 'Failed to update case' }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get('id');
    if (!caseId) {
      return json({ error: 'Case ID required' }, { status: 400 });
    }

    await db.delete(cases).where(eq(cases.id, caseId));
    return json({ success: true, message: 'Case deleted' });
  } catch (error) {
    console.error('Error deleting case:', error);
    return json({ error: 'Failed to delete case' }, { status: 500 });
  }
};