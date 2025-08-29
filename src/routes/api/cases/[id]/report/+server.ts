import { json, error } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { cases, reports } from '$lib/server/schema';
import { eq, and } from 'drizzle-orm';
import type { RequestHandler } from './$types';

// GET /api/cases/[id]/report - Get report for case
export const GET: RequestHandler = async ({ locals, params }): Promise<any> => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  const caseId = Number(params.id);
  
  if (isNaN(caseId)) {
    throw error(400, 'Invalid case ID');
  }

  try {
    // Verify case ownership
    const [caseData] = await db
      .select()
      .from(cases)
      .where(and(eq(cases.id, caseId), eq(cases.userId, locals.user.id)))
      .limit(1);

    if (!caseData) {
      throw error(404, 'Case not found or access denied');
    }

    // Get report
    const [report] = await db
      .select()
      .from(reports)
      .where(eq(reports.caseId, caseId))
      .limit(1);

    return json({
      success: true,
      data: report || null
    });

  } catch (err: any) {
    console.error('Error fetching report:', err);
    throw error(500, 'Failed to fetch report');
  }
};

// POST /api/cases/[id]/report - Create or update report
export const POST: RequestHandler = async ({ locals, params, request }): Promise<any> => {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  const caseId = Number(params.id);
  
  if (isNaN(caseId)) {
    throw error(400, 'Invalid case ID');
  }

  try {
    const body = await request.json();
    const { title, summary, doc = {} } = body;

    if (!title) {
      throw error(400, 'Report title is required');
    }

    // Verify case ownership
    const [caseData] = await db
      .select()
      .from(cases)
      .where(and(eq(cases.id, caseId), eq(cases.userId, locals.user.id)))
      .limit(1);

    if (!caseData) {
      throw error(404, 'Case not found or access denied');
    }

    // Check if report exists
    const [existingReport] = await db
      .select()
      .from(reports)
      .where(eq(reports.caseId, caseId))
      .limit(1);

    let report;

    if (existingReport) {
      // Update existing report
      [report] = await db
        .update(reports)
        .set({
          title,
          summary: summary || '',
          doc: doc || {},
          updatedAt: new Date()
        })
        .where(eq(reports.id, existingReport.id))
        .returning();
    } else {
      // Create new report
      [report] = await db
        .insert(reports)
        .values({
          title,
          summary: summary || '',
          doc: doc || {},
          caseId,
          createdAt: new Date(),
          updatedAt: new Date()
        })
        .returning();
    }

    return json({
      success: true,
      data: report
    });

  } catch (err: any) {
    console.error('Error saving report:', err);
    throw error(500, 'Failed to save report');
  }
};