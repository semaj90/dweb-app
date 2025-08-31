import { db } from '$lib/server/db';
import { cases, evidence } from '$lib/server/db/index';
import { eq } from 'drizzle-orm';
import { error, redirect } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals, params }): Promise<any> => {
  // Check if user is authenticated
  if (!locals.user) {
    throw redirect(302, '/auth/login');
  }

  const caseId = Number(params.id);
  
  // Validate case ID
  if (isNaN(caseId)) {
    throw error(400, 'Invalid case ID');
  }

  try {
    // Fetch the case
    const [caseData] = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!caseData) {
      throw error(404, 'Case not found');
    }

    // Check if user has access to this case
    if (caseData.userId !== locals.user.id) {
      throw error(403, 'Access denied');
    }

    // Fetch evidence for this case
    const evidenceList = await db
      .select()
      .from(evidence)
      .where(eq(evidence.caseId, caseId))
      .orderBy(evidence.createdAt);

    // Fetch report for this case (if exists)
    const [report] = await db
      .select()
      .from(reports)
      .where(eq(reports.caseId, caseId))
      .limit(1);

    return {
      user: locals.user,
      case: caseData,
      evidence: evidenceList,
      report: report || null
    };

  } catch (err: any) {
    console.error('Error loading case data:', err);
    
    // Re-throw known errors
    if (err.status) {
      throw err;
    }
    
    // Handle unknown errors
    throw error(500, 'Failed to load case data');
  }
};