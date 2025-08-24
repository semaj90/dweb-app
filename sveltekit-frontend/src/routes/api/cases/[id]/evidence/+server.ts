// src/routes/api/cases/[id]/evidence/+server.ts
// API endpoint to get evidence for a specific case

import { json } from '@sveltejs/kit';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { eq, desc } from 'drizzle-orm';
import { evidenceTable } from '$lib/server/schema.js';
import type { RequestHandler } from './$types.js';

const sql = postgres(process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db');
const db = drizzle(sql);

export const GET: RequestHandler = async ({ params }) => {
  try {
    const { id: caseId } = params;
    
    if (!caseId) {
      return json({ error: 'Case ID is required' }, { status: 400 });
    }

    // Get all evidence for the case, ordered by upload date (newest first)
    const evidence = await db
      .select()
      .from(evidenceTable)
      .where(eq(evidenceTable.case_id, caseId))
      .orderBy(desc(evidenceTable.uploaded_at));

    return json({
      success: true,
      evidence,
      count: evidence.length
    });

  } catch (error) {
    console.error('Error fetching evidence:', error);
    return json({ 
      error: 'Failed to fetch evidence',
      details: error.message 
    }, { status: 500 });
  }
};