import type { RequestHandler } from '@sveltejs/kit';
import { json, error } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { evidence } from '$lib/server/db/schema-postgres-enhanced';
import { eq } from 'drizzle-orm';

export const DELETE: RequestHandler = async ({ params }) => {
  try {
    const evidenceId = params.id;
    
    if (!evidenceId) {
      throw error(400, 'Evidence ID is required');
    }

    // Delete evidence from database
    const deletedEvidence = await db
      .delete(evidence)
      .where(eq(evidence.id, evidenceId))
      .returning();

    if (deletedEvidence.length === 0) {
      throw error(404, 'Evidence not found');
    }

    return json({ 
      success: true, 
      message: 'Evidence deleted successfully',
      id: evidenceId 
    }, { status: 200 });
  } catch (err) {
    console.error('Evidence deletion error:', err);
    
    if (err instanceof Response) {
      throw err;
    }
    
    return json({ 
      success: false, 
      error: 'Failed to delete evidence' 
    }, { status: 500 });
  }
};