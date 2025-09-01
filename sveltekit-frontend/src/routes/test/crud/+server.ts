import type { RequestHandler } from './$types.js';

// API Route Handler for CRUD operations
import { json, error } from '@sveltejs/kit';

// Database imports
import { db } from '$lib/server/db/index';
import { cases } from '$lib/server/db/schema-postgres';
import { eq } from '$lib/server/db/index';
import { URL } from "url";

// DELETE request handler
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      throw error(400, 'Case ID is required');
    }

    console.log('🗑️ API: Deleting case:', caseId);

    // Delete the case
    const [deletedCase] = await db
      .delete(cases)
      .where(eq(cases.id, caseId))
      .returning();

    if (!deletedCase) {
      throw error(404, 'Case not found');
    }

    console.log('✅ API: Case deleted successfully:', deletedCase.id);

    return json({
      success: true,
      message: 'Case deleted successfully',
      deletedId: deletedCase.id
    });

  } catch (err: any) {
    console.error('❌ API: Error deleting case:', err);
    
    if (err.status) {
      throw err; // Re-throw SvelteKit errors
    }
    
    throw error(500, `Internal server error: ${err.message}`);
  }
};

// GET request handler for health check
export const GET: RequestHandler = async () => {
  try {
    // Simple health check
    const startTime = Date.now();
    await db.select().from(cases).limit(1);
    const responseTime = Date.now() - startTime;

    return json({
      status: 'healthy',
      database: {
        connected: true,
        responseTime
      },
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    console.error('❌ API Health check failed:', err);
    
    return json({
      status: 'unhealthy',
      database: {
        connected: false,
        error: err.message
      },
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
};