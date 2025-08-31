import type { RequestHandler } from './$types';

/**
 * AI Summary Save Endpoint
 * Saves legal AI analysis results to PostgreSQL with audit trail
 */

import { json } from '@sveltejs/kit';
import { getUser } from '$lib/server/auth';
import { db } from '$lib/server/db';
import { cases, aiAnalyses } from '$lib/server/db/schema-unified';
import { eq } from 'drizzle-orm';

export interface SaveSummaryRequest {
  caseId: string;
  summary: string;
  metadata?: {
    analysisType?: string;
    model?: string;
    confidence?: number;
    processingTime?: number;
    tokenCount?: number;
    sources?: any[];
  };
}

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    // Authentication check
    const { user } = await getUser({ request, cookies } as any);
    if (!user) {
      return json({ error: 'Authentication required' }, { status: 401 });
    }

    // Parse request body
    const body: SaveSummaryRequest = await request.json();
    const { caseId, summary, metadata = {} } = body;

    // Validate required fields
    if (!caseId || !summary) {
      return json({ 
        error: 'Missing required fields: caseId, summary' 
      }, { status: 400 });
    }

    // Verify case exists and user has access
    const caseRecord = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (caseRecord.length === 0) {
      return json({ error: 'Case not found' }, { status: 404 });
    }

    // Check if user has access to this case
    const userCase = caseRecord[0];
    if (userCase.userId !== user.id && user.role !== 'admin') {
      return json({ error: 'Access denied' }, { status: 403 });
    }

    // Save AI analysis to database
    const analysisRecord = await db
      .insert(aiAnalyses)
      .values({
        caseId,
        userId: user.id,
        analysisType: metadata.analysisType || 'summary',
        model: metadata.model || 'gemma3-legal:latest',
        summary,
        confidence: metadata.confidence || 0.85,
        processingTime: metadata.processingTime || 0,
        tokenCount: metadata.tokenCount || 0,
        sources: metadata.sources || [],
        metadata: {
          userRole: user.role,
          userSpecialties: user.legalSpecialties || [],
          timestamp: new Date().toISOString(),
          ...metadata
        }
      })
      .returning();

    // Update case with latest analysis timestamp
    await db
      .update(cases)
      .set({ 
        updatedAt: new Date(),
        lastAnalysisAt: new Date()
      })
      .where(eq(cases.id, caseId));

    // Log the save operation
    console.log('AI analysis saved:', {
      analysisId: analysisRecord[0].id,
      caseId,
      userId: user.id,
      model: metadata.model,
      confidence: metadata.confidence
    });

    return json({
      success: true,
      analysisId: analysisRecord[0].id,
      message: 'Summary saved successfully'
    });

  } catch (error: any) {
    console.error('Summary save error:', error);
    
    return json({
      error: 'Failed to save summary',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};