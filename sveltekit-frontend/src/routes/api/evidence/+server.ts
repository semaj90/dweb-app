// @ts-nocheck
import { json } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler
import {
db } from "$lib/server/db/drizzle";
// Orphaned content: import { evidence, cases
import {
eq, and, or, ilike, count, desc, asc } from "drizzle-orm";
// Orphaned content: import type { Evidence
import {
URL } from "url";

export const GET: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get('caseId');
    const type = url.searchParams.get('type');
    const search = url.searchParams.get('search');
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    const offset = (page - 1) * limit;
    
    // Build where conditions
    const whereConditions = [];
    
    if (caseId) {
      whereConditions.push(eq(evidence.caseId, caseId));
    }
    
    if (type) {
      whereConditions.push(eq(evidence.evidenceType, type));
    }
    
    if (search) {
      whereConditions.push(
        or(
          ilike(evidence.title, `%${search}%`),
          ilike(evidence.description, `%${search}%`),
          ilike(evidence.summary, `%${search}%`)
        )
      );
    }

    // Get evidence with pagination
    const evidenceQuery = db
      .select({
        id: evidence.id,
        caseId: evidence.caseId,
        criminalId: evidence.criminalId,
        title: evidence.title,
        description: evidence.description,
        evidenceType: evidence.evidenceType,
        fileType: evidence.fileType,
        subType: evidence.subType,
        fileUrl: evidence.fileUrl,
        fileName: evidence.fileName,
        fileSize: evidence.fileSize,
        mimeType: evidence.mimeType,
        hash: evidence.hash,
        tags: evidence.tags,
        chainOfCustody: evidence.chainOfCustody,
        collectedAt: evidence.collectedAt,
        collectedBy: evidence.collectedBy,
        location: evidence.location,
        labAnalysis: evidence.labAnalysis,
        aiAnalysis: evidence.aiAnalysis,
        aiTags: evidence.aiTags,
        aiSummary: evidence.aiSummary,
        summary: evidence.summary,
        isAdmissible: evidence.isAdmissible,
        confidentialityLevel: evidence.confidentialityLevel,
        canvasPosition: evidence.canvasPosition,
        uploadedBy: evidence.uploadedBy,
        uploadedAt: evidence.uploadedAt,
        updatedAt: evidence.updatedAt
      })
      .from(evidence)
      .orderBy(desc(evidence.uploadedAt))
      .limit(limit)
      .offset(offset);

    // Add where conditions if any
    if (whereConditions.length > 0) {
      evidenceQuery.where(and(...whereConditions));
    }

    const evidenceResults = await evidenceQuery;
    
    // Get total count for pagination
    const totalQuery = db
      .select({ count: count() })
      .from(evidence);
      
    if (whereConditions.length > 0) {
      totalQuery.where(and(...whereConditions));
    }
    
    const [{ count: totalCount }] = await totalQuery;

    return json({
      evidence: evidenceResults,
      total: totalCount,
      page,
      limit,
      totalPages: Math.ceil(totalCount / limit),
      filters: { caseId, type, search }
    });
  } catch (error) {
    console.error('Error fetching evidence:', error);
    return json(
      { error: 'Failed to fetch evidence' },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const data = await request.json();
    const { 
      caseId,
      criminalId,
      title,
      description,
      evidenceType,
      fileType,
      subType,
      fileUrl,
      fileName,
      fileSize,
      mimeType,
      hash,
      tags = [],
      chainOfCustody = [],
      collectedAt,
      collectedBy,
      location,
      labAnalysis = {},
      aiAnalysis = {},
      aiTags = [],
      aiSummary,
      summary,
      isAdmissible = true,
      confidentialityLevel = 'standard',
      canvasPosition = {},
      uploadedBy
    } = data;
    
    if (!caseId || !title || !evidenceType) {
      return json(
        { error: 'Case ID, title, and evidence type are required' },
        { status: 400 }
      );
    }

    // Verify case exists
    const caseExists = await db
      .select({ id: cases.id })
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);
      
    if (caseExists.length === 0) {
      return json(
        { error: 'Case not found' },
        { status: 404 }
      );
    }

    // Insert new evidence
    const [newEvidence] = await db
      .insert(evidence)
      .values({
        caseId,
        criminalId,
        title,
        description,
        evidenceType,
        fileType,
        subType,
        fileUrl,
        fileName,
        fileSize,
        mimeType,
        hash,
        tags,
        chainOfCustody,
        collectedAt: collectedAt ? new Date(collectedAt) : null,
        collectedBy,
        location,
        labAnalysis,
        aiAnalysis,
        aiTags,
        aiSummary,
        summary,
        isAdmissible,
        confidentialityLevel,
        canvasPosition,
        uploadedBy
      })
      .returning();
    
    return json(newEvidence, { status: 201 });
  } catch (error) {
    console.error('Error creating evidence:', error);
    return json(
      { error: 'Failed to create evidence' },
      { status: 500 }
    );
  }
};

// Note: PUT and DELETE handlers should be in /api/evidence/[id]/+server.ts