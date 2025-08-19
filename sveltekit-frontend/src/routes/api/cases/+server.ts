// @ts-nocheck
import { json } from '@sveltejs/kit';
// Orphaned content: import type { RequestHandler
import {
db } from "$lib/server/db/drizzle";
// Orphaned content: import { cases, evidence, reports
import {
eq, and, or, ilike, count, desc, asc } from "drizzle-orm";
// Orphaned content: import type { Case
import {
URL } from "url";

// Case with computed counts interface
interface CaseWithCounts extends Case {
  evidenceCount: number;
  documentsCount: number;
}

export const GET: RequestHandler = async ({ url }) => {
  try {
    const search = url.searchParams.get('search');
    const status = url.searchParams.get('status');
    const priority = url.searchParams.get('priority');
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    const offset = (page - 1) * limit;
    
    // Build where conditions
    const whereConditions = [];
    
    if (search) {
      whereConditions.push(
        or(
          ilike(cases.title, `%${search}%`),
          ilike(cases.description, `%${search}%`)
        )
      );
    }
    
    if (status) {
      whereConditions.push(eq(cases.status, status));
    }
    
    if (priority) {
      whereConditions.push(eq(cases.priority, priority));
    }

    // Get cases with evidence and document counts
    const casesQuery = db
      .select({
        id: cases.id,
        caseNumber: cases.caseNumber,
        title: cases.title,
        name: cases.name,
        description: cases.description,
        incidentDate: cases.incidentDate,
        location: cases.location,
        priority: cases.priority,
        status: cases.status,
        category: cases.category,
        dangerScore: cases.dangerScore,
        estimatedValue: cases.estimatedValue,
        jurisdiction: cases.jurisdiction,
        leadProsecutor: cases.leadProsecutor,
        assignedTeam: cases.assignedTeam,
        tags: cases.tags,
        aiSummary: cases.aiSummary,
        aiTags: cases.aiTags,
        metadata: cases.metadata,
        createdBy: cases.createdBy,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt,
        closedAt: cases.closedAt,
        evidenceCount: count(evidence.id),
        documentsCount: count(reports.id)
      })
      .from(cases)
      .leftJoin(evidence, eq(cases.id, evidence.caseId))
      .leftJoin(reports, eq(cases.id, reports.caseId))
      .groupBy(cases.id)
      .orderBy(desc(cases.updatedAt))
      .limit(limit)
      .offset(offset);

    // Add where conditions if any
    if (whereConditions.length > 0) {
      casesQuery.where(and(...whereConditions));
    }

    const caseResults = await casesQuery;
    
    // Get total count for pagination
    const totalQuery = db
      .select({ count: count() })
      .from(cases);
      
    if (whereConditions.length > 0) {
      totalQuery.where(and(...whereConditions));
    }
    
    const [{ count: totalCount }] = await totalQuery;

    return json({
      cases: caseResults,
      total: totalCount,
      page,
      limit,
      totalPages: Math.ceil(totalCount / limit),
      filters: { search, status, priority }
    });
  } catch (error) {
    console.error('Error fetching cases:', error);
    return json(
      { error: 'Failed to fetch cases' },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const data = await request.json();
    const { 
      title, 
      description, 
      priority = 'medium',
      name,
      category,
      location,
      incidentDate,
      jurisdiction,
      tags = [],
      assignedTeam = [],
      createdBy 
    } = data;
    
    if (!title) {
      return json(
        { error: 'Title is required' },
        { status: 400 }
      );
    }
    
    // Generate unique case number
    const caseNumberPrefix = new Date().getFullYear().toString();
    const [{ caseCount }] = await db
      .select({ caseCount: count() })
      .from(cases);
      
    const caseNumber = `CASE-${caseNumberPrefix}-${String(caseCount + 1).padStart(4, '0')}`;

    // Insert new case
    const [newCase] = await db
      .insert(cases)
      .values({
        caseNumber,
        title,
        name: name || title,
        description,
        priority,
        status: 'open',
        category,
        location,
        incidentDate: incidentDate ? new Date(incidentDate) : null,
        jurisdiction,
        tags,
        assignedTeam,
        createdBy,
        dangerScore: 0,
        estimatedValue: null,
        leadProsecutor: null,
        aiSummary: null,
        aiTags: [],
        metadata: {}
      })
      .returning();
    
    // Return with computed counts (initially 0)
    const caseWithCounts: CaseWithCounts = {
      ...newCase,
      evidenceCount: 0,
      documentsCount: 0
    };
    
    return json(caseWithCounts, { status: 201 });
  } catch (error) {
    console.error('Error creating case:', error);
    return json(
      { error: 'Failed to create case' },
      { status: 500 }
    );
  }
};

// Note: PUT and DELETE handlers should be in /api/cases/[caseId]/+server.ts

// Export types for use in other files
export type { CaseWithCounts };