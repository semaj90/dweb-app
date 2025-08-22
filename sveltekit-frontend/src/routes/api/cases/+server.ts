import type { RequestHandler } from '@sveltejs/kit';
import { json, error } from "@sveltejs/kit";
import { z } from "zod";
import { db } from '$lib/database/connection';
import { cases, users } from '$lib/database/schema';
import { eq, desc, ilike, and, count } from 'drizzle-orm';

// Enhanced case schema for creation/updates
const caseSchema = z.object({
  title: z.string().min(1, "Case title is required"),
  description: z.string().optional(),
  priority: z.enum(["low", "medium", "high", "critical"]).default("medium"),
  status: z.enum(["active", "closed", "archived"]).default("active")
});

// GET - List cases with advanced search and filtering
export const GET: RequestHandler = async ({ url, locals }) => {
  const startTime = Date.now();
  
  try {
    // Extract query parameters
    const search = url.searchParams.get("search");
    const status = url.searchParams.get("status");
    const priority = url.searchParams.get("priority");
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");

    console.log("🔍 Cases search request:", { search, status, priority, limit, offset });

    // Build query conditions
    const conditions = [];
    
    if (status) {
      conditions.push(eq(cases.status, status));
    }
    if (priority) {
      conditions.push(eq(cases.priority, priority));
    }
    if (search && search.trim()) {
      conditions.push(ilike(cases.title, `%${search}%`));
    }

    // Execute database query
    const [results, totalCount] = await Promise.all([
      db.select({
        id: cases.id,
        title: cases.title,
        description: cases.description,
        status: cases.status,
        priority: cases.priority,
        caseNumber: cases.caseNumber,
        createdAt: cases.createdAt,
        updatedAt: cases.updatedAt,
        createdBy: users.firstName,
        createdByLastName: users.lastName
      })
      .from(cases)
      .leftJoin(users, eq(cases.createdBy, users.id))
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(cases.createdAt))
      .limit(limit)
      .offset(offset),
      
      db.select({ count: count() })
      .from(cases)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
    ]);

    const total = totalCount[0]?.count || 0;

    const response = {
      success: true,
      data: results,
      pagination: {
        total,
        limit,
        offset,
        hasNext: offset + limit < total,
        hasPrev: offset > 0
      },
      search: search ? {
        term: search,
        resultsCount: results.length,
        processingTime: Date.now() - startTime
      } : null,
      processingTime: Date.now() - startTime
    };

    console.log(`✅ Cases search completed: ${results.length}/${total} results in ${Date.now() - startTime}ms`);
    
    return json(response);

  } catch (err) {
    console.error("❌ Cases search error:", err);
    return json({
      success: false,
      error: "Failed to search cases",
      message: err instanceof Error ? err.message : "Unknown error"
    }, { status: 500 });
  }
};

// POST - Create new case
export const POST: RequestHandler = async ({ request, locals }) => {
  const startTime = Date.now();
  
  try {
    // Mock user for now - replace with real auth when available
    const mockUser = {
      id: 'mock-user-id',
      firstName: 'Detective',
      lastName: 'Smith'
    };

    const body = await request.json();
    console.log("📝 Creating new case:", { title: body.title, priority: body.priority });

    // Validate input
    const validationResult = caseSchema.safeParse(body);
    if (!validationResult.success) {
      return json({
        success: false,
        error: "Invalid case data",
        details: validationResult.error.flatten()
      }, { status: 400 });
    }

    const caseData = validationResult.data;

    // Generate case number
    const caseNumber = await generateCaseNumber();

    // Insert into database
    const [newCase] = await db.insert(cases).values({
      title: caseData.title,
      description: caseData.description,
      priority: caseData.priority,
      status: caseData.status,
      caseNumber,
      createdBy: mockUser.id,
      assignedTo: mockUser.id
    }).returning();

    console.log("✅ Case created successfully:", {
      id: newCase.id,
      caseNumber: newCase.caseNumber,
      processingTime: Date.now() - startTime
    });

    return json({
      success: true,
      message: "Case created successfully",
      data: newCase
    }, { status: 201 });

  } catch (err) {
    console.error("❌ Case creation error:", err);
    return json({
      success: false,
      error: "Failed to create case",
      message: err instanceof Error ? err.message : "Unknown error"
    }, { status: 500 });
  }
};

// Helper functions
async function generateCaseNumber(): Promise<string> {
  const year = new Date().getFullYear();
  
  // Get count of all cases (simplified for now)
  const caseCount = await db.select({ count: count() }).from(cases);
  
  const sequence = (caseCount[0]?.count || 0) + 1;
  return `CR-${year}-${sequence.toString().padStart(3, '0')}`;
}