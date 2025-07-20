import { cases } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { desc, like, or, eq, and, sql } from "drizzle-orm";
import { db } from "$lib/server/db/index";

export const GET: RequestHandler = async ({ locals, url }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const search = url.searchParams.get("search") || "";
    const status = url.searchParams.get("status") || "";
    const priority = url.searchParams.get("priority") || "";
    const sortBy = url.searchParams.get("sortBy") || "updatedAt";
    const sortOrder = url.searchParams.get("sortOrder") || "desc";

    // Build query with filters
    let query = db.select().from(cases);

    // Add search filter
    if (search) {
      query = query.where(
        or(
          like(cases.title, `%${search}%`),
          like(cases.description, `%${search}%`),
          like(cases.caseNumber, `%${search}%`),
          like(cases.location, `%${search}%`),
        ),
      );
    }
    // Add status filter
    if (status) {
      query = query.where(
        search
          ? and(
              or(
                like(cases.title, `%${search}%`),
                like(cases.description, `%${search}%`),
                like(cases.caseNumber, `%${search}%`),
                like(cases.location, `%${search}%`),
              ),
              eq(cases.status, status),
            )
          : eq(cases.status, status),
      );
    }
    // Add priority filter
    if (priority) {
      const existingWhere = search || status;
      query = query.where(
        existingWhere
          ? and(
              search
                ? or(
                    like(cases.title, `%${search}%`),
                    like(cases.description, `%${search}%`),
                    like(cases.caseNumber, `%${search}%`),
                    like(cases.location, `%${search}%`),
                  )
                : sql`TRUE`,
              status ? eq(cases.status, status) : sql`TRUE`,
              eq(cases.priority, priority),
            )
          : eq(cases.priority, priority),
      );
    }
    // Add sorting
    const orderColumn =
      sortBy === "title"
        ? cases.title
        : sortBy === "caseNumber"
          ? cases.caseNumber
          : sortBy === "priority"
            ? cases.priority
            : sortBy === "status"
              ? cases.status
              : sortBy === "createdAt"
                ? cases.createdAt
                : cases.updatedAt;

    query = query.orderBy(
      sortOrder === "asc" ? orderColumn : desc(orderColumn),
    );

    // Add pagination
    query = query.limit(limit).offset(offset);

    const caseResults = await query;

    // Get total count for pagination
    const totalCountResult = await db
      .select({ count: sql<number>`count(*)` })
      .from(cases);

    const totalCount = totalCountResult[0]?.count || 0;

    return json({
      cases: caseResults,
      totalCount,
      hasMore: offset + limit < totalCount,
    });
  } catch (error) {
    console.error("Error fetching cases:", error);
    return json({ error: "Failed to fetch cases" }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    // Validate required fields
    if (!data.title || !data.caseNumber) {
      return json(
        { error: "Title and case number are required" },
        { status: 400 },
      );
    }
    // Check if case number already exists
    const existingCase = await db
      .select()
      .from(cases)
      .where(eq(cases.caseNumber, data.caseNumber))
      .limit(1);

    if (existingCase.length > 0) {
      return json({ error: "Case number already exists" }, { status: 409 });
    }
    // Map frontend data to schema fields
    const caseData = {
      title: data.title,
      description: data.description || "",
      caseNumber: data.caseNumber,
      name: data.name || data.title, // Use title as fallback for name
      incidentDate: data.incidentDate ? new Date(data.incidentDate) : null,
      location: data.location || "",
      status: data.status || "open",
      priority: data.priority || "medium",
      category: data.category || "",
      dangerScore: data.dangerScore || 0,
      estimatedValue: data.estimatedValue || null,
      jurisdiction: data.jurisdiction || "",
      leadProsecutor: data.leadProsecutor || locals.user.id,
      assignedTeam: data.assignedTeam || [],
      tags: data.tags || [],
      aiSummary: data.aiSummary || null,
      aiTags: data.aiTags || [],
      metadata: data.metadata || {},
      createdBy: locals.user.id,
    };

    const [newCase] = await db.insert(cases).values(caseData).returning();

    return json(newCase, { status: 201 });
  } catch (error) {
    console.error("Error creating case:", error);
    return json({ error: "Failed to create case" }, { status: 500 });
  }
};
