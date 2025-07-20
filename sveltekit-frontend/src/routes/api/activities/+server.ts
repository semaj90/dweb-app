import { caseActivities } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { and, desc, eq, like, or, sql } from "drizzle-orm";
import { db } from "$lib/server/db/index";

export const GET: RequestHandler = async ({ locals, url }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const caseId = url.searchParams.get("caseId");
    const activityType = url.searchParams.get("activityType");
    const status = url.searchParams.get("status");
    const priority = url.searchParams.get("priority");
    const assignedTo = url.searchParams.get("assignedTo");
    const search = url.searchParams.get("search") || "";
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const sortBy = url.searchParams.get("sortBy") || "scheduledFor";
    const sortOrder = url.searchParams.get("sortOrder") || "asc";

    // Build query with filters
    let query = db.select().from(caseActivities);
    const filters: any[] = [];

    // Add case filter
    if (caseId) {
      filters.push(eq(caseActivities.caseId, caseId));
    }
    // Add activity type filter
    if (activityType) {
      filters.push(eq(caseActivities.activityType, activityType));
    }
    // Add status filter
    if (status) {
      filters.push(eq(caseActivities.status, status));
    }
    // Add priority filter
    if (priority) {
      filters.push(eq(caseActivities.priority, priority));
    }
    // Add assigned user filter
    if (assignedTo) {
      filters.push(eq(caseActivities.assignedTo, assignedTo));
    }
    // Add search filter
    if (search) {
      filters.push(
        or(
          like(caseActivities.title, `%${search}%`),
          like(caseActivities.description, `%${search}%`),
        ),
      );
    }
    // Apply filters
    if (filters.length > 0) {
      query = query.where(and(...filters));
    }
    // Add sorting
    const orderColumn =
      sortBy === "title"
        ? caseActivities.title
        : sortBy === "activityType"
          ? caseActivities.activityType
          : sortBy === "status"
            ? caseActivities.status
            : sortBy === "priority"
              ? caseActivities.priority
              : sortBy === "completedAt"
                ? caseActivities.completedAt
                : sortBy === "createdAt"
                  ? caseActivities.createdAt
                  : sortBy === "updatedAt"
                    ? caseActivities.updatedAt
                    : caseActivities.scheduledFor;

    query = query.orderBy(
      sortOrder === "asc" ? orderColumn : desc(orderColumn),
    );

    // Add pagination
    query = query.limit(limit).offset(offset);

    const activityResults = await query;

    // Get total count for pagination
    let countQuery = db
      .select({ count: sql<number>`count(*)` })
      .from(caseActivities);
    if (filters.length > 0) {
      countQuery = countQuery.where(and(...filters));
    }
    const totalCountResult = await countQuery;
    const totalCount = totalCountResult[0]?.count || 0;

    return json({
      activities: activityResults,
      totalCount,
      hasMore: offset + limit < totalCount,
      pagination: {
        limit,
        offset,
        total: totalCount,
      },
    });
  } catch (error) {
    console.error("Error fetching activities:", error);
    return json({ error: "Failed to fetch activities" }, { status: 500 });
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
    if (!data.caseId || !data.title || !data.activityType) {
      return json(
        { error: "Case ID, title, and activity type are required" },
        { status: 400 },
      );
    }
    // Map frontend data to schema fields
    const activityData = {
      caseId: data.caseId,
      activityType: data.activityType,
      title: data.title.trim(),
      description: data.description?.trim() || null,
      scheduledFor: data.scheduledFor ? new Date(data.scheduledFor) : null,
      completedAt: data.completedAt ? new Date(data.completedAt) : null,
      status: data.status || "pending",
      priority: data.priority || "medium",
      assignedTo: data.assignedTo || null,
      relatedEvidence: data.relatedEvidence || [],
      relatedCriminals: data.relatedCriminals || [],
      metadata: data.metadata || {},
      createdBy: locals.user.id,
    };

    const [newActivity] = await db
      .insert(caseActivities)
      .values(activityData)
      .returning();

    return json(newActivity, { status: 201 });
  } catch (error) {
    console.error("Error creating activity:", error);
    return json({ error: "Failed to create activity" }, { status: 500 });
  }
};
