import {
  aiReports,
  canvasStates,
  reports,
} from "$lib/server/db/schema-postgres";
import type { RequestEvent } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { and, desc, eq, like, or, sql } from "drizzle-orm";
import { db } from "$lib/server/db/index";

export async function GET({ url, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const caseId = url.searchParams.get("caseId");
    const reportType = url.searchParams.get("reportType");
    const status = url.searchParams.get("status");
    const search = url.searchParams.get("search") || "";
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const sortBy = url.searchParams.get("sortBy") || "updatedAt";
    const sortOrder = url.searchParams.get("sortOrder") || "desc";

    // Use the simpler reports table if aiReports doesn't exist, with fallback
    let query;
    try {
      query = db.select().from(aiReports);
    } catch (error) {
      console.warn("aiReports table not found, using reports table");
      query = db.select().from(reports);
    }
    const conditions: any[] = [];

    // Add filters
    if (caseId) {
      conditions.push(eq(reports.caseId, caseId));
    }
    if (reportType) {
      conditions.push(eq(reports.reportType, reportType));
    }
    if (status) {
      conditions.push(eq(reports.status, status));
    }
    // Add search filter
    if (search) {
      conditions.push(
        or(
          like(reports.title, `%${search}%`),
          like(reports.content, `%${search}%`),
        ),
      );
    }
    // Apply filters
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }
    // Add sorting
    const orderColumn =
      sortBy === "title"
        ? reports.title
        : sortBy === "reportType"
          ? reports.reportType
          : sortBy === "status"
            ? reports.status
            : sortBy === "createdAt"
              ? reports.createdAt
              : reports.updatedAt;

    query = query.orderBy(
      sortOrder === "asc" ? orderColumn : desc(orderColumn),
    );

    // Add pagination
    query = query.limit(limit).offset(offset);

    const reportResults = await query;

    // Get total count for pagination
    let countQuery = db.select({ count: sql<number>`count(*)` }).from(reports);
    if (conditions.length > 0) {
      countQuery = countQuery.where(and(...conditions));
    }
    const totalCountResult = await countQuery;
    const totalCount = totalCountResult[0]?.count || 0;

    // Get associated canvas states for each report
    const enrichedReports = await Promise.all(
      reportResults.map(async (report: any) => {
        try {
          const canvasState = await db
            .select()
            .from(canvasStates)
            .where(eq(canvasStates.caseId, report.caseId))
            .limit(1);

          return {
            ...report,
            canvasState: canvasState[0] || null,
          };
        } catch (error) {
          console.warn("Error fetching canvas state:", error);
          return {
            ...report,
            canvasState: null,
          };
        }
      }),
    );

    return json({
      reports: enrichedReports,
      totalCount,
      hasMore: offset + limit < totalCount,
      pagination: {
        limit,
        offset,
        total: totalCount,
      },
    });
  } catch (error) {
    console.error("Error fetching reports:", error);
    return json({ error: "Failed to fetch reports" }, { status: 500 });
  }
}
export async function POST({ request, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    // Validate required fields
    if (!data.title || !data.caseId) {
      return json({ error: "Title and case ID are required" }, { status: 400 });
    }
    // Calculate word count and estimated read time
    const textContent = data.content
      ? data.content.replace(/<[^>]*>/g, "").trim()
      : "";
    const wordCount = textContent
      .split(/\s+/)
      .filter((word: string) => word.length > 0).length;

    // Map data to the reports table schema
    const reportData = {
      title: data.title,
      content: data.content || "",
      caseId: data.caseId,
      reportType: data.reportType || "case_summary",
      status: data.status || "draft",
      isPublic: data.isPublic || false,
      tags: data.tags || [],
      metadata: {
        ...(data.metadata || {}),
        wordCount,
        estimatedReadTime: Math.ceil(wordCount / 200),
        summary: data.summary || "",
        confidentialityLevel: data.confidentialityLevel || "restricted",
        jurisdiction: data.jurisdiction || "",
        sections: data.sections || [],
        aiSummary: data.aiSummary || null,
        aiTags: data.aiTags || [],
        templateId: data.templateId || null,
      },
      createdBy: locals.user.id,
    };

    const [newReport] = await db.insert(reports).values(reportData).returning();

    return json(newReport, { status: 201 });
  } catch (error) {
    console.error("Error creating report:", error);
    return json({ error: "Failed to create report" }, { status: 500 });
  }
}
export async function PUT({ request, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    if (!data.id) {
      return json({ error: "Report ID is required" }, { status: 400 });
    }
    // Check if report exists
    const existingReport = await db
      .select()
      .from(reports)
      .where(eq(reports.id, data.id))
      .limit(1);

    if (!existingReport.length) {
      return json({ error: "Report not found" }, { status: 404 });
    }
    // Calculate word count and estimated read time
    const textContent = data.content
      ? data.content.replace(/<[^>]*>/g, "").trim()
      : "";
    const wordCount = textContent
      .split(/\s+/)
      .filter((word: string) => word.length > 0).length;

    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Only update provided fields
    if (data.title !== undefined) updateData.title = data.title;
    if (data.content !== undefined) updateData.content = data.content;
    if (data.reportType !== undefined) updateData.reportType = data.reportType;
    if (data.status !== undefined) updateData.status = data.status;
    if (data.isPublic !== undefined) updateData.isPublic = data.isPublic;
    if (data.tags !== undefined) updateData.tags = data.tags;

    // Update metadata with new calculated values
    if (data.content !== undefined || data.metadata !== undefined) {
      const currentMetadata = (existingReport[0].metadata as any) || {};
      updateData.metadata = {
        ...currentMetadata,
        ...(data.metadata || {}),
        wordCount,
        estimatedReadTime: Math.ceil(wordCount / 200),
      };
    }
    const [updatedReport] = await db
      .update(reports)
      .set(updateData)
      .where(eq(reports.id, data.id))
      .returning();

    return json(updatedReport);
  } catch (error) {
    console.error("Error updating report:", error);
    return json({ error: "Failed to update report" }, { status: 500 });
  }
}
export async function DELETE({ url, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const reportId = url.searchParams.get("id");
    if (!reportId) {
      return json({ error: "Report ID is required" }, { status: 400 });
    }
    // Check if report exists
    const existingReport = await db
      .select()
      .from(reports)
      .where(eq(reports.id, reportId))
      .limit(1);

    if (!existingReport.length) {
      return json({ error: "Report not found" }, { status: 404 });
    }
    // Delete the report (cascade will handle related records)
    const [deletedReport] = await db
      .delete(reports)
      .where(eq(reports.id, reportId))
      .returning();

    return json({ success: true, deletedReport });
  } catch (error) {
    console.error("Error deleting report:", error);
    return json({ error: "Failed to delete report" }, { status: 500 });
  }
}
// PATCH endpoint for partial updates
export async function PATCH({ request, url, locals }: RequestEvent) {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const reportId = url.searchParams.get("id");
    if (!reportId) {
      return json({ error: "Report ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if report exists
    const existingReport = await db
      .select()
      .from(reports)
      .where(eq(reports.id, reportId))
      .limit(1);

    if (!existingReport.length) {
      return json({ error: "Report not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Handle specific patch operations
    if (data.operation === "publish") {
      updateData.status = "published";
      updateData.isPublic = data.isPublic || false;
    } else if (data.operation === "archive") {
      updateData.status = "archived";
    } else if (data.operation === "draft") {
      updateData.status = "draft";
    } else if (data.operation === "addTag") {
      const currentTags = (existingReport[0].tags as string[]) || [];
      if (!currentTags.includes(data.tag)) {
        updateData.tags = [...currentTags, data.tag];
      }
    } else if (data.operation === "removeTag") {
      const currentTags = (existingReport[0].tags as string[]) || [];
      updateData.tags = currentTags.filter((tag) => tag !== data.tag);
    } else {
      // Regular field updates
      Object.keys(data).forEach((key) => {
        if (key !== "operation") {
          updateData[key] = data[key];
        }
      });
    }
    const [updatedReport] = await db
      .update(reports)
      .set(updateData)
      .where(eq(reports.id, reportId))
      .returning();

    return json(updatedReport);
  } catch (error) {
    console.error("Error patching report:", error);
    return json({ error: "Failed to update report" }, { status: 500 });
  }
}
