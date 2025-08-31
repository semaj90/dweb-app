import { reports } from '$lib/server/db/schema-postgres';
import { db } from '$lib/server/db/index';
import { eq } from 'drizzle-orm';
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ params, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const reportId = params.reportId;
    if (!reportId) {
      return json({ error: "Report ID is required" }, { status: 400 });
    }
    const reportResult = await db
      .select()
      .from(reports)
      .where(eq(reports.id, reportId))
      .limit(1);

    if (!reportResult.length) {
      return json({ error: "Report not found" }, { status: 404 });
    }
    return json(reportResult[0]);
  } catch (error: any) {
    console.error("Error fetching report:", error);
    return json({ error: "Failed to fetch report" }, { status: 500 });
  }
};

export const PUT: RequestHandler = async ({ params, request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const reportId = params.reportId;
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
      .where(eq(reports.id, reportId))
      .returning();

    return json(updatedReport);
  } catch (error: any) {
    console.error("Error updating report:", error);
    return json({ error: "Failed to update report" }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ params, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const reportId = params.reportId;
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
  } catch (error: any) {
    console.error("Error deleting report:", error);
    return json({ error: "Failed to delete report" }, { status: 500 });
  }
};

export const PATCH: RequestHandler = async ({ params, request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const reportId = params.reportId;
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
  } catch (error: any) {
    console.error("Error patching report:", error);
    return json({ error: "Failed to update report" }, { status: 500 });
  }
};
