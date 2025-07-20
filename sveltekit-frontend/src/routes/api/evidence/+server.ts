import { evidence } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { and, desc, eq, like, or, sql } from "drizzle-orm";
import { createClient } from "redis";
import { db } from "$lib/server/db/index";

// Redis client for real-time updates
let redisClient: any = null;

async function initRedis() {
  if (!redisClient) {
    try {
      redisClient = createClient({
        url: process.env.REDIS_URL || "redis://localhost:6379",
      });
      await redisClient.connect();
    } catch (error) {
      console.error("Redis connection failed:", error);
    }
  }
}
async function publishEvidenceUpdate(type: string, data: any, userId?: string) {
  if (redisClient) {
    try {
      await redisClient.publish(
        "evidence_update",
        JSON.stringify({
          type,
          timestamp: new Date().toISOString(),
          userId,
          ...data,
        })
      );
    } catch (error) {
      console.error("Failed to publish evidence update:", error);
    }
  }
}
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const caseId = url.searchParams.get("caseId");
    const criminalId = url.searchParams.get("criminalId");
    const evidenceType = url.searchParams.get("evidenceType");
    const search = url.searchParams.get("search") || "";
    const limit = parseInt(url.searchParams.get("limit") || "100");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const sortBy = url.searchParams.get("sortBy") || "uploadedAt";
    const sortOrder = url.searchParams.get("sortOrder") || "desc";

    // Build query with filters
    let query = db.select().from(evidence);
    const filters = [];

    // Add filters if provided
    if (caseId) {
      filters.push(eq(evidence.caseId, caseId));
    }
    if (criminalId) {
      filters.push(eq(evidence.criminalId, criminalId));
    }
    if (evidenceType) {
      filters.push(eq(evidence.evidenceType, evidenceType));
    }
    // Add search filter
    if (search) {
      filters.push(
        or(
          like(evidence.title, `%${search}%`),
          like(evidence.description, `%${search}%`),
          like(evidence.fileName, `%${search}%`),
          like(evidence.summary, `%${search}%`)
        )
      );
    }
    // Apply filters
    if (filters.length > 0) {
      query = query.where(and(...filters));
    }
    // Add sorting
    const orderColumn =
      sortBy === "title"
        ? evidence.title
        : sortBy === "evidenceType"
          ? evidence.evidenceType
          : sortBy === "fileSize"
            ? evidence.fileSize
            : sortBy === "collectedAt"
              ? evidence.collectedAt
              : evidence.uploadedAt;

    query = query.orderBy(
      sortOrder === "asc" ? orderColumn : desc(orderColumn)
    );

    // Add pagination
    query = query.limit(limit).offset(offset);

    const evidenceResults = await query;

    // Get total count for pagination
    let countQuery = db.select({ count: sql<number>`count(*)` }).from(evidence);
    if (filters.length > 0) {
      countQuery = countQuery.where(and(...filters));
    }
    const totalCountResult = await countQuery;
    const totalCount = totalCountResult[0]?.count || 0;

    return json({
      evidence: evidenceResults,
      totalCount,
      hasMore: offset + limit < totalCount,
      pagination: {
        limit,
        offset,
        total: totalCount,
      },
    });
  } catch (error) {
    console.error("Error fetching evidence:", error);
    return json({ error: "Failed to fetch evidence" }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    await initRedis();

    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    // Validate required fields
    if (!data.title || !data.evidenceType) {
      return json(
        { error: "Title and evidence type are required" },
        { status: 400 }
      );
    }
    // Map frontend data to schema fields
    const evidenceData = {
      title: data.title,
      description: data.description || "",
      caseId: data.caseId || null,
      criminalId: data.criminalId || null,
      evidenceType: data.evidenceType,
      fileType: data.fileType || null,
      subType: data.subType || null,
      fileUrl: data.fileUrl || null,
      fileName: data.fileName || null,
      fileSize: data.fileSize || null,
      mimeType: data.mimeType || null,
      hash: data.hash || null,
      tags: data.tags || [],
      chainOfCustody: data.chainOfCustody || [],
      collectedAt: data.collectedAt ? new Date(data.collectedAt) : null,
      collectedBy: data.collectedBy || null,
      location: data.location || null,
      labAnalysis: data.labAnalysis || {},
      aiAnalysis: data.aiAnalysis || {},
      aiTags: data.aiTags || [],
      aiSummary: data.aiSummary || null,
      summary: data.summary || null,
      isAdmissible: data.isAdmissible !== undefined ? data.isAdmissible : true,
      confidentialityLevel: data.confidentialityLevel || "standard",
      canvasPosition: data.canvasPosition || {},
      uploadedBy: locals.user.id,
    };

    const [newEvidence] = await db
      .insert(evidence)
      .values(evidenceData)
      .returning();

    // Publish real-time update
    await publishEvidenceUpdate(
      "EVIDENCE_CREATED",
      {
        evidenceId: newEvidence.id,
        data: newEvidence,
      },
      locals.user.id
    );

    return json(newEvidence, { status: 201 });
  } catch (error) {
    console.error("Error creating evidence:", error);
    return json({ error: "Failed to create evidence" }, { status: 500 });
  }
};

export const PATCH: RequestHandler = async ({ request, url, locals }) => {
  try {
    await initRedis();

    const evidenceId = url.searchParams.get("id");
    if (!evidenceId) {
      return json({ error: "Evidence ID is required" }, { status: 400 });
    }
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    // Check if evidence exists
    const existingEvidence = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    if (!existingEvidence.length) {
      return json({ error: "Evidence not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Map frontend fields to schema fields - only update provided fields
    if (data.title !== undefined) updateData.title = data.title;
    if (data.description !== undefined)
      updateData.description = data.description;
    if (data.caseId !== undefined) updateData.caseId = data.caseId;
    if (data.criminalId !== undefined) updateData.criminalId = data.criminalId;
    if (data.evidenceType !== undefined)
      updateData.evidenceType = data.evidenceType;
    if (data.fileType !== undefined) updateData.fileType = data.fileType;
    if (data.subType !== undefined) updateData.subType = data.subType;
    if (data.fileUrl !== undefined) updateData.fileUrl = data.fileUrl;
    if (data.fileName !== undefined) updateData.fileName = data.fileName;
    if (data.fileSize !== undefined) updateData.fileSize = data.fileSize;
    if (data.mimeType !== undefined) updateData.mimeType = data.mimeType;
    if (data.hash !== undefined) updateData.hash = data.hash;
    if (data.tags !== undefined) updateData.tags = data.tags;
    if (data.chainOfCustody !== undefined)
      updateData.chainOfCustody = data.chainOfCustody;
    if (data.collectedAt !== undefined) {
      updateData.collectedAt = data.collectedAt
        ? new Date(data.collectedAt)
        : null;
    }
    if (data.collectedBy !== undefined)
      updateData.collectedBy = data.collectedBy;
    if (data.location !== undefined) updateData.location = data.location;
    if (data.labAnalysis !== undefined)
      updateData.labAnalysis = data.labAnalysis;
    if (data.aiAnalysis !== undefined) updateData.aiAnalysis = data.aiAnalysis;
    if (data.aiTags !== undefined) updateData.aiTags = data.aiTags;
    if (data.aiSummary !== undefined) updateData.aiSummary = data.aiSummary;
    if (data.summary !== undefined) updateData.summary = data.summary;
    if (data.isAdmissible !== undefined)
      updateData.isAdmissible = data.isAdmissible;
    if (data.confidentialityLevel !== undefined)
      updateData.confidentialityLevel = data.confidentialityLevel;
    if (data.canvasPosition !== undefined)
      updateData.canvasPosition = data.canvasPosition;

    // Update evidence in database
    const [updatedEvidence] = await db
      .update(evidence)
      .set(updateData)
      .where(eq(evidence.id, evidenceId))
      .returning();

    // Publish real-time update
    await publishEvidenceUpdate(
      "EVIDENCE_UPDATED",
      {
        evidenceId,
        changes: updateData,
        data: updatedEvidence,
      },
      locals.user.id
    );

    return json({
      success: true,
      evidence: updatedEvidence,
    });
  } catch (error) {
    console.error("Error updating evidence:", error);
    return json({ error: "Failed to update evidence" }, { status: 500 });
  }
};

// PATCH endpoint to update evidence order for drag-drop persistence
export const PATCH: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
    });
  }
  const { updates } = await request.json(); // [{id, order}, ...]
  if (!Array.isArray(updates)) {
    return new Response(JSON.stringify({ error: "Missing updates array" }), {
      status: 400,
    });
  }
  for (const { id, order } of updates) {
    await db.update(evidence).set({ order }).where(eq(evidence.id, id));
  }
  return new Response(JSON.stringify({ success: true }), { status: 200 });
};

export const DELETE: RequestHandler = async ({ url, locals }) => {
  try {
    await initRedis();

    const evidenceId = url.searchParams.get("id");
    if (!evidenceId) {
      return json({ error: "Evidence ID is required" }, { status: 400 });
    }
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    // Get evidence before deletion for real-time update
    const [existingEvidence] = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId));

    if (!existingEvidence) {
      return json({ error: "Evidence not found" }, { status: 404 });
    }
    // Delete evidence from database
    const [deletedEvidence] = await db
      .delete(evidence)
      .where(eq(evidence.id, evidenceId))
      .returning();

    // Publish real-time update
    await publishEvidenceUpdate(
      "EVIDENCE_DELETED",
      {
        evidenceId,
        data: deletedEvidence,
      },
      locals.user.id
    );

    return json({
      success: true,
      deletedEvidence,
    });
  } catch (error) {
    console.error("Error deleting evidence:", error);
    return json({ error: "Failed to delete evidence" }, { status: 500 });
  }
};
