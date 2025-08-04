import { cases } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { db } from "$lib/server/db/drizzle";

export const GET: RequestHandler = async ({ params, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const caseId = params.caseId;
    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    const caseResult = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!caseResult.length) {
      return json({ error: "Case not found" }, { status: 404 });
    }
    return json(caseResult[0]);
  } catch (error) {
    console.error("Error fetching case:", error);
    return json({ error: "Failed to fetch case" }, { status: 500 });
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
    const caseId = params.caseId;
    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if case exists
    const existingCase = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!existingCase.length) {
      return json({ error: "Case not found" }, { status: 404 });
    }
    // If updating case number, check for duplicates
    if (data.caseNumber && data.caseNumber !== existingCase[0].caseNumber) {
      const duplicateCase = await db
        .select()
        .from(cases)
        .where(eq(cases.caseNumber, data.caseNumber))
        .limit(1);

      if (duplicateCase.length > 0) {
        return json({ error: "Case number already exists" }, { status: 409 });
      }
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Map frontend fields to schema fields - only update provided fields
    if (data.title !== undefined) updateData.title = data.title;
    if (data.description !== undefined)
      updateData.description = data.description;
    if (data.caseNumber !== undefined) updateData.caseNumber = data.caseNumber;
    if (data.name !== undefined) updateData.name = data.name;
    if (data.incidentDate !== undefined) {
      updateData.incidentDate = data.incidentDate
        ? new Date(data.incidentDate)
        : null;
    }
    if (data.location !== undefined) updateData.location = data.location;
    if (data.status !== undefined) updateData.status = data.status;
    if (data.priority !== undefined) updateData.priority = data.priority;
    if (data.category !== undefined) updateData.category = data.category;
    if (data.dangerScore !== undefined)
      updateData.dangerScore = data.dangerScore;
    if (data.estimatedValue !== undefined)
      updateData.estimatedValue = data.estimatedValue;
    if (data.jurisdiction !== undefined)
      updateData.jurisdiction = data.jurisdiction;
    if (data.leadProsecutor !== undefined)
      updateData.leadProsecutor = data.leadProsecutor;
    if (data.assignedTeam !== undefined)
      updateData.assignedTeam = data.assignedTeam;
    if (data.tags !== undefined) updateData.tags = data.tags;
    if (data.aiSummary !== undefined) updateData.aiSummary = data.aiSummary;
    if (data.aiTags !== undefined) updateData.aiTags = data.aiTags;
    if (data.metadata !== undefined) updateData.metadata = data.metadata;
    if (data.closedAt !== undefined) {
      updateData.closedAt = data.closedAt ? new Date(data.closedAt) : null;
    }
    const [updatedCase] = await db
      .update(cases)
      .set(updateData)
      .where(eq(cases.id, caseId))
      .returning();

    return json(updatedCase);
  } catch (error) {
    console.error("Error updating case:", error);
    return json({ error: "Failed to update case" }, { status: 500 });
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
    const caseId = params.caseId;
    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    // Check if case exists
    const existingCase = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!existingCase.length) {
      return json({ error: "Case not found" }, { status: 404 });
    }
    // Delete the case (cascade will handle related records)
    const [deletedCase] = await db
      .delete(cases)
      .where(eq(cases.id, caseId))
      .returning();

    return json({ success: true, deletedCase });
  } catch (error) {
    console.error("Error deleting case:", error);
    return json({ error: "Failed to delete case" }, { status: 500 });
  }
};

// PATCH endpoint for partial updates (like status changes)
export const PATCH: RequestHandler = async ({ params, request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const caseId = params.caseId;
    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if case exists
    const existingCase = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!existingCase.length) {
      return json({ error: "Case not found" }, { status: 404 });
    }
    // For PATCH, only update the specific fields provided
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Handle specific patch operations
    if (data.operation === "close") {
      updateData.status = "closed";
      updateData.closedAt = new Date();
    } else if (data.operation === "reopen") {
      updateData.status = "open";
      updateData.closedAt = null;
    } else if (data.operation === "archive") {
      updateData.status = "archived";
    } else if (data.operation === "updatePriority") {
      updateData.priority = data.priority;
    } else if (data.operation === "addTag") {
      const currentTags = (existingCase[0].tags as string[]) || [];
      if (!currentTags.includes(data.tag)) {
        updateData.tags = [...currentTags, data.tag];
      }
    } else if (data.operation === "removeTag") {
      const currentTags = (existingCase[0].tags as string[]) || [];
      updateData.tags = currentTags.filter((tag) => tag !== data.tag);
    } else {
      // Regular field updates
      Object.keys(data).forEach((key) => {
        if (key !== "operation") {
          updateData[key] = data[key];
        }
      });
    }
    const [updatedCase] = await db
      .update(cases)
      .set(updateData)
      .where(eq(cases.id, caseId))
      .returning();

    return json(updatedCase);
  } catch (error) {
    console.error("Error patching case:", error);
    return json({ error: "Failed to update case" }, { status: 500 });
  }
};
