import { json } from "@sveltejs/kit";
import { caseActivities } from "$lib/server/db/schema-postgres";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ params, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const activityId = params.activityId;
    if (!activityId) {
      return json({ error: "Activity ID is required" }, { status: 400 });
    }
    const activityResult = await db
      .select()
      .from(caseActivities)
      .where(eq(caseActivities.id, activityId))
      .limit(1);

    if (!activityResult.length) {
      return json({ error: "Activity not found" }, { status: 404 });
    }
    return json(activityResult[0]);
  } catch (error: any) {
    console.error("Error fetching activity:", error);
    return json({ error: "Failed to fetch activity" }, { status: 500 });
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
    const activityId = params.activityId;
    if (!activityId) {
      return json({ error: "Activity ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if activity exists
    const existingActivity = await db
      .select()
      .from(caseActivities)
      .where(eq(caseActivities.id, activityId))
      .limit(1);

    if (!existingActivity.length) {
      return json({ error: "Activity not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Map frontend fields to schema fields - only update provided fields
    if (data.title !== undefined) updateData.title = data.title.trim();
    if (data.description !== undefined)
      updateData.description = data.description?.trim() || null;
    if (data.activityType !== undefined)
      updateData.activityType = data.activityType;
    if (data.scheduledFor !== undefined) {
      updateData.scheduledFor = data.scheduledFor
        ? new Date(data.scheduledFor)
        : null;
    }
    if (data.completedAt !== undefined) {
      updateData.completedAt = data.completedAt
        ? new Date(data.completedAt)
        : null;
    }
    if (data.status !== undefined) updateData.status = data.status;
    if (data.priority !== undefined) updateData.priority = data.priority;
    if (data.assignedTo !== undefined) updateData.assignedTo = data.assignedTo;
    if (data.relatedEvidence !== undefined)
      updateData.relatedEvidence = data.relatedEvidence;
    if (data.relatedCriminals !== undefined)
      updateData.relatedCriminals = data.relatedCriminals;
    if (data.metadata !== undefined) updateData.metadata = data.metadata;

    const [updatedActivity] = await db
      .update(caseActivities)
      .set(updateData)
      .where(eq(caseActivities.id, activityId))
      .returning();

    return json(updatedActivity);
  } catch (error: any) {
    console.error("Error updating activity:", error);
    return json({ error: "Failed to update activity" }, { status: 500 });
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
    const activityId = params.activityId;
    if (!activityId) {
      return json({ error: "Activity ID is required" }, { status: 400 });
    }
    // Check if activity exists
    const existingActivity = await db
      .select()
      .from(caseActivities)
      .where(eq(caseActivities.id, activityId))
      .limit(1);

    if (!existingActivity.length) {
      return json({ error: "Activity not found" }, { status: 404 });
    }
    // Delete the activity
    const [deletedActivity] = await db
      .delete(caseActivities)
      .where(eq(caseActivities.id, activityId))
      .returning();

    return json({ success: true, deletedActivity });
  } catch (error: any) {
    console.error("Error deleting activity:", error);
    return json({ error: "Failed to delete activity" }, { status: 500 });
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
    const activityId = params.activityId;
    if (!activityId) {
      return json({ error: "Activity ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if activity exists
    const existingActivity = await db
      .select()
      .from(caseActivities)
      .where(eq(caseActivities.id, activityId))
      .limit(1);

    if (!existingActivity.length) {
      return json({ error: "Activity not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Handle specific patch operations
    if (data.operation === "complete") {
      updateData.status = "completed";
      updateData.completedAt = new Date();
    } else if (data.operation === "reopen") {
      updateData.status = "pending";
      updateData.completedAt = null;
    } else if (data.operation === "reschedule") {
      updateData.scheduledFor = data.scheduledFor
        ? new Date(data.scheduledFor)
        : null;
    } else if (data.operation === "reassign") {
      updateData.assignedTo = data.assignedTo;
    } else if (data.operation === "updatePriority") {
      updateData.priority = data.priority;
    } else if (data.operation === "addEvidence") {
      const currentEvidence =
        (existingActivity[0].relatedEvidence as string[]) || [];
      if (!currentEvidence.includes(data.evidenceId)) {
        updateData.relatedEvidence = [...currentEvidence, data.evidenceId];
      }
    } else if (data.operation === "removeEvidence") {
      const currentEvidence =
        (existingActivity[0].relatedEvidence as string[]) || [];
      updateData.relatedEvidence = currentEvidence.filter(
        (id) => id !== data.evidenceId,
      );
    } else if (data.operation === "addCriminal") {
      const currentCriminals =
        (existingActivity[0].relatedCriminals as string[]) || [];
      if (!currentCriminals.includes(data.criminalId)) {
        updateData.relatedCriminals = [...currentCriminals, data.criminalId];
      }
    } else if (data.operation === "removeCriminal") {
      const currentCriminals =
        (existingActivity[0].relatedCriminals as string[]) || [];
      updateData.relatedCriminals = currentCriminals.filter(
        (id) => id !== data.criminalId,
      );
    } else {
      // Regular field updates
      Object.keys(data).forEach((key) => {
        if (key !== "operation") {
          updateData[key] = data[key];
        }
      });
    }
    const [updatedActivity] = await db
      .update(caseActivities)
      .set(updateData)
      .where(eq(caseActivities.id, activityId))
      .returning();

    return json(updatedActivity);
  } catch (error: any) {
    console.error("Error patching activity:", error);
    return json({ error: "Failed to update activity" }, { status: 500 });
  }
};
