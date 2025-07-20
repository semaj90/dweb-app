import { db } from "$lib/server/db/index";
import { cases, evidence } from "$lib/server/db/unified-schema";
import { fail } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import type { Actions, PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ url, parent }) => {
  // Wait for layout data (ensures user is authenticated)
  const layoutData = await parent();

  const caseIdToView = url.searchParams.get("view");

  if (!caseIdToView) {
    return {
      activeCase: null,
      caseEvidence: [],
      userCases: layoutData.userCases,
      caseStats: layoutData.caseStats,
    };
  }
  // Fetch the active case with full details
  const activeCase = await db.query.cases.findFirst({
    where: eq(cases.id, caseIdToView),
    with: {
      // Add relationships if defined in your schema
    },
  });

  if (!activeCase) {
    return {
      activeCase: null,
      caseEvidence: [],
      userCases: layoutData.userCases,
      caseStats: layoutData.caseStats,
    };
  }
  // Fetch evidence for this case
  const caseEvidence = await db.query.evidence.findMany({
    where: eq(evidence.caseId, caseIdToView),
    orderBy: (evidence, { desc }) => [desc(evidence.collectedAt)],
  });

  return {
    activeCase,
    caseEvidence,
    userCases: layoutData.userCases,
    caseStats: layoutData.caseStats,
  };
};

export const actions: Actions = {
  // Add evidence to case
  addEvidence: async ({ request, locals }) => {
    const session = await locals.auth.validate();
    if (!session) {
      return fail(401, { message: "Unauthorized" });
    }
    const formData = await request.formData();
    const caseId = formData.get("caseId") as string;
    const title = formData.get("title") as string;
    const description = formData.get("description") as string;
    const type = formData.get("type") as string;

    if (!caseId || !title || !type) {
      return fail(400, { message: "Missing required fields" });
    }
    try {
      const newEvidence = await db
        .insert(evidence)
        .values({
          caseId,
          title,
          description,
          type,
          collectedAt: new Date(),
          metadata: {},
        })
        .returning();

      return { success: true, evidence: newEvidence[0] };
    } catch (error) {
      console.error("Failed to add evidence:", error);
      return fail(500, { message: "Failed to add evidence" });
    }
  },

  // Update evidence
  updateEvidence: async ({ request, locals }) => {
    const session = await locals.auth.validate();
    if (!session) {
      return fail(401, { message: "Unauthorized" });
    }
    const formData = await request.formData();
    const evidenceId = formData.get("evidenceId") as string;
    const title = formData.get("title") as string;
    const description = formData.get("description") as string;
    const type = formData.get("type") as string;

    if (!evidenceId || !title || !type) {
      return fail(400, { message: "Missing required fields" });
    }
    try {
      const updatedEvidence = await db
        .update(evidence)
        .set({
          title,
          description,
          type,
          updatedAt: new Date(),
        })
        .where(eq(evidence.id, evidenceId))
        .returning();

      return { success: true, evidence: updatedEvidence[0] };
    } catch (error) {
      console.error("Failed to update evidence:", error);
      return fail(500, { message: "Failed to update evidence" });
    }
  },

  // Delete evidence
  deleteEvidence: async ({ request, locals }) => {
    const session = await locals.auth.validate();
    if (!session) {
      return fail(401, { message: "Unauthorized" });
    }
    const formData = await request.formData();
    const evidenceId = formData.get("evidenceId") as string;

    if (!evidenceId) {
      return fail(400, { message: "Missing evidence ID" });
    }
    try {
      await db.delete(evidence).where(eq(evidence.id, evidenceId));
      return { success: true };
    } catch (error) {
      console.error("Failed to delete evidence:", error);
      return fail(500, { message: "Failed to delete evidence" });
    }
  },
};
