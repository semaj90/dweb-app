// @ts-nocheck
import { canvasStates, cases, evidence } from "$lib/server/db/schema-postgres";
import { redirect } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ locals, url }) => {
  if (!locals.user) {
    throw redirect(302, "/login");
  }
  const user = locals.user;

  // Get report ID from query params
  const reportId = url.searchParams.get("reportId");

  let canvasState = null;
  let reportData = null;
  let evidenceData: any[] = [];

  if (reportId) {
    // Load existing canvas state
    const [canvasStateResult] = await db
      .select()
      .from(canvasStates)
      .where(eq(canvasStates.caseId, reportId))
      .limit(1);

    if (canvasStateResult) {
      canvasState = canvasStateResult.canvasData;
    }
    // Load associated evidence
    evidenceData = await db
      .select()
      .from(evidence)
      .where(eq(evidence.caseId, reportId));

    // Load report/case data
    const [caseResult] = await db
      .select()
      .from(cases)
      .where(eq(cases.id, reportId))
      .limit(1);

    reportData = caseResult;
  }
  return {
    user,
    reportId,
    canvasState,
    reportData,
    evidence: evidenceData,
  };
};
