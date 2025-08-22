import type { PageServerLoad } from "./$types";

import { cases, evidence } from "$lib/server/db/schema-postgres";
// Orphaned content: import {

import { eq } from "drizzle-orm";
// Orphaned content: import {

export const load: PageServerLoad = async ({ params }) => {
  const caseId = params.id;

  try {
    // Load case details
    const [caseData] = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!caseData) {
      throw error(404, "Case not found");
    }
    // Load evidence for this case
    const evidenceList = await db
      .select()
      .from(evidence)
      .where(eq(evidence.caseId, caseId));

    return {
      case: caseData,
      evidence: evidenceList,
    };
  } catch (err) {
    console.error("Error loading case data:", err);
    throw error(500, "Failed to load case data");
  }
};
