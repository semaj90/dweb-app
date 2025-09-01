import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import { caseActivities, cases, evidence, statutes } from "$lib/server/db/index";
import { eq, sql, ilike } from "drizzle-orm";
import { QdrantClient } from "@qdrant/js-client-rest";
import type { RequestHandler } from './$types';


// Environment variables fallback
const env = process.env || {};

const qdrantClient = new QdrantClient({
  url: env.QDRANT_URL || "http://localhost:6333",
});
const NLP_SERVICE_URL = env.LLM_SERVICE_URL || "http://localhost:8000";

// Add 'task' type for recommendations
export interface Recommendation {
  id: string;
  type:
    | "missing_info"
    | "link_case"
    | "link_statute"
    | "investigative_step"
    | "task";
  title: string;
  description: string;
  confidence: number;
  actionData: Record<string, any>;
}
export const GET: RequestHandler = async ({ params, locals }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  const { caseId } = params;
  if (!caseId) {
    return json({ error: "Case ID is required" }, { status: 400 });
  }
  const recommendations: Recommendation[] = [];

  try {
    // 1. Fetch the current case data
    const currentCaseResults = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!currentCaseResults.length) {
      return json({ error: "Case not found" }, { status: 404 });
    }

    const currentCase = currentCaseResults[0];
    // 2. Recommendation: Check for missing information
    if (!currentCase.description || currentCase.description.length < 50) {
      recommendations.push({
        id: "rec-desc",
        type: "missing_info",
        title: "Expand Case Description",
        description:
          "A more detailed description will improve AI analysis and case clarity.",
        confidence: 0.9,
        actionData: { field: "description" },
      });
    }
    // Check for missing evidence
    const evidenceCount = await db
      .select({ count: sql<number>`count(*)` })
      .from(evidence)
      .where(eq(evidence.caseId, caseId));
    if (evidenceCount[0]?.count === 0) {
      recommendations.push({
        id: "rec-evidence",
        type: "missing_info",
        title: "Add Evidence",
        description:
          "This case has no evidence attached. Upload relevant documents, images, or other files.",
        confidence: 0.95,
        actionData: { relation: "evidence" },
      });
    }
    // Check for activities
    const activitiesCount = await db
      .select({ count: sql<number>`count(*)` })
      .from(caseActivities)
      .where(eq(caseActivities.caseId, caseId));
    if (currentCase.status === "open" && activitiesCount[0]?.count === 0) {
      recommendations.push({
        id: "rec-first-activity",
        type: "task",
        title: "Schedule First Activity",
        description:
          "This case is open but has no activities. Consider scheduling an initial review or evidence collection.",
        confidence: 0.8,
        actionData: { activityType: "initial_review" },
      });
    }
    // 3. Recommendation: Find similar cases via Qdrant (simplified)
    try {
      const textToEmbed =
        currentCase.aiSummary || currentCase.description || currentCase.title;
      const nlpResponse = await fetch(
        `${NLP_SERVICE_URL}/analyze-criminal-actions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textToEmbed }),
        }
      );

      if (nlpResponse.ok) {
        const nlpData = await nlpResponse.json();
        const caseEmbedding = nlpData.embedding;

        const searchResult = await qdrantClient.search("prosecutor_cases", {
          vector: caseEmbedding,
          limit: 5,
          filter: { must_not: [{ key: "id", match: { value: caseId } }] }, // Exclude self
          with_payload: true,
        });

        for (const hit of searchResult) {
          recommendations.push({
            id: `rec-case-${hit.id}`,
            type: "link_case",
            title: `Review Similar Case: ${hit.payload?.title || "Untitled"}`,
            description: `This case has a similarity score of ${hit.score.toFixed(
              2
            )}. It may contain related evidence or criminals.`,
            confidence: hit.score,
            actionData: {
              caseId: hit.id,
              title: hit.payload?.title,
              summary: hit.payload?.aiSummary,
            },
          });
        }
      }
    } catch (vectorError) {
      console.warn("Vector search failed:", vectorError);
      // Continue without vector recommendations
    }
    // 4. Example: Suggest reviewing statutes if case has specific tags
    if (
      currentCase.aiTags &&
      Array.isArray(currentCase.aiTags) &&
      currentCase.aiTags.includes("fraud")
    ) {
      const fraudStatutes = await db
        .select()
        .from(statutes)
        .where(ilike(statutes.title, "%fraud%"))
        .limit(3);
      fraudStatutes.forEach((statute) => {
        recommendations.push({
          id: `rec-statute-${statute.id}`,
          type: "link_statute",
          title: `Review Statute: ${statute.title} (${statute.code})`,
          description: `This statute may be relevant to the fraud aspects of this case.`,
          confidence: 0.75,
          actionData: { statuteId: statute.id, title: statute.title },
        });
      });
    }
    return json({
      success: true,
      recommendations: recommendations
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 10),
    });
  } catch (error: any) {
    console.error("Error generating recommendations:", error);
    return json(
      { error: "Failed to generate recommendations" },
      { status: 500 }
    );
  }
};
