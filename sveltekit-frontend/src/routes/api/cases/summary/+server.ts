import type { Case } from "$lib/types";

import {
  caseEmbeddings,
  chatEmbeddings,
  evidence,
  evidenceVectors,
} from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { count, desc, eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";
import { qdrantService } from "$lib/server/services/qdrant-service";
import VectorService from "$lib/server/services/vector-service";
import { ollamaService } from "$lib/services/ollama-service";
import type { RequestHandler } from "./$types";

interface CaseSummaryRequest {
  caseId: string;
  includeEvidence?: boolean;
  includeTimeline?: boolean;
  analysisDepth?: "basic" | "comprehensive" | "detailed";
  regenerate?: boolean;
}
interface CaseSummaryResponse {
  success: boolean;
  summary?: {
    aiGenerated: boolean;
    overview: string;
    keyFindings: string[];
    recommendations: string[];
    riskAssessment: {
      level: "low" | "medium" | "high";
      factors: string[];
    };
    timeline: Array<{
      date: Date;
      event: string;
      importance: "low" | "medium" | "high";
    }>;
    evidence: {
      total: number;
      admissible: number;
      questionable: number;
      inadmissible: number;
    };
    nextSteps: string[];
    confidence: number;
    generatedAt: Date;
  };
  analytics?: {
    evidenceCount: number;
    documentsReviewed: number;
    witnessesInterviewed: number;
    daysActive: number;
    completionPercentage: number;
  };
  error?: string;
}
export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    // Get user session
    const sessionId = cookies.get("session_id");
    if (!sessionId) {
      return json(
        { success: false, error: "Authentication required" },
        { status: 401 },
      );
    }
    const body: CaseSummaryRequest = await request.json();
    const {
      caseId,
      includeEvidence = true,
      includeTimeline = true,
      analysisDepth = "comprehensive",
      regenerate = false,
    } = body;

    // Validate input
    if (!caseId) {
      return json(
        {
          success: false,
          error: "caseId is required",
        },
        { status: 400 },
      );
    }
    // Check for existing summary if not regenerating
    if (!regenerate) {
      const existingSummary = await db
        .select()
        .from(caseEmbeddings)
        .where(eq(caseEmbeddings.caseId, caseId))
        .orderBy(desc(caseEmbeddings.createdAt))
        .limit(1);

      if (
        existingSummary.length > 0 &&
        (existingSummary[0].metadata as any)?.summary
      ) {
        return json({
          success: true,
          summary: (existingSummary[0].metadata as any).summary,
        } as CaseSummaryResponse);
      }
    }
    // Gather case data
    const caseData = await gatherCaseData(
      caseId,
      includeEvidence,
      includeTimeline,
    );

    // Generate AI summary
    const summary = await generateAISummary(caseData, analysisDepth);

    // Store summary as embedding
    const summaryText = `Case Summary: ${summary.overview}. Key Findings: ${summary.keyFindings.join(". ")}. Recommendations: ${summary.recommendations.join(". ")}.`;

    await VectorService.storeCaseEmbedding({
      caseId,
      content: summaryText,
      embedding: await VectorService.generateEmbedding(summaryText, {
        model: "ollama",
      }),
      summary_type: "ai_generated",
      metadata: {
        summary,
        analysisDepth,
        generatedAt: new Date(),
        includeEvidence,
        includeTimeline,
      },
    });

    // Calculate analytics
    const analytics = await calculateCaseAnalytics(caseId);

    return json({
      success: true,
      summary,
      analytics,
    } as CaseSummaryResponse);
  } catch (error) {
    console.error("Case summary generation error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      } as CaseSummaryResponse,
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async ({ url, cookies }) => {
  try {
    // Get user session
    const sessionId = cookies.get("session_id");
    if (!sessionId) {
      return json(
        { success: false, error: "Authentication required" },
        { status: 401 },
      );
    }
    const caseId = url.searchParams.get("caseId");

    if (!caseId) {
      return json(
        {
          success: false,
          error: "caseId is required",
        },
        { status: 400 },
      );
    }
    // Get latest summary
    const summaryRecord = await db
      .select()
      .from(caseEmbeddings)
      .where(eq(caseEmbeddings.caseId, caseId))
      .orderBy(desc(caseEmbeddings.createdAt))
      .limit(1);

    if (summaryRecord.length === 0) {
      return json(
        {
          success: false,
          error: "No summary found for this case",
        },
        { status: 404 },
      );
    }
    const analytics = await calculateCaseAnalytics(caseId);

    return json({
      success: true,
      summary: (summaryRecord[0].metadata as any)?.summary,
      analytics,
    } as CaseSummaryResponse);
  } catch (error) {
    console.error("Case summary retrieval error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      } as CaseSummaryResponse,
      { status: 500 },
    );
  }
};

async function gatherCaseData(
  caseId: string,
  includeEvidence: boolean,
  includeTimeline: boolean,
) {
  const data: any = { caseId };

  if (includeEvidence) {
    // Get evidence data
    const evidenceData = await db
      .select()
      .from(evidenceVectors)
      .innerJoin(evidence, eq(evidenceVectors.evidenceId, evidence.id))
      .where(eq(evidence.caseId, caseId));

    data.evidence = evidenceData.map((e) => ({
      id: e.evidenceId,
      content: e.content,
      metadata: e.metadata,
      createdAt: e.createdAt,
    }));

    // Get evidence analytics from Qdrant
    const evidenceAnalytics = await qdrantService.getEvidenceAnalytics(caseId);
    data.evidenceAnalytics = evidenceAnalytics;
  }
  if (includeTimeline) {
    // Get chat/interaction history for timeline
    const interactions = await db
      .select()
      .from(chatEmbeddings)
      .where(eq(chatEmbeddings.conversationId, `case_${caseId}`))
      .orderBy(desc(chatEmbeddings.createdAt));

    data.timeline = interactions.map((i) => ({
      date: i.createdAt,
      event: i.content.substring(0, 100) + "...",
      type: i.role,
      importance: determineImportance(i.content),
    }));
  }
  return data;
}
async function generateAISummary(caseData: any, depth: string) {
  const evidenceText =
    caseData.evidence?.map((e: any) => e.content).join("\n") || "";
  const timelineText =
    caseData.timeline?.map((t: any) => `${t.date}: ${t.event}`).join("\n") ||
    "";

  const analysisPrompt = `
As a legal expert, generate a comprehensive case summary based on the following data:

CASE ID: ${caseData.caseId}

EVIDENCE DATA:
${evidenceText.substring(0, 4000)}

TIMELINE DATA:
${timelineText.substring(0, 2000)}

EVIDENCE ANALYTICS:
- Total Evidence: ${caseData.evidenceAnalytics?.totalEvidence || 0}
- Evidence by Type: ${JSON.stringify(caseData.evidenceAnalytics?.evidenceByType || {})}
- Top Tags: ${caseData.evidenceAnalytics?.topTags?.map((t: any) => t.tag).join(", ") || "None"}

Generate a ${depth} analysis with the following JSON structure:
{
  "aiGenerated": true,
  "overview": "2-3 paragraph case overview",
  "keyFindings": ["finding1", "finding2", "finding3"],
  "recommendations": ["recommendation1", "recommendation2", "recommendation3"],
  "riskAssessment": {
    "level": "low|medium|high",
    "factors": ["factor1", "factor2"]
  },
  "timeline": [
    {
      "date": "2024-01-01T00:00:00.000Z",
      "event": "Key event description",
      "importance": "low|medium|high"
}
  ],
  "evidence": {
    "total": 0,
    "admissible": 0,
    "questionable": 0,
    "inadmissible": 0
  },
  "nextSteps": ["step1", "step2", "step3"],
  "confidence": 0.85,
  "generatedAt": "${new Date().toISOString()}"
}
Focus on:
- Legal accuracy and compliance
- Evidence evaluation and admissibility
- Strategic recommendations
- Risk factors and mitigation
- Actionable next steps

Respond with valid JSON only.`;

  const response = await ollamaService.generateResponse(analysisPrompt, {
    model: "gemma3-legal",
    max_tokens: 2000,
    temperature: 0.3,
  });

  if (response.response) {
    try {
      const summary = JSON.parse(response.response);

      // Add calculated evidence metrics
      if (caseData.evidenceAnalytics) {
        summary.evidence = {
          total: caseData.evidenceAnalytics.totalEvidence,
          admissible: Math.round(
            caseData.evidenceAnalytics.totalEvidence * 0.7,
          ), // Estimated
          questionable: Math.round(
            caseData.evidenceAnalytics.totalEvidence * 0.2,
          ),
          inadmissible: Math.round(
            caseData.evidenceAnalytics.totalEvidence * 0.1,
          ),
        };
      }
      return summary;
    } catch (parseError) {
      console.error("Failed to parse AI summary:", parseError);
      return generateFallbackSummary(caseData);
    }
  }
  return generateFallbackSummary(caseData);
}
function generateFallbackSummary(caseData: any) {
  return {
    aiGenerated: false,
    overview: `Case ${caseData.caseId} contains ${caseData.evidence?.length || 0} evidence items and requires manual review for comprehensive analysis.`,
    keyFindings: [
      "Evidence collection in progress",
      "Manual analysis required",
    ],
    recommendations: [
      "Conduct thorough evidence review",
      "Engage legal experts",
      "Update case documentation",
    ],
    riskAssessment: {
      level: "medium" as const,
      factors: ["Incomplete analysis", "Requires manual review"],
    },
    timeline: caseData.timeline?.slice(0, 5) || [],
    evidence: {
      total: caseData.evidenceAnalytics?.totalEvidence || 0,
      admissible: 0,
      questionable: 0,
      inadmissible: 0,
    },
    nextSteps: [
      "Complete evidence analysis",
      "Generate detailed summary",
      "Review with legal team",
    ],
    confidence: 0.5,
    generatedAt: new Date(),
  };
}
async function calculateCaseAnalytics(caseId: string) {
  // Get evidence count
  const evidenceCount = await db
    .select({ count: count() })
    .from(evidenceVectors)
    .innerJoin(evidence, eq(evidenceVectors.evidenceId, evidence.id))
    .where(eq(evidence.caseId, caseId));

  // Get interaction count (as proxy for documents reviewed)
  const interactionCount = await db
    .select({ count: count() })
    .from(chatEmbeddings)
    .where(eq(chatEmbeddings.conversationId, `case_${caseId}`));

  // Calculate other metrics (these would come from a proper case management system)
  const evidence = evidenceCount[0]?.count || 0;
  const interactions = interactionCount[0]?.count || 0;

  return {
    evidenceCount: evidence,
    documentsReviewed: interactions,
    witnessesInterviewed: Math.floor(evidence * 0.3), // Estimated
    daysActive: 30, // Would calculate from case creation date
    completionPercentage: Math.min(
      95,
      Math.floor((evidence + interactions) * 10),
    ),
  };
}
function determineImportance(content: string): "low" | "medium" | "high" {
  const highPriorityKeywords = [
    "critical",
    "urgent",
    "evidence",
    "witness",
    "court",
    "trial",
  ];
  const mediumPriorityKeywords = [
    "review",
    "analysis",
    "investigation",
    "statement",
  ];

  const lowerContent = content.toLowerCase();

  if (highPriorityKeywords.some((keyword) => lowerContent.includes(keyword))) {
    return "high";
  }
  if (
    mediumPriorityKeywords.some((keyword) => lowerContent.includes(keyword))
  ) {
    return "medium";
  }
  return "low";
}
