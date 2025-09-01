
import { evidence } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types.js';
import { URL } from "url";


export const POST: RequestHandler = async ({ request, locals }) => {
  const user = locals.user;
  if (!user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    const body = await request.json();
    const { evidenceId, eventId, valid, feedback, corrections } = body;

    if (!evidenceId) {
      return json({ error: "Evidence ID is required" }, { status: 400 });
    }
    // Get the current evidence record
    const evidenceRecords = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    if (evidenceRecords.length === 0) {
      return json({ error: "Evidence not found" }, { status: 404 });
    }
    const evidenceRecord = evidenceRecords[0];

    // Parse existing AI analysis
    let aiAnalysis: any = {};
    try {
      aiAnalysis = JSON.parse(evidenceRecord.aiAnalysis as string || "{}");
    } catch (e: any) {
      console.warn("Failed to parse AI analysis JSON");
    }
    // Create validation record
    const validation = {
      evidenceId,
      eventId: eventId || null,
      userId: user.id,
      valid: Boolean(valid),
      feedback: feedback || null,
      corrections: corrections || null,
      timestamp: new Date().toISOString(),
    };

    // Update AI analysis with validation feedback
    if (!aiAnalysis.validations) {
      aiAnalysis.validations = [];
    }
    aiAnalysis.validations.push(validation);

    // If there are corrections, apply them to the analysis
    if (corrections && typeof corrections === "object") {
      if (corrections.summary) {
        aiAnalysis.summary = corrections.summary;
      }
      if (corrections.tags && Array.isArray(corrections.tags)) {
        aiAnalysis.tags = corrections.tags;
      }
      if (corrections.evidenceType) {
        aiAnalysis.evidenceType = corrections.evidenceType;
      }
    }
    // Calculate validation confidence score
    const validValidations = aiAnalysis.validations.filter(
      (v) => v.valid,
    ).length;
    const totalValidations = aiAnalysis.validations.length;
    aiAnalysis.validationScore =
      totalValidations > 0 ? validValidations / totalValidations : 0;

    // Update the evidence record
    const updateData: any = {
      aiAnalysis: JSON.stringify(aiAnalysis),
      updatedAt: new Date().toISOString(),
    };

    // Update summary if corrected
    if (corrections?.summary) {
      updateData.aiSummary = corrections.summary;
    }
    // Update tags if corrected
    if (corrections?.tags) {
      updateData.aiTags = JSON.stringify(corrections.tags);
    }
    // Update evidence type if corrected
    if (corrections?.evidenceType) {
      updateData.evidenceType = corrections.evidenceType;
    }
    await db
      .update(evidence)
      .set(updateData)
      .where(eq(evidence.id, evidenceId));

    // Store validation in separate table (if you want to track all validations)
    // This would require creating a separate validations table
    // For now, we store it in the aiAnalysis JSON

    const response = {
      success: true,
      validation,
      updatedAnalysis: aiAnalysis,
      message: valid
        ? "Validation recorded successfully"
        : "Correction recorded successfully",
    };

    return json(response);
  } catch (error: any) {
    console.error("Validation error:", error);

    return json(
      {
        success: false,
        error: "Failed to record validation",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async ({ url, locals }) => {
  const user = locals.user;
  if (!user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    const evidenceId = url.searchParams.get("evidenceId");

    if (!evidenceId) {
      return json({ error: "Evidence ID is required" }, { status: 400 });
    }
    // Get validation history for this evidence
    const evidenceRecords = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    if (evidenceRecords.length === 0) {
      return json({ error: "Evidence not found" }, { status: 404 });
    }
    const evidenceRecord = evidenceRecords[0];

    let aiAnalysis: any = {};
    try {
      aiAnalysis = JSON.parse(evidenceRecord.aiAnalysis as string || "{}");
    } catch (e: any) {
      console.warn("Failed to parse AI analysis JSON");
    }
    const validations = (aiAnalysis as any).validations || [];

    return json({
      success: true,
      evidenceId,
      validations,
      validationScore: (aiAnalysis as any).validationScore || 0,
      totalValidations: validations.length,
      validValidations: validations.filter((v) => v.valid).length,
    });
  } catch (error: any) {
    console.error("Get validations error:", error);

    return json(
      {
        success: false,
        error: "Failed to retrieve validations",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};
