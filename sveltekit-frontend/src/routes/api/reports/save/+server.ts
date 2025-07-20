import { aiReports } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { randomUUID } from "crypto";
import { and, eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      caseId,
      reportType,
      title,
      content,
      richTextContent,
      metadata,
      canvasElements,
    } = await request.json();

    if (!caseId || !title || !content) {
      return json(
        {
          error: "Case ID, title, and content are required",
        },
        { status: 400 },
      );
    }
    const reportData = {
      id: randomUUID(),
      caseId,
      reportType: reportType || "case_notes",
      title,
      content,
      richTextContent: richTextContent || null,
      metadata: {
        generatedAt: new Date().toISOString(),
        modelUsed: "gemma3-legal",
        confidence: metadata?.confidence || 0.85,
        keyPoints: metadata?.keyPoints || [],
        recommendations: metadata?.recommendations || [],
        riskFactors: metadata?.riskFactors || [],
        ...metadata,
      },
      canvasElements: canvasElements || [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    const [savedReport] = await db
      .insert(aiReports)
      .values(reportData)
      .returning();

    return json({
      success: true,
      report: savedReport,
      message: "Report saved successfully",
    });
  } catch (error) {
    console.error("Report save error:", error);
    return json(
      {
        error: "Failed to save report",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get("caseId");
    const reportType = url.searchParams.get("reportType");

    if (!caseId) {
      return json({ error: "Case ID is required" }, { status: 400 });
    }
    let query = db.select().from(aiReports).where(eq(aiReports.caseId, caseId));

    if (reportType) {
      query = query.where(
        and(eq(aiReports.caseId, caseId), eq(aiReports.reportType, reportType)),
      );
    }
    const reports = await query.orderBy(aiReports.createdAt);

    return json({
      reports,
    });
  } catch (error) {
    console.error("Reports load error:", error);
    return json(
      {
        error: "Failed to load reports",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};
