
import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import { eq, and } from "drizzle-orm";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ params, request, locals }) => {
  if (!locals.user) {
    return json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: caseId } = params;
  const { reportType = "case_summary", includeEvidence = true } =
    await request.json();

  try {
    // Verify case ownership and fetch data
    const caseData = await db
      .select()
      .from(cases)
      .where(and(eq(cases.id, caseId), eq(cases.createdBy, locals.user.id)))
      .limit(1);

    if (!caseData.length) {
      return json(
        { error: "Case not found or access denied" },
        { status: 404 }
      );
    }

    const caseRecord = caseData[0];

    // Fetch related evidence if requested
    let evidenceData: any[] = [];
    if (includeEvidence) {
      evidenceData = await db
        .select()
        .from(evidence)
        .where(eq(evidence.caseId, caseId));
    }

    // Generate report content
    const reportContent = {
      case: caseRecord,
      evidence: evidenceData,
      generatedAt: new Date().toISOString(),
      generatedBy: locals.user.id,
    };

    // Create report record
    const newReport = await db
      .insert(reports)
      .values({
        title: `${reportType} - ${caseRecord.title}`,
        content: JSON.stringify(reportContent),
        caseId: caseRecord.id,
        reportType,
        status: "completed",
        createdBy: locals.user.id,
      })
      .returning();

    return json({
      success: true,
      report: newReport[0],
    });
  } catch (error: any) {
    console.error("Report generation failed:", error);
    return json({ error: "Report generation failed" }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ params, locals }) => {
  if (!locals.user) {
    return json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const userReports = await db
      .select()
      .from(reports)
      .where(eq(reports.createdBy, locals.user.id));

    return json({ reports: userReports });
  } catch (error: any) {
    return json({ error: "Failed to fetch reports" }, { status: 500 });
  }
};
