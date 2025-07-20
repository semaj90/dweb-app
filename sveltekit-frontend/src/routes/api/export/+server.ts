import { db } from "$lib/server/db/index";
import { cases, evidence } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { count, desc, sql } from "drizzle-orm";
import { z } from "zod";
import type { RequestHandler } from "./$types";

// Export request schema
const ExportRequestSchema = z.object({
  format: z.enum(["json", "csv", "xml"]).default("json"),
  includeEvidence: z.boolean().default(true),
  includeCases: z.boolean().default(true),
  includeAnalytics: z.boolean().default(false),
  dateRange: z
    .object({
      from: z.string().optional(),
      to: z.string().optional(),
    })
    .optional(),
  caseIds: z.array(z.string()).optional(),
});

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    // Check authentication
    const sessionId = cookies.get("session_id");
    if (!sessionId) {
      return json(
        { success: false, error: "Authentication required" },
        { status: 401 },
      );
    }
    const body = await request.json();
    const validatedData = ExportRequestSchema.parse(body);

    const {
      format,
      includeEvidence,
      includeCases,
      includeAnalytics,
      dateRange,
      caseIds,
    } = validatedData;

    let exportData: any = {
      metadata: {
        exportedAt: new Date().toISOString(),
        format,
        includeEvidence,
        includeCases,
        includeAnalytics,
      },
    };

    // Export cases
    if (includeCases) {
      let caseQuery = db.select().from(cases);

      if (caseIds?.length) {
        caseQuery = caseQuery.where(sql`${cases.id} IN (${caseIds.join(",")})`);
      }
      if (dateRange?.from) {
        caseQuery = caseQuery.where(
          sql`${cases.createdAt} >= ${dateRange.from}`,
        );
      }
      if (dateRange?.to) {
        caseQuery = caseQuery.where(sql`${cases.createdAt} <= ${dateRange.to}`);
      }
      const casesData = await caseQuery.orderBy(desc(cases.createdAt));
      exportData.cases = casesData;
    }
    // Export evidence
    if (includeEvidence) {
      let evidenceQuery = db.select().from(evidence);

      if (caseIds?.length) {
        evidenceQuery = evidenceQuery.where(
          sql`${evidence.caseId} IN (${caseIds.join(",")})`,
        );
      }
      if (dateRange?.from) {
        evidenceQuery = evidenceQuery.where(
          sql`${evidence.createdAt} >= ${dateRange.from}`,
        );
      }
      if (dateRange?.to) {
        evidenceQuery = evidenceQuery.where(
          sql`${evidence.createdAt} <= ${dateRange.to}`,
        );
      }
      const evidenceData = await evidenceQuery.orderBy(
        desc(evidence.createdAt),
      );
      exportData.evidence = evidenceData;
    }
    // Export analytics
    if (includeAnalytics) {
      const analytics = {
        totalCases: await db.select({ count: count() }).from(cases),
        totalEvidence: await db.select({ count: count() }).from(evidence),
        casesByStatus: await db
          .select({
            status: cases.status,
            count: count(),
          })
          .from(cases)
          .groupBy(cases.status),
        evidenceByType: await db
          .select({
            type: evidence.evidenceType,
            count: count(),
          })
          .from(evidence)
          .groupBy(evidence.evidenceType),
      };
      exportData.analytics = analytics;
    }
    // Format data based on requested format
    let responseData: string;
    let contentType: string;
    let fileName: string;

    switch (format) {
      case "csv":
        responseData = convertToCSV(exportData);
        contentType = "text/csv";
        fileName = `legal-data-export-${new Date().toISOString().split("T")[0]}.csv`;
        break;

      case "xml":
        responseData = convertToXML(exportData);
        contentType = "application/xml";
        fileName = `legal-data-export-${new Date().toISOString().split("T")[0]}.xml`;
        break;

      default: // json
        responseData = JSON.stringify(exportData, null, 2);
        contentType = "application/json";
        fileName = `legal-data-export-${new Date().toISOString().split("T")[0]}.json`;
    }
    return new Response(responseData, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `attachment; filename="${fileName}"`,
        "Content-Length": responseData.length.toString(),
      },
    });
  } catch (error) {
    console.error("Export error:", error);
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Export failed",
      },
      { status: 500 },
    );
  }
};

function convertToCSV(data: any): string {
  let csv = "";

  // Export cases as CSV
  if (data.cases?.length > 0) {
    csv += "CASES\n";
    const caseHeaders = Object.keys(data.cases[0]).join(",");
    csv += caseHeaders + "\n";

    for (const caseItem of data.cases) {
      const row = Object.values(caseItem)
        .map((value) =>
          typeof value === "string" ? `"${value.replace(/"/g, '""')}"` : value,
        )
        .join(",");
      csv += row + "\n";
    }
    csv += "\n";
  }
  // Export evidence as CSV
  if (data.evidence?.length > 0) {
    csv += "EVIDENCE\n";
    const evidenceHeaders = Object.keys(data.evidence[0]).join(",");
    csv += evidenceHeaders + "\n";

    for (const evidenceItem of data.evidence) {
      const row = Object.values(evidenceItem)
        .map((value) =>
          typeof value === "string" ? `"${value.replace(/"/g, '""')}"` : value,
        )
        .join(",");
      csv += row + "\n";
    }
  }
  return csv;
}
function convertToXML(data: any): string {
  let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<legal_data_export>\n';

  xml += `  <metadata>\n`;
  xml += `    <exported_at>${data.metadata.exportedAt}</exported_at>\n`;
  xml += `    <format>${data.metadata.format}</format>\n`;
  xml += `  </metadata>\n`;

  if (data.cases?.length > 0) {
    xml += "  <cases>\n";
    for (const caseItem of data.cases) {
      xml += "    <case>\n";
      for (const [key, value] of Object.entries(caseItem)) {
        xml += `      <${key}>${escapeXml(String(value))}</${key}>\n`;
      }
      xml += "    </case>\n";
    }
    xml += "  </cases>\n";
  }
  if (data.evidence?.length > 0) {
    xml += "  <evidence>\n";
    for (const evidenceItem of data.evidence) {
      xml += "    <evidence_item>\n";
      for (const [key, value] of Object.entries(evidenceItem)) {
        xml += `      <${key}>${escapeXml(String(value))}</${key}>\n`;
      }
      xml += "    </evidence_item>\n";
    }
    xml += "  </evidence>\n";
  }
  if (data.analytics) {
    xml += "  <analytics>\n";
    xml += `    <total_cases>${data.analytics.totalCases[0]?.count || 0}</total_cases>\n`;
    xml += `    <total_evidence>${data.analytics.totalEvidence[0]?.count || 0}</total_evidence>\n`;
    xml += "  </analytics>\n";
  }
  xml += "</legal_data_export>";
  return xml;
}
function escapeXml(unsafe: string): string {
  return unsafe.replace(/[<>&'"]/g, function (c) {
    switch (c) {
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case "&":
        return "&amp;";
      case "'":
        return "&apos;";
      case '"':
        return "&quot;";
      default:
        return c;
    }
  });
}
