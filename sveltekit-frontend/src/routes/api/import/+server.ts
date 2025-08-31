
import { db } from "$lib/server/db/index";
import { cases, criminals, evidence } from "drizzle-orm";
import { json } from "@sveltejs/kit";
import type { RequestHandler } from './$types';



export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const importType = formData.get("type") as string;
    const overwriteExisting = formData.get("overwrite") === "true";

    if (!file) {
      return json({ error: "No file provided" }, { status: 400 });
    }
    const fileContent = await file.text();
    let data: any;

    // Parse file based on type
    try {
      switch (file.type) {
        case "application/json":
          data = JSON.parse(fileContent);
          break;
        case "text/csv":
          data = parseCSV(fileContent);
          break;
        case "application/xml":
        case "text/xml":
          data = parseXML(fileContent);
          break;
        default:
          // Try to parse as JSON by default
          data = JSON.parse(fileContent);
      }
    } catch (parseError) {
      return json(
        { error: "Invalid file format or corrupted data" },
        { status: 400 }
      );
    }
    const results = {
      imported: 0,
      updated: 0,
      skipped: 0,
      errors: [] as string[],
    };

    // Process import based on type
    switch (importType) {
      case "cases":
        await importCases(data, overwriteExisting, results);
        break;
      case "evidence":
        await importEvidence(data, overwriteExisting, results);
        break;
      case "criminals":
        await importParticipants(data, overwriteExisting, results);
        break;
      case "all":
        if (data.cases)
          await importCases(data.cases, overwriteExisting, results);
        if (data.evidence)
          await importEvidence(data.evidence, overwriteExisting, results);
        if (data.criminals)
          await importParticipants(data.criminals, overwriteExisting, results);
        break;
      default:
        return json({ error: "Invalid import type" }, { status: 400 });
    }
    return json({
      success: true,
      results,
      message: `Import completed: ${results.imported} imported, ${results.updated} updated, ${results.skipped} skipped`,
    });
  } catch (error: any) {
    console.error("Import error:", error);
    return json(
      {
        error: error instanceof Error ? error.message : "Import failed",
      },
      { status: 500 }
    );
  }
};

async function importCases(
  casesData: any[],
  overwriteExisting: boolean,
  results: any
): Promise<any> {
  if (!Array.isArray(casesData)) {
    results.errors.push("Cases data must be an array");
    return;
  }
  for (const caseData of casesData) {
    try {
      // Validate required fields
      if (!caseData.title || !caseData.status) {
        results.errors.push(
          `Case missing required fields: ${JSON.stringify(caseData)}`
        );
        results.skipped++;
        continue;
      }
      // Check if case exists
      const existingCase = caseData.id
        ? await db
            .select()
            .from(cases)
            .where(eq(cases.id, caseData.id))
            .limit(1)
        : [];

      if (existingCase.length > 0) {
        if (overwriteExisting) {
          await db
            .update(cases)
            .set({
              title: caseData.title,
              description: caseData.description,
              status: caseData.status,
              priority: caseData.priority,
              updatedAt: new Date(),
            })
            .where(eq(cases.id, caseData.id));
          results.updated++;
        } else {
          results.skipped++;
        }
      } else {
        // Create new case - map to correct schema fields
        const newCase = {
          title: caseData.title,
          caseNumber: caseData.caseNumber || `CASE-${Date.now()}`,
          description: caseData.description || "",
          status: caseData.status || "open",
          priority: caseData.priority || "medium",
          createdAt: caseData.created_at
            ? new Date(caseData.created_at)
            : new Date(),
          updatedAt: new Date(),
        };

        await db.insert(cases).values(newCase);
        results.imported++;
      }
    } catch (error: any) {
      results.errors.push(
        `Error importing case: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      results.skipped++;
    }
  }
}
async function importEvidence(
  evidenceData: any[],
  overwriteExisting: boolean,
  results: any
): Promise<any> {
  if (!Array.isArray(evidenceData)) {
    results.errors.push("Evidence data must be an array");
    return;
  }
  for (const evidenceItem of evidenceData) {
    try {
      // Validate required fields
      if (
        !evidenceItem.case_id ||
        !evidenceItem.type ||
        !evidenceItem.description
      ) {
        results.errors.push(
          `Evidence missing required fields: ${JSON.stringify(evidenceItem)}`
        );
        results.skipped++;
        continue;
      }
      // Check if evidence exists
      const existingEvidence = evidenceItem.id
        ? await db
            .select()
            .from(evidence)
            .where(eq(evidence.id, evidenceItem.id))
            .limit(1)
        : [];

      if (existingEvidence.length > 0) {
        if (overwriteExisting) {
          await db
            .update(evidence)
            .set({
              caseId: evidenceItem.case_id,
              evidenceType: evidenceItem.type,
              description: evidenceItem.description,
              fileUrl: evidenceItem.file_path,
              updatedAt: new Date(),
            })
            .where(eq(evidence.id, evidenceItem.id));
          results.updated++;
        } else {
          results.skipped++;
        }
      } else {
        // Create new evidence - map to correct schema fields
        const newEvidence = {
          caseId: evidenceItem.case_id,
          title: evidenceItem.title || "Imported Evidence",
          evidenceType: evidenceItem.type || "document",
          description: evidenceItem.description,
          fileUrl: evidenceItem.file_path || null,
          createdAt: evidenceItem.created_at
            ? new Date(evidenceItem.created_at)
            : new Date(),
          updatedAt: new Date(),
          // Only include fields that exist in the schema
          tags: evidenceItem.tags || [],
          chainOfCustody: evidenceItem.chain_of_custody || [],
          labAnalysis: evidenceItem.lab_analysis || {},
          aiAnalysis: evidenceItem.ai_analysis || {},
          aiTags: evidenceItem.ai_tags || [],
          aiSummary: evidenceItem.ai_summary || null,
          summary: evidenceItem.summary || null,
          isAdmissible:
            evidenceItem.is_admissible !== undefined
              ? evidenceItem.is_admissible
              : true,
          confidentialityLevel:
            evidenceItem.confidentiality_level || "standard",
          canvasPosition: evidenceItem.canvas_position || {},
          uploadedBy: evidenceItem.uploaded_by || null,
          uploadedAt: evidenceItem.uploaded_at
            ? new Date(evidenceItem.uploaded_at)
            : new Date(),
        };

        await db.insert(evidence).values(newEvidence);
        results.imported++;
      }
    } catch (error: any) {
      results.errors.push(
        `Error importing evidence: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      results.skipped++;
    }
  }
}
async function importParticipants(
  participantsData: any[],
  overwriteExisting: boolean,
  results: any
): Promise<any> {
  if (!Array.isArray(participantsData)) {
    results.errors.push("Participants data must be an array");
    return;
  }
  for (const participant of participantsData) {
    try {
      // Validate required fields
      if (!participant.case_id || !participant.name || !participant.role) {
        results.errors.push(
          `Participant missing required fields: ${JSON.stringify(participant)}`
        );
        results.skipped++;
        continue;
      }
      // Check if participant exists
      const existingParticipant = participant.id
        ? await db
            .select()
            .from(criminals)
            .where(eq(criminals.id, participant.id))
            .limit(1)
        : [];

      if (existingParticipant.length > 0) {
        if (overwriteExisting) {
          await db
            .update(criminals)
            .set({
              firstName: (participant.name || "").split(" ")[0] || "Unknown",
              lastName:
                (participant.name || "").split(" ").slice(1).join(" ") ||
                "Unknown",
              notes: participant.role ? `Role: ${participant.role}` : null,
              email: participant.contact_info?.email || null,
              phone: participant.contact_info?.phone || null,
              updatedAt: new Date(),
            })
            .where(eq(criminals.id, participant.id));
          results.updated++;
        } else {
          results.skipped++;
        }
      } else {
        // Create new participant - map to correct schema fields
        const nameParts = (participant.name || "").split(" ");
        const newParticipant = {
          firstName: nameParts[0] || "Unknown",
          lastName: nameParts.slice(1).join(" ") || "Unknown",
          email: participant.contact_info?.email || null,
          phone: participant.contact_info?.phone || null,
          notes: participant.role ? `Role: ${participant.role}` : null,
          createdAt: participant.created_at
            ? new Date(participant.created_at)
            : new Date(),
          updatedAt: new Date(),
        };

        await db.insert(criminals).values(newParticipant);
        results.imported++;
      }
    } catch (error: any) {
      results.errors.push(
        `Error importing participant: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      results.skipped++;
    }
  }
}
function parseCSV(csvContent: string): unknown[] {
  const lines = csvContent.split("\n");
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""));
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const values = line.split(",").map((v) => v.trim().replace(/"/g, ""));
    const row: any = {};

    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });

    data.push(row);
  }
  return data;
}
function parseXML(xmlContent: string): unknown {
  // Simple XML parser - in production, use a proper XML parser like 'fast-xml-parser'
  try {
    // This is a simplified parser - for production use a proper XML library
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlContent, "text/xml");

    // Convert XML to JSON structure
    const result: any = {};

    function xmlToJson(node: any): unknown {
      const obj: any = {};

      if (node.children && node.children.length > 0) {
        for (const child of node.children) {
          const childName = child.tagName;
          const childValue = xmlToJson(child);

          if (obj[childName]) {
            if (Array.isArray(obj[childName])) {
              obj[childName].push(childValue);
            } else {
              obj[childName] = [obj[childName], childValue];
            }
          } else {
            obj[childName] = childValue;
          }
        }
      } else {
        return node.textContent || "";
      }
      return obj;
    }
    return xmlToJson(xmlDoc.documentElement);
  } catch (error: any) {
    throw new Error("Invalid XML format");
  }
}
