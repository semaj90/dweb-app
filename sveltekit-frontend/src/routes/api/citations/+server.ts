
import { citationPoints } from "$lib/server/db/unified-schema";
import type { RequestEvent } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types.js';
import { URL } from "url";


// Sample citations for when database is not available or for demo data
const sampleCitations = [
  {
    id: "1",
    title: "Miranda v. Arizona",
    content:
      "The Court held that both inculpatory and exculpatory statements made in response to interrogation by a defendant in police custody will be admissible at trial only if the prosecution can show that the defendant was informed of the right to consult with an attorney.",
    author: "U.S. Supreme Court",
    date: "1966",
    source: "384 U.S. 436",
    type: "case",
    tags: ["criminal procedure", "constitutional law", "miranda rights"],
    createdAt: new Date("2024-01-15"),
    updatedAt: new Date("2024-01-15"),
  },
  {
    id: "2",
    title: "Federal Rules of Evidence Rule 404",
    content:
      "Evidence of a person's character or character trait is not admissible to prove that on a particular occasion the person acted in accordance with the character or trait.",
    source: "Fed. R. Evid. 404",
    type: "statute",
    tags: ["evidence", "character evidence", "federal rules"],
    createdAt: new Date("2024-01-16"),
    updatedAt: new Date("2024-01-16"),
  },
  {
    id: "3",
    title: "Daubert v. Merrell Dow Pharmaceuticals",
    content:
      "The Federal Rules of Evidence, not Frye, provide the standard for admitting expert scientific testimony in federal court.",
    author: "U.S. Supreme Court",
    date: "1993",
    source: "509 U.S. 579",
    type: "case",
    tags: ["expert testimony", "scientific evidence", "daubert standard"],
    createdAt: new Date("2024-01-17"),
    updatedAt: new Date("2024-01-17"),
  },
];

// UUID validation regex
const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function isValidUUID(uuid: string): boolean {
  return UUID_REGEX.test(uuid);
}
function isDemoCase(caseId: string): boolean {
  return caseId === "demo-case-123" || caseId.startsWith("demo-");
}
export async function GET({ url }: RequestEvent): Promise<any> {
  const caseId = url.searchParams.get("caseId");
  const reportId = url.searchParams.get("reportId");
  const type = url.searchParams.get("type");
  const search = url.searchParams.get("search");
  const bookmarked = url.searchParams.get("bookmarked");
  const recent = url.searchParams.get("recent");
  const limit = parseInt(url.searchParams.get("limit") || "10");
  const offset = parseInt(url.searchParams.get("offset") || "0");

  try {
    // Handle demo cases or when database is not available
    if (!db || (caseId && isDemoCase(caseId))) {
      let filteredCitations = sampleCitations;

      // Apply filters to sample data
      if (search) {
        const query = search.toLowerCase();
        filteredCitations = sampleCitations.filter(
          (citation) =>
            citation.title.toLowerCase().includes(query) ||
            citation.content.toLowerCase().includes(query) ||
            citation.author?.toLowerCase().includes(query) ||
            citation.source?.toLowerCase().includes(query) ||
            citation.tags?.some((tag) => tag.toLowerCase().includes(query)),
        );
      }
      if (type) {
        filteredCitations = filteredCitations.filter(
          (citation) => citation.type === type,
        );
      }
      if (recent === "true") {
        filteredCitations = filteredCitations.slice(0, 5);
      }
      return json({
        citations: filteredCitations.slice(offset, offset + limit),
        recentCitations: sampleCitations.slice(0, 5),
        total: filteredCitations.length,
      });
    }
    // Validate UUID for database queries
    if (caseId && !isValidUUID(caseId)) {
      return json(
        {
          error: "Invalid case ID format. Expected UUID format.",
          citations: [],
          total: 0,
        },
        { status: 400 },
      );
    }
    if (reportId && !isValidUUID(reportId)) {
      return json(
        {
          error: "Invalid report ID format. Expected UUID format.",
          citations: [],
          total: 0,
        },
        { status: 400 },
      );
    }
    let query = db.select().from(citationPoints);
    const conditions: any[] = [];

    if (caseId) {
      conditions.push(eq(citationPoints.caseId, caseId));
    }
    if (reportId) {
      conditions.push(eq(citationPoints.reportId, reportId));
    }
    if (type) {
      conditions.push(eq(citationPoints.type, type));
    }
    if (search) {
      conditions.push(like(citationPoints.text, `%${search}%`));
    }
    if (bookmarked === "true") {
      conditions.push(eq(citationPoints.isBookmarked, true));
    }
    if (conditions.length > 0) {
      query = query.where(and(...conditions)) as any;
    }
    const results = await query
      .orderBy(
        desc(citationPoints.relevanceScore),
        desc(citationPoints.createdAt),
      )
      .limit(limit)
      .offset(offset);

    return json({
      citations: results,
      total: results.length,
    });
  } catch (error: any) {
    console.error("Error fetching citation points:", error);

    // Return sample data as fallback
    return json({
      citations: sampleCitations.slice(offset, offset + limit),
      total: sampleCitations.length,
      error: "Database error, showing sample data",
    });
  }
}
export async function POST({ request }: RequestEvent): Promise<any> {
  try {
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    // Validate required fields
    if (!data.text || !data.source) {
      return json({ error: "Text and source are required" }, { status: 400 });
    }
    // Validate UUIDs if provided
    if (data.caseId && !isDemoCase(data.caseId) && !isValidUUID(data.caseId)) {
      return json({ error: "Invalid case ID format" }, { status: 400 });
    }
    if (data.reportId && !isValidUUID(data.reportId)) {
      return json({ error: "Invalid report ID format" }, { status: 400 });
    }
    if (data.evidenceId && !isValidUUID(data.evidenceId)) {
      return json({ error: "Invalid evidence ID format" }, { status: 400 });
    }
    if (data.statuteId && !isValidUUID(data.statuteId)) {
      return json({ error: "Invalid statute ID format" }, { status: 400 });
    }
    const citationData = {
      text: data.text,
      source: data.source,
      page: data.page || null,
      context: data.context || "",
      type: data.type || "statute",
      jurisdiction: data.jurisdiction || "",
      tags: data.tags || [],
      caseId: data.caseId && !isDemoCase(data.caseId) ? data.caseId : null,
      reportId: data.reportId || null,
      evidenceId: data.evidenceId || null,
      statuteId: data.statuteId || null,
      aiSummary: data.aiSummary || null,
      relevanceScore: data.relevanceScore || "0.0",
      metadata: data.metadata || {},
      isBookmarked: data.isBookmarked || false,
      usageCount: 0,
      createdBy: "1", // Default user ID for now
    };

    const [newCitation] = await db
      .insert(citationPoints)
      .values(citationData)
      .returning();

    return json(newCitation, { status: 201 });
  } catch (error: any) {
    console.error("Error creating citation point:", error);
    return json({ error: "Failed to create citation point" }, { status: 500 });
  }
}
export async function PUT({ request }: RequestEvent): Promise<any> {
  try {
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const data = await request.json();

    if (!data.id) {
      return json({ error: "Citation ID is required" }, { status: 400 });
    }
    if (!isValidUUID(data.id)) {
      return json({ error: "Invalid citation ID format" }, { status: 400 });
    }
    // Check if citation exists
    const existingCitation = await db
      .select()
      .from(citationPoints)
      .where(eq(citationPoints.id, data.id))
      .limit(1);

    if (!existingCitation.length) {
      return json({ error: "Citation not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      text: data.text,
      source: data.source,
      page: data.page,
      context: data.context,
      type: data.type,
      jurisdiction: data.jurisdiction,
      tags: data.tags,
      caseId: data.caseId,
      reportId: data.reportId,
      evidenceId: data.evidenceId,
      statuteId: data.statuteId,
      aiSummary: data.aiSummary,
      relevanceScore: data.relevanceScore,
      metadata: data.metadata,
      isBookmarked: data.isBookmarked,
      usageCount: data.usageCount,
      updatedAt: new Date().toISOString(),
    };

    // Remove undefined values
    Object.keys(updateData).forEach((key) => {
      if (updateData[key] === undefined) {
        delete updateData[key];
      }
    });

    const [updatedCitation] = await db
      .update(citationPoints)
      .set(updateData)
      .where(eq(citationPoints.id, data.id))
      .returning();

    return json(updatedCitation);
  } catch (error: any) {
    console.error("Error updating citation point:", error);
    return json({ error: "Failed to update citation point" }, { status: 500 });
  }
}
export async function DELETE({ url }: RequestEvent): Promise<any> {
  try {
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const citationId = url.searchParams.get("id");
    if (!citationId) {
      return json({ error: "Citation ID is required" }, { status: 400 });
    }
    if (!isValidUUID(citationId)) {
      return json({ error: "Invalid citation ID format" }, { status: 400 });
    }
    // Check if citation exists
    const existingCitation = await db
      .select()
      .from(citationPoints)
      .where(eq(citationPoints.id, citationId))
      .limit(1);

    if (!existingCitation.length) {
      return json({ error: "Citation not found" }, { status: 404 });
    }
    // Delete the citation
    await db.delete(citationPoints).where(eq(citationPoints.id, citationId));

    return json({ success: true });
  } catch (error: any) {
    console.error("Error deleting citation point:", error);
    return json({ error: "Failed to delete citation point" }, { status: 500 });
  }
}
