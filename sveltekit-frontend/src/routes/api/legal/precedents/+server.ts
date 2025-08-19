import { URL } from "url";
// @ts-nocheck
// Legal Precedents API - Context7 Enhanced Legal Research
import { db } from "$lib/server/db/index";
// Orphaned content: import {
type RequestHandler, json
import { legalPrecedents } from "$lib/server/db/schema-postgres";
import { eq, desc, and, like, or, sql, , interface PrecedentSearchRequest {,   query: string;,   jurisdiction?: string;,   court?: string;,   yearFrom?: number;,   yearTo?: number;,   legalPrinciples?: string[];,   limit?: number; } from

interface PrecedentSearchResponse {
  precedents: Array<{
    id: string;
    caseTitle: string;
    citation: string;
    court: string;
    year: number;
    jurisdiction: string;
    summary: string;
    relevanceScore: number;
    legalPrinciples: string[];
    linkedCases: string[];
  }>;
  totalCount: number;
  searchTerms: string[];
  processingTime: number;
}

export const GET: RequestHandler = async ({ url }) => {
  const startTime = Date.now();

  try {
    const searchParams = url.searchParams;
    const query = searchParams.get("query") || "";
    const jurisdiction = searchParams.get("jurisdiction");
    const court = searchParams.get("court");
    const yearFrom = searchParams.get("yearFrom")
      ? parseInt(searchParams.get("yearFrom")!)
      : undefined;
    const yearTo = searchParams.get("yearTo")
      ? parseInt(searchParams.get("yearTo")!)
      : undefined;
    const limit = parseInt(searchParams.get("limit") || "20");
    const offset = parseInt(searchParams.get("offset") || "0");

    // Build search conditions
    const conditions = [];

    if (query) {
      conditions.push(
        or(
          like(legalPrecedents.caseTitle, `%${query}%`),
          like(legalPrecedents.summary, `%${query}%`),
          like(legalPrecedents.fullText, `%${query}%`),
        ),
      );
    }

    if (jurisdiction) {
      conditions.push(eq(legalPrecedents.jurisdiction, jurisdiction));
    }

    if (court) {
      conditions.push(like(legalPrecedents.court, `%${court}%`));
    }

    if (yearFrom) {
      conditions.push(sql`${legalPrecedents.year} >= ${yearFrom}`);
    }

    if (yearTo) {
      conditions.push(sql`${legalPrecedents.year} <= ${yearTo}`);
    }

    // Execute search query
    const precedentsQuery = db
      .select()
      .from(legalPrecedents)
      .limit(limit)
      .offset(offset)
      .orderBy(
        desc(legalPrecedents.relevanceScore),
        desc(legalPrecedents.year),
      );

    if (conditions.length > 0) {
      precedentsQuery.where(and(...conditions));
    }

    const precedents = await precedentsQuery;

    // Get total count for pagination
    const countQuery = db
      .select({ count: sql`count(*)` })
      .from(legalPrecedents);

    if (conditions.length > 0) {
      countQuery.where(and(...conditions));
    }

    const [{ count }] = await countQuery;

    const response: PrecedentSearchResponse = {
      precedents: precedents.map((prec) => ({
        id: prec.id,
        caseTitle: prec.caseTitle,
        citation: prec.citation,
        court: prec.court || "",
        year: prec.year || 0,
        jurisdiction: prec.jurisdiction || "",
        summary: prec.summary || "",
        relevanceScore: parseFloat(prec.relevanceScore?.toString() || "0"),
        legalPrinciples: (prec.legalPrinciples as string[]) || [],
        linkedCases: (prec.linkedCases as string[]) || [],
      })),
      totalCount: parseInt(count.toString()),
      searchTerms: query.split(" ").filter((term) => term.length > 2),
      processingTime: Date.now() - startTime,
    };

    return json(response);
  } catch (error) {
    console.error("Error searching legal precedents:", error);
    return json(
      { error: "Failed to search legal precedents" },
      { status: 500 },
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const precedentData = await request.json();

    // Validate required fields
    if (!precedentData.caseTitle || !precedentData.citation) {
      return json(
        { error: "Missing required fields: caseTitle, citation" },
        { status: 400 },
      );
    }

    // Create new precedent record
    const [newPrecedent] = await db
      .insert(legalPrecedents)
      .values({
        caseTitle: precedentData.caseTitle,
        citation: precedentData.citation,
        court: precedentData.court,
        year: precedentData.year,
        jurisdiction: precedentData.jurisdiction,
        summary: precedentData.summary,
        fullText: precedentData.fullText,
        embedding: precedentData.embedding, // Vector embedding for similarity search
        relevanceScore: precedentData.relevanceScore || 0.5,
        legalPrinciples: precedentData.legalPrinciples || [],
        linkedCases: precedentData.linkedCases || [],
      })
      .returning();

    return json(newPrecedent, { status: 201 });
  } catch (error) {
    console.error("Error creating legal precedent:", error);
    return json({ error: "Failed to create legal precedent" }, { status: 500 });
  }
};

// Get similar precedents using vector similarity (placeholder for future implementation)
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { precedentId, queryVector } = await request.json();

    if (!precedentId || !queryVector) {
      return json(
        { error: "Missing required fields: precedentId, queryVector" },
        { status: 400 },
      );
    }

    // This would typically use vector similarity search with pgvector
    // For now, return related precedents based on legal principles
    const basePrecedent = await db
      .select()
      .from(legalPrecedents)
      .where(eq(legalPrecedents.id, precedentId))
      .limit(1);

    if (basePrecedent.length === 0) {
      return json({ error: "Precedent not found" }, { status: 404 });
    }

    const legalPrinciples =
      (basePrecedent[0].legalPrinciples as string[]) || [];

    // Find similar precedents based on legal principles
    const similarPrecedents = await db
      .select()
      .from(legalPrecedents)
      .where(
        and(
          sql`${legalPrecedents.id} != ${precedentId}`,
          // This would be replaced with proper vector similarity in production
          legalPrinciples.length > 0
            ? or(
                ...legalPrinciples.map((principle) =>
                  like(legalPrecedents.summary, `%${principle}%`),
                ),
              )
            : sql`1=1`,
        ),
      )
      .orderBy(desc(legalPrecedents.relevanceScore))
      .limit(10);

    return json({
      basePrecedent: basePrecedent[0],
      similarPrecedents,
      similarityMethod: "legal_principles",
      count: similarPrecedents.length,
    });
  } catch (error) {
    console.error("Error finding similar precedents:", error);
    return json(
      { error: "Failed to find similar precedents" },
      { status: 500 },
    );
  }
};
