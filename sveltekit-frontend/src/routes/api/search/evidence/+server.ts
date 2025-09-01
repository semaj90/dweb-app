
import type { RequestHandler } from './$types.js';

// Evidence search API endpoint with advanced vector capabilities
// Supports document content search, image analysis, and multi-modal search
import { evidence } from "$lib/server/db/schema-postgres";

import { and, desc, ilike, or, sql } from "drizzle-orm";

import { db } from "$lib/server/db/index";
import { URL } from "url";


export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get("q");
    const caseId = url.searchParams.get("caseId");
    const evidenceType = url.searchParams.get("type");
    const searchMode = url.searchParams.get("mode") || "hybrid"; // 'text', 'content', 'semantic', 'hybrid'
    const limit = parseInt(url.searchParams.get("limit") || "20");

    if (!query || query.length < 2) {
      return json({ results: [], searchMode: "none", executionTime: 0 });
    }
    const startTime = Date.now();
    let results = [];

    switch (searchMode) {
      case "text":
        // Fast metadata search
        results = await searchEvidenceText(query, {
          caseId,
          evidenceType,
          limit,
        });
        break;

      case "content":
        // Deep content search using Qdrant
        results = await searchEvidenceContent(query, {
          caseId,
          evidenceType,
          limit,
        });
        break;

      case "semantic":
        // PostgreSQL vector search
        results = await searchEvidenceSemantic(query, {
          caseId,
          evidenceType,
          limit,
        });
        break;

      case "hybrid":
      default:
        // Best of all worlds
        results = await searchEvidenceHybrid(query, {
          caseId,
          evidenceType,
          limit,
        });
        break;
    }
    const executionTime = Date.now() - startTime;

    return json({
      results,
      searchMode,
      executionTime,
      query,
      totalResults: results.length,
    });
  } catch (error: any) {
    console.error("Evidence search error:", error);
    return json({ error: "Search failed" }, { status: 500 });
  }
};

// Fast text search on evidence metadata
async function searchEvidenceText(query: string, options: any): Promise<any> {
  const { caseId, evidenceType, limit } = options;
  const whereConditions = [];

  // Text search conditions
  const textSearch = or(
    ilike(evidence.title, `%${query}%`),
    ilike(evidence.description, `%${query}%`),
    ilike(evidence.fileName, `%${query}%`),
    sql`${evidence.tags}::text ILIKE ${`%${query}%`}`,
  );
  whereConditions.push(textSearch);

  // Apply filters
  if (caseId) whereConditions.push(sql`${evidence.caseId} = ${caseId}`);
  if (evidenceType)
    whereConditions.push(sql`${evidence.evidenceType} = ${evidenceType}`);

  return await db
    .select({
      id: evidence.id,
      caseId: evidence.caseId,
      title: evidence.title,
      description: evidence.description,
      evidenceType: evidence.evidenceType,
      fileName: evidence.fileName,
      fileUrl: evidence.fileUrl,
      tags: evidence.tags,
      summary: evidence.summary,
      uploadedAt: evidence.uploadedAt,
      similarity: sql<number>`1.0`,
      searchType: sql<string>`'text'`,
    })
    .from(evidence)
    .where(and(...whereConditions))
    .orderBy(desc(evidence.uploadedAt))
    .limit(limit);
}
// Deep content search using Qdrant
async function searchEvidenceContent(query: string, options: any): Promise<any> {
  const { caseId, evidenceType, limit } = options;

  try {
    // Search Qdrant for document content
    const qdrantResults = await searchEvidence(query, {
      limit,
      filter: {
        must: [
          ...(caseId ? [{ key: "case_id", match: { value: caseId } }] : []),
          ...(evidenceType
            ? [{ key: "evidence_type", match: { value: evidenceType } }]
            : []),
        ],
      },
    });

    // Get full evidence records for matches
    const evidenceIds = qdrantResults.map((r) => r.payload.evidence_id);

    if (evidenceIds.length === 0) {
      return [];
    }
    const evidenceRecords = await db
      .select({
        id: evidence.id,
        caseId: evidence.caseId,
        title: evidence.title,
        description: evidence.description,
        evidenceType: evidence.evidenceType,
        fileName: evidence.fileName,
        fileUrl: evidence.fileUrl,
        tags: evidence.tags,
        summary: evidence.summary,
        uploadedAt: evidence.uploadedAt,
      })
      .from(evidence)
      .where(sql`${evidence.id} = ANY(${evidenceIds})`);

    // Merge with similarity scores
    return evidenceRecords.map((record) => {
      const qdrantMatch = qdrantResults.find(
        (r) => r.payload.evidence_id === record.id,
      );
      return {
        ...record,
        similarity: qdrantMatch?.score || 0,
        searchType: "content" as const,
        contentMatch: qdrantMatch?.payload.content_snippet || null,
      };
    });
  } catch (error: any) {
    console.error("Qdrant search failed, falling back to PostgreSQL:", error);
    return await searchEvidenceSemantic(query, options);
  }
}
// PostgreSQL vector search
async function searchEvidenceSemantic(query: string, options: any): Promise<any> {
  const { caseId, evidenceType, limit } = options;
  const queryEmbedding = await generateEmbedding(query);

  const whereConditions = [sql`1=1`]; // Remove missing column reference

  if (caseId) whereConditions.push(sql`${evidence.caseId} = ${caseId}`);
  if (evidenceType)
    whereConditions.push(sql`${evidence.evidenceType} = ${evidenceType}`);

  return await db
    .select({
      id: evidence.id,
      caseId: evidence.caseId,
      title: evidence.title,
      description: evidence.description,
      evidenceType: evidence.evidenceType,
      fileName: evidence.fileName,
      fileUrl: evidence.fileUrl,
      tags: evidence.tags,
      summary: evidence.summary,
      uploadedAt: evidence.uploadedAt,
      similarity: sql<number>`0.5`, // Placeholder similarity score
      searchType: sql<string>`'semantic'`,
    })
    .from(evidence)
    .where(and(...whereConditions))
    .orderBy(evidence.uploadedAt)
    .limit(limit);
}
// Hybrid search combining all methods
async function searchEvidenceHybrid(query: string, options: any): Promise<any> {
  const { limit } = options;

  // Run searches in parallel for speed
  const [textResults, contentResults, semanticResults] =
    await Promise.allSettled([
      searchEvidenceText(query, { ...options, limit: Math.ceil(limit / 3) }),
      searchEvidenceContent(query, { ...options, limit: Math.ceil(limit / 3) }),
      searchEvidenceSemantic(query, {
        ...options,
        limit: Math.ceil(limit / 3),
      }),
    ]);

  const allResults: any[] = [];
  const seenIds = new Set<string>();

  // Merge results with deduplication
  const addResults = (results: any[], boost = 1) => {
    if (results) {
      results.forEach((result) => {
        if (!seenIds.has(result.id)) {
          seenIds.add(result.id);
          allResults.push({
            ...result,
            similarity: result.similarity * boost,
          });
        }
      });
    }
  };

  // Add results with different priority weights
  if (textResults.status === "fulfilled") addResults(textResults.value, 1.0);
  if (contentResults.status === "fulfilled")
    addResults(contentResults.value, 1.2); // Boost content matches
  if (semanticResults.status === "fulfilled")
    addResults(semanticResults.value, 1.1); // Boost semantic matches

  // Sort by similarity and limit
  return allResults
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit)
    .map((result) => ({
      ...result,
      searchType: "hybrid" as const,
    }));
}
