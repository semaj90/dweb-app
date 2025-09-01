
import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/pg";
import { sql } from "drizzle-orm";
import type { RequestHandler } from './$types';
import { legalDocuments } from "$lib/server/db/schema-postgres";
import { nomicEmbeddings } from "$lib/services/nomic-embedding-service";

// Use Nomic embeddings with 768 dimensions (Nomic's default)
const EMBEDDING_DIMENSION = 768;

import { enhancedSearchWithNeo4j, type UserContext, type Neo4jPathContext } from "$lib/ai/custom-reranker";
import { mcpContext72GetLibraryDocs } from "$lib/mcp-context72-get-library-docs";
import { URL } from "url";

export const POST: RequestHandler = async ({ request }) => {
  const {
    query,
    userContext = {},
    neo4jContext = {},
    limit = 10,
    threshold = 0.3,
  } = await request.json();

  if (!query) {
    return json({ error: "Query is required" }, { status: 400 });
  }

  try {
    // Use enhanced reranker with Neo4j context if provided
    const reranked = await enhancedSearchWithNeo4j(
      query,
      userContext as UserContext,
      neo4jContext as Neo4jPathContext,
      limit * 2
    );
    // Enrich with memory and docs for final scoring
    const memory = await accessMemoryMCP(query, userContext);
    const docs = await mcpContext72GetLibraryDocs("svelte", "runes");
    // Final scoring pass
    const filteredResults = reranked
      .map((result) => {
        let score = result.rerankScore;
        if (memory.some((m) => m.relatedId === result.id)) score += 1;
        if (docs && docs.includes(result.intent)) score += 1;
        return { ...result, finalScore: score };
      })
      .filter((result) => (result.confidence ?? 0) / 100 >= threshold)
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, limit)
      .map((result) => ({
        id: result.id,
        title: result.metadata?.title || "",
        documentType: result.metadata?.documentType || "",
        content: result.content,
        similarity: result.confidence / 100,
        caseId: result.metadata?.caseId || "",
        rerankScore: result.rerankScore,
        finalScore: result.finalScore,
        intent: result.intent,
        timeOfDay: result.timeOfDay,
        position: result.position,
      }));

    // Analytics logging (async, non-blocking)
    analyticsLog({
      event: "semantic_search",
      query,
      userContext,
      neo4jContext,
      results: filteredResults.map((r) => r.id),
      timestamp: Date.now(),
    });

    return json({
      success: true,
      results: filteredResults,
      total: filteredResults.length,
      query,
      options: { limit, threshold },
    });
  } catch (error: any) {
    console.error("Semantic search error:", error);
    analyticsLog({
      event: "semantic_search_error",
      query,
      error: error.message,
      timestamp: Date.now(),
    });
    return json(
      { error: "Failed to perform semantic search", success: false },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const query = url.searchParams.get("q");
  const limit = parseInt(url.searchParams.get("limit") || "10");
  const threshold = parseFloat(url.searchParams.get("threshold") || "0.3");

  if (!query) {
    return json({ error: "Query parameter 'q' is required" }, { status: 400 });
  }

  try {
    // Generate embedding for the query using Nomic
    const embeddingResult = await nomicEmbeddings.embed(query);
    const queryEmbedding = embeddingResult.embedding;
    const embeddingString = `[${queryEmbedding.join(",")}]`;

    // Perform vector similarity search
    const results = await db
      .select({
        id: legalDocuments.id,
        title: legalDocuments.title,
        documentType: legalDocuments.documentType,
        summary: legalDocuments.summary,
        fullText: legalDocuments.fullText,
        caseId: legalDocuments.caseId,
        similarity: sql<number>`1 - (embedding <=> ${embeddingString}::vector)`,
      })
      .from(legalDocuments)
      .where(sql`embedding IS NOT NULL AND is_active = true`)
      .orderBy(sql`embedding <=> ${embeddingString}::vector`)
      .limit(limit);

    // Filter by similarity threshold
    const filteredResults = results
      .filter((result) => result.similarity >= threshold)
      .map((result) => ({
        id: result.id,
        title: result.title,
        documentType: result.documentType,
        content: result.summary || result.fullText || "",
        similarity: result.similarity,
        caseId: result.caseId,
      }));

    return json({
      success: true,
      results: filteredResults,
      total: filteredResults.length,
      query,
      options: { limit, threshold },
    });
  } catch (error: any) {
    console.error("Semantic search GET error:", error);
    return json(
      { error: "Failed to perform semantic search", success: false },
      { status: 500 }
    );
  }
};
