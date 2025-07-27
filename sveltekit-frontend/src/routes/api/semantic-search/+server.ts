import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { db } from "$lib/server/db/pg";
import { legalDocs } from "$lib/server/db/schema-postgres";
import { sql } from "drizzle-orm";
import { nomicEmbeddings } from "$lib/ai/nomic-embeddings";

// Use Nomic embeddings with 768 dimensions (Nomic's default)
const EMBEDDING_DIMENSION = 768;

export const POST: RequestHandler = async ({ request }) => {
  const { query, limit = 10, threshold = 0.3 } = await request.json();

  if (!query) {
    return json({ error: "Query is required" }, { status: 400 });
  }

  try {
    // Generate embedding for the query using Nomic
    const embeddingResult = await nomicEmbeddings.embed(query);
    const queryEmbedding = embeddingResult.embedding;
    const embeddingString = `[${queryEmbedding.join(',')}]`;

    // Perform vector similarity search
    const results = await db
      .select({
        id: legalDocs.id,
        title: legalDocs.title,
        documentType: legalDocs.documentType,
        summary: legalDocs.summary,
        fullText: legalDocs.fullText,
        caseId: legalDocs.caseId,
        similarity: sql<number>`1 - (embedding <=> ${embeddingString}::vector)`,
      })
      .from(legalDocs)
      .where(sql`embedding IS NOT NULL AND is_active = true`)
      .orderBy(sql`embedding <=> ${embeddingString}::vector`)
      .limit(limit);

    // Filter by similarity threshold
    const filteredResults = results
      .filter(result => result.similarity >= threshold)
      .map(result => ({
        id: result.id,
        title: result.title,
        documentType: result.documentType,
        content: result.summary || result.fullText || '',
        similarity: result.similarity,
        caseId: result.caseId,
      }));

    return json({
      success: true,
      results: filteredResults,
      total: filteredResults.length,
      query,
      options: { limit, threshold }
    });

  } catch (error) {
    console.error("Semantic search error:", error);
    return json(
      { error: "Failed to perform semantic search", success: false },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const query = url.searchParams.get('q');
  const limit = parseInt(url.searchParams.get('limit') || '10');
  const threshold = parseFloat(url.searchParams.get('threshold') || '0.3');

  if (!query) {
    return json({ error: "Query parameter 'q' is required" }, { status: 400 });
  }

  try {
    // Generate embedding for the query using Nomic
    const embeddingResult = await nomicEmbeddings.embed(query);
    const queryEmbedding = embeddingResult.embedding;
    const embeddingString = `[${queryEmbedding.join(',')}]`;

    // Perform vector similarity search
    const results = await db
      .select({
        id: legalDocs.id,
        title: legalDocs.title,
        documentType: legalDocs.documentType,
        summary: legalDocs.summary,
        fullText: legalDocs.fullText,
        caseId: legalDocs.caseId,
        similarity: sql<number>`1 - (embedding <=> ${embeddingString}::vector)`,
      })
      .from(legalDocs)
      .where(sql`embedding IS NOT NULL AND is_active = true`)
      .orderBy(sql`embedding <=> ${embeddingString}::vector`)
      .limit(limit);

    // Filter by similarity threshold
    const filteredResults = results
      .filter(result => result.similarity >= threshold)
      .map(result => ({
        id: result.id,
        title: result.title,
        documentType: result.documentType,
        content: result.summary || result.fullText || '',
        similarity: result.similarity,
        caseId: result.caseId,
      }));

    return json({
      success: true,
      results: filteredResults,
      total: filteredResults.length,
      query,
      options: { limit, threshold }
    });

  } catch (error) {
    console.error("Semantic search GET error:", error);
    return json(
      { error: "Failed to perform semantic search", success: false },
      { status: 500 }
    );
  }
};
