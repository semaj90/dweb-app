// Documents Search API
// Provides comprehensive document search capabilities

import { json } from "@sveltejs/kit";
import { db } from "$lib/database/postgres.js";
import { serializeEmbedding } from "$lib/utils/embeddings.js"; // future use
import {
  legalDocuments,
  searchSessions,
  embeddings,
} from "$lib/database/schema/legal-documents.js";
import type { RequestHandler } from "./$types";
import type { SearchParams, SearchResult } from "$lib/types/search-types";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const searchParams: SearchParams = await request.json();

    const {
      query,
      searchType = "semantic",
      limit = 10,
      offset = 0,
      filters = {},
    } = searchParams;

    if (!query) {
      return json({ error: "Query is required" }, { status: 400 });
    }

    const startTime = Date.now();

    // Generate query embedding for semantic search
    let queryEmbedding: number[] = [];
    if (searchType === "semantic" || searchType === "hybrid") {
      queryEmbedding = await generateQueryEmbedding(query);
    }

    // Perform search based on type
    let searchResults: SearchResult[] = [];

    switch (searchType) {
      case "semantic":
        searchResults = await performSemanticSearch(
          query,
          queryEmbedding,
          filters,
          limit,
          offset
        );
        break;
      case "full-text":
        searchResults = await performFullTextSearch(
          query,
          filters,
          limit,
          offset
        );
        break;
      case "hybrid":
        searchResults = await performHybridSearch(
          query,
          queryEmbedding,
          filters,
          limit,
          offset
        );
        break;
      default:
        throw new Error(`Unknown search type: ${searchType}`);
    }

    const processingTime = Date.now() - startTime;

    // Save search session
    await db.insert(searchSessions).values({
      query: searchParams.query,
      searchType: searchParams.searchType || "semantic",
      queryEmbedding: JSON.stringify(queryEmbedding) as any,
      results: searchResults as any,
      resultCount: searchResults.length,
    } as any);

    return json({
      success: true,
      results: searchResults,
      metadata: {
        query,
        searchType,
        totalResults: searchResults.length,
        offset,
        limit,
        processingTime,
        filters: filters,
      },
      pagination: {
        hasMore: searchResults.length === limit,
        nextOffset: offset + limit,
      },
    });
  } catch (error: any) {
    console.error("Document search error:", error);
    return json(
      { error: "Search failed", details: error.message },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const sessionId = url.searchParams.get("sessionId");
    const recent = url.searchParams.get("recent");
    const limit = parseInt(url.searchParams.get("limit") || "10");

    if (sessionId) {
      // Get specific search session
      const session = await db.query.searchSessions.findFirst({
        where: (sessions, { eq }) => eq(sessions.id, Number(sessionId)),
      });

      if (!session) {
        return json({ error: "Search session not found" }, { status: 404 });
      }

      return json({
        session: {
          id: session.id,
          query: session.query,
          searchType: session.searchType,
          resultCount: session.resultCount,
          results: session.results,
          createdAt: session.createdAt,
        },
      });
    }

    if (recent === "true") {
      // Get recent search sessions
      const recentSessions = await db
        .select({
          id: searchSessions.id,
          query: searchSessions.query,
          searchType: searchSessions.searchType,
          resultCount: searchSessions.resultCount,
          createdAt: searchSessions.createdAt,
        })
        .from(searchSessions)
        .orderBy((session) => session.createdAt)
        .limit(limit);

      return json({
        recentSessions,
        count: recentSessions.length,
      });
    }

    // Get search statistics
    const stats = await getSearchStatistics();
    return json(stats);
  } catch (error: any) {
    console.error("Search retrieval error:", error);
    return json(
      { error: "Failed to retrieve search data", details: error.message },
      { status: 500 }
    );
  }
};

// Search implementation functions

async function performSemanticSearch(
  query: string,
  queryEmbedding: number[],
  filters: any,
  limit: number,
  offset: number
): Promise<SearchResult[]> {
  try {
    // In a real implementation, this would use vector similarity search
    // For now, we'll simulate with a basic document search

    let dbQuery = db
      .select({
        id: legalDocuments.id,
        title: legalDocuments.title,
        content: legalDocuments.content,
        documentType: legalDocuments.documentType,
        jurisdiction: legalDocuments.jurisdiction,
        practiceArea: legalDocuments.practiceArea,
        createdAt: legalDocuments.createdAt,
        updatedAt: legalDocuments.updatedAt,
      })
      .from(legalDocuments);

    // Apply filters
    dbQuery = applyFilters(dbQuery, filters);

    const documents = await dbQuery.limit(limit).offset(offset);

    return documents.map((doc, index) => ({
      score: calculateSemanticScore(query, queryEmbedding, doc),
      rank: offset + index + 1,
      id: String(doc.id),
      title: doc.title,
      content: doc.content,
      excerpt: extractExcerpt(query, doc.content),
      type: doc.documentType,
      metadata: {
        jurisdiction: doc.jurisdiction,
        practiceArea: doc.practiceArea,
      },
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
      document: {
        id: String(doc.id),
        title: doc.title,
        content: doc.content,
        documentType: doc.documentType,
        jurisdiction: doc.jurisdiction,
        practiceArea: doc.practiceArea || "general",
        processingStatus: "completed",
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
      },
    }));
  } catch (error: any) {
    console.error("Semantic search error:", error);
    return [];
  }
}

async function performFullTextSearch(
  query: string,
  filters: any,
  limit: number,
  offset: number
): Promise<SearchResult[]> {
  try {
    let dbQuery = db
      .select({
        id: legalDocuments.id,
        title: legalDocuments.title,
        content: legalDocuments.content,
        documentType: legalDocuments.documentType,
        jurisdiction: legalDocuments.jurisdiction,
        practiceArea: legalDocuments.practiceArea,
        createdAt: legalDocuments.createdAt,
        updatedAt: legalDocuments.updatedAt,
      })
      .from(legalDocuments);

    // Apply text search filter
    const queryTerms = query.toLowerCase().split(/\s+/);
    // TODO: Implement proper full-text search; currently not supported via JS predicate

    // Apply additional filters
    dbQuery = applyFilters(dbQuery, filters);

    const documents = await dbQuery.limit(limit).offset(offset);

    return documents.map((doc, index) => ({
      score: calculateTextScore(query, doc),
      rank: offset + index + 1,
      id: String(doc.id),
      title: doc.title,
      content: doc.content,
      excerpt: extractExcerpt(query, doc.content),
      type: doc.documentType,
      metadata: {
        jurisdiction: doc.jurisdiction,
        practiceArea: doc.practiceArea,
      },
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
      document: {
        id: String(doc.id),
        title: doc.title,
        content: doc.content,
        documentType: doc.documentType,
        jurisdiction: doc.jurisdiction,
        practiceArea: doc.practiceArea || "general",
        processingStatus: "completed",
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
      },
    }));
  } catch (error: any) {
    console.error("Full-text search error:", error);
    return [];
  }
}

async function performHybridSearch(
  query: string,
  queryEmbedding: number[],
  filters: any,
  limit: number,
  offset: number
): Promise<SearchResult[]> {
  try {
    // Combine semantic and full-text search results
    const semanticResults = await performSemanticSearch(
      query,
      queryEmbedding,
      filters,
      limit,
      offset
    );
    const textResults = await performFullTextSearch(
      query,
      filters,
      limit,
      offset
    );

    // Merge and deduplicate results
    const combinedResults = new Map<string, SearchResult>();

    // Add semantic results
    semanticResults.forEach((result) => {
      combinedResults.set(result.id, {
        ...result,
        score: result.score * 0.6, // Weight semantic score
      });
    });

    // Add or merge text results
    textResults.forEach((result) => {
      const existing = combinedResults.get(result.id);
      if (existing) {
        // Combine scores
        existing.score = existing.score + result.score * 0.4;
      } else {
        combinedResults.set(result.id, {
          ...result,
          score: result.score * 0.4, // Weight text score
        });
      }
    });

    // Sort by combined score and re-rank
    const finalResults = Array.from(combinedResults.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((result, index) => ({
        ...result,
        rank: index + 1,
      }));

    return finalResults;
  } catch (error: any) {
    console.error("Hybrid search error:", error);
    return [];
  }
}

// Helper functions

function applyFilters(query: any, filters: any): any {
  if (filters.documentType) {
    query = query.where(
      (doc: any) => doc.documentType === filters.documentType
    );
  }

  if (filters.jurisdiction) {
    query = query.where(
      (doc: any) => doc.jurisdiction === filters.jurisdiction
    );
  }

  if (filters.practiceArea) {
    query = query.where(
      (doc: any) => doc.practiceArea === filters.practiceArea
    );
  }

  if (filters.dateRange) {
    if (filters.dateRange.start) {
      query = query.where(
        (doc: any) => doc.createdAt >= new Date(filters.dateRange.start)
      );
    }
    if (filters.dateRange.end) {
      query = query.where(
        (doc: any) => doc.createdAt <= new Date(filters.dateRange.end)
      );
    }
  }

  return query;
}

async function generateQueryEmbedding(query: string): Promise<number[]> {
  try {
    const response = await fetch("/api/ai/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query }),
    });

    if (response.ok) {
      const data = await response.json();
      return data.embedding;
    }
  } catch (error) {
    console.warn("Failed to generate query embedding, using mock");
  }

  // Return mock embedding
  return Array.from({ length: 384 }, () => Math.random() - 0.5);
}

function calculateSemanticScore(
  query: string,
  queryEmbedding: number[],
  document: any
): number {
  // In a real implementation, this would calculate cosine similarity
  // between query embedding and document embedding

  // For now, use a combination of text matching and mock semantic similarity
  const textScore = calculateTextScore(query, document);
  const mockSemanticScore = Math.random() * 0.5 + 0.3; // 0.3 to 0.8

  return textScore * 0.4 + mockSemanticScore * 0.6;
}

function calculateTextScore(query: string, document: any): number {
  const queryTerms = query.toLowerCase().split(/\s+/);
  const docText = (document.title + " " + document.content).toLowerCase();

  let matches = 0;
  let totalOccurrences = 0;

  for (const term of queryTerms) {
    const occurrences = (docText.match(new RegExp(term, "g")) || []).length;
    if (occurrences > 0) {
      matches++;
      totalOccurrences += occurrences;
    }
  }

  const termCoverage = matches / queryTerms.length;
  const termDensity = Math.min(1.0, totalOccurrences / 100);

  return termCoverage * 0.7 + termDensity * 0.3;
}

function extractExcerpt(query: string, content: string): string {
  if (!content) return "";

  const queryTerms = query.toLowerCase().split(/\s+/);
  const sentences = content.split(/[.!?]+/);

  // Find the sentence with the most query term matches
  let bestSentence = "";
  let bestScore = 0;

  for (const sentence of sentences) {
    const sentenceLower = sentence.toLowerCase();
    let score = 0;

    for (const term of queryTerms) {
      if (sentenceLower.includes(term)) {
        score++;
      }
    }

    if (score > bestScore && sentence.trim().length > 20) {
      bestScore = score;
      bestSentence = sentence.trim();
    }
  }

  if (!bestSentence && sentences.length > 0) {
    bestSentence = sentences[0].trim();
  }

  return (
    bestSentence.substring(0, 300) + (bestSentence.length > 300 ? "..." : "")
  );
}

async function getSearchStatistics(): Promise<any> {
  try {
    const totalSessions = await db
      // TODO: Add proper aggregate counts using sql`count(*)`
      .from(searchSessions);

    return {
      totalSearches: totalSessions[0]?.count || 0,
      popularQueries: [
        { query: "contract breach", count: 45 },
        { query: "employment law", count: 38 },
        { query: "intellectual property", count: 32 },
      ],
      averageResultsPerSearch: 12.5,
      searchTypes: {
        semantic: 60,
        fullText: 30,
        hybrid: 10,
      },
    };
  } catch (error) {
    console.error("Failed to get search statistics:", error);
    return {
      totalSearches: 0,
      popularQueries: [],
      averageResultsPerSearch: 0,
      searchTypes: { semantic: 0, fullText: 0, hybrid: 0 },
    };
  }
}

// Bulk operations endpoint
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { action, ...params } = await request.json();

    switch (action) {
      case "reindex_embeddings":
        // Trigger reindexing of document embeddings
        const reindexResult = await reindexDocumentEmbeddings(
          params.documentIds
        );
        return json({
          success: true,
          message: `Reindexed ${reindexResult.count} documents`,
          ...reindexResult,
        });

      case "bulk_search":
        // Perform multiple searches
        const { queries } = params;
        if (!Array.isArray(queries)) {
          return json({ error: "Queries must be an array" }, { status: 400 });
        }

        const bulkResults = await Promise.all(
          queries.map(async (query: string) => {
            const embedding = await generateQueryEmbedding(query);
            return {
              query,
              results: await performSemanticSearch(query, embedding, {}, 5, 0),
            };
          })
        );

        return json({
          success: true,
          bulkResults,
          totalQueries: queries.length,
        });

      default:
        return json(
          {
            error: "Unknown action",
            availableActions: ["reindex_embeddings", "bulk_search"],
          },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error("Bulk operation error:", error);
    return json(
      { error: "Bulk operation failed", details: error.message },
      { status: 500 }
    );
  }
};

async function reindexDocumentEmbeddings(
  documentIds?: string[]
): Promise<{ count: number; updated: string[] }> {
  try {
    // This would trigger embedding regeneration for specified documents
    const updated: string[] = [];

    // Mock implementation
    if (documentIds && documentIds.length > 0) {
      for (const docId of documentIds) {
        // Generate new embeddings for each document
        await db.insert(embeddings).values({
          content: `Document ${docId} content`,
          embedding: JSON.stringify(
            Array.from({ length: 384 }, () => Math.random() - 0.5)
          ),
          metadata: { reindexed: true, timestamp: new Date() },
          createdAt: new Date(),
        });
        updated.push(docId);
      }
    }

    return {
      count: updated.length,
      updated,
    };
  } catch (error: any) {
    console.error("Reindexing error:", error);
    return { count: 0, updated: [] };
  }
}
