
import { json } from "@sveltejs/kit";
import { enhancedSearchWithNeo4j } from "$lib/ai/custom-reranker";
import { mcpContext72GetLibraryDocs } from "$lib/mcp-context72-get-library-docs";
import type { RequestHandler } from './$types';


// Enhanced RAG endpoint with reranker, Neo4j, memory, and docs
export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      query,
      userContext,
      neo4jContext,
      limit = 8,
    } = await request.json();
    // Run enhanced search with Neo4j context
    const reranked = await enhancedSearchWithNeo4j(
      query,
      userContext,
      neo4jContext,
      limit * 2
    );
    // Enrich with memory and docs for final scoring
    const memory = await accessMemoryMCP(query, userContext);
    const docs = await mcpContext72GetLibraryDocs("svelte", "runes");
    // Final scoring pass
    const highScoreRecommendations = reranked
      .map((result) => {
        let score = result.rerankScore;
        if (memory.some((m) => m.relatedId === result.id)) score += 1;
        if (docs && docs.includes(result.intent)) score += 1;
        return { ...result, finalScore: score };
      })
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, limit);
    // Compose RAG answer (stub: use top result)
    const answer = highScoreRecommendations[0]?.content || "[No answer found]";
    return json({
      answer,
      references: highScoreRecommendations.map((r) => ({
        id: r.id,
        score: r.finalScore,
      })),
      confidence: highScoreRecommendations[0]?.finalScore || 0,
      highScoreRecommendations,
    });
  } catch (error: any) {
    return json(
      { error: error.message || "Failed to run enhanced RAG" },
      { status: 500 }
    );
  }
};
