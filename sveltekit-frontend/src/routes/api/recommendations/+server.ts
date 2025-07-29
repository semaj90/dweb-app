import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { enhancedSearchWithNeo4j } from "$lib/ai/custom-reranker";
import { accessMemoryMCP } from "$lib/utils/copilot-self-prompt";
import { mcpContext72GetLibraryDocs } from "$lib/mcp-context72-get-library-docs";

// Recommendation endpoint using enhanced reranker, Neo4j, and memory
export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      query,
      userContext,
      neo4jContext,
      limit = 5,
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
    const recommendations = reranked
      .map((result) => {
        let score = result.rerankScore;
        if (memory.some((m) => m.relatedId === result.id)) score += 1;
        if (docs && docs.includes(result.intent)) score += 1;
        return { ...result, finalScore: score };
      })
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, limit);
    return json({ recommendations });
  } catch (error: any) {
    return json(
      { error: error.message || "Failed to generate recommendations" },
      { status: 500 }
    );
  }
};
