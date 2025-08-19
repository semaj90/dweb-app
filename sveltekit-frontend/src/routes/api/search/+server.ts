// @ts-nocheck
import { json } from "@sveltejs/kit";
// Orphaned content: import type { RequestHandler
import {
vectorSearch } from "$lib/server/search/vector-search";
// Orphaned content: import { URL

export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get("q");
    const limit = parseInt(url.searchParams.get("limit") || "20");
    const threshold = parseFloat(url.searchParams.get("threshold") || "0.7");
    const type = url.searchParams.get("type");
    const caseId = url.searchParams.get("caseId");

    if (!query) {
      return json({ error: "Query parameter is required" }, { status: 400 });
    }
    const filters: Record<string, any> = {};
    if (type) filters.type = type;
    if (caseId) filters.caseId = caseId;

    const searchResults = await vectorSearch(query, {
      limit,
      threshold,
      filters,
      useCache: true,
      fallbackToQdrant: true,
    });

    return json({
      success: true,
      data: searchResults,
    });
  } catch (error) {
    console.error("Search API error:", error);
    return json(
      {
        error: error instanceof Error ? error.message : "Search failed",
      },
      { status: 500 },
    );
  }
};
