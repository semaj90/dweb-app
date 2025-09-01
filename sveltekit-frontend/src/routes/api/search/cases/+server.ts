
import type { RequestHandler } from './$types.js';

// Optimized case search API endpoint
// Supports multiple search strategies with automatic fallbacks
import { json } from "@sveltejs/kit";
import { and, desc, ilike, or, sql } from "drizzle-orm";
import { db, isPostgreSQL } from "$lib/server/db/index";
import { URL } from "url";


export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get("q");
    const limit = parseInt(url.searchParams.get("limit") || "20");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const searchType = url.searchParams.get("type") || "hybrid"; // 'text', 'semantic', 'hybrid'

    const filters = {
      status: url.searchParams.get("status"),
      priority: url.searchParams.get("priority"),
      category: url.searchParams.get("category"),
    };

    if (!query || query.length < 2) {
      return json({
        results: [],
        searchType: "none",
        executionTime: 0,
        total: 0,
        message: "Query too short",
      });
    }

    const startTime = Date.now();
    let results = [];

    // For now, use text search only until vector search is properly configured
    results = await searchCasesText(query, limit + offset, filters);

    const executionTime = Date.now() - startTime;

    return json({
      results: results.slice(offset, offset + limit),
      searchType: "text",
      executionTime,
      total: results.length,
      query,
      filters,
      fromCache: false,
    });
  } catch (error: any) {
    console.error("Case search error:", error);
    return json(
      {
        results: [],
        error: "Search failed",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Fast text-based search using SQL LIKE
async function searchCasesText(
  query: string,
  limit: number,
  filters: any,
): Promise<any[]> {
  try {
    const whereConditions = [
      or(
        ilike(cases.title, `%${query}%`),
        ilike(cases.description, `%${query}%`),
        ilike(cases.caseNumber, `%${query}%`),
      ),
    ];

    // Add filters
    if (filters.status) {
      whereConditions.push(sql`${cases.status} = ${filters.status}`);
    }
    if (filters.priority) {
      whereConditions.push(sql`${cases.priority} = ${filters.priority}`);
    }
    if (filters.category) {
      whereConditions.push(sql`${cases.category} = ${filters.category}`);
    }

    const results = await db
      .select()
      .from(cases)
      .where(and(...whereConditions))
      .orderBy(desc(cases.createdAt))
      .limit(limit);

    return results.map((case_) => ({
      ...case_,
      searchScore: 1.0,
      matchType: "text",
    }));
  } catch (error: any) {
    console.error("Text search failed:", error);
    return [];
  }
}
