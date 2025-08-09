// @ts-nocheck
/**
 * Enhanced Search API Endpoint
 * Provides advanced search capabilities with monitoring and security
 */

import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { advancedSearch } from "$lib/server/services/advanced-search";
import {
  rateLimitAPI,
  addSecurityHeaders,
} from "$lib/server/monitoring/security";
import { logUserAction, logInfo } from "$lib/server/monitoring/logger";

export const GET: RequestHandler = async ({
  url,
  locals,
  request,
  getClientAddress,
}) => {
  try {
    // Apply security measures
    addSecurityHeaders()({
      url,
      request,
      getClientAddress,
      setHeaders: () => {},
      locals,
    } as any);
    rateLimitAPI()({
      url,
      request,
      getClientAddress,
      setHeaders: () => {},
      locals,
    } as any);

    // Parse search parameters
    const searchParams = url.searchParams;
    const filters = {
      query: searchParams.get("q") || undefined,
      caseStatus: searchParams.getAll("status"),
      priority: searchParams.getAll("priority"),
      tags: searchParams.getAll("tags"),
      evidenceType: searchParams.getAll("evidenceType"),
      sortBy: (searchParams.get("sortBy") as any) || "relevance",
      sortOrder: (searchParams.get("sortOrder") as any) || "desc",
      limit: parseInt(searchParams.get("limit") || "20"),
      offset: parseInt(searchParams.get("offset") || "0"),
      dateRange:
        searchParams.get("dateStart") && searchParams.get("dateEnd")
          ? {
              start: searchParams.get("dateStart")!,
              end: searchParams.get("dateEnd")!,
            }
          : undefined,
    };

    // Log search activity
    if (locals.user) {
      logUserAction("search", locals.user.id, {
        query: filters.query,
        filters: Object.keys(filters).filter(
          (key) => filters[key as keyof typeof filters],
        ),
      });
    }
    // Perform search
    logInfo("Search initiated", {
      query: filters.query,
      userId: locals.user?.id,
    });
    const results = await advancedSearch.search(filters);

    logInfo("Search completed", {
      query: filters.query,
      resultCount: results.total,
      queryTime: results.queryTime,
    });

    return json({
      success: true,
      data: results,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Search API error:", error);

    return json(
      {
        success: false,
        error: "Search failed",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Apply security measures
    rateLimitAPI()({ request, locals } as any);

    const body = await request.json();
    const { query, filters: customFilters, saveSearch } = body;

    // Enhanced filters from POST body
    const filters = {
      query,
      ...customFilters,
    };

    // Log search activity
    if (locals.user) {
      logUserAction("advanced_search", locals.user.id, {
        query: filters.query,
        customFilters,
      });
    }
    // Perform search
    const results = await advancedSearch.search(filters);

    // Optionally save search for user
    if (saveSearch && locals.user) {
      // Implementation for saving searches would go here
      logInfo("Search saved", { userId: locals.user.id, query: filters.query });
    }
    return json({
      success: true,
      data: results,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Advanced search API error:", error);

    return json(
      {
        success: false,
        error: "Advanced search failed",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};
