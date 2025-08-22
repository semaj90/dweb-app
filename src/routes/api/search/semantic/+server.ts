
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { aiPipeline } from "$lib/ai/processing-pipeline";
import { z } from "zod";

/**
 * Semantic Search API Endpoint
 * Provides AI-powered vector similarity search across legal documents
 */

const searchRequestSchema = z.object({
  query: z.string().min(1).max(1000),
  limit: z.number().int().min(1).max(100).optional().default(10),
  documentType: z
    .enum([
      "contract",
      "motion",
      "evidence",
      "correspondence",
      "brief",
      "regulation",
      "case_law",
    ])
    .optional(),
  practiceArea: z
    .enum([
      "corporate",
      "litigation",
      "intellectual_property",
      "employment",
      "real_estate",
      "criminal",
      "family",
      "tax",
      "immigration",
      "environmental",
    ])
    .optional(),
  jurisdiction: z.enum(["federal", "state", "local"]).optional(),
  minSimilarity: z.number().min(0).max(1).optional().default(0.6),
  useCache: z.boolean().optional().default(true),
  includeContent: z.boolean().optional().default(false),
  includeAnalysis: z.boolean().optional().default(true),
});

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();

    // Validate request
    const searchRequest = searchRequestSchema.safeParse(body);

    if (!searchRequest.success) {
      return json(
        {
          success: false,
          error: "Invalid request parameters",
          details: searchRequest.error.issues,
        },
        { status: 400 }
      );
    }

    const {
      query,
      limit,
      documentType,
      practiceArea,
      jurisdiction,
      minSimilarity,
      useCache,
      includeContent,
      includeAnalysis,
    } = searchRequest.data;

    const startTime = Date.now();

    // Perform semantic search
    const searchOptions = {
      limit,
      documentType,
      practiceArea,
      jurisdiction,
      useCache,
    };

    const results = await aiPipeline.semanticSearch(query, searchOptions);

    // Filter by similarity threshold
    const filteredResults = results.filter(
      (result) => result.similarity >= minSimilarity
    );

    // Process results based on options
    const processedResults = filteredResults.map((result) => {
      const processedResult: unknown = {
        id: result.id,
        title: result.title,
        documentType: result.documentType,
        practiceArea: result.practiceArea,
        jurisdiction: result.jurisdiction,
        similarity: result.similarity,
        createdAt: result.createdAt,
        fileName: result.fileName,
        fileSize: result.fileSize,
      };

      // Include content preview if requested
      if (includeContent) {
        processedResult.contentPreview =
          result.content.substring(0, 500) +
          (result.content.length > 500 ? "..." : "");
      }

      // Include analysis if requested and available
      if (includeAnalysis && result.analysisResults) {
        processedResult.analysis = {
          confidence: result.analysisResults.confidenceLevel,
          entities: result.analysisResults.entities?.slice(0, 10) || [],
          keyTerms: result.analysisResults.keyTerms?.slice(0, 10) || [],
          risks: result.analysisResults.risks?.length || 0,
          sentiment: result.analysisResults.sentimentScore,
        };
      }

      return processedResult;
    });

    // Calculate search metrics
    const searchTime = Date.now() - startTime;
    const totalResults = filteredResults.length;
    const avgSimilarity =
      totalResults > 0
        ? filteredResults.reduce((sum, r) => sum + r.similarity, 0) /
          totalResults
        : 0;

    return json({
      success: true,
      query,
      results: processedResults,
      metadata: {
        totalResults,
        searchTime,
        avgSimilarity,
        minSimilarity,
        filters: {
          documentType: documentType || null,
          practiceArea: practiceArea || null,
          jurisdiction: jurisdiction || null,
        },
        cached: useCache,
      },
    });
  } catch (error: unknown) {
    console.error("Semantic search error:", error);

    return json(
      {
        success: false,
        error:
          error instanceof Error
            ? (error as any)?.message || "Unknown error"
            : "Search failed",
        details: process.env.NODE_ENV === "development" ? error : undefined,
      },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    // Simple GET endpoint for quick searches
    const query = url.searchParams.get("q");
    const limit = Math.min(parseInt(url.searchParams.get("limit") || "10"), 50);
    const documentType = url.searchParams.get("type");
    const practiceArea = url.searchParams.get("area");

    if (!query) {
      return json({ error: "Query parameter required" }, { status: 400 });
    }

    const searchOptions: unknown = { limit, useCache: true };

    if (documentType) searchOptions.documentType = documentType;
    if (practiceArea) searchOptions.practiceArea = practiceArea;

    const results = await aiPipeline.semanticSearch(query, searchOptions);

    // Simplified response for GET requests
    const simplifiedResults = results.map((result) => ({
      id: result.id,
      title: result.title,
      documentType: result.documentType,
      similarity: result.similarity,
      preview: result.content.substring(0, 200) + "...",
    }));

    return json({
      success: true,
      query,
      results: simplifiedResults,
      count: results.length,
    });
  } catch (error: unknown) {
    console.error("GET search error:", error);

    return json(
      {
        success: false,
        error: "Search failed",
      },
      { status: 500 }
    );
  }
};
