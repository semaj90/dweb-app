
import type { RequestHandler } from './$types';

/**
 * Vector Intelligence Semantic Analysis API
 * Provides advanced semantic analysis of legal documents and content
 */

import { json, error } from "@sveltejs/kit";
import { vectorIntelligenceService } from "$lib/services/vector-intelligence-service.js";

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const body = await request.json();
    const { content, options = {} } = body;

    if (!content || typeof content !== "string") {
      throw error(400, "Content is required and must be a string");
    }

    if (content.length > 50000) {
      throw error(400, "Content too large. Maximum 50,000 characters allowed");
    }

    console.log(
      `🔬 Performing semantic analysis on ${content.length} characters...`,
    );

    // Perform comprehensive semantic analysis
    const analysis = await vectorIntelligenceService.analyzeSemantics(content);

    // Get system health for metadata
    const systemHealth = await vectorIntelligenceService.getSystemHealth();

    return json({
      success: true,
      analysis,
      metadata: {
        contentLength: content.length,
        processingTime: Date.now(),
        systemHealth: {
          status: systemHealth.systemHealth,
          confidence: systemHealth.modelConfidence,
        },
      },
    });
  } catch (err: any) {
    console.error("❌ Semantic analysis API error:", err);

    const errorMessage = err instanceof Error ? err.message : "Unknown error";
    const statusCode =
      err && typeof err === "object" && "status" in err
        ? (err as any).status
        : 500;

    throw error(statusCode, errorMessage);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  return json({
    message: "Vector Intelligence Semantic Analysis API - Phase 4",
    endpoints: {
      "POST /api/vector/analyze": "Perform semantic analysis on content",
    },
    parameters: {
      content: "Text content to analyze (required, max 50,000 chars)",
      options: "Analysis options (optional)",
    },
    analysisFeatures: [
      "Entity extraction (persons, organizations, locations, dates, legal concepts)",
      "Theme identification and weighting",
      "Relationship mapping between entities",
      "Sentiment analysis with legal context",
      "Complexity assessment (readability, technical level, legal complexity)",
      "Keyword and phrase extraction",
      "Document structure analysis",
    ],
    entityTypes: [
      "person",
      "organization",
      "location",
      "date",
      "legal_concept",
    ],
    complexityMetrics: ["readability", "technicalLevel", "legalComplexity"],
    usage: {
      example: {
        method: "POST",
        url: "/api/vector/analyze",
        body: {
          content:
            "The defendant, John Smith, signed the contract on January 15, 2024...",
          options: {
            extractEntities: true,
            analyzeSentiment: true,
            assessComplexity: true,
          },
        },
      },
    },
  });
};
