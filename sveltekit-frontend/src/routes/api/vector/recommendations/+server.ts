import type { RequestHandler } from '@sveltejs/kit';
// @ts-nocheck
/**
 * Vector Intelligence Recommendations API
 * Provides intelligent recommendations using vector analysis and machine learning
 */

import { json, error } from "@sveltejs/kit";
import { vectorIntelligenceService } from "$lib/services/vector-intelligence-service.js";
import { URL } from "url";

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const body = await request.json();
    const recommendationRequest: RecommendationRequest = {
      context: body.context || "",
      userProfile: body.userProfile,
      currentCase: body.currentCase,
      preferences: body.preferences,
    };

    if (!recommendationRequest.context) {
      throw error(400, "Context is required for generating recommendations");
    }

    console.log(
      `🎯 Generating intelligent recommendations for context: "${recommendationRequest.context.substring(0, 100)}..."`,
    );

    // Generate intelligent recommendations
    const recommendations =
      await vectorIntelligenceService.generateRecommendations(
        recommendationRequest,
      );

    // Get system health for metadata
    const systemHealth = await vectorIntelligenceService.getSystemHealth();

    return json({
      success: true,
      context: recommendationRequest.context,
      recommendations,
      metadata: {
        totalRecommendations: recommendations.length,
        processingTime: Date.now(),
        systemHealth: {
          status: systemHealth.systemHealth,
          confidence: systemHealth.modelConfidence,
        },
        personalization: {
          userRole: recommendationRequest.userProfile?.role || "unknown",
          hasPreferences: !!recommendationRequest.preferences,
          hasCurrentCase: !!recommendationRequest.currentCase,
        },
      },
    });
  } catch (err) {
    console.error("❌ Recommendations API error:", err);

    const errorMessage = err instanceof Error ? err.message : "Unknown error";
    const statusCode =
      err && typeof err === "object" && "status" in err
        ? (err as any).status
        : 500;

    throw error(statusCode, errorMessage);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const context = url.searchParams.get("context");
  const role = url.searchParams.get("role") as
    | "prosecutor"
    | "detective"
    | "admin"
    | "user"
    | null;
  const caseId = url.searchParams.get("caseId");

  if (!context) {
    // Return API documentation
    return json({
      message: "Vector Intelligence Recommendations API - Phase 4",
      endpoints: {
        "POST /api/vector/recommendations":
          "Generate personalized recommendations",
        "GET /api/vector/recommendations?context=query":
          "Quick recommendations via query parameter",
      },
      parameters: {
        context: "Context for recommendations (required)",
        role: "User role for personalization (optional)",
        caseId: "Current case ID for context (optional)",
      },
      supportedRoles: ["prosecutor", "detective", "admin", "user"],
      recommendationTypes: ["action", "insight", "warning", "opportunity"],
      categories: [
        "investigation",
        "legal_analysis",
        "evidence_review",
        "case_strategy",
        "workflow",
      ],
    });
  }

  try {
    // Build recommendation request from query parameters
    const recommendationRequest: RecommendationRequest = {
      context,
      userProfile: role
        ? {
            role,
            experience: "senior", // Default to senior for GET requests
            specialization: [],
          }
        : undefined,
      currentCase: caseId
        ? {
            id: caseId,
            type: "general",
            priority: "medium",
            status: "active",
          }
        : undefined,
    };

    const recommendations =
      await vectorIntelligenceService.generateRecommendations(
        recommendationRequest,
      );

    return json({
      success: true,
      context,
      recommendations,
      metadata: {
        totalRecommendations: recommendations.length,
        processingTime: Date.now(),
      },
    });
  } catch (err) {
    console.error("❌ Recommendations GET error:", err);
    throw error(
      500,
      err instanceof Error ? err.message : "Recommendations failed",
    );
  }
};
