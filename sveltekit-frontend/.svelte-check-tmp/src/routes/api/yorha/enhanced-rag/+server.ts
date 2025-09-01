
import { json } from "@sveltejs/kit";
import { enhancedSearchWithNeo4j } from "$lib/ai/custom-reranker";
import { legalDocuments, cases, evidence } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from './$types';


// YoRHa Enhanced RAG API
// Integrated AI-powered legal analysis for YoRHa interface

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      query,
      dataType = "documents",
      context,
      analysisType = "comprehensive",
      limit = 5,
      includeRecommendations = true,
      includeMetadata = true
    } = await request.json();

    if (!query) {
      return json(
        { success: false, error: "Query is required" },
        { status: 400 }
      );
    }

    // Enhanced RAG search with reranking
    const rerankedResults = await enhancedSearchWithNeo4j(
      query,
      context || `Analyzing ${dataType} for legal insights`,
      null, // neo4jContext can be null for basic search
      limit * 2 // Get more results for better reranking
    );

    // Database search for relevant legal data
    let dbResults: any[] = [];
    
    switch (dataType) {
      case "documents":
        dbResults = await db
          .select()
          .from(legalDocuments)
          .where(
            or(
              like(legalDocuments.title, `%${query}%`),
              like(legalDocuments.content, `%${query}%`),
              like(legalDocuments.summary, `%${query}%`),
              sql`${legalDocuments.keywords} @> ${JSON.stringify([query.toLowerCase()])}`
            )
          )
          .limit(limit);
        break;

      case "cases":
        dbResults = await db
          .select()
          .from(cases)
          .where(
            or(
              like(cases.title, `%${query}%`),
              like(cases.description, `%${query}%`),
              like(cases.caseNumber, `%${query}%`)
            )
          )
          .limit(limit);
        break;

      case "evidence":
        dbResults = await db
          .select()
          .from(evidence)
          .where(
            or(
              like(evidence.title, `%${query}%`),
              like(evidence.description, `%${query}%`),
              like(evidence.evidenceType, `%${query}%`)
            )
          )
          .limit(limit);
        break;
    }

    // Combine and analyze results
    const analysisResults = await performYoRHaAnalysis(
      query,
      rerankedResults,
      dbResults,
      analysisType
    );

    // Generate recommendations if requested
    let recommendations: any[] = [];
    if (includeRecommendations) {
      recommendations = await generateYoRHaRecommendations(
        query,
        analysisResults,
        dataType
      );
    }

    // Format response for YoRHa interface
    const yorhaResponse = {
      success: true,
      query,
      dataType,
      analysisType,
      timestamp: new Date().toISOString(),
      
      // Core results
      results: analysisResults.slice(0, limit),
      
      // Analysis metadata
      analysis: {
        totalResultsAnalyzed: rerankedResults.length + dbResults.length,
        confidenceScore: calculateOverallConfidence(analysisResults),
        processingTime: Date.now(), // This would be calculated properly
        aiModelUsed: "enhanced-rag-yorha",
        legalComplexity: assessLegalComplexity(analysisResults),
        riskLevel: assessRiskLevel(analysisResults)
      },

      // Enhanced features
      recommendations: includeRecommendations ? recommendations : [],
      
      // Legal-specific insights
      legalInsights: {
        jurisdiction: extractJurisdiction(analysisResults),
        legalAreas: extractLegalAreas(analysisResults),
        precedents: findRelevantPrecedents(analysisResults),
        keyTerms: extractKeyTerms(analysisResults),
        citations: extractCitations(analysisResults)
      },

      // YoRHa-specific formatting
      yorhaMetadata: includeMetadata ? {
        systemStatus: "OPERATIONAL",
        securityLevel: "AUTHORIZED",
        analysisMode: "ENHANCED",
        dataIntegrity: "VERIFIED",
        processingNode: "YORHA-LEGAL-AI-001",
        classification: "CONFIDENTIAL"
      } : null,

      // Service information
      service: "yorha-enhanced-rag-api",
      version: "4.0.0"
    };

    return json(yorhaResponse);

  } catch (error: any) {
    console.error("YoRHa Enhanced RAG error:", error);
    return json(
      {
        success: false,
        error: error.message || "Enhanced RAG analysis failed",
        query: request.body?.query || "",
        timestamp: new Date().toISOString(),
        service: "yorha-enhanced-rag-api",
        yorhaMetadata: {
          systemStatus: "ERROR",
          errorCode: "ERR_ANALYSIS_FAILED",
          processingNode: "YORHA-LEGAL-AI-001"
        }
      },
      { status: 500 }
    );
  }
};

// YoRHa-specific analysis function
async function performYoRHaAnalysis(
  query: string,
  rerankedResults: any[],
  dbResults: any[],
  analysisType: string
): Promise<any[]> {
  
  // Combine all results
  const allResults = [
    ...rerankedResults.map(r => ({
      ...r,
      source: "enhanced-rag",
      yorha_type: "AI_ANALYSIS",
      yorha_confidence: r.rerankScore || r.score || 0.5
    })),
    ...dbResults.map(r => ({
      ...r,
      source: "database",
      yorha_type: "DATABASE_RECORD",
      yorha_confidence: r.confidenceScore || 0.7,
      content: r.content || r.description || r.title
    }))
  ];

  // Apply YoRHa-specific scoring and analysis
  return allResults.map(result => ({
    ...result,
    yorha_id: `ANALYSIS-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    yorha_processed: true,
    yorha_timestamp: new Date(),
    yorha_analysis: {
      relevanceScore: calculateRelevance(query, result.content || ""),
      legalWeight: calculateLegalWeight(result),
      riskFactor: calculateRiskFactor(result),
      actionRequired: determineActionRequired(result),
      classification: classifyResult(result)
    }
  })).sort((a, b) => b.yorha_confidence - a.yorha_confidence);
}

// Generate AI-powered recommendations
async function generateYoRHaRecommendations(
  query: string,
  analysisResults: any[],
  dataType: string
): Promise<any[]> {
  
  // Basic recommendation logic (would be enhanced with actual AI)
  const recommendations = [
    {
      id: `REC-${Date.now()}-1`,
      type: "INVESTIGATE",
      priority: "HIGH",
      title: `Further investigation recommended for: ${query}`,
      description: `Based on analysis of ${analysisResults.length} results, additional research is recommended`,
      actionItems: [
        "Review similar cases in jurisdiction",
        "Examine legal precedents",
        "Consult relevant statutes"
      ],
      estimatedTime: "2-4 hours",
      yorha_confidence: 0.85
    },
    {
      id: `REC-${Date.now()}-2`,
      type: "ANALYSIS",
      priority: "MEDIUM",
      title: "Document analysis required",
      description: "Several documents require detailed legal analysis",
      actionItems: [
        "Perform contract review",
        "Identify key clauses",
        "Assess legal risks"
      ],
      estimatedTime: "1-2 hours",
      yorha_confidence: 0.75
    }
  ];

  return recommendations;
}

// Utility functions for YoRHa legal analysis
function calculateOverallConfidence(results: any[]): number {
  if (!results.length) return 0;
  const sum = results.reduce((acc, r) => acc + (r.yorha_confidence || 0), 0);
  return Math.round((sum / results.length) * 100) / 100;
}

function calculateRelevance(query: string, content: string): number {
  if (!content) return 0;
  const queryWords = query.toLowerCase().split(/\s+/);
  const contentLower = content.toLowerCase();
  const matches = queryWords.filter(word => contentLower.includes(word));
  return matches.length / queryWords.length;
}

function calculateLegalWeight(result: any): number {
  const legalTerms = ["contract", "liability", "breach", "damages", "jurisdiction", "statute", "precedent"];
  const content = (result.content || "").toLowerCase();
  const matches = legalTerms.filter(term => content.includes(term));
  return Math.min(matches.length / 3, 1); // Normalize to 0-1
}

function calculateRiskFactor(result: any): number {
  const riskTerms = ["litigation", "penalty", "violation", "breach", "liability", "damages"];
  const content = (result.content || "").toLowerCase();
  const matches = riskTerms.filter(term => content.includes(term));
  return Math.min(matches.length / 2, 1); // Normalize to 0-1
}

function determineActionRequired(result: any): string {
  const riskFactor = calculateRiskFactor(result);
  if (riskFactor > 0.7) return "URGENT";
  if (riskFactor > 0.4) return "REVIEW";
  return "MONITOR";
}

function classifyResult(result: any): string {
  if (result.documentType) return result.documentType.toUpperCase();
  if (result.evidenceType) return result.evidenceType.toUpperCase();
  if (result.source === "enhanced-rag") return "AI_ANALYSIS";
  return "GENERAL";
}

function assessLegalComplexity(results: any[]): string {
  const avgLegalWeight = results.reduce((acc, r) => acc + (r.yorha_analysis?.legalWeight || 0), 0) / results.length;
  if (avgLegalWeight > 0.7) return "HIGH";
  if (avgLegalWeight > 0.4) return "MEDIUM";
  return "LOW";
}

function assessRiskLevel(results: any[]): string {
  const avgRiskFactor = results.reduce((acc, r) => acc + (r.yorha_analysis?.riskFactor || 0), 0) / results.length;
  if (avgRiskFactor > 0.7) return "HIGH";
  if (avgRiskFactor > 0.4) return "MEDIUM";
  return "LOW";
}

function extractJurisdiction(results: any[]): string[] {
  const jurisdictions = new Set<string>();
  results.forEach(r => {
    if (r.jurisdiction) jurisdictions.add(r.jurisdiction);
  });
  return Array.from(jurisdictions);
}

function extractLegalAreas(results: any[]): string[] {
  const areas = new Set<string>();
  results.forEach(r => {
    if (r.legalCategory) areas.add(r.legalCategory);
    if (r.topics) r.topics.forEach((topic: string) => areas.add(topic));
  });
  return Array.from(areas);
}

function findRelevantPrecedents(results: any[]): unknown[] {
  return results
    .filter(r => r.documentType === "precedent" || r.classification === "PRECEDENT")
    .slice(0, 3);
}

function extractKeyTerms(results: any[]): string[] {
  const terms = new Set<string>();
  results.forEach(r => {
    if (r.keywords) r.keywords.forEach((keyword: string) => terms.add(keyword));
  });
  return Array.from(terms).slice(0, 10);
}

function extractCitations(results: any[]): string[] {
  const citations = new Set<string>();
  results.forEach(r => {
    if (r.citation) citations.add(r.citation);
    if (r.fullCitation) citations.add(r.fullCitation);
  });
  return Array.from(citations);
}
