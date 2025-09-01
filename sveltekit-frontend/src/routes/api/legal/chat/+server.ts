import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';


// Legal AI Chat API - Context7 Enhanced with Gemma3 Legal
import { db } from "$lib/server/db/index";
import { URL } from "url";

// Mock interfaces for now
export interface InsertLegalAnalysisSession {
  userId: string;
  prompt: string;
  response: string;
  caseId?: string;
}

export interface LegalChatRequest {
  prompt: string;
  caseId?: string;
  userId: string;
  sessionType?:
    | "case_analysis"
    | "legal_research"
    | "document_review"
    | "precedent_search";
  context?: {
    caseDetails?: unknown;
    evidenceIds?: string[];
    requestedAnalysis?: string[];
  };
}

export interface LegalChatResponse {
  sessionId: string;
  analysis: string;
  confidence: number;
  sources: Array<{
    type: "document" | "precedent" | "statute";
    id: string;
    title: string;
    relevance: number;
    excerpt: string;
  }>;
  recommendations: string[];
  processingTime: number;
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const {
      prompt,
      caseId,
      userId,
      sessionType = "case_analysis",
      context,
    }: LegalChatRequest = await request.json();

    if (!prompt || !userId) {
      return json(
        { error: "Missing required fields: prompt, userId" },
        { status: 400 }
      );
    }

    // 1. Search for relevant legal documents and precedents
    const relevantSources = await findRelevantLegalSources(prompt, caseId);

    // 2. Generate legal analysis using Gemma3 Legal model
    const analysisResult = await generateLegalAnalysis(
      prompt,
      relevantSources,
      context
    );

    // 3. Create analysis session record
    const sessionInsert: InsertLegalAnalysisSession = {
      caseId: caseId || null,
      userId,
      sessionType,
      analysisPrompt: prompt,
      analysisResult: analysisResult.analysis,
      confidenceLevel: String(analysisResult.confidence),
      sourcesUsed: relevantSources.map((source) => ({
        type: source.type,
        id: source.id,
        title: source.title || source.caseTitle || source.citation,
        relevance: source.relevanceScore || 0.85,
      })),
      model: "gemma3-legal",
      processingTime: Date.now() - startTime,
      isActive: true,
    };
    const [session] = await db
      .insert(legalAnalysisSessions)
      .values(sessionInsert)
      .returning();

    const response: LegalChatResponse = {
      sessionId: session?.id || "",
      analysis: analysisResult.analysis,
      confidence: analysisResult.confidence,
      sources: relevantSources.map((source) => ({
        type: source.type,
        id: source.id,
        title: source.title || source.caseTitle || source.citation,
        relevance: source.relevanceScore || 0.85,
        excerpt: source.summary || source.content?.substring(0, 200) + "...",
      })),
      recommendations: analysisResult.recommendations,
      processingTime: session?.processingTime ?? 0,
    };

    return json(response);
  } catch (error: any) {
    console.error("Legal chat error:", error);
    return json(
      { error: "Failed to process legal analysis request" },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const searchParams = url.searchParams;
    const caseId = searchParams.get("caseId");
    const userId = searchParams.get("userId");
    const sessionType = searchParams.get("sessionType");
    const limit = parseInt(searchParams.get("limit") || "20");

    const conditions = [];
    if (caseId) conditions.push(eq(legalAnalysisSessions.caseId, caseId));
    if (userId) conditions.push(eq(legalAnalysisSessions.userId, userId));
    if (sessionType)
      conditions.push(eq(legalAnalysisSessions.sessionType, sessionType));

    const sessions = await db
      .select()
      .from(legalAnalysisSessions)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(legalAnalysisSessions.createdAt))
      .limit(limit);

    return json(sessions);
  } catch (error: any) {
    console.error("Error fetching legal analysis sessions:", error);
    return json(
      { error: "Failed to fetch analysis sessions" },
      { status: 500 }
    );
  }
};

// Helper function to find relevant legal sources
async function findRelevantLegalSources(prompt: string, caseId?: string): Promise<any> {
  const sources = [];

  try {
    // Search legal documents
    const documents = await db
      .select()
      .from(legalDocuments)
      .where(
        caseId
          ? and(
              eq(legalDocuments.caseId, caseId),
              like(legalDocuments.content, `%${prompt}%`)
            )
          : like(legalDocuments.content, `%${prompt}%`)
      )
      .limit(5);

    sources.push(...documents.map((doc) => ({ ...doc, type: "document" })));

    // Search legal precedents (vector similarity would be ideal here)
    const precedents = await db
      .select()
      .from(legalPrecedents)
      .where(like(legalPrecedents.summary, `%${prompt}%`))
      .limit(3);

    sources.push(...precedents.map((prec) => ({ ...prec, type: "precedent" })));
  } catch (error: any) {
    console.warn("Error searching legal sources:", error);
  }

  return sources;
}

// Helper function to generate legal analysis using Gemma3
async function generateLegalAnalysis(
  prompt: string,
  sources: any[],
  context?: unknown
): Promise<any> {
  try {
    // Construct analysis prompt with legal context
    const legalPrompt = `
As a legal AI assistant specialized in prosecutor case analysis, analyze the following:

QUERY: ${prompt}

RELEVANT SOURCES:
${sources
  .map(
    (source) => `
- ${source.type.toUpperCase()}: ${source.title || source.caseTitle || source.citation}
  Summary: ${source.summary || source.content?.substring(0, 300)}
`
  )
  .join("\n")}

CASE CONTEXT:
${context ? JSON.stringify(context, null, 2) : "No additional context provided"}

Please provide:
1. Legal Analysis (comprehensive analysis of the query)
2. Confidence Level (0.0-1.0)
3. Key Recommendations (3-5 actionable items)
4. Supporting Evidence from the provided sources

Format your response as structured JSON.
`;

    // In a real implementation, this would call the Gemma3 Legal model via Ollama
    // For now, return a structured response
    const analysisResult = {
      analysis: `Based on the legal query and available sources, the analysis indicates several key considerations for the prosecution. The relevant precedents and documents suggest a strong foundation for the case, with particular attention needed to evidence handling and procedural requirements.

Key legal principles identified:
- Chain of custody requirements must be strictly maintained
- All evidence must meet admissibility standards under current jurisdiction
- Procedural deadlines and notification requirements are critical

The case appears to have merit based on the documented evidence and applicable legal standards.`,
      confidence: 0.87,
      recommendations: [
        "Review all evidence for chain of custody documentation",
        "Verify compliance with procedural notification requirements",
        "Prepare responses to anticipated defense challenges",
        "Consider additional expert witness testimony if needed",
        "Document all procedural steps for appellate protection",
      ],
    };

    return analysisResult;
  } catch (error: any) {
    console.error("Error generating legal analysis:", error);
    throw new Error("Failed to generate legal analysis");
  }
}
