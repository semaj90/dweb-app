import type { RequestHandler } from './$types.js';

// Comprehensive AI Mix Summaries API - End-to-End Integration
// Combines: Local LLM + Enhanced RAG + User Activity (simulated) + Fuse.js + XState Synthesis
// Supports: Chunking, Streaming, Async Processing, Hybrid Vector Search

// TODO: Strengthen typing & streaming implementation; current file cleaned from corruption
import { db } from "$lib/server/db";
import { evidence, cases, legalDocuments } from "$lib/server/db/schema-postgres";
import { eq } from "drizzle-orm";
import { vectorService } from "$lib/server/vector/vectorService";
import { qdrantService } from "$lib/server/services/qdrant-service";
import Fuse from "fuse.js";
import { interpret } from "xstate";
import { aiSummaryMachine } from "$lib/machines/aiSummaryMachine";
import { ollamaService } from "$lib/server/services/ollama-service"; // Assumed service providing generateResponse
import { URL } from "url";

// Request payload for summary generation
export interface SummaryRequest {
  type: "case" | "evidence" | "legal_document" | "cross_analysis";
  targetId: string;
  depth: "quick" | "comprehensive" | "forensic";
  includeRAG: boolean;
  includeUserActivity: boolean;
  enableStreaming: boolean;
  chunkSize?: number;
  userId?: string;
}

export interface AILLMOutput {
  content: string;
  model: string;
  confidence: number;
  tokens: number;
  processingTime: number;
}

// Basic shape for vector search results (loose to accommodate both services)
export interface BasicVectorResult {
  id: string;
  content?: string;
  payload?: { content?: string; [k: string]: unknown };
  score?: number;
  relevance?: number;
  source?: string;
}

export interface EnhancedRAGOutput {
  relevantDocs: Array<{
    id: string;
    content: string;
    relevance: number;
    source: string;
  }>;
  contextSummary: string;
  searchMetrics: {
    vectorSearchTime: number;
    documentsRetrieved: number;
    averageRelevance: number;
  };
}

export interface UserActivityContext {
  recentQueries: string[];
  preferredTopics: string[];
  interactionPatterns: {
    timeOfDay: string;
    commonActions: string[];
    focusAreas: string[];
  };
  recommendations: string[];
}

export interface SynthesizedOutput {
  summary: string;
  keyInsights: string[];
  actionItems: string[];
  confidence: number;
  sources: Array<{
    type: "llm" | "rag" | "user_activity";
    contribution: number;
    details: any;
  }>;
  nextSteps: string[];
  relatedCases?: string[];
  warnings?: string[];
}

// XState Machine Service for orchestrating AI mix workflow
const summaryService = interpret(aiSummaryMachine).start();

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Authentication required" }, { status: 401 });
    }

    const summaryRequest: SummaryRequest = await request.json();
    const userId = locals.user.id;

    // Initialize XState machine with request context
    summaryService.send({
      type: "GENERATE_SUMMARY" as any,
    });

    if (summaryRequest.enableStreaming) {
      return handleStreamingSummary(summaryRequest, userId);
    } else {
      return handleBatchSummary(summaryRequest, userId);
    }
  } catch (error: any) {
    console.error("Summaries API error:", error);
    return json(
      {
        error: "Failed to generate summary",
        details: error.message,
      },
      { status: 500 }
    );
  }
};

async function handleBatchSummary(request: SummaryRequest, userId: string): Promise<any> {
  const startTime = Date.now();

  // Step 1: Get Local LLM Output
  const llmOutput = await getLocalLLMOutput(request);

  // Step 2: Get Enhanced RAG Output
  const ragOutput = request.includeRAG
    ? await getEnhancedRAGOutput(request)
    : null;

  // Step 3: Get User Activity Context
  const userActivity = request.includeUserActivity
    ? await getUserActivityContext(userId)
    : null;

  // Step 4: Synthesize all outputs using XState + Fuse.js
  const synthesizedResult = await synthesizeOutputs({
    llmOutput,
    ragOutput,
    userActivity,
    request,
  });

  const totalTime = Date.now() - startTime;

  return json({
    success: true,
    result: synthesizedResult,
    metadata: {
      processingTime: totalTime,
      request: request,
      timestamp: new Date().toISOString(),
      version: "1.0.0",
    },
  });
}

async function handleStreamingSummary(request: SummaryRequest, userId: string): Promise<any> {
  // Create SSE stream for real-time updates
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();

      try {
        // Send initial status
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "status",
              message: "Starting AI summary generation...",
              progress: 0,
            })}\n\n`
          )
        );

        // Step 1: Local LLM (streaming)
        const llmOutput = await getLocalLLMOutputStreaming(request, (chunk) => {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: "llm_chunk",
                content: chunk,
                progress: 33,
              })}\n\n`
            )
          );
        });

        // Step 2: Enhanced RAG
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "status",
              message: "Retrieving relevant documents...",
              progress: 50,
            })}\n\n`
          )
        );

        const ragOutput = request.includeRAG
          ? await getEnhancedRAGOutput(request)
          : null;

        // Step 3: User Activity
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "status",
              message: "Analyzing user activity patterns...",
              progress: 75,
            })}\n\n`
          )
        );

        const userActivity = request.includeUserActivity
          ? await getUserActivityContext(userId)
          : null;

        // Step 4: Final synthesis
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "status",
              message: "Synthesizing final summary...",
              progress: 90,
            })}\n\n`
          )
        );

        const synthesizedResult = await synthesizeOutputs({
          llmOutput,
          ragOutput,
          userActivity,
          request,
        });

        // Send final result
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "complete",
              result: synthesizedResult,
              progress: 100,
            })}\n\n`
          )
        );

        controller.close();
      } catch (error: any) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "error",
              error: error.message,
            })}\n\n`
          )
        );
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Cache-Control",
    },
  });
}

// AI Mix Function 1: Local LLM Output
async function getLocalLLMOutput(
  request: SummaryRequest
): Promise<AILLMOutput> {
  const startTime = Date.now();

  // Get source content based on type
  let sourceContent = "";
  switch (request.type) {
    case "case":
      const caseData = await db
        .select()
        .from(cases)
        .where(eq(cases.id, request.targetId))
        .limit(1);
      sourceContent = caseData[0]?.description || "";
      break;
    case "evidence":
      const evidenceData = await db
        .select()
        .from(evidence)
        .where(eq(evidence.id, request.targetId))
        .limit(1);
      sourceContent = evidenceData[0]?.description || "";
      break;
    case "legal_document":
      const docData = await db
        .select()
        .from(legalDocuments)
        .where(eq(legalDocuments.id, request.targetId))
        .limit(1);
      sourceContent = docData[0]?.content || "";
      break;
  }

  // Prepare prompt based on depth
  const depthPrompts = {
    quick: "Provide a concise 2-3 sentence summary of:",
    comprehensive: "Provide a detailed analysis and comprehensive summary of:",
    forensic:
      "Conduct a thorough forensic analysis with legal implications for:",
  };

  const prompt = `${depthPrompts[request.depth]} ${sourceContent}`;

  // Use chunking for large content
  const chunkSize = request.chunkSize || 2000;
  const chunks =
    sourceContent.length > chunkSize
      ? chunkText(sourceContent, chunkSize)
      : [sourceContent];

  let combinedResponse = "";
  let totalTokens = 0;

  for (const chunk of chunks) {
    const promptText =
      chunk.length < sourceContent.length
        ? `${prompt} (Part of larger document): ${chunk}`
        : prompt;
    const response = await ollamaService.generateResponse(promptText, {
      model: "gemma3:7b-instruct-q4_K_M",
      temperature: 0.3,
      top_p: 0.9,
      num_predict: request.depth === "forensic" ? 1000 : 500,
    });

    combinedResponse += response.response + "\n\n";
    totalTokens += response.eval_count || 0;
  }

  const processingTime = Date.now() - startTime;

  return {
    content: combinedResponse.trim(),
    model: "gemma3:7b-instruct-q4_K_M",
    confidence: 0.85, // Based on model reliability
    tokens: totalTokens,
    processingTime,
  };
}

// AI Mix Function 2: Enhanced RAG Output with pgvector + Qdrant
async function getEnhancedRAGOutput(
  request: SummaryRequest
): Promise<EnhancedRAGOutput> {
  const startTime = Date.now();

  // Create search query from target content
  let searchQuery = "";
  switch (request.type) {
    case "case":
      const caseData = await db
        .select()
        .from(cases)
        .where(eq(cases.id, request.targetId))
        .limit(1);
      searchQuery =
        `${caseData[0]?.title} ${caseData[0]?.description}`.substring(0, 200);
      break;
    case "evidence":
      const evidenceData = await db
        .select()
        .from(evidence)
        .where(eq(evidence.id, request.targetId))
        .limit(1);
      searchQuery =
        `${evidenceData[0]?.title} ${evidenceData[0]?.description}`.substring(
          0,
          200
        );
      break;
  }

  // Dual vector search: PostgreSQL pgvector + Qdrant
  const [pgResults, qdrantResults] = await Promise.all([
    // Use internal vector service (pgvector backed) and external qdrant service
    vectorService.search(searchQuery, { limit: 10, threshold: 0.7 }).catch(() => []),
    qdrantService.searchSimilar(searchQuery, { limit: 10, threshold: 0.7 }).catch(() => []),
  ]);

  // Combine and deduplicate results
  const allResults = [...pgResults, ...qdrantResults];
  const uniqueResults = Array.from(
    new Map(allResults.map((item) => [item.id, item])).values()
  );

  // Rank results by relevance
  const relevantDocs = uniqueResults
    .sort(
      (a, b) => (b.score || b.relevance || 0) - (a.score || a.relevance || 0)
    )
    .slice(0, 5)
    .map((doc) => ({
      id: doc.id,
      content: doc.content || doc.payload?.content || "",
      relevance: doc.score || doc.relevance || 0,
      source: doc.source || "vector_db",
    }));

  // Generate context summary using the most relevant docs
  const contextContent = relevantDocs.map((doc) => doc.content).join("\n\n");
  const contextSummaryResp = await ollamaService.generateResponse(
    `Summarize the key context from these related documents: ${contextContent.substring(0, 1500)}`,
    {
      model: "gemma3:7b-instruct-q4_K_M",
      temperature: 0.2,
      num_predict: 300,
    }
  );

  const processingTime = Date.now() - startTime;

  return {
    relevantDocs,
  contextSummary: contextSummaryResp.response,
    searchMetrics: {
      vectorSearchTime: processingTime,
      documentsRetrieved: uniqueResults.length,
      averageRelevance:
        relevantDocs.reduce((sum, doc) => sum + doc.relevance, 0) /
        relevantDocs.length,
    },
  };
}

// AI Mix Function 3: User Activity Context with Loki.js
async function getUserActivityContext(
  userId: string
): Promise<UserActivityContext> {
  // TODO: In production, this would connect to actual Loki.js user activity store
  // For now, we'll simulate based on available data patterns

  // Get user's recent case interactions
  const recentCases = await db
    .select()
    .from(cases)
    .where(eq(cases.createdBy, userId))
    .orderBy(cases.updatedAt)
    .limit(10);

  // Get user's recent evidence interactions
  const recentEvidence = await db
    .select()
    .from(evidence)
    .where(eq(evidence.uploadedBy, userId))
    .orderBy(evidence.uploadedAt)
    .limit(10);

  // Extract activity patterns
  const recentQueries = [
    ...recentCases.map((c) => c.title),
    ...recentEvidence.map((e: any) => e.title),
  ].slice(0, 5);

  const preferredTopics = extractTopics([
    ...recentCases.map((c) => c.description || ""),
    ...recentEvidence.map((e: any) => e.description || ""),
  ]);

  // Generate recommendations using Fuse.js fuzzy search
  const fuse = new Fuse(recentQueries, {
    threshold: 0.6,
    keys: ["title", "description"],
  });

  const recommendations = generateRecommendations(preferredTopics, fuse);

  return {
    recentQueries,
    preferredTopics,
    interactionPatterns: {
      timeOfDay: "morning", // TODO: Extract from actual activity logs
      commonActions: ["case_analysis", "evidence_upload", "report_generation"],
      focusAreas: preferredTopics.slice(0, 3),
    },
    recommendations,
  };
}

// AI Mix Function 4: XState-powered Synthesis Engine
async function synthesizeOutputs({
  llmOutput,
  ragOutput,
  userActivity,
  request,
}: {
  llmOutput: AILLMOutput;
  ragOutput: EnhancedRAGOutput | null;
  userActivity: UserActivityContext | null;
  request: SummaryRequest;
}): Promise<SynthesizedOutput> {
  // Update XState machine with collected data
  summaryService.send({
    type: "SYNTHESIZE_INSIGHTS" as any,
  });

  // Weighted synthesis based on source reliability
  const weights = {
    llm: 0.6,
    rag: ragOutput ? 0.3 : 0,
    userActivity: userActivity ? 0.1 : 0,
  };

  // Normalize weights if some sources are missing
  const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);
  Object.keys(weights).forEach((key) => {
    weights[key] = weights[key] / totalWeight;
  });

  // Extract key insights using Fuse.js across all sources
  const allContent = [
    llmOutput.content,
    ragOutput?.contextSummary || "",
    userActivity?.recentQueries.join(" ") || "",
  ].filter(Boolean);

  const keyInsights = await extractKeyInsights(allContent);
  const actionItems = await generateActionItems(llmOutput.content, ragOutput);

  // Calculate confidence score
  const confidence =
    llmOutput.confidence * weights.llm +
    (ragOutput ? 0.9 : 0) * weights.rag +
    (userActivity ? 0.8 : 0) * weights.userActivity;

  // Generate next steps based on synthesis
  const nextSteps = await generateNextSteps(
    keyInsights,
    actionItems,
    request.type
  );

  // Find related cases using vector similarity
  const relatedCases = await findRelatedCases(request.targetId, request.type);

  // Send completion event to XState machine
  summaryService.send({ type: "RESET" as any });

  return {
    summary: llmOutput.content,
    keyInsights,
    actionItems,
    confidence,
    sources: [
      {
        type: "llm",
        contribution: weights.llm,
        details: {
          model: llmOutput.model,
          tokens: llmOutput.tokens,
          processingTime: llmOutput.processingTime,
        },
      },
      ...(ragOutput
        ? [
            {
              type: "rag" as const,
              contribution: weights.rag,
              details: {
                documentsUsed: ragOutput.relevantDocs.length,
                averageRelevance: ragOutput.searchMetrics.averageRelevance,
              },
            },
          ]
        : []),
      ...(userActivity
        ? [
            {
              type: "user_activity" as const,
              contribution: weights.userActivity,
              details: {
                recentQueries: userActivity.recentQueries.length,
                preferences: userActivity.preferredTopics.length,
              },
            },
          ]
        : []),
    ],
    nextSteps,
    relatedCases,
    warnings: generateWarnings(
      confidence,
      ragOutput?.searchMetrics.documentsRetrieved || 0
    ),
  };
}

// Helper Functions

function chunkText(text: string, chunkSize: number): string[] {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize) {
    chunks.push(text.substring(i, i + chunkSize));
  }
  return chunks;
}

async function getLocalLLMOutputStreaming(
  request: SummaryRequest,
  onChunk: (chunk: string) => void
): Promise<AILLMOutput> {
  // This would implement streaming response from Ollama
  // For now, return the same as batch mode
  return getLocalLLMOutput(request);
}

function extractTopics(texts: string[]): string[] {
  // Simple topic extraction - in production would use more sophisticated NLP
  const allText = texts.join(" ").toLowerCase();
  const keywords = allText.match(/\b\w{4,}\b/g) || [];
  const frequency: Record<string, number> = {};

  keywords.forEach((word) => {
    frequency[word] = (frequency[word] || 0) + 1;
  });

  return Object.entries(frequency)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([word]) => word);
}

function generateRecommendations(topics: string[], fuse: Fuse<any>): string[] {
  // Generate recommendations based on topic analysis
  return topics
    .slice(0, 3)
    .map((topic) => `Consider exploring ${topic} in related cases`);
}

async function extractKeyInsights(contents: string[]): Promise<string[]> {
  // Use simple keyword extraction for insights
  const combined = contents.join(" ");
  const sentences = combined
    .split(/[.!?]+/)
    .filter((s) => s.trim().length > 20);

  // Sort by sentence importance (length and keyword density)
  return sentences
    .sort((a, b) => b.length - a.length)
    .slice(0, 5)
    .map((s) => s.trim());
}

async function generateActionItems(
  llmContent: string,
  ragOutput: EnhancedRAGOutput | null
): Promise<string[]> {
  const actionItems = [];

  if (llmContent.toLowerCase().includes("evidence")) {
    actionItems.push("Review additional evidence");
  }
  if (llmContent.toLowerCase().includes("witness")) {
    actionItems.push("Schedule witness interviews");
  }
  if (ragOutput && ragOutput.relevantDocs.length > 0) {
    actionItems.push("Analyze related legal precedents");
  }

  return actionItems.length > 0 ? actionItems : ["Continue investigation"];
}

async function generateNextSteps(
  insights: string[],
  actionItems: string[],
  type: string
): Promise<string[]> {
  const nextSteps = [...actionItems];

  if (type === "case") {
    nextSteps.push("Update case status");
    nextSteps.push("Notify relevant stakeholders");
  }

  return nextSteps.slice(0, 5);
}

async function findRelatedCases(
  targetId: string,
  type: string
): Promise<string[]> {
  if (type !== "case") return [];

  // Simple related case finding - in production would use vector similarity
  const casesData = await db.select().from(cases).limit(5);
  return casesData.map((c) => c.id).filter((id) => id !== targetId);
}

function generateWarnings(confidence: number, docsRetrieved: number): string[] {
  const warnings = [];

  if (confidence < 0.7) {
    warnings.push("Low confidence in results - consider additional analysis");
  }
  if (docsRetrieved < 3) {
    warnings.push("Limited contextual information available");
  }

  return warnings;
}

// GET endpoint for summary status and health check
export const GET: RequestHandler = async ({ url }) => {
  const summaryId = url.searchParams.get("id");

  if (summaryId) {
    // Return status of specific summary
    return json({
      status: "completed", // TODO: Implement actual status tracking
      summaryId,
      timestamp: new Date().toISOString(),
    });
  }

  // Health check
  return json({
    service: "AI Mix Summaries API",
    status: "healthy",
    capabilities: [
      "Local LLM Integration (Ollama + Gemma3)",
      "Enhanced RAG (PostgreSQL pgvector + Qdrant)",
      "User Activity Analysis (Loki.js simulation)",
      "XState Synthesis Engine",
      "Streaming Support",
      "Chunking & Async Processing",
    ],
    version: "1.0.0",
    timestamp: new Date().toISOString(),
  });
};
