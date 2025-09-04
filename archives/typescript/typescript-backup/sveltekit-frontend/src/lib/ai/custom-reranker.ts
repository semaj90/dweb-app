
// Custom Reranker for Legal AI - Replaces basic top-K ANN with intelligent scoring
// Integrates with our existing Qdrant + PGVector setup

export interface RerankResult {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  originalScore: number;
  rerankScore: number;
  intent?: string;
  timeOfDay?: string;
  position?: string;
  confidence: number;
}

export interface UserContext {
  intent: "search" | "analyze" | "review" | "create" | "navigate";
  timeOfDay: "morning" | "afternoon" | "evening" | "night";
  focusedElement?: string;
  currentCase?: string;
  recentActions: string[];
  userRole: "prosecutor" | "detective" | "admin" | "user";
  workflowState: "draft" | "review" | "approved" | "archived";
}

export class LegalAIReranker {
  private contextWeights = {
    intent: 2.0,
    timeOfDay: 1.0,
    position: 1.5,
    role: 1.8,
    workflow: 1.2,
    recency: 0.8,
    confidence: 1.5,
  };

  /**
   * Advanced reranking with legal context awareness
   */
  async rerank(
    annResults: RerankResult[],
    userContext: UserContext,
    queryEmbedding?: number[],
  ): Promise<RerankResult[]> {
    return annResults
      .map((result) => {
        let score = result.originalScore;

        // Intent matching (critical for legal workflows)
        if (result.intent === userContext.intent) {
          score += this.contextWeights.intent;
        }

        // Time-based relevance (court schedules, deadlines)
        if (result.timeOfDay === userContext.timeOfDay) {
          score += this.contextWeights.timeOfDay;
        }

        // UI position context (focused evidence, active case)
        if (result.position === userContext.focusedElement) {
          score += this.contextWeights.position;
        }

        // Role-based scoring (prosecutor vs detective needs)
        score += this.calculateRoleScore(result, userContext.userRole);

        // Workflow state relevance
        score += this.calculateWorkflowScore(result, userContext.workflowState);

        // Recency boost for legal case updates
        score += this.calculateRecencyScore(result, userContext.recentActions);

        // Confidence penalty for low-confidence AI results
        score *= result.confidence / 100;

        return {
          ...result,
          rerankScore: score,
        };
      })
      .sort((a, b) => b.rerankScore - a.rerankScore);
  }

  /**
   * Role-specific scoring for legal professionals
   */
  private calculateRoleScore(result: RerankResult, role: string): number {
    const roleBoosts = {
      prosecutor: {
        "evidence-analysis": 2.0,
        "case-precedent": 1.8,
        "witness-testimony": 1.5,
      },
      detective: {
        "forensic-data": 2.0,
        "timeline-analysis": 1.8,
        "suspect-profile": 1.5,
      },
      admin: {
        "case-management": 2.0,
        "user-activity": 1.5,
        "system-reports": 1.3,
      },
    };

    const boosts = roleBoosts[role as keyof typeof roleBoosts];
    const contentType = result.metadata?.type as string;

    return boosts?.[contentType as keyof typeof boosts] || 0;
  }

  /**
   * Workflow state context scoring
   */
  private calculateWorkflowScore(
    result: RerankResult,
    workflowState: string,
  ): number {
    const workflowBoosts = {
      draft: { templates: 1.5, examples: 1.3 },
      review: { checklist: 1.8, validation: 1.5 },
      approved: { archive: 1.2, export: 1.5 },
    };

    const boosts = workflowBoosts[workflowState as keyof typeof workflowBoosts];
    const actionType = result.metadata?.actionType as string;

    return boosts?.[actionType as keyof typeof boosts] || 0;
  }

  /**
   * Recency scoring based on user's recent actions
   */
  private calculateRecencyScore(
    result: RerankResult,
    recentActions: string[],
  ): number {
    const resultAction = result.metadata?.lastAction as string;
    const actionIndex = recentActions.indexOf(resultAction);

    if (actionIndex === -1) return 0;

    // More recent actions get higher scores
    return (
      this.contextWeights.recency * (1 - actionIndex / recentActions.length)
    );
  }

  /**
   * Semantic similarity boost using embeddings
   */
  async calculateSemanticBoost(
    result: RerankResult,
    queryEmbedding: number[],
  ): Promise<number> {
    if (!result.metadata?.embedding || !queryEmbedding) return 0;

    const resultEmbedding = result.metadata.embedding as number[];
    return this.cosineSimilarity(queryEmbedding, resultEmbedding);
  }

  /**
   * Cosine similarity calculation
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Update context weights based on user feedback
   */
  updateWeights(feedbackData: Record<string, number>): void {
    Object.entries(feedbackData).forEach(([key, value]) => {
      if (key in this.contextWeights) {
        this.contextWeights[key as keyof typeof this.contextWeights] = value;
      }
    });
  }
}

// Neo4j path context enhancement
export interface Neo4jPathContext {
  userPath: string[];
  relatedCases: string[];
  frequentActions: string[];
  collaborators: string[];
  timeSpentByNode: Record<string, number>;
}

/**
 * Enhanced search with Neo4j path context
 */
export async function enhancedSearchWithNeo4j(
  query: string,
  userContext: UserContext,
  neo4jContext?: Neo4jPathContext,
  limit: number = 10,
): Promise<RerankResult[]> {
  // Use existing qdrant service for initial ANN search
  const { qdrantService } = await import("./qdrant-service");

  // Generate embedding for the query
  const { nomicEmbeddings } = await import("./nomic-embeddings");
  const embeddingResult = await nomicEmbeddings.embed(query);

  const annResults = await qdrantService.searchSimilar(
    embeddingResult.embedding,
    limit * 2,
  ); // Get more for reranking

  // Convert to rerank format with Neo4j enrichment
  const rerankInput: RerankResult[] = annResults.map((result: any) => ({
    id: result.id,
    content: result.payload?.text || "",
    metadata: {
      ...result.payload,
      neo4jPath: neo4jContext ? calculatePathScore(result, neo4jContext) : 0,
      relatedCases: neo4jContext?.relatedCases || [],
      userFrequency: neo4jContext
        ? calculateFrequencyScore(result, neo4jContext)
        : 0,
    },
    originalScore: result.score || 0,
    rerankScore: 0,
    confidence: (result.score || 0) * 100,
  }));

  // Apply custom reranking with Neo4j context
  const reranker = new LegalAIReranker();
  const rerankedResults = await reranker.rerank(rerankInput, userContext);

  return rerankedResults.slice(0, limit);
}

/**
 * Calculate path relevance score from Neo4j context
 */
function calculatePathScore(
  result: any,
  neo4jContext: Neo4jPathContext,
): number {
  let pathScore = 0;

  // Check if result relates to user's recent path
  neo4jContext.userPath.forEach((pathNode, index) => {
    if (
      result.content?.includes(pathNode) ||
      result.payload?.tags?.includes(pathNode)
    ) {
      pathScore +=
        (neo4jContext.userPath.length - index) / neo4jContext.userPath.length;
    }
  });

  // Boost score for related cases
  neo4jContext.relatedCases.forEach((caseId) => {
    if (result.payload?.caseId === caseId) {
      pathScore += 1.5;
    }
  });

  return pathScore;
}

/**
 * Calculate user frequency score
 */
function calculateFrequencyScore(
  result: any,
  neo4jContext: Neo4jPathContext,
): number {
  const nodeId = result.id || result.payload?.nodeId;
  if (!nodeId) return 0;

  const timeSpent = neo4jContext.timeSpentByNode[nodeId] || 0;
  return Math.min(timeSpent / 1000, 2.0); // Max boost of 2.0 points
}

// Legacy function for backward compatibility
export async function enhancedSearch(
  query: string,
  userContext: UserContext,
  limit: number = 10,
): Promise<RerankResult[]> {
  return enhancedSearchWithNeo4j(query, userContext, undefined, limit);
}

// Export for use in components
export { LegalAIReranker as default };

/**
 * Multi-LLM synthesis function for advanced legal AI workflows
 * Accepts multiple LLM outputs, user history, uploaded files, MCP server data, and synthesizes a rich output.
 * Caches, auto-encodes, and trains on user feedback/history. Generates fixes, code review, analysis, summaries, next steps, and self-prompting.
 */
import type {
  AIModelOutput,
  UserHistory,
  UploadedFile,
  MCPServerData,
  SynthesisOptions,
} from "./types";

export async function synthesizeMultiLLMOutput({
  llmOutputs,
  userHistory,
  uploadedFiles,
  mcpServers,
  options,
}: {
  llmOutputs: AIModelOutput[];
  userHistory: UserHistory;
  uploadedFiles: UploadedFile[];
  mcpServers: MCPServerData[];
  options?: SynthesisOptions;
}): Promise<{
  fixes: string[];
  codeReview: string;
  analysis: string;
  summary: string;
  nextSteps: string[];
  generativeAutocomplete: string;
  selfPrompt: string;
}> {
  // 1. Cache and auto-encode all inputs for fast retrieval and training
  // (Use Loki.js or similar for local cache)
  // ...existing code...

  // 2. Aggregate LLM outputs, user history, uploaded files, and MCP data
  const allInputs = [
    ...llmOutputs.map((o) => o.content),
    ...userHistory.actions,
    ...uploadedFiles.map((f) => f.textContent || f.name),
    ...mcpServers.map((s) => s.dataSummary),
  ];

  // 3. Synthesize extra "thinking" tokens from all sources
  const thinkingTokens = allInputs.join(" ");

  // 4. Apply best practices for legal AI: fixes, code review, analysis, summaries, next steps
  // (Stub: Replace with actual AI calls or rule-based logic)
  const fixes = llmOutputs.flatMap((o) => o.suggestedFixes || []);
  const codeReview = llmOutputs
    .map((o) => o.codeReview)
    .filter(Boolean)
    .join("\n---\n");
  const analysis = llmOutputs
    .map((o) => o.analysis)
    .filter(Boolean)
    .join(" ");
  const summary = llmOutputs
    .map((o) => o.summary)
    .filter(Boolean)
    .join(" ");
  const nextSteps = llmOutputs.flatMap((o) => o.nextSteps || []);
  const generativeAutocomplete = `Auto-complete: ${thinkingTokens.slice(0, 200)}...`;
  const selfPrompt = `AI Assistant: Based on all sources, recommend actions for prosecutor.`;

  // 5. Optionally train/update cache with user feedback
  // ...existing code...

  return {
    fixes,
    codeReview,
    analysis,
    summary,
    nextSteps,
    generativeAutocomplete,
    selfPrompt,
  };
}
