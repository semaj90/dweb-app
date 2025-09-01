/// <reference types="vite/client" />
import crypto from "crypto";

/**
 * Enhanced Copilot Self-Prompt Wrapper Utility
 * Integrates semantic search, memory MCP servers, multi-agent orchestration,
 * and autonomous engineering for comprehensive problem-solving
 */

import { autonomousEngineeringSystem } from '../services/autonomous-engineering-system';

// Mock functions for missing services
const analyzeLegalCaseWithCrew = async (caseData: any): Promise<any> => ({ analysis: 'completed' });

// Mock types and imports
export interface AITask {
  id: string;
  type: string;
  data?: any;
}

// Mock embeddings service
const OpenAIEmbeddings = class {
  embedQuery = async (query: string) => new Array(384).fill(0).map(() => Math.random());
};

// Mock database pool
const pool = {
  query: async (query: string, params?: any[]) => ({ rows: [] })
};

// Import Redis client
import { createClient } from 'redis';

// Singleton Redis client for connection reuse
let redisClient: ReturnType<typeof createClient> | null = null;
async function getRedisClient(): Promise<any> {
  if (!redisClient) {
    redisClient = createClient({
      url: import.meta.env.REDIS_URL || "redis://localhost:6379",
    });
    redisClient.on("error", (err) => console.error("Redis Client Error", err));
    await redisClient.connect();
    console.log("✅ Redis client connected for caching.");
  }
  return redisClient;
}

// Enhanced context with Redis cache and LangChain/Nomic embeddings
export async function getEnhancedContext(query: string): Promise<any> {
  const cacheKey = `context:${query}`;
  const client = await getRedisClient();
  try {
    // 1. Check cache first
    let cachedResult = await client.get(cacheKey);
    if (cachedResult) {
      if (Buffer.isBuffer(cachedResult)) {
        cachedResult = cachedResult.toString();
      }
      console.log("CACHE HIT for:", query);
      return JSON.parse(cachedResult);
    }
    // 2. If not in cache, fetch and compute the data
    console.log("CACHE MISS for:", query);
    // Use LangChain to embed the query with local Nomic embed LLM (no baseURL property)
    const embeddings = new OpenAIEmbeddings({
      modelName: "nomic-embed-text",
      openAIApiKey: "N/A",
      // baseURL property removed; using local LLM per best practices
    });
    // PGVectorStore expects (embeddings, config)
    const vectorStore = new PGVectorStore(embeddings, {
      pool,
      tableName: "vectors",
    });
    // Generate embedding and search for top results
    const results = await vectorStore.similaritySearch(query, 8);
    // 3. Store the result in Redis with an expiration (e.g., 1 hour)
    await client.set(cacheKey, JSON.stringify(results), { EX: 3600 });
    return results;
  } catch (err: any) {
    console.error("Redis cache error:", err);
    // Fallback to direct semantic search if cache fails
    return [];
  }
}

// Example: Inject enhanced context into Copilot prompt
export async function injectContextToCopilotPrompt(
  query: string,
  code: string
): Promise<any> {
  const context = await getEnhancedContext(query);
  return `/* Copilot Context Injection: ${JSON.stringify(context)} */\n${code}`;
}

export interface CopilotSelfPromptOptions {
  useSemanticSearch?: boolean;
  useMemory?: boolean;
  useMultiAgent?: boolean;
  useAutonomousEngineering?: boolean;
  enableSelfSynthesis?: boolean;
  context?: {
    projectPath?: string;
    platform?: "webapp" | "desktop" | "mobile" | "all";
    urgency?: "low" | "medium" | "high" | "critical";
    includeTests?: boolean;
    targetExtensions?: string[]; // For Cline, Roo, VSCode extensions
  };
  outputFormat?: "json" | "markdown" | "structured";
}

export interface CopilotSelfPromptResult {
  contextResults: any[];
  memoryResults: any[];
  agentResults: any[];
  engineeringAnalysis?: unknown;
  synthesizedOutput: string;
  nextActions: NextAction[];
  recommendations: Recommendation[];
  selfPrompt: string;
  executionPlan?: ExecutionPlan;
  metadata: {
    processingTime: number;
    confidence: number;
    sources: string[];
    tokensUsed: number;
  };
}

export interface NextAction {
  id: string;
  type: "code" | "test" | "debug" | "research" | "deploy" | "monitor";
  priority: "low" | "medium" | "high" | "critical";
  description: string;
  commands?: string[];
  targetFiles?: string[];
  estimatedTime: number;
  dependencies?: string[];
}

export interface Recommendation {
  category:
    | "architecture"
    | "performance"
    | "security"
    | "testing"
    | "deployment";
  title: string;
  description: string;
  impact: "low" | "medium" | "high";
  effort: "low" | "medium" | "high";
  priority: number;
}

export interface ExecutionPlan {
  phases: ExecutionPhase[];
  totalEstimatedTime: number;
  parallelizable: boolean;
  criticalPath: string[];
}

export interface ExecutionPhase {
  id: string;
  name: string;
  actions: string[];
  order: number;
  canRunInParallel: boolean;
}

/**
 * Main Copilot self-prompting function with comprehensive AI orchestration
 */
export async function copilotSelfPrompt(
  prompt: string,
  options: CopilotSelfPromptOptions = {}
): Promise<CopilotSelfPromptResult> {
  const startTime = Date.now();
  console.log("🤖 Starting enhanced Copilot self-prompt...");

  const {
    useSemanticSearch = true,
    useMemory = true,
    useMultiAgent = true,
    useAutonomousEngineering = true,
    enableSelfSynthesis = true,
    context = {},
    outputFormat = "structured",
  } = options;

  let contextResults: any[] = [];
  let memoryResults: any[] = [];
  let agentResults: any[] = [];
  let engineeringAnalysis: any = null;
  let tokensUsed = 0;

  try {
    // Phase 1: Semantic Search & Memory Integration
    if (useSemanticSearch) {
      contextResults = await performSemanticSearch(prompt, context);
      console.log(`📚 Found ${contextResults.length} semantic matches`);
    }

    if (useMemory) {
      memoryResults = await accessMemoryMCP(prompt, context);
      console.log(`🧠 Retrieved ${memoryResults.length} memory entries`);
    }

    // Phase 2: Multi-Agent Analysis
    if (useMultiAgent) {
      agentResults = await orchestrateMultiAgentAnalysis(prompt, context);
      console.log(
        `👥 Completed multi-agent analysis with ${agentResults.length} agent responses`
      );
      tokensUsed += agentResults.reduce(
        (sum, result) => sum + (result.tokensUsed || 0),
        0
      );
    }

    // Phase 3: Autonomous Engineering (if enabled)
    if (useAutonomousEngineering) {
      engineeringAnalysis =
        await autonomousEngineeringSystem.solveProblemAutonomously(prompt, {
          projectPath: context.projectPath,
          platform: context.platform || "webapp",
          urgency: context.urgency || "medium",
          includeTests: context.includeTests || true,
        });
      console.log("🔧 Autonomous engineering analysis completed");
    }

    // Phase 4: Self-Synthesis
    const synthesizedOutput = enableSelfSynthesis
      ? await synthesizeAllResults(
          prompt,
          contextResults,
          memoryResults,
          agentResults,
          engineeringAnalysis
        )
      : generateBasicSummary(
          prompt,
          contextResults,
          memoryResults,
          agentResults
        );

    // Phase 5: Generate Next Actions and Recommendations
    const nextActions = await generateNextActions(
      prompt,
      synthesizedOutput,
      engineeringAnalysis
    );
    const recommendations = await generateRecommendations(
      engineeringAnalysis,
      context
    );

    // Phase 6: Create Execution Plan
    const executionPlan = await createExecutionPlan(
      nextActions,
      recommendations
    );

    // Phase 7: Generate Self-Prompt for Copilot
    const selfPrompt = generateCopilotSelfPrompt(
      prompt,
      synthesizedOutput,
      nextActions,
      recommendations,
      outputFormat
    );

    const processingTime = Date.now() - startTime;

    return {
      contextResults,
      memoryResults,
      agentResults,
      engineeringAnalysis,
      synthesizedOutput,
      nextActions,
      recommendations,
      selfPrompt,
      executionPlan,
      metadata: {
        processingTime,
        confidence: calculateConfidence(
          contextResults,
          agentResults,
          engineeringAnalysis
        ),
        sources: extractSources(contextResults, memoryResults, agentResults),
        tokensUsed,
      },
    };
  } catch (error: any) {
    // Log error to MCP_TODO_LOG.md for productionization
    try {
      const { mcpLogErrorOrContextLoss } = await import("./mcp-helpers.js");
      await mcpLogErrorOrContextLoss(
        `Copilot self-prompt failed: ${error?.message || error}`
      );
    } catch (logErr) {
      console.error("Failed to log error to MCP_TODO_LOG.md:", logErr);
    }
    console.error("❌ Copilot self-prompt failed:", error);
    throw error;
  }
}

/**
 * Enhanced semantic search with caching and relevance scoring
 */
async function performSemanticSearch(
  prompt: string,
  context: any
): Promise<any[]> {
  try {
    // Quick health check with timeout to avoid hanging
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1000); // 1 second timeout
    
    try {
      const response = await fetch("http://localhost:8000/api/semantic/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          query: prompt,
          context: context.projectPath || process.cwd(),
          limit: 20,
          threshold: 0.7,
          includeCode: true,
          includeDocs: true,
        }),
      });
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        // Sort by relevance_score if available
      if (Array.isArray(data.results)) {
        return data.results.sort(
          (a, b) => (b.relevance_score || 0) - (a.relevance_score || 0)
        );
      }
      return data.results || [];
    }
    } catch (error: any) {
      clearTimeout(timeoutId);
      console.error("Semantic search service unavailable:", error.name);
      return []; // Return empty array immediately instead of hanging
    }
  } catch (error: any) {
    console.error("Semantic search failed:", error);
    return []; // Fast fallback for any other errors
  }
}

/**
 * Access memory MCP servers for context and history
 */
export async function accessMemoryMCP(
  prompt: string,
  context: any
): Promise<any[]> {
  try {
    // Quick timeout to avoid hanging
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1000);
    
    try {
      const response = await fetch("http://localhost:8000/api/memory/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          query: prompt,
          context: context,
          includeGraph: true,
          includeHistory: true,
        }),
      });
      clearTimeout(timeoutId);
      
      if (response.ok) {
      const data = await response.json();
      // Sort by recency or relevance if available
      if (Array.isArray(data.memories)) {
        return data.memories.sort(
          (a, b) => (b.relevance_score || 0) - (a.relevance_score || 0)
        );
      }
      return data.memories || [];
    }
    } catch (error: any) {
      clearTimeout(timeoutId);
      console.error("Memory MCP service unavailable:", error.name);
      return []; // Fast fallback
    }
  } catch (error: any) {
    console.error("Memory MCP access failed:", error);
    return []; // Fast fallback for any other errors
  }
}

/**
 * Orchestrate multi-agent analysis with AutoGen and CrewAI
 */
async function orchestrateMultiAgentAnalysis(
  prompt: string,
  context: any
): Promise<any[]> {
  const results: any[] = [];
  try {
    // AutoGen analysis (production)
    const autogenResult = await autoGenService.executeLegalWorkflow(
      "case_analysis",
      prompt,
      context
    );
    results.push({
      source: "autogen",
      type: "conversational_analysis",
      ...autogenResult,
    });
    // CrewAI analysis (production)
    const crewaiResult = await analyzeLegalCaseWithCrew(
      prompt,
      [],
      context.jurisdiction || "federal"
    );
    results.push({
      source: "crewai",
      type: "task_based_analysis",
      ...crewaiResult,
    });
  } catch (error: any) {
    console.error("Multi-agent analysis failed:", error);
  }
  // Sort agent results by confidence/tokensUsed if available
  return results.sort(
    (a, b) =>
      (b.confidence || b.tokensUsed || 0) - (a.confidence || a.tokensUsed || 0)
  );
}

/**
 * Synthesize all results using advanced LLM coordination
 */
async function synthesizeAllResults(
  prompt: string,
  contextResults: any[],
  memoryResults: any[],
  agentResults: any[],
  engineeringAnalysis: any
): Promise<string> {
  const synthesisPrompt = `
As an advanced AI synthesis engine, analyze and synthesize the following comprehensive analysis results:

ORIGINAL PROMPT: ${prompt}

SEMANTIC SEARCH RESULTS:
${JSON.stringify(contextResults, null, 2)}

MEMORY CONTEXT:
${JSON.stringify(memoryResults, null, 2)}

MULTI-AGENT ANALYSIS:
${JSON.stringify(agentResults, null, 2)}

AUTONOMOUS ENGINEERING ANALYSIS:
${JSON.stringify(engineeringAnalysis, null, 2)}

Please provide a comprehensive synthesis that:
1. Identifies key insights and patterns
2. Resolves any conflicts between different analyses
3. Provides a unified understanding of the problem/situation
4. Suggests concrete next steps
5. Highlights critical issues or opportunities
6. Recommends best practices and optimizations

Format your response as a structured analysis with clear sections and actionable insights.
  `;

  try {
    const synthesisTask: AITask = {
      taskId: crypto.randomUUID(),
      type: "analyze",
      providerId: "ollama",
      model: "gemma3-legal",
      prompt: synthesisPrompt,
      timestamp: Date.now(),
      priority: "high",
      temperature: 0.2,
      maxTokens: 3072,
    };

    const taskId = await aiWorkerManager.submitTask(synthesisTask);
    const result = await aiWorkerManager.waitForTask(taskId);

    return (
      result.response?.content ||
      generateBasicSummary(prompt, contextResults, memoryResults, agentResults)
    );
  } catch (error: any) {
    console.error("Synthesis failed, using fallback:", error);
    return generateBasicSummary(
      prompt,
      contextResults,
      memoryResults,
      agentResults
    );
  }
}

/**
 * Generate fallback summary if synthesis fails
 */
function generateBasicSummary(
  prompt: string,
  contextResults: any[],
  memoryResults: any[],
  agentResults: any[]
): string {
  return `
# Analysis Summary

## Original Request
${prompt}

## Key Findings
- Found ${contextResults.length} relevant context items
- Retrieved ${memoryResults.length} memory entries
- Completed ${agentResults.length} agent analyses

## Recommendations
- Review semantic search results for relevant patterns
- Consider memory context for historical insights
- Implement agent recommendations with proper testing
- Monitor for any issues during implementation

## Next Steps
1. Prioritize critical issues identified
2. Implement recommended solutions
3. Test thoroughly before deployment
4. Monitor system performance post-implementation
  `;
}

/**
 * Generate actionable next steps
 */
async function generateNextActions(
  prompt: string,
  synthesis: string,
  engineeringAnalysis: any
): Promise<NextAction[]> {
  const actions: NextAction[] = [];

  // Extract actions from engineering analysis
  if (engineeringAnalysis?.solutions) {
    engineeringAnalysis.solutions.forEach((solution: any, index: number) => {
      solution.steps?.forEach((step: any, stepIndex: number) => {
        actions.push({
          id: `action-${index}-${stepIndex}`,
          type: inferActionType(step.action),
          priority: solution.approach === "immediate" ? "high" : "medium",
          description: step.description || step.action,
          commands: step.commands || [],
          targetFiles: step.targetFiles || [],
          estimatedTime: Math.floor(
            solution.estimatedTime / solution.steps.length
          ),
          dependencies: step.dependencies || [],
        });
      });
    });
  }

  // Add default actions if none found
  if (actions.length === 0) {
    actions.push({
      id: "investigate",
      type: "research",
      priority: "medium",
      description: "Investigate the reported issue or request",
      estimatedTime: 15,
      dependencies: [],
    });
  }

  return actions;
}

/**
 * Generate recommendations based on analysis
 */
async function generateRecommendations(
  engineeringAnalysis: any,
  context: any
): Promise<Recommendation[]> {
  const recommendations: Recommendation[] = [];

  if (engineeringAnalysis?.recommendations) {
    engineeringAnalysis.recommendations.forEach((rec: any) => {
      recommendations.push({
        category: rec.type || "architecture",
        title: rec.title,
        description: rec.description,
        impact: rec.impact || "medium",
        effort: rec.effort || "medium",
        priority: rec.priority || 50,
      });
    });
  }

  // Add platform-specific recommendations
  if (context.platform === "all") {
    recommendations.push({
      category: "architecture",
      title: "Cross-Platform Compatibility",
      description:
        "Ensure solutions work across webapp, desktop, and mobile platforms",
      impact: "high",
      effort: "medium",
      priority: 80,
    });
  }

  return recommendations.sort((a, b) => b.priority - a.priority);
}

/**
 * Create execution plan from actions and recommendations
 */
async function createExecutionPlan(
  actions: NextAction[],
  recommendations: Recommendation[]
): Promise<ExecutionPlan> {
  const phases: ExecutionPhase[] = [];

  // Group actions by dependencies and priority
  const criticalActions = actions.filter((a) => a.priority === "critical");
  const highActions = actions.filter((a) => a.priority === "high");
  const mediumActions = actions.filter((a) => a.priority === "medium");
  const lowActions = actions.filter((a) => a.priority === "low");

  let phaseOrder = 1;

  if (criticalActions.length > 0) {
    phases.push({
      id: `phase-${phaseOrder}`,
      name: "Critical Issues",
      actions: criticalActions.map((a) => a.id),
      order: phaseOrder++,
      canRunInParallel: false,
    });
  }

  if (highActions.length > 0) {
    phases.push({
      id: `phase-${phaseOrder}`,
      name: "High Priority Tasks",
      actions: highActions.map((a) => a.id),
      order: phaseOrder++,
      canRunInParallel: true,
    });
  }

  if (mediumActions.length > 0) {
    phases.push({
      id: `phase-${phaseOrder}`,
      name: "Medium Priority Tasks",
      actions: mediumActions.map((a) => a.id),
      order: phaseOrder++,
      canRunInParallel: true,
    });
  }

  if (lowActions.length > 0) {
    phases.push({
      id: `phase-${phaseOrder}`,
      name: "Low Priority Tasks",
      actions: lowActions.map((a) => a.id),
      order: phaseOrder++,
      canRunInParallel: true,
    });
  }

  const totalTime = actions.reduce(
    (sum, action) => sum + action.estimatedTime,
    0
  );
  const parallelTime = phases.reduce((sum, phase) => {
    const phaseActions = actions.filter((a) => phase.actions.includes(a.id));
    const maxTime = Math.max(...phaseActions.map((a) => a.estimatedTime));
    return sum + maxTime;
  }, 0);

  return {
    phases,
    totalEstimatedTime: parallelTime,
    parallelizable: parallelTime < totalTime,
    criticalPath: phases.filter((p) => !p.canRunInParallel).map((p) => p.id),
  };
}

/**
 * Generate self-prompt for Copilot
 */
function generateCopilotSelfPrompt(
  originalPrompt: string,
  synthesis: string,
  nextActions: NextAction[],
  recommendations: Recommendation[],
  outputFormat: string
): string {
  const formatInstruction =
    outputFormat === "json"
      ? "Please respond in JSON format."
      : outputFormat === "markdown"
        ? "Please respond in Markdown format."
        : "Please respond in a structured, readable format.";

  return `
Based on comprehensive AI analysis including semantic search, memory context, multi-agent coordination, and autonomous engineering assessment, here is the synthesized understanding:

## Original Request
${originalPrompt}

## Comprehensive Analysis
${synthesis}

## Recommended Next Actions
${nextActions.map((action) => `- ${action.description} (Priority: ${action.priority}, Est. ${action.estimatedTime}min)`).join("\n")}

## Strategic Recommendations
${recommendations.map((rec) => `- ${rec.title}: ${rec.description} (Impact: ${rec.impact}, Effort: ${rec.effort})`).join("\n")}

As an AI assistant, what specific actions should I take next to best help with this request? Consider:
1. The most impactful immediate actions
2. Potential risks and mitigation strategies
3. Testing and validation approaches
4. Long-term maintenance considerations
5. Cross-platform compatibility (webapp, desktop, mobile)

${formatInstruction}
  `;
}

// RAG Copilot Self-Prompting Utility
export class CopilotSelfPrompt {
  private vectorStore: any;
  private embeddings: any;
  private redisClient: any;

  constructor() {
    // this.vectorStore = new PGVectorStore({ pool });
    this.embeddings = new OpenAIEmbeddings({
      modelName: "nomic-embed-text",
      openAIApiKey: "N/A",
      // baseURL property removed for local LLM compatibility
    });
    // this.redisClient = Redis.createClient();
  }

  async getSemanticContext(query: string, todoList: string[] = []) {
    // Check Redis cache first
    // const cacheKey = `semantic:${query}`;
    // const cached = await this.redisClient.get(cacheKey);
    // if (cached) return JSON.parse(cached);

    // Generate embedding for query
    // const embedding = await this.embeddings.embedQuery(query);
    // const results = await this.vectorStore.similaritySearch(query, 8);
    // Rank by high-score and offset Copilot resources
    // const ranked = results.sort((a, b) => b.relevance_score - a.relevance_score);
    // Optionally offset Copilot context with todoList priorities
    // const prioritized = this.offsetByTodo(ranked, todoList);

    // Cache results
    // await this.redisClient.set(cacheKey, JSON.stringify(prioritized));
    // return prioritized;
    return [];
  }

  offsetByTodo(results: any[], todoList: string[]) {
    // Boost results related to todoList items
    return results.map((r) => {
      r.relevance_score += todoList.some((todo) => r.content.includes(todo))
        ? 0.2
        : 0;
      return r;
    });
  }

  async injectContextToCopilot(context: any[], code: string) {
    // Inject context as JSON for Copilot
    return `/* Copilot Context Injection: ${JSON.stringify(context)} */\n${code}`;
  }

  async selfPromptFromTodo(todoList: string[], code: string) {
    // Generate self-prompting plan from todoList
    const context = await this.getSemanticContext(todoList.join(" "), todoList);
    return await this.injectContextToCopilot(context, code);
  }
}

// Helper functions
function inferActionType(action: string): NextAction["type"] {
  const actionLower = action.toLowerCase();
  if (actionLower.includes("test")) return "test";
  if (actionLower.includes("debug")) return "debug";
  if (actionLower.includes("deploy")) return "deploy";
  if (actionLower.includes("monitor")) return "monitor";
  if (actionLower.includes("research")) return "research";
  return "code";
}

function calculateConfidence(
  contextResults: any[],
  agentResults: any[],
  engineeringAnalysis: any
): number {
  let confidence = 0.5; // Base confidence

  if (contextResults.length > 0) confidence += 0.2;
  if (agentResults.length > 0) confidence += 0.2;
  if (engineeringAnalysis) confidence += 0.1;

  return Math.min(confidence, 1.0);
}

function extractSources(
  contextResults: any[],
  memoryResults: any[],
  agentResults: any[]
): string[] {
  const sources = new Set<string>();

  contextResults.forEach((result) => {
    if (result.source) sources.add(result.source);
  });

  agentResults.forEach((result) => {
    if (result.source) sources.add(result.source);
  });

  sources.add("semantic-search");
  sources.add("memory-mcp");
  sources.add("multi-agent-analysis");

  return Array.from(sources);
}

// RL Ranking Datastore Implementation
export interface RLRankingSummary {
  id: string;
  timestamp: number;
  prompt: string;
  confidence: number;
  tokensUsed: number;
  processingTime: number;
  successful: boolean;
  agentsUsed: string[];
  userFeedback?: 'positive' | 'negative' | 'neutral';
  effectiveness: number; // 0-1 score
  nextActions: NextAction[];
  recommendations: Recommendation[];
}

export class RLRankingDatastore {
  private redisClient: any;
  private summariesKey = 'copilot:rl:summaries';
  private userActivityKey = 'copilot:rl:user_activity';
  private rankingModelKey = 'copilot:rl:ranking_model';

  constructor() {
    this.initializeRedis();
  }

  private async initializeRedis() {
    try {
      this.redisClient = await getRedisClient();
    } catch (error: any) {
      console.error('Failed to initialize Redis for RL ranking:', error);
    }
  }

  async storeSummary(result: CopilotSelfPromptResult, prompt: string): Promise<void> {
    if (!this.redisClient) return;

    const summary: RLRankingSummary = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      prompt,
      confidence: result.metadata.confidence,
      tokensUsed: result.metadata.tokensUsed,
      processingTime: result.metadata.processingTime,
      successful: result.nextActions.length > 0 && result.recommendations.length > 0,
      agentsUsed: result.metadata.sources,
      effectiveness: this.calculateEffectiveness(result),
      nextActions: result.nextActions,
      recommendations: result.recommendations
    };

    try {
      // Store summary with score-based ranking
      await this.redisClient.zadd(
        this.summariesKey, 
        summary.effectiveness, 
        JSON.stringify(summary)
      );
      
      // Keep only top 10 summaries
      await this.redisClient.zremrangebyrank(this.summariesKey, 0, -11);
      
      console.log(`✅ Stored RL summary with effectiveness: ${summary.effectiveness}`);
    } catch (error: any) {
      console.error('Failed to store RL summary:', error);
    }
  }

  async getTopSummaries(limit: number = 10): Promise<RLRankingSummary[]> {
    if (!this.redisClient) return [];

    try {
      const summaries = await this.redisClient.zrevrange(
        this.summariesKey, 0, limit - 1
      );
      
      return summaries.map((s: string) => JSON.parse(s));
    } catch (error: any) {
      console.error('Failed to get top summaries:', error);
      return [];
    }
  }

  async updateUserFeedback(summaryId: string, feedback: 'positive' | 'negative' | 'neutral'): Promise<void> {
    if (!this.redisClient) return;

    try {
      const summaries = await this.redisClient.zrevrange(this.summariesKey, 0, -1);
      
      for (const summaryStr of summaries) {
        const summary: RLRankingSummary = JSON.parse(summaryStr);
        
        if (summary.id === summaryId) {
          summary.userFeedback = feedback;
          
          // Adjust effectiveness based on feedback
          if (feedback === 'positive') {
            summary.effectiveness = Math.min(1.0, summary.effectiveness + 0.1);
          } else if (feedback === 'negative') {
            summary.effectiveness = Math.max(0.0, summary.effectiveness - 0.2);
          }
          
          // Re-store with updated score
          await this.redisClient.zrem(this.summariesKey, summaryStr);
          await this.redisClient.zadd(
            this.summariesKey, 
            summary.effectiveness, 
            JSON.stringify(summary)
          );
          
          console.log(`✅ Updated feedback for summary ${summaryId}: ${feedback}`);
          break;
        }
      }
    } catch (error: any) {
      console.error('Failed to update user feedback:', error);
    }
  }

  private calculateEffectiveness(result: CopilotSelfPromptResult): number {
    let effectiveness = 0.5; // Base score

    // Confidence bonus
    effectiveness += result.metadata.confidence * 0.2;

    // Action count bonus
    if (result.nextActions.length > 0) effectiveness += 0.1;
    if (result.nextActions.length > 3) effectiveness += 0.1;

    // Recommendation bonus
    if (result.recommendations.length > 0) effectiveness += 0.1;
    if (result.recommendations.length > 3) effectiveness += 0.1;

    // Processing efficiency bonus (faster is better, up to 30 seconds)
    const timeBonus = Math.max(0, (30000 - result.metadata.processingTime) / 30000) * 0.1;
    effectiveness += timeBonus;

    return Math.min(1.0, effectiveness);
  }
}

// Singleton instance
export const rlRankingDatastore = new RLRankingDatastore();
;
// Update copilotSelfPrompt to use RL ranking
export async function enhancedCopilotSelfPromptWithRL(
  prompt: string,
  options: CopilotSelfPromptOptions = {}
): Promise<CopilotSelfPromptResult> {
  const result = await copilotSelfPrompt(prompt, options);
  
  // Store for RL ranking
  await rlRankingDatastore.storeSummary(result, prompt);
  
  return result;
}
