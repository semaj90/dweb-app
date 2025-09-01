/// <reference types="vite/client" />
import * as fs from "fs/promises";

/**
 * Copilot Orchestration Wrapper
 * Self-prompts after using MCP memory/codebase tools
 */

// Service imports with fallbacks
let autoGenService: any = null;
let legalTeam: any = null;

// Initialize services with fallbacks
try {
  const { autoGenService: autoGenSvc } = await import(
    "$lib/services/autogen-service"
  ).catch(() => ({ autoGenService: null }));
  autoGenService = autoGenSvc;
} catch {
  // Service not available
}

try {
  const { AutogenLegalTeam } = await import(
    "$lib/ai/autogen-legal-agents"
  ).catch(() => ({ AutogenLegalTeam: null }));
  legalTeam = AutogenLegalTeam ? new AutogenLegalTeam() : null;
} catch {
  // Team not available
}

// --- Type Definitions Export ---
// Export all relevant interfaces for easy import in other files and for Copilot/agent visibility

// --- Agent Orchestration Types ---
export interface AgentResult {
  agent: string;
  result: any;
}

export interface MCPContextAnalysis {
  query: string;
  context: any;
  suggestions: string[];
  confidence: number;
}

export interface AutoMCPSuggestion {
  type: "enhancement" | "correction" | "alternative";
  original: string;
  suggested: string;
  reasoning: string;
  confidence: number;
}

export interface OrchestrationOptions {
  useMemory?: boolean;
  useCodebase?: boolean;
  useChangedFiles?: boolean;
  useReadGraph?: boolean;
  useSemanticSearch?: boolean;
  useMultiAgent?: boolean;
  logErrors?: boolean;
  synthesizeOutputs?: boolean;
  directoryPath?: string;
  context?: unknown;
  agents?: string[]; // e.g. ["autogen", "crewai", "copilot", "claude"]
}

// --- Agent Registry for Extensible Orchestration ---
const agentRegistry: Record<
  string,
  (prompt: string, context?: unknown) => Promise<AgentResult>
> = {
  autogen: async (prompt, context) => {
    try {
      if (autoGenService) {
        return {
          agent: "autogen",
          result: await autoGenService.executeLegalWorkflow(
            "legal_research",
            prompt,
            context || {}
          ),
        };
      } else {
        return {
          agent: "autogen",
          result: `AutoGen agent (mock): Analyzed "${prompt}" - would provide legal research workflow results`,
        };
      }
    } catch (error: any) {
      return {
        agent: "autogen",
        result: `AutoGen agent error: ${error}`,
      };
    }
  },
  crewai: async (prompt) => {
    try {
      if (legalTeam) {
        return {
          agent: "crewai",
          result: await legalTeam.analyzeCase({
            query: prompt,
            analysisType: "legal_research",
            priority: "medium",
          }),
        };
      } else {
        throw new Error(`CrewAI legal team not available for prompt: ${prompt}`);
      }
    } catch (error: any) {
      return {
        agent: "crewai",
        result: `CrewAI agent error: ${error}`,
      };
    }
  },
  // Add Copilot and Claude agent implementations
  copilot: async (prompt, context) => {
    try {
      // Use Ollama for copilot-style responses
      const ollamaUrl =
        typeof window !== "undefined"
          ? "http://localhost:11434"
          : import.meta.env.OLLAMA_BASE_URL || "http://localhost:11434";

      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemma2:2b",
          prompt: `As a coding assistant, analyze and provide suggestions for: ${prompt}`,
          stream: false,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return {
          agent: "copilot",
          result: data.response || `Copilot analysis for: ${prompt}`,
        };
      } else {
        throw new Error("Ollama request failed");
      }
    } catch (error: any) {
      return {
        agent: "copilot",
        result: `Copilot agent (mock): Code analysis for "${prompt}" - would provide coding suggestions and optimizations`,
      };
    }
  },
  claude: async (prompt, context) => {
    try {
      // Use legal-optimized model for Claude-style responses
      const ollamaUrl =
        typeof window !== "undefined"
          ? "http://localhost:11434"
          : import.meta.env.OLLAMA_BASE_URL || "http://localhost:11434";

      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemma2-legal",
          prompt: `As a legal AI assistant, provide detailed analysis for: ${prompt}`,
          stream: false,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return {
          agent: "claude",
          result: data.response || `Claude legal analysis for: ${prompt}`,
        };
      } else {
        throw new Error("Legal model request failed");
      }
    } catch (error: any) {
      return {
        agent: "claude",
        result: `Claude agent (mock): Legal analysis for "${prompt}" - would provide detailed legal insights and case analysis`,
      };
    }
  },
  // RAG agent for enhanced retrieval
  rag: async (prompt, context) => {
    try {
      const ragUrl =
        typeof window !== "undefined"
          ? "http://localhost:5173"
          : "http://localhost:5173";

      const response = await fetch(`${ragUrl}/api/rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "query",
          query: prompt,
          context: context,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return {
          agent: "rag",
          result: data.result || `RAG analysis for: ${prompt}`,
        };
      } else {
        throw new Error("RAG request failed");
      }
    } catch (error: any) {
      return {
        agent: "rag",
        result: `RAG agent (mock): Enhanced retrieval for "${prompt}" - would provide context-aware document analysis`,
      };
    }
  },
};

/**
 * Main Orchestration Wrapper
 * Now supports dynamic agent selection (autogen, crewai, copilot, claude, etc)
 */
export async function copilotOrchestrator(
  prompt: string,
  options: OrchestrationOptions = {}
): Promise<any> {
  let results: any = {};

  // Step 1: Semantic Search
  if (options.useSemanticSearch) {
    results.semantic = await semanticSearch(prompt);
  }

  // Step 2: Memory MCP Server
  if (options.useMemory) {
    results.memory = await mcpMemoryReadGraph();
  }

  // Step 3: Codebase Analysis
  if (options.useCodebase) {
    results.codebase = await mcpCodebaseAnalyze(prompt);
  }

  // Step 4: Changed Files
  if (options.useChangedFiles) {
    results.changedFiles = await getChangedFiles();
  }

  // Step 5: Directory Reading
  if (options.directoryPath) {
    results.directory = await mcpReadDirectory(options.directoryPath);
  }

  // Step 6: Multi-Agent Orchestration (dynamic agent registry)
  if (options.useMultiAgent || (options.agents && options.agents.length > 0)) {
    const agentsToRun =
      options.agents && options.agents.length > 0
        ? options.agents
        : ["autogen", "crewai"];
    results.agentResults = [];
    for (const agent of agentsToRun) {
      if (agentRegistry[agent]) {
        try {
          const agentResult = await agentRegistry[agent](
            prompt,
            options.context
          );
          results.agentResults.push(agentResult);
        } catch (err: any) {
          results.agentResults.push({ agent, error: String(err) });
        }
      } else {
        results.agentResults.push({ agent, error: "Agent not registered" });
      }
    }
  }

  // Step 7: Log Errors and Synthesize Outputs
  if (options.logErrors) {
    results.errorLog = await mcpReadErrorLog();
    results.criticalErrors = await mcpRankErrors(results.errorLog);
  }
  if (options.synthesizeOutputs) {
    results.synthesized = synthesizeLLMOutputs(results);
  }

  // Step 8: Rank and Suggest Best Practices
  results.bestPractices = await mcpSuggestBestPractices(results);

  // Step 9: Compose self-prompt for Copilot/agentic action
  results.selfPrompt = `Given the following results, what is the best next action?\n\n${JSON.stringify(
    results,
    null,
    2
  )}\n\nPrompt: ${prompt}`;

  return results;
}

/**
 * MCP Context7 Helper Functions
 * Utility functions for interacting with Context7 MCP tools
 */

export interface MCPToolRequest {
  tool:
    | "analyze-stack"
    | "generate-best-practices"
    | "suggest-integration"
    | "resolve-library-id"
    | "get-library-docs"
    | "unsloth-best-practices"
    | "rag-query"
    | "rag-upload-document"
    | "rag-get-stats"
    | "rag-analyze-relevance"
    | "rag-integration-guide";
  component?: string;
  context?: "legal-ai" | "gaming-ui" | "performance" | "llm" | "unsloth";
  area?: "performance" | "security" | "ui-ux" | "gpu" | "low-memory";
  feature?: string;
  requirements?: string;
  library?: string;
  topic?: string;
  // RAG-specific properties
  query?: string;
  maxResults?: number;
  confidenceThreshold?: number;
  caseId?: string;
  documentTypes?: string[];
  filePath?: string;
  documentType?: string;
  title?: string;
  documentId?: string;
  integrationType?:
    | "api-integration"
    | "component-integration"
    | "search-ui"
    | "document-upload";
}

export interface MCPResponse {
  success: boolean;
  data?: unknown;
  error?: string;
}

/**
 * Generate a natural language prompt for MCP tools
 */
export function generateMCPPrompt(request: MCPToolRequest): string {
  const {
    tool,
    component,
    context,
    area,
    feature,
    requirements,
    library,
    topic,
    query,
    maxResults,
    confidenceThreshold,
    caseId,
    documentTypes,
    filePath,
    documentType,
    title,
    documentId,
    integrationType,
  } = request;

  switch (tool) {
    case "analyze-stack":
      if (!component)
        throw new Error("Component is required for analyze-stack");
      return `analyze ${component}${context ? ` with context ${context}` : ""}`;

    case "generate-best-practices":
      if (!area)
        throw new Error("Area is required for generate-best-practices");
      return `generate best practices for ${area}`;

    case "suggest-integration":
      if (!feature)
        throw new Error("Feature is required for suggest-integration");
      return `suggest integration for ${feature}${requirements ? ` with requirements ${requirements}` : ""}`;

    case "resolve-library-id":
      if (!library)
        throw new Error("Library is required for resolve-library-id");
      return `resolve library id for ${library}`;

    case "get-library-docs":
      if (!library) throw new Error("Library is required for get-library-docs");
      return `get library docs for ${library}${topic ? ` topic ${topic}` : ""}`;

    case "rag-query":
      if (!query) throw new Error("Query is required for rag-query");
      return `rag query "${query}"${caseId ? ` for case ${caseId}` : ""}${maxResults ? ` max results ${maxResults}` : ""}`;

    case "rag-upload-document":
      if (!filePath)
        throw new Error("File path is required for rag-upload-document");
      return `upload document "${filePath}"${caseId ? ` to case ${caseId}` : ""}${documentType ? ` as ${documentType}` : ""}`;

    case "rag-get-stats":
      return "get rag system statistics";

    case "rag-analyze-relevance":
      if (!query || !documentId)
        throw new Error(
          "Query and document ID are required for rag-analyze-relevance"
        );
      return `analyze relevance of document ${documentId} for query "${query}"`;

    case "rag-integration-guide":
      if (!integrationType)
        throw new Error(
          "Integration type is required for rag-integration-guide"
        );
      return `get rag integration guide for ${integrationType}`;

    default:
      throw new Error(`Unknown tool: ${tool}`);
  }
}

/**
 * Validate MCP tool request
 */
export function validateMCPRequest(request: MCPToolRequest): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!request.tool) {
    errors.push("Tool is required");
  }

  switch (request.tool) {
    case "analyze-stack":
      if (!request.component)
        errors.push("Component is required for analyze-stack");
      if (
        request.context &&
        !["legal-ai", "gaming-ui", "performance"].includes(request.context)
      ) {
        errors.push("Context must be one of: legal-ai, gaming-ui, performance");
      }
      break;

    case "generate-best-practices":
      if (!request.area)
        errors.push("Area is required for generate-best-practices");
      if (
        request.area &&
        !["performance", "security", "ui-ux"].includes(request.area)
      ) {
        errors.push("Area must be one of: performance, security, ui-ux");
      }
      break;

    case "suggest-integration":
      if (!request.feature)
        errors.push("Feature is required for suggest-integration");
      break;

    case "resolve-library-id":
      if (!request.library)
        errors.push("Library is required for resolve-library-id");
      break;

    case "get-library-docs":
      if (!request.library)
        errors.push("Library is required for get-library-docs");
      break;

    case "rag-query":
      if (!request.query) errors.push("Query is required for rag-query");
      if (
        request.maxResults &&
        (request.maxResults < 1 || request.maxResults > 50)
      )
        errors.push("Max results must be between 1 and 50");
      if (
        request.confidenceThreshold &&
        (request.confidenceThreshold < 0 || request.confidenceThreshold > 1)
      )
        errors.push("Confidence threshold must be between 0 and 1");
      break;

    case "rag-upload-document":
      if (!request.filePath)
        errors.push("File path is required for rag-upload-document");
      break;

    case "rag-get-stats":
      // No validation needed
      break;

    case "rag-analyze-relevance":
      if (!request.query)
        errors.push("Query is required for rag-analyze-relevance");
      if (!request.documentId)
        errors.push("Document ID is required for rag-analyze-relevance");
      break;

    case "rag-integration-guide":
      if (!request.integrationType)
        errors.push("Integration type is required for rag-integration-guide");
      if (
        request.integrationType &&
        ![
          "api-integration",
          "component-integration",
          "search-ui",
          "document-upload",
        ].includes(request.integrationType)
      ) {
        errors.push(
          "Integration type must be one of: api-integration, component-integration, search-ui, document-upload"
        );
      }
      break;
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Common MCP queries for the legal AI stack
 */
export const commonMCPQueries = {
  // Stack Analysis
  analyzeSvelteKit: (): MCPToolRequest => ({
    tool: "analyze-stack",
    component: "sveltekit",
    context: "legal-ai",
  }),

  analyzeDrizzle: (): MCPToolRequest => ({
    tool: "analyze-stack",
    component: "drizzle",
    context: "legal-ai",
  }),

  analyzeUnoCSS: (): MCPToolRequest => ({
    tool: "analyze-stack",
    component: "unocss",
    context: "performance",
  }),

  // Best Practices
  performanceBestPractices: (): MCPToolRequest => ({
    tool: "generate-best-practices",
    area: "performance",
  }),

  securityBestPractices: (): MCPToolRequest => ({
    tool: "generate-best-practices",
    area: "security",
  }),

  uiUxBestPractices: (): MCPToolRequest => ({
    tool: "generate-best-practices",
    area: "ui-ux",
  }),

  unslothBestPractices: (): MCPToolRequest => ({
    tool: "unsloth-best-practices",
  }),

  // Integration Suggestions
  aiChatIntegration: (): MCPToolRequest => ({
    tool: "suggest-integration",
    feature: "AI chat component",
    requirements: "legal compliance and audit trails",
  }),

  documentUploadIntegration: (): MCPToolRequest => ({
    tool: "suggest-integration",
    feature: "document upload system",
    requirements: "security and virus scanning",
  }),

  gamingUIIntegration: (): MCPToolRequest => ({
    tool: "suggest-integration",
    feature: "gaming-style UI components",
    requirements: "professional legal interface",
  }),

  // Library Documentation
  svelteKitRouting: (): MCPToolRequest => ({
    tool: "get-library-docs",
    library: "sveltekit",
    topic: "routing",
  }),

  bitsUIDialog: (): MCPToolRequest => ({
    tool: "get-library-docs",
    library: "bits-ui",
    topic: "dialog",
  }),

  drizzleSchema: (): MCPToolRequest => ({
    tool: "get-library-docs",
    library: "drizzle",
    topic: "schema",
  }),

  // RAG System Queries
  ragStats: (): MCPToolRequest => ({
    tool: "rag-get-stats",
  }),

  ragLegalQuery: (query: string, caseId?: string): MCPToolRequest => ({
    tool: "rag-query",
    query,
    caseId,
    maxResults: 10,
    confidenceThreshold: 0.7,
    documentTypes: ["contract", "case_law", "statute", "evidence"],
  }),

  ragContractAnalysis: (query: string): MCPToolRequest => ({
    tool: "rag-query",
    query,
    maxResults: 5,
    confidenceThreshold: 0.8,
    documentTypes: ["contract", "agreement"],
  }),

  ragCaseLawSearch: (query: string): MCPToolRequest => ({
    tool: "rag-query",
    query,
    maxResults: 15,
    confidenceThreshold: 0.75,
    documentTypes: ["case_law", "judgment", "precedent"],
  }),

  ragEvidenceSearch: (query: string, caseId: string): MCPToolRequest => ({
    tool: "rag-query",
    query,
    caseId,
    maxResults: 20,
    confidenceThreshold: 0.6,
    documentTypes: ["evidence", "exhibit", "testimony"],
  }),

  ragApiIntegration: (): MCPToolRequest => ({
    tool: "rag-integration-guide",
    integrationType: "api-integration",
  }),

  ragComponentIntegration: (): MCPToolRequest => ({
    tool: "rag-integration-guide",
    integrationType: "component-integration",
  }),

  ragSearchUI: (): MCPToolRequest => ({
    tool: "rag-integration-guide",
    integrationType: "search-ui",
  }),

  ragDocumentUpload: (): MCPToolRequest => ({
    tool: "rag-integration-guide",
    integrationType: "document-upload",
  }),
};

/**
 * Format MCP response for display
 */
export function formatMCPResponse(response: any): string {
  if (typeof response === "string") {
    return response;
  }

  if (response?.content) {
    if (Array.isArray(response.content)) {
      return response.content
        .map((item: any) => item.text || item.content || String(item))
        .join("\n");
    }
    return String(response.content);
  }

  return JSON.stringify(response, null, 2);
}

/**
 * Quick access to MCP resources
 */
export const mcpResources = {
  stackOverview: "context7://stack-overview",
  integrationGuide: "context7://integration-guide",
  performanceTips: "context7://performance-tips",
} as const;

/**
 * Generate Claude Code prompt for MCP tool usage
 */
export function generateClaudePrompt(request: MCPToolRequest): string {
  const validation = validateMCPRequest(request);
  if (!validation.valid) {
    throw new Error(`Invalid request: ${validation.errors.join(", ")}`);
  }

  const prompt = generateMCPPrompt(request);
  return `Please use the Context7 MCP tools to ${prompt}.`;
}

// Unsloth Best Practices
export function getUnslothBestPractices(): string {
  return `# Unsloth Best Practices\n\n- Use Unsloth for ultra-fast, low-memory fine-tuning\n- Supports LoRA, QLoRA, and quantized models\n- Use with Ollama for efficient serving\n- Monitor training logs for memory spikes\n- Use context7 to fetch Unsloth docs and integration patterns\n- Integrate with SvelteKit backend for custom training workflows\n`;
}

// Stub implementations for missing MCP and agent functions
// Production: Integrate with Context7 MCP semantic search
export async function semanticSearch(query: string): Promise<any> {
  try {
    // Use the real semantic search endpoint
    const response = await fetch("http://localhost:3000/api/semantic-search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error: any) {
    console.error("semanticSearch error:", error);
    return [{ error: String(error) }];
  }
}

// Production: Integrate with MCP memory server
export async function mcpMemoryReadGraph(): Promise<any> {
  try {
    // TODO: Replace with real MCP memory server call when available
    // For now, return mock structure that matches expected format
    return [
      {
        node: "legal-workflow-memory",
        relations: ["case-evidence", "document-analysis"],
        value: "Context7 memory graph integration ready",
      },
    ];
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}

// Enhanced Context7 MCP codebase analysis
export async function mcpCodebaseAnalyze(prompt: string): Promise<any> {
  // Node.js/ESM alias import is not supported in CJS build. Provide fallback for CJS/Node.js usage.
  try {
    // Fallback: Return a stub/mock result for Node.js/require usage
    return [
      {
        analysis: `Codebase analysis for: ${prompt}`,
        context7LibraryId: "context7-sveltekit",
        documentation:
          "SvelteKit routing documentation (stub for CJS build)...",
        recommendations: [
          "Use SvelteKit file-based routing for legal document workflows",
          "Implement API routes for AI agent integration",
          "Consider server-side rendering for legal compliance",
        ],
      },
    ];
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}

// Production: Integrate with MCP get_changed_files
export async function getChangedFiles(): Promise<any> {
  try {
    // TODO: Implement MCP SDK integration when available
    return ["file1.ts", "file2.svelte"];
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}

// Production: Integrate with MCP directory reading
export async function mcpReadDirectory(path: string): Promise<any> {
  try {
    // TODO: Implement MCP SDK integration when available
    return [`Read directory: ${path}`];
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}

// Production: Autogen agent orchestration (stub, replace with real API integration if available)
const autogenServiceFallback = {
  async runAgents(prompt: string, context?: unknown) {
    // TODO: Replace with real Autogen API call
    return { agent: "autogen", result: `AutoGen agent result for: ${prompt}` };
  },
};

// Production: CrewAI agent orchestration (stub, replace with real API integration if available)
const crewAIService = {
  async analyzeLegalCaseWithCrew(prompt: string) {
    // TODO: Replace with real CrewAI API call
    return { agent: "crewai", result: `CrewAI agent result for: ${prompt}` };
  },
};

// Production: Read error log from MCP and append to MCP_TODO_LOG.md
const MCP_TODO_LOG_PATH = "../../../../MCP_TODO_LOG.md";

export async function mcpReadErrorLog(): Promise<any> {
  try {
    // Read the error log file
    const log = await fs.readFile(MCP_TODO_LOG_PATH, "utf-8");
    return log.split(/\r?\n/).filter(Boolean);
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}

// Append error or lost context to MCP_TODO_LOG.md
export async function mcpLogErrorOrContextLoss(message: string): Promise<any> {
  const entry = `- [${new Date().toISOString()}] ${message}\n`;
  try {
    await fs.appendFile(MCP_TODO_LOG_PATH, entry, "utf-8");
  } catch (error: any) {
    console.error("Failed to log MCP error/context loss:", error);
  }
}

// Production: Rank errors using MCP
export async function mcpRankErrors(errorLog: any): Promise<any> {
  try {
    // TODO: Implement MCP SDK integration when available
    return ["Critical error: Example ranked error"];
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}

// Production: Synthesize LLM outputs
export function synthesizeLLMOutputs(results: any) {
  // Combine and format results for display or further processing
  return JSON.stringify(results, null, 2);
}

// Production: Suggest best practices using Microsoft Docs via MCP
export async function mcpSuggestBestPractices(results: any): Promise<any> {
  try {
    // TODO: Implement MCP SDK integration when available
    return [
      "Best practice: Always use Drizzle ORM",
      "Best practice: Use SSR for sensitive data",
    ];
  } catch (error: any) {
    return [{ error: String(error) }];
  }
}
