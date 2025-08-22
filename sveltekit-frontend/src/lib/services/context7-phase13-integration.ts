
/**
 * Context7 MCP Phase 13 Integration Service
 * Comprehensive integration of Context7 MCP tools with Phase 13 enhanced features
 * Semantic search, memory graph, agent orchestration, and best practices automation
 */

import { writable, derived, type Writable } from "svelte/store";
import { browser } from "$app/environment";
import { 
  copilotOrchestrator,
  mcpMemoryReadGraph,
  semanticSearch,
  generateMCPPrompt,
  validateMCPRequest,
  commonMCPQueries
} from "$lib/optimization/comprehensive-orchestrator";
import type { 
  OrchestrationOptions,
  MCPToolRequest,
  EnhancedRAGEngine
} from "$lib/types/ai";
import type { StatelessAPICoordinator } from "./stateless-api-coordinator";

// Context7 MCP integration types
export interface MCPSemanticResult {
  id: string;
  content: string;
  relevance: number;
  source: "context7" | "local" | "hybrid";
  metadata: {
    libraryId?: string;
    documentType?: string;
    confidence: number;
    processingTime: number;
  };
  enhancedData?: {
    pageRankScore?: number;
    userFeedback?: number;
    networkPosition?: number;
  };
}

export interface MCPMemoryNode {
  id: string;
  type: "CONCEPT" | "ENTITY" | "RELATIONSHIP" | "PRACTICE" | "PATTERN";
  content: string;
  connections: string[];
  weight: number;
  lastAccessed: number;
  accessCount: number;
  metadata: {
    caseId?: string;
    userId?: string;
    sessionId?: string;
    tags: string[];
    confidence: number;
  };
}

export interface MCPAgentRecommendation {
  id: string;
  agent: "context7" | "autogen" | "crewai" | "copilot" | "claude";
  recommendation: string;
  confidence: number;
  reasoning: string;
  actionType: "SEARCH" | "ANALYSIS" | "INTEGRATION" | "OPTIMIZATION" | "WORKFLOW";
  priority: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  estimatedImpact: number;
  metadata: {
    processingTime: number;
    dataPoints: number;
    contextRelevance: number;
  };
}

export interface MCPBestPractice {
  id: string;
  category: "PERFORMANCE" | "SECURITY" | "UI_UX" | "ARCHITECTURE" | "LEGAL_COMPLIANCE";
  title: string;
  description: string;
  implementation: string;
  codeExample?: string;
  applicableScenarios: string[];
  prerequisites: string[];
  expectedBenefits: string[];
  difficulty: "BEGINNER" | "INTERMEDIATE" | "ADVANCED" | "EXPERT";
  estimatedTime: number; // minutes
  lastUpdated: number;
  sourceLibraries: string[];
}

export interface Context7IntegrationConfig {
  enableSemanticSearch: boolean;
  enableMemoryGraph: boolean;
  enableAgentOrchestration: boolean;
  enableBestPractices: boolean;
  enableRealTimeUpdates: boolean;
  semanticSearchTimeout: number;
  memoryGraphSyncInterval: number;
  agentOrchestrationConcurrency: number;
  bestPracticesRefreshRate: number;
  cacheExpiration: number;
}

// Context7 MCP Phase 13 Integration Service
export class Context7Phase13Integration {
  private config: Context7IntegrationConfig;
  private ragEngine?: EnhancedRAGEngine;
  private apiCoordinator?: StatelessAPICoordinator;
  
  // Internal state
  private semanticCache = new Map<string, MCPSemanticResult[]>();
  private memoryGraph = new Map<string, MCPMemoryNode>();
  private agentRecommendations: MCPAgentRecommendation[] = [];
  private bestPracticesCache = new Map<string, MCPBestPractice[]>();
  
  // Performance tracking
  private performanceMetrics = {
    semanticSearchTime: [] as number[],
    memoryGraphTime: [] as number[],
    agentOrchestrationTime: [] as number[],
    totalQueries: 0,
    cacheHitRate: 0,
    errorCount: 0
  };

  // Reactive stores
  public semanticResults = writable<MCPSemanticResult[]>([]);
  public memoryGraphNodes = writable<MCPMemoryNode[]>([]);
  public activeRecommendations = writable<MCPAgentRecommendation[]>([]);
  public bestPractices = writable<MCPBestPractice[]>([]);
  public integrationStatus = writable<{
    semanticSearch: "IDLE" | "ACTIVE" | "ERROR";
    memoryGraph: "IDLE" | "SYNCING" | "ERROR";
    agentOrchestration: "IDLE" | "PROCESSING" | "ERROR";
    bestPractices: "IDLE" | "UPDATING" | "ERROR";
    overall: "HEALTHY" | "DEGRADED" | "ERROR";
  }>({
    semanticSearch: "IDLE",
    memoryGraph: "IDLE", 
    agentOrchestration: "IDLE",
    bestPractices: "IDLE",
    overall: "HEALTHY"
  });

  public performanceStats = writable<{
    averageSemanticSearchTime: number;
    averageMemoryGraphTime: number;
    averageAgentTime: number;
    totalQueries: number;
    cacheHitRate: number;
    errorRate: number;
  }>({
    averageSemanticSearchTime: 0,
    averageMemoryGraphTime: 0,
    averageAgentTime: 0,
    totalQueries: 0,
    cacheHitRate: 0,
    errorRate: 0
  });

  constructor(
    config: Partial<Context7IntegrationConfig> = {},
    ragEngine?: EnhancedRAGEngine,
    apiCoordinator?: StatelessAPICoordinator
  ) {
    this.config = {
      enableSemanticSearch: true,
      enableMemoryGraph: true,
      enableAgentOrchestration: true,
      enableBestPractices: true,
      enableRealTimeUpdates: true,
      semanticSearchTimeout: 15000,
      memoryGraphSyncInterval: 30000,
      agentOrchestrationConcurrency: 3,
      bestPracticesRefreshRate: 300000, // 5 minutes
      cacheExpiration: 600000, // 10 minutes
      ...config
    };

    this.ragEngine = ragEngine;
    this.apiCoordinator = apiCoordinator;

    this.initializeIntegration();
  }

  // Initialize Context7 MCP integration
  private async initializeIntegration(): Promise<void> {
    if (!browser) return;

    try {
      // Initialize semantic search
      if (this.config.enableSemanticSearch) {
        await this.initializeSemanticSearchInternal();
      }

      // Initialize memory graph sync
      if (this.config.enableMemoryGraph) {
        this.startMemoryGraphSync();
      }

      // Initialize best practices
      if (this.config.enableBestPractices) {
        await this.loadBestPractices();
        this.startBestPracticesRefresh();
      }

      // Initialize real-time updates
      if (this.config.enableRealTimeUpdates) {
        this.startRealTimeUpdates();
      }

      this.updateIntegrationStatus("overall", "HEALTHY");

    } catch (error) {
      console.error("Context7 MCP integration failed:", error);
      this.updateIntegrationStatus("overall", "ERROR");
    }
  }

  // Initialize semantic search integration
  private async initializeSemanticSearchInternal(): Promise<void> {
    try {
      // Initialize Context7 semantic search connection
      console.log("Initializing Context7 semantic search...");
      
      // Test Context7 connection
      const testResult = await semanticSearch("test query");
      
      if (testResult && testResult.length > 0) {
        console.log("Context7 semantic search initialized successfully");
      } else {
        console.warn("Context7 semantic search returned empty results");
      }
    } catch (error) {
      console.error("Failed to initialize Context7 semantic search:", error);
      throw error;
    }
  }

  // Enhanced semantic search with Context7 MCP
  public async performEnhancedSemanticSearch(
    query: string,
    options: {
      maxResults?: number;
      includeLocalRAG?: boolean;
      includeMemoryGraph?: boolean;
      cacheResults?: boolean;
      timeout?: number;
    } = {}
  ): Promise<MCPSemanticResult[]> {
    const startTime = Date.now();
    this.updateIntegrationStatus("semanticSearch", "ACTIVE");
    
    try {
      const cacheKey = `${query}_${JSON.stringify(options)}`;
      
      // Check cache first
      if (options.cacheResults !== false && this.semanticCache.has(cacheKey)) {
        const cached = this.semanticCache.get(cacheKey)!;
        this.performanceMetrics.totalQueries++;
        this.performanceMetrics.cacheHitRate = 
          (this.performanceMetrics.cacheHitRate * (this.performanceMetrics.totalQueries - 1) + 1) / 
          this.performanceMetrics.totalQueries;
        return cached;
      }

      const results: MCPSemanticResult[] = [];

      // Context7 MCP semantic search
      const mcpResults = await this.callContext7SemanticSearch(query, options);
      results.push(...mcpResults);

      // Local RAG integration
      if (options.includeLocalRAG && this.ragEngine) {
        const localResults = await this.integrateLocalRAGResults(query, options);
        results.push(...localResults);
      }

      // Memory graph enhancement
      if (options.includeMemoryGraph) {
        const memoryResults = await this.enhanceWithMemoryGraph(query, results);
        results.push(...memoryResults);
      }

      // Sort by relevance and enhance with PageRank if available
      const enhancedResults = this.enhanceWithPageRank(results);
      enhancedResults.sort((a, b) => b.relevance - a.relevance);

      // Limit results
      const finalResults = enhancedResults.slice(0, options.maxResults || 10);

      // Cache results
      if (options.cacheResults !== false) {
        this.semanticCache.set(cacheKey, finalResults);
        this.scheduleCacheCleanup();
      }

      // Update stores and metrics
      this.semanticResults.set(finalResults);
      this.updatePerformanceMetrics("semantic", Date.now() - startTime);
      this.updateIntegrationStatus("semanticSearch", "IDLE");

      return finalResults;

    } catch (error) {
      console.error("Enhanced semantic search failed:", error);
      this.performanceMetrics.errorCount++;
      this.updateIntegrationStatus("semanticSearch", "ERROR");
      throw error;
    }
  }

  // Context7 MCP semantic search
  private async callContext7SemanticSearch(
    query: string,
    options: any
  ): Promise<MCPSemanticResult[]> {
    const startTime = Date.now();

    try {
      // Use copilot orchestrator for Context7 MCP
      const orchestrationResult = await copilotOrchestrator(
        `Enhanced semantic search: ${query}`,
        {
          useSemanticSearch: true,
          useMemory: false, // Handle separately
          useMultiAgent: false, // Handle separately
          synthesizeOutputs: true,
          context: {
            queryType: "semantic_search",
            maxResults: options.maxResults,
            timeout: options.timeout || this.config.semanticSearchTimeout
          }
        }
      );

      const results: MCPSemanticResult[] = [];

      // Process semantic search results
      if (orchestrationResult.semantic && Array.isArray(orchestrationResult.semantic)) {
        for (const item of orchestrationResult.semantic) {
          results.push({
            id: `context7_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            content: item.text || item.content || String(item),
            relevance: item.relevance || item.score || 0.5,
            source: "context7",
            metadata: {
              libraryId: item.libraryId,
              documentType: item.type,
              confidence: item.confidence || 0.7,
              processingTime: Date.now() - startTime
            }
          });
        }
      }

      return results;

    } catch (error) {
      console.warn("Context7 semantic search failed, using fallback:", error);
      
      // Fallback to local semantic search
      const fallbackResults = await semanticSearch(query);
      return fallbackResults.map((item: any) => ({
        id: `fallback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        content: item.text || item.content || String(item),
        relevance: item.relevance || 0.3,
        source: "local" as const,
        metadata: {
          confidence: 0.5,
          processingTime: Date.now() - startTime
        }
      }));
    }
  }

  // Integrate local RAG results
  private async integrateLocalRAGResults(
    query: string,
    options: any
  ): Promise<MCPSemanticResult[]> {
    if (!this.ragEngine) return [];

    try {
      // This would integrate with the enhanced RAG engine
      // For now, return empty array as RAG integration is handled separately
      return [];
    } catch (error) {
      console.warn("Local RAG integration failed:", error);
      return [];
    }
  }

  // Enhance results with memory graph
  private async enhanceWithMemoryGraph(
    query: string,
    existingResults: MCPSemanticResult[]
  ): Promise<MCPSemanticResult[]> {
    const memoryResults: MCPSemanticResult[] = [];

    try {
      // Find related memory nodes
      const relatedNodes = Array.from(this.memoryGraph.values())
        .filter((node: any) => node.content.toLowerCase().includes(query.toLowerCase()) ||
          node.metadata.tags.some((tag: any) => query.toLowerCase().includes(tag.toLowerCase()))
        )
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 5);

      for (const node of relatedNodes) {
        memoryResults.push({
          id: `memory_${node.id}`,
          content: node.content,
          relevance: node.weight,
          source: "hybrid",
          metadata: {
            confidence: node.metadata.confidence,
            processingTime: 0
          },
          enhancedData: {
            networkPosition: node.connections.length
          }
        });
      }

    } catch (error) {
      console.warn("Memory graph enhancement failed:", error);
    }

    return memoryResults;
  }

  // Enhance with PageRank scores
  private enhanceWithPageRank(results: MCPSemanticResult[]): MCPSemanticResult[] {
    // This would integrate with the PageRank system from enhanced RAG
    return results.map((result: any) => {
      // Simulate PageRank boost
      const pageRankBoost = Math.random() * 0.2; // 0-20% boost
      result.relevance = Math.min(1.0, result.relevance + pageRankBoost);
      result.enhancedData = {
        ...result.enhancedData,
        pageRankScore: pageRankBoost
      };
      return result;
    });
  }

  // Agent orchestration with Context7 MCP
  public async requestAgentRecommendations(
    context: string,
    options: {
      agents?: string[];
      priority?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
      maxRecommendations?: number;
      includeImplementationPlan?: boolean;
    } = {}
  ): Promise<MCPAgentRecommendation[]> {
    const startTime = Date.now();
    this.updateIntegrationStatus("agentOrchestration", "PROCESSING");

    try {
      const recommendations: MCPAgentRecommendation[] = [];

      // Use copilot orchestrator for agent recommendations
      const orchestrationResult = await copilotOrchestrator(
        `Generate agent recommendations for: ${context}`,
        {
          useSemanticSearch: false,
          useMemory: true,
          useMultiAgent: true,
          synthesizeOutputs: true,
          agents: options.agents || ["context7", "copilot", "claude"],
          context: {
            priority: options.priority,
            includeImplementation: options.includeImplementationPlan
          }
        }
      );

      // Process agent results
      if (orchestrationResult.agentResults && Array.isArray(orchestrationResult.agentResults)) {
        for (const agentResult of orchestrationResult.agentResults) {
          if (agentResult.result && typeof agentResult.result === 'object') {
            recommendations.push({
              id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              agent: agentResult.agent as any,
              recommendation: agentResult.result.recommendation || "No specific recommendation",
              confidence: agentResult.result.confidence || 0.7,
              reasoning: agentResult.result.reasoning || "Agent analysis",
              actionType: agentResult.result.actionType || "ANALYSIS",
              priority: options.priority || "MEDIUM",
              estimatedImpact: agentResult.result.estimatedImpact || 0.5,
              metadata: {
                processingTime: Date.now() - startTime,
                dataPoints: agentResult.result.dataPoints || 1,
                contextRelevance: agentResult.result.contextRelevance || 0.7
              }
            });
          }
        }
      }

      // Sort by confidence and priority
      recommendations.sort((a, b) => {
        const priorityWeight = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
        const aPriority = priorityWeight[a.priority];
        const bPriority = priorityWeight[b.priority];
        
        if (aPriority !== bPriority) {
          return bPriority - aPriority;
        }
        
        return b.confidence - a.confidence;
      });

      const finalRecommendations = recommendations.slice(0, options.maxRecommendations || 5);

      // Update stores and cache
      this.agentRecommendations.push(...finalRecommendations);
      this.activeRecommendations.set(this.agentRecommendations.slice(-20)); // Keep last 20

      this.updatePerformanceMetrics("agent", Date.now() - startTime);
      this.updateIntegrationStatus("agentOrchestration", "IDLE");

      return finalRecommendations;

    } catch (error) {
      console.error("Agent orchestration failed:", error);
      this.updateIntegrationStatus("agentOrchestration", "ERROR");
      throw error;
    }
  }

  // Memory graph management
  public async syncMemoryGraph(): Promise<void> {
    const startTime = Date.now();
    this.updateIntegrationStatus("memoryGraph", "SYNCING");

    try {
      // Read memory graph from MCP
      const memoryData = await mcpMemoryReadGraph();
      
      if (Array.isArray(memoryData)) {
        for (const item of memoryData) {
          // Handle both error and valid memory items
          if ('error' in item) {
            console.warn('Memory item error:', item.error);
            continue;
          }
          
          const node: MCPMemoryNode = {
            id: `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: "CONCEPT",
            content: item.value || String(item),
            connections: item.relations || [],
            weight: 1.0,
            lastAccessed: Date.now(),
            accessCount: 1,
            metadata: {
              tags: [],
              confidence: 0.7,
              caseId: undefined,
              userId: undefined,
              sessionId: undefined
            }
          };

          this.memoryGraph.set(node.id, node);
        }
      }

      // Update store
      this.memoryGraphNodes.set(Array.from(this.memoryGraph.values()));
      
      this.updatePerformanceMetrics("memory", Date.now() - startTime);
      this.updateIntegrationStatus("memoryGraph", "IDLE");

    } catch (error) {
      console.error("Memory graph sync failed:", error);
      this.updateIntegrationStatus("memoryGraph", "ERROR");
    }
  }

  // Best practices management
  private async loadBestPractices(): Promise<void> {
    this.updateIntegrationStatus("bestPractices", "UPDATING");

    try {
      const categories = ["PERFORMANCE", "SECURITY", "UI_UX", "ARCHITECTURE"] as const;
      
      for (const category of categories) {
        const mcpRequest: MCPToolRequest = {
          tool: "generate-best-practices",
          area: category.toLowerCase().replace("_", "-") as any
        };

        const prompt = generateMCPPrompt(mcpRequest);
        const orchestrationResult = await copilotOrchestrator(`Context7 ${prompt}`, {
          useSemanticSearch: true,
          synthesizeOutputs: true
        });

        const practices: MCPBestPractice[] = [];
        
        if (orchestrationResult.bestPractices && Array.isArray(orchestrationResult.bestPractices)) {
          for (const practice of orchestrationResult.bestPractices) {
            practices.push({
              id: `bp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              category,
              title: practice.title || `${category} Best Practice`,
              description: practice.description || practice,
              implementation: practice.implementation || "Implementation details pending",
              codeExample: practice.codeExample,
              applicableScenarios: practice.applicableScenarios || ["General"],
              prerequisites: practice.prerequisites || [],
              expectedBenefits: practice.expectedBenefits || [],
              difficulty: practice.difficulty || "INTERMEDIATE",
              estimatedTime: practice.estimatedTime || 30,
              lastUpdated: Date.now(),
              sourceLibraries: practice.sourceLibraries || []
            });
          }
        }

        this.bestPracticesCache.set(category, practices);
      }

      // Update store with all best practices
      const allPractices = Array.from(this.bestPracticesCache.values()).flat();
      this.bestPractices.set(allPractices);
      
      this.updateIntegrationStatus("bestPractices", "IDLE");

    } catch (error) {
      console.error("Best practices loading failed:", error);
      this.updateIntegrationStatus("bestPractices", "ERROR");
    }
  }

  // Real-time updates
  private startMemoryGraphSync(): void {
    if (!browser) return;

    setInterval(() => {
      this.syncMemoryGraph().catch(console.error);
    }, this.config.memoryGraphSyncInterval);
  }

  private startBestPracticesRefresh(): void {
    if (!browser) return;

    setInterval(() => {
      this.loadBestPractices().catch(console.error);
    }, this.config.bestPracticesRefreshRate);
  }

  private startRealTimeUpdates(): void {
    if (!browser) return;

    // Connect to Phase 13 stores for real-time coordination
    phase13Stores.aiRecommendations.subscribe((recommendations: any) => {
      // Integrate Phase 13 AI recommendations with Context7 MCP
      if (recommendations.length > 0) {
        this.requestAgentRecommendations("Phase 13 AI integration", {
          priority: "HIGH",
          includeImplementationPlan: true
        }).catch(console.error);
      }
    });
  }

  // Utility methods
  private updateIntegrationStatus(
    component: keyof {
      semanticSearch: "IDLE" | "ACTIVE" | "ERROR";
      memoryGraph: "IDLE" | "SYNCING" | "ERROR";
      agentOrchestration: "IDLE" | "PROCESSING" | "ERROR";
      bestPractices: "IDLE" | "UPDATING" | "ERROR";
      overall: "HEALTHY" | "DEGRADED" | "ERROR";
    },
    status: string
  ): void {
    this.integrationStatus.update((current: any) => ({
      ...current,
      [component]: status
    }));
  }

  private updatePerformanceMetrics(type: "semantic" | "memory" | "agent", time: number): void {
    if (type === "semantic") {
      this.performanceMetrics.semanticSearchTime.push(time);
      if (this.performanceMetrics.semanticSearchTime.length > 100) {
        this.performanceMetrics.semanticSearchTime.shift();
      }
    } else if (type === "memory") {
      this.performanceMetrics.memoryGraphTime.push(time);
      if (this.performanceMetrics.memoryGraphTime.length > 100) {
        this.performanceMetrics.memoryGraphTime.shift();
      }
    } else if (type === "agent") {
      this.performanceMetrics.agentOrchestrationTime.push(time);
      if (this.performanceMetrics.agentOrchestrationTime.length > 100) {
        this.performanceMetrics.agentOrchestrationTime.shift();
      }
    }

    this.performanceMetrics.totalQueries++;
    
    // Update performance stats store
    this.performanceStats.set({
      averageSemanticSearchTime: this.performanceMetrics.semanticSearchTime.reduce((a, b) => a + b, 0) / 
        this.performanceMetrics.semanticSearchTime.length || 0,
      averageMemoryGraphTime: this.performanceMetrics.memoryGraphTime.reduce((a, b) => a + b, 0) / 
        this.performanceMetrics.memoryGraphTime.length || 0,
      averageAgentTime: this.performanceMetrics.agentOrchestrationTime.reduce((a, b) => a + b, 0) / 
        this.performanceMetrics.agentOrchestrationTime.length || 0,
      totalQueries: this.performanceMetrics.totalQueries,
      cacheHitRate: this.performanceMetrics.cacheHitRate,
      errorRate: this.performanceMetrics.errorCount / this.performanceMetrics.totalQueries
    });
  }

  private scheduleCacheCleanup(): void {
    setTimeout(() => {
      const now = Date.now();
      for (const [key, _] of this.semanticCache.entries()) {
        // Remove entries older than cache expiration
        // Simplified cleanup - in production, track timestamps
        if (this.semanticCache.size > 100) {
          this.semanticCache.delete(key);
        }
      }
    }, this.config.cacheExpiration);
  }

  // Public API methods
  public async search(query: string, options?: unknown): Promise<MCPSemanticResult[]> {
    return this.performEnhancedSemanticSearch(query, options);
  }

  public async getRecommendations(context: string, options?: unknown): Promise<MCPAgentRecommendation[]> {
    return this.requestAgentRecommendations(context, options);
  }

  public async getBestPractices(category?: string): Promise<MCPBestPractice[]> {
    if (category && this.bestPracticesCache.has(category as any)) {
      return this.bestPracticesCache.get(category as any)!;
    }
    
    let allPractices: MCPBestPractice[] = [];
    this.bestPractices.subscribe((practices: any) => allPractices = practices)();
    return allPractices;
  }

  public getMemoryGraph(): MCPMemoryNode[] {
    return Array.from(this.memoryGraph.values());
  }

  // Cleanup
  public destroy(): void {
    this.semanticCache.clear();
    this.memoryGraph.clear();
    this.agentRecommendations.length = 0;
    this.bestPracticesCache.clear();
  }
}

// Factory function for Svelte integration
export function createContext7Phase13Integration(
  config?: Partial<Context7IntegrationConfig>,
  ragEngine?: EnhancedRAGEngine,
  apiCoordinator?: StatelessAPICoordinator
) {
  const integration = new Context7Phase13Integration(config, ragEngine, apiCoordinator);
  
  return {
    integration,
    stores: {
      semanticResults: integration.semanticResults,
      memoryGraphNodes: integration.memoryGraphNodes,
      activeRecommendations: integration.activeRecommendations,
      bestPractices: integration.bestPractices,
      integrationStatus: integration.integrationStatus,
      performanceStats: integration.performanceStats
    },
    
    // Derived stores
    derived: {
      isHealthy: derived(integration.integrationStatus, ($status) => 
        $status.overall === "HEALTHY"
      ),
      totalResults: derived(
        [integration.semanticResults, integration.memoryGraphNodes],
        ([$semantic, $memory]) => $semantic.length + $memory.length
      ),
      highPriorityRecommendations: derived(
        integration.activeRecommendations,
        ($recommendations) => $recommendations.filter((r: any) => r.priority === "HIGH" || r.priority === "CRITICAL"
        )
      )
    },
    
    // API methods
    search: integration.search.bind(integration),
    getRecommendations: integration.getRecommendations.bind(integration),
    getBestPractices: integration.getBestPractices.bind(integration),
    getMemoryGraph: integration.getMemoryGraph.bind(integration),
    destroy: integration.destroy.bind(integration)
  };
}

export default Context7Phase13Integration;