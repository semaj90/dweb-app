/**
 * Enhanced RAG Service - Production Implementation
 * Integrates Ollama, MCP orchestration, and self-organizing RAG
 */

import { ollamaService } from './ollama-service.js';
import { copilotSelfPrompt, type CopilotSelfPromptOptions } from '../utils/copilot-self-prompt.js';
import { copilotOrchestrator } from '../utils/mcp-helpers.js';
import { EnhancedRAGSelfOrganizing, type EnhancedRAGConfig } from './enhanced-rag-self-organizing.js';

export interface EnhancedRAGServiceConfig {
  ollamaModel: string;
  embeddingModel: string;
  maxTokens: number;
  temperature: number;
  mcpEnabled: boolean;
  contextSearchEnabled: boolean;
  memoryGraphEnabled: boolean;
  selfOrganizingEnabled: boolean;
  clusteringThreshold: number;
  adaptiveLearningRate: number;
  agentOrchestrationEnabled: boolean;
  agents: string[];
  cacheEnabled: boolean;
  maxConcurrentQueries: number;
  timeout: number;
}

export class EnhancedRAGService {
  private config: EnhancedRAGServiceConfig;
  private ragSystem?: EnhancedRAGSelfOrganizing;
  private initialized = false;
  private queryQueue: Array<() => Promise<any>> = [];
  private activeQueries = 0;

  constructor(config?: Partial<EnhancedRAGServiceConfig>) {
    this.config = {
      ollamaModel: 'gemma3-legal',
      embeddingModel: 'nomic-embed-text',
      maxTokens: 1024,
      temperature: 0.1,
      mcpEnabled: true,
      contextSearchEnabled: true,
      memoryGraphEnabled: true,
      selfOrganizingEnabled: true,
      clusteringThreshold: 0.7,
      adaptiveLearningRate: 0.1,
      agentOrchestrationEnabled: true,
      agents: ['autogen', 'crewai', 'claude', 'rag'],
      cacheEnabled: true,
      maxConcurrentQueries: 3,
      timeout: 30000,
      ...config
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Initialize Ollama service
      const isAvailable = await ollamaService.initialize();
      
      if (!isAvailable) {
        console.warn('⚠️ Ollama not available, using fallback mode');
      } else {
        const hasModel = ollamaService.getGemma3Model();
        if (!hasModel) {
          console.log('📦 Importing Gemma3 model...');
          await ollamaService.importGGUF(
            'C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app\\gemma3Q4_K_M\\mohf16-Q4_K_M.gguf',
            'gemma3-legal'
          );
        }
      }

      // Initialize self-organizing RAG
      if (this.config.selfOrganizingEnabled) {
        const ragConfig: Partial<EnhancedRAGConfig> = {
          embeddingDim: 384,
          chunkSize: 512,
          chunkOverlap: 64,
          maxDocuments: 10000,
          similarityThreshold: this.config.clusteringThreshold,
          selfOrganizingEnabled: true,
          adaptiveFeedbackWeight: this.config.adaptiveLearningRate
        };

        this.ragSystem = new EnhancedRAGSelfOrganizing(undefined, ragConfig);
      }

      // Verify MCP integration
      if (this.config.mcpEnabled) {
        try {
          await copilotOrchestrator('Test MCP integration', {
            useSemanticSearch: this.config.contextSearchEnabled,
            useMemory: this.config.memoryGraphEnabled,
            useMultiAgent: this.config.agentOrchestrationEnabled,
            agents: this.config.agents
          });
        } catch (error) {
          console.warn('⚠️ MCP integration issues:', error);
          this.config.mcpEnabled = false;
        }
      }

      this.initialized = true;
      console.log('✅ Enhanced RAG Service initialized');
    } catch (error) {
      console.error('❌ Enhanced RAG Service initialization failed:', error);
      throw error;
    }
  }

  async query(
    query: string,
    options: {
      useContextRAG?: boolean;
      useSelfPrompting?: boolean;
      useMultiAgent?: boolean;
      documentTypes?: string[];
      maxResults?: number;
    } = {}
  ): Promise<{
    results: any[];
    analysis: string;
    recommendations: string[];
    metadata: {
      processingTime: number;
      sources: string[];
      confidence: number;
      ragScore: number;
    };
  }> {
    if (!this.initialized) await this.initialize();

    if (this.activeQueries >= this.config.maxConcurrentQueries) {
      return new Promise((resolve, reject) => {
        this.queryQueue.push(async () => {
          try {
            const result = await this.query(query, options);
            resolve(result);
          } catch (error) {
            reject(error);
          }
        });
      });
    }

    this.activeQueries++;
    const startTime = Date.now();

    try {
      const {
        useContextRAG = true,
        useSelfPrompting = true,
        useMultiAgent = true
      } = options;

      let results: any[] = [];
      let analysis = '';
      let recommendations: string[] = [];
      const sources = new Set<string>();

      // Phase 1: Enhanced RAG
      if (useContextRAG && this.ragSystem) {
        try {
          const ragResults = await this.ragSystem.query(query, {
            intent: 'research',
            constraints: {
              documentTypes: options.documentTypes,
              maxResults: options.maxResults || 10
            }
          });

          results = ragResults.chunks;
          analysis = ragResults.llmAnalysis.summary;
          recommendations = ragResults.llmAnalysis.recommendations;
          sources.add('enhanced-rag');
        } catch (error) {
          console.warn('⚠️ RAG query failed:', error);
        }
      }

      // Phase 2: MCP orchestration
      if (this.config.mcpEnabled) {
        try {
          const mcpResults = await copilotOrchestrator(query, {
            useSemanticSearch: true,
            useMemory: true,
            useCodebase: true,
            useMultiAgent: useMultiAgent,
            agents: this.config.agents,
            synthesizeOutputs: true
          });

          if (mcpResults.semantic) {
            results = [...results, ...mcpResults.semantic];
            sources.add('mcp-semantic');
          }

          if (mcpResults.agentResults) {
            sources.add('multi-agent');
          }
        } catch (error) {
          console.warn('⚠️ MCP orchestration failed:', error);
        }
      }

      // Phase 3: Self-prompting
      if (useSelfPrompting) {
        try {
          const selfPromptOptions: CopilotSelfPromptOptions = {
            useSemanticSearch: true,
            useMemory: true,
            useMultiAgent: useMultiAgent,
            useAutonomousEngineering: true,
            enableSelfSynthesis: true,
            context: {
              platform: 'webapp',
              urgency: 'medium'
            }
          };

          const selfPromptResult = await copilotSelfPrompt(query, selfPromptOptions);
          
          analysis = selfPromptResult.synthesizedOutput || analysis;
          recommendations = [
            ...recommendations,
            ...selfPromptResult.recommendations.map(r => r.description)
          ];
          sources.add('self-prompting');
        } catch (error) {
          console.warn('⚠️ Self-prompting failed:', error);
        }
      }

      // Phase 4: LLM synthesis
      const finalSynthesis = await this.synthesizeResults(query, results, analysis, recommendations);

      const processingTime = Date.now() - startTime;

      return {
        results: results.slice(0, options.maxResults || 10),
        analysis: finalSynthesis.analysis,
        recommendations: finalSynthesis.recommendations,
        metadata: {
          processingTime,
          sources: Array.from(sources),
          confidence: finalSynthesis.confidence,
          ragScore: this.calculateRAGScore(results)
        }
      };

    } finally {
      this.activeQueries--;
      
      if (this.queryQueue.length > 0) {
        const nextQuery = this.queryQueue.shift();
        if (nextQuery) {
          nextQuery().catch(console.error);
        }
      }
    }
  }

  private async synthesizeResults(
    query: string,
    results: any[],
    analysis: string,
    recommendations: string[]
  ): Promise<{
    analysis: string;
    recommendations: string[];
    confidence: number;
  }> {
    try {
      if (!ollamaService.getIsAvailable()) {
        return {
          analysis: analysis || `Analysis for query: ${query}`,
          recommendations: recommendations.length > 0 ? recommendations : ['Review available documentation'],
          confidence: 0.5
        };
      }

      const synthesisPrompt = `
Legal AI Analysis Synthesis

Query: ${query}
Available Data: ${results.length} relevant documents
Current Analysis: ${analysis}
Recommendations: ${recommendations.join(', ')}

Provide:
1. Comprehensive legal analysis
2. Actionable recommendations
3. Confidence assessment

Response:`;

      const response = await ollamaService.generate(synthesisPrompt, {
        temperature: this.config.temperature,
        maxTokens: this.config.maxTokens,
        system: 'You are a specialized legal AI assistant.'
      });

      return {
        analysis: response || analysis,
        recommendations: this.extractRecommendations(response) || recommendations,
        confidence: this.calculateConfidence(results, response)
      };

    } catch (error) {
      console.warn('Synthesis failed:', error);
      return {
        analysis,
        recommendations,
        confidence: 0.7
      };
    }
  }

  private extractRecommendations(text: string): string[] {
    const lines = text.split('\n');
    const recommendations: string[] = [];
    
    for (const line of lines) {
      if (line.match(/^\d+\.|\-|\•/) && line.toLowerCase().includes('recommend')) {
        recommendations.push(line.replace(/^\d+\.|\-|\•\s*/, '').trim());
      }
    }
    
    return recommendations.slice(0, 5);
  }

  private calculateConfidence(results: any[], synthesis: string): number {
    let confidence = 0.5;
    
    if (results.length > 5) confidence += 0.2;
    if (synthesis.length > 200) confidence += 0.1;
    if (synthesis.includes('statute') || synthesis.includes('precedent')) confidence += 0.2;
    
    return Math.min(0.95, confidence);
  }

  private calculateRAGScore(results: any[]): number {
    if (results.length === 0) return 0;
    
    const avgRelevance = results.reduce((sum, r) => {
      return sum + (r.contextualRelevance || r.feedbackScore || 0.5);
    }, 0) / results.length;
    
    return avgRelevance;
  }

  async addDocument(content: string, metadata: any): Promise<string[]> {
    if (!this.initialized) await this.initialize();
    
    if (!this.ragSystem) {
      console.warn('RAG system not available');
      return [];
    }
    
    try {
      return await this.ragSystem.addDocument(content, metadata);
    } catch (error) {
      console.error('Document addition failed:', error);
      return [];
    }
  }

  async getSystemStatus(): Promise<{
    initialized: boolean;
    ollamaAvailable: boolean;
    activeQueries: number;
    queuedQueries: number;
    ragSystemReady: boolean;
    mcpEnabled: boolean;
  }> {
    return {
      initialized: this.initialized,
      ollamaAvailable: ollamaService.getIsAvailable(),
      activeQueries: this.activeQueries,
      queuedQueries: this.queryQueue.length,
      ragSystemReady: !!this.ragSystem,
      mcpEnabled: this.config.mcpEnabled
    };
  }

  async shutdown(): Promise<void> {
    if (this.ragSystem) {
      await this.ragSystem.shutdown();
    }
    this.initialized = false;
  }
}

export function createEnhancedRAGService(config?: Partial<EnhancedRAGServiceConfig>) {
  return new EnhancedRAGService(config);
}

export const enhancedRAGService = createEnhancedRAGService({
  ollamaModel: 'gemma3-legal',
  embeddingModel: 'nomic-embed-text',
  mcpEnabled: true,
  contextSearchEnabled: true,
  memoryGraphEnabled: true,
  selfOrganizingEnabled: true,
  agentOrchestrationEnabled: true,
  agents: ['autogen', 'crewai', 'claude', 'rag'],
  cacheEnabled: true
});

export default EnhancedRAGService;