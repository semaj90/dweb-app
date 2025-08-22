/**
 * AutoGen Agent Backend Implementation
 * Coordinates multiple AI agents for complex legal analysis
 */

import { AutogenLegalTeam } from '../sveltekit-frontend/src/lib/ai/autogen-legal-agents.js';
import { context7Service } from '../sveltekit-frontend/src/lib/services/context7Service.js';

export interface AutoGenAgentConfig {
  ollamaEndpoint: string;
  useGPU: boolean;
  maxAgents: number;
  conversationTimeout: number;
}

export interface AutoGenAgentRequest {
  prompt: string;
  context?: unknown;
  options?: {
    analysisType?: 'case_review' | 'evidence_analysis' | 'legal_research' | 'prosecution_strategy';
    priority?: 'low' | 'medium' | 'high' | 'urgent';
    caseId?: string;
    evidenceIds?: string[];
    includeContext7?: boolean;
    autoFix?: boolean;
  };
}

export interface AutoGenAgentResponse {
  output: string;
  score: number;
  metadata: {
    agentsUsed: number;
    conversationRounds: number;
    analysisType: string;
    processingTime: number;
    confidence: number;
    context7Enhanced: boolean;
    autoFixApplied?: boolean;
  };
}

export class AutoGenAgent {
  private config: AutoGenAgentConfig;
  private legalTeam: AutogenLegalTeam;

  constructor(config: AutoGenAgentConfig) {
    this.config = config;
    this.legalTeam = new AutogenLegalTeam({
      ollamaEndpoint: config.ollamaEndpoint,
      useGPU: config.useGPU
    });
  }

  async execute(request: AutoGenAgentRequest): Promise<AutoGenAgentResponse> {
    const startTime = Date.now();
    
    try {
      let enhancedContext = request.context || {};
      let context7Enhanced = false;
      let autoFixApplied = false;

      // Enhance with Context7 analysis if requested
      if (request.options?.includeContext7) {
        const analysis = await context7Service.analyzeComponent('autogen', 'legal-ai');
        enhancedContext.context7Analysis = {
          recommendations: analysis.recommendations,
          bestPractices: analysis.bestPractices,
          integration: analysis.integration
        };
        context7Enhanced = true;
      }

      // Apply auto-fix if requested
      if (request.options?.autoFix) {
        const autoFixResult = await context7Service.autoFixCodebase({
          dryRun: false
        });
        
        enhancedContext.autoFixResults = {
          filesFixed: autoFixResult.summary.filesFixed,
          totalIssues: autoFixResult.summary.totalIssues,
          improvements: autoFixResult.configImprovements
        };
        autoFixApplied = true;
      }

      // Prepare legal analysis request
      const analysisRequest = {
        query: request.prompt,
        analysisType: request.options?.analysisType || 'legal_research',
        priority: request.options?.priority || 'medium',
        caseId: request.options?.caseId,
        evidenceIds: request.options?.evidenceIds,
        ...enhancedContext
      };

      // Execute multi-agent analysis
      const result = await this.legalTeam.analyzeCase(analysisRequest);
      
      const processingTime = Date.now() - startTime;

      return {
        output: result.finalAnalysis,
        score: this.calculateScore(result),
        metadata: {
          agentsUsed: this.extractAgentCount(result.agentConversations),
          conversationRounds: result.agentConversations.length,
          analysisType: request.options?.analysisType || 'legal_research',
          processingTime,
          confidence: result.confidence,
          context7Enhanced,
          autoFixApplied
        }
      };

    } catch (error) {
      console.error('AutoGen agent execution failed:', error);
      
      return {
        output: `AutoGen Analysis Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        score: 0,
        metadata: {
          agentsUsed: 0,
          conversationRounds: 0,
          analysisType: request.options?.analysisType || 'legal_research',
          processingTime: Date.now() - startTime,
          confidence: 0,
          context7Enhanced: false
        }
      };
    }
  }

  private calculateScore(result: unknown): number {
    let score = 0.5; // Base score

    // Confidence-based scoring
    score += result.confidence * 0.3;

    // Quality indicators
    if (result.recommendations && result.recommendations.length > 0) score += 0.1;
    if (result.sources && result.sources.length > 0) score += 0.1;
    if (result.finalAnalysis.length > 200) score += 0.1;

    return Math.min(1.0, score);
  }

  private extractAgentCount(conversations: unknown[]): number { // TODO-AUTO: Replace unknown[] with proper Conversation interface - create type { agent: string, message: string, timestamp: Date }
    const agents = new Set();
    conversations.forEach(conv => {
      if (conv.agent) agents.add(conv.agent);
    });
    return agents.size;
  }
}

// Factory function for creating AutoGen agent instances
export function createAutoGenAgent(config?: Partial<AutoGenAgentConfig>): AutoGenAgent {
  const defaultConfig: AutoGenAgentConfig = {
    ollamaEndpoint: process.env.OLLAMA_ENDPOINT || 'http://localhost:11434',
    useGPU: true,
    maxAgents: 5,
    conversationTimeout: 120000
  };

  return new AutoGenAgent({ ...defaultConfig, ...config });
}

// Export singleton instance
export const autoGenAgent = createAutoGenAgent();