
import type { RequestHandler } from './$types';

/**
 * Agent Orchestrator API Endpoint
 * Coordinates multiple AI agents with Context7 MCP integration and auto-fix capabilities
 */


import { autoGenAgent } from '../../../../../agents/autogen-agent';

import { enhancedRAGService } from '../../../../../rag/enhanced-rag-service';

export interface AgentOrchestrationRequest {
  prompt: string;
  context?: unknown;
  agents?: string[]; // ['claude', 'autogen', 'crewai', 'rag']
  options?: {
    includeContext7?: boolean;
    autoFix?: boolean;
    autoFixArea?: string;
    parallel?: boolean;
    timeout?: number;
    caseId?: string;
    priority?: 'low' | 'medium' | 'high' | 'urgent';
  };
}

export interface AgentOrchestrationResponse {
  success: boolean;
  results: Array<{
    agent: string;
    output: string;
    score: number;
    metadata: any;
    error?: string;
  }>;
  synthesis: {
    bestResult: string;
    consensusScore: number;
    recommendations: string[];
    nextSteps: string[];
  };
  orchestrationMetadata: {
    totalProcessingTime: number;
    agentsUsed: number;
    context7Enhanced: boolean;
    autoFixApplied: boolean;
    timestamp: string;
  };
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  
  try {
    const requestData: AgentOrchestrationRequest = await request.json();
    
    const {
      prompt,
      context = {},
      agents = ['claude', 'autogen', 'crewai', 'rag'],
      options = {}
    } = requestData;

    // Validate request
    if (!prompt || prompt.trim().length === 0) {
      return json({
        success: false,
        error: 'Prompt is required',
        results: [],
        synthesis: {
          bestResult: '',
          consensusScore: 0,
          recommendations: [],
          nextSteps: []
        },
        orchestrationMetadata: {
          totalProcessingTime: Date.now() - startTime,
          agentsUsed: 0,
          context7Enhanced: false,
          autoFixApplied: false,
          timestamp: new Date().toISOString()
        }
      }, { status: 400 });
    }

    const results: any[] = [];
    let context7Enhanced = false;
    let autoFixApplied = false;

    // Apply Context7 analysis and auto-fix if requested
    if (options.includeContext7) {
      const analysis = await context7Service.analyzeComponent('agent-orchestrator', 'legal-ai');
      context.context7Analysis = analysis;
      context7Enhanced = true;
    }

    if (options.autoFix) {
      const autoFixResult = await context7Service.autoFixCodebase({
        area: options.autoFixArea as any,
        dryRun: false
      });
      context.autoFixResults = autoFixResult;
      autoFixApplied = true;
    }

    // Execute agents based on configuration
    const agentPromises: Promise<any>[] = [];

    if (agents.includes('claude')) {
      const claudePromise = claudeAgent.execute({
        prompt,
        context,
        options: {
          includeContext7: options.includeContext7,
          autoFix: options.autoFix,
          area: options.autoFixArea
        }
      }).then((result: any) => ({
        agent: 'claude',
        ...result,
        error: undefined
      })).catch((error: any) => ({
        agent: 'claude',
        output: '',
        score: 0,
        metadata: {},
        error: error.message
      }));
      
      agentPromises.push(claudePromise);
    }

    if (agents.includes('autogen')) {
      const autogenPromise = autoGenAgent.execute({
        prompt,
        context,
        options: {
          analysisType: 'legal_research',
          priority: options.priority || 'medium',
          caseId: options.caseId,
          includeContext7: options.includeContext7,
          autoFix: options.autoFix
        }
      }).then((result: any) => ({
        agent: 'autogen',
        ...result,
        error: undefined
      })).catch((error: any) => ({
        agent: 'autogen',
        output: '',
        score: 0,
        metadata: {},
        error: error.message
      }));
      
      agentPromises.push(autogenPromise);
    }

    if (agents.includes('crewai')) {
      const crewaiPromise = crewAIAgent.execute({
        prompt,
        context,
        options: {
          crewType: 'legal_research',
          includeContext7: options.includeContext7,
          autoFix: options.autoFix
        }
      }).then((result: any) => ({
        agent: 'crewai',
        ...result,
        error: undefined
      })).catch((error: any) => ({
        agent: 'crewai',
        output: '',
        score: 0,
        metadata: {},
        error: error.message
      }));
      
      agentPromises.push(crewaiPromise);
    }

    if (agents.includes('rag')) {
      const ragPromise = enhancedRAGService.query({
        query: prompt,
        context,
        options: {
          caseId: options.caseId,
          includeContext7: options.includeContext7,
          autoFix: options.autoFix,
          maxResults: 5,
          confidenceThreshold: 0.7
        }
      }).then((result: any) => ({
        agent: 'rag',
        ...result,
        error: undefined
      })).catch((error: any) => ({
        agent: 'rag',
        output: '',
        score: 0,
        sources: [],
        metadata: {},
        error: error.message
      }));
      
      agentPromises.push(ragPromise);
    }

    // Execute agents (parallel or sequential based on options)
    if (options.parallel !== false) {
      // Execute in parallel with timeout
      const timeout = options.timeout || 30000;
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Agent execution timeout')), timeout)
      );
      
      try {
        const agentResults = await Promise.race([
          Promise.allSettled(agentPromises),
          timeoutPromise
        ]) as PromiseSettledResult<any>[];

        agentResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            results.push({
              agent: agents[index] || 'unknown',
              output: '',
              score: 0,
              metadata: {},
              error: result.reason?.message || 'Agent execution failed'
            });
          }
        });
      } catch (error: any) {
        // Timeout occurred, collect partial results
        console.error('Agent orchestration timeout:', error);
        results.push({
          agent: 'orchestrator',
          output: '',
          score: 0,
          metadata: {},
          error: 'Execution timeout - partial results may be available'
        });
      }
    } else {
      // Execute sequentially
      for (const agentPromise of agentPromises) {
        try {
          const result = await agentPromise;
          results.push(result);
        } catch (error: any) {
          results.push({
            agent: 'unknown',
            output: '',
            score: 0,
            metadata: {},
            error: error instanceof Error ? error.message : 'Unknown error'
          });
        }
      }
    }

    // Synthesize results
    const synthesis = synthesizeResults(results, prompt);
    
    const totalProcessingTime = Date.now() - startTime;

    const response: AgentOrchestrationResponse = {
      success: true,
      results,
      synthesis,
      orchestrationMetadata: {
        totalProcessingTime,
        agentsUsed: results.length,
        context7Enhanced,
        autoFixApplied,
        timestamp: new Date().toISOString()
      }
    };

    return json(response);

  } catch (error: any) {
    console.error('Agent orchestration failed:', error);
    
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown orchestration error',
      results: [],
      synthesis: {
        bestResult: '',
        consensusScore: 0,
        recommendations: ['Check agent configurations', 'Verify service availability'],
        nextSteps: ['Review error logs', 'Test individual agents']
      },
      orchestrationMetadata: {
        totalProcessingTime: Date.now() - startTime,
        agentsUsed: 0,
        context7Enhanced: false,
        autoFixApplied: false,
        timestamp: new Date().toISOString()
      }
    }, { status: 500 });
  }
};

function synthesizeResults(results: any[], originalPrompt: string) {
  // Find best result by score
  const validResults = results.filter((r: any) => !r.error && r.score > 0);
  
  if (validResults.length === 0) {
    return {
      bestResult: 'No valid results from agents',
      consensusScore: 0,
      recommendations: ['Check agent configurations', 'Review error logs'],
      nextSteps: ['Test individual agent endpoints', 'Verify Context7 integration']
    };
  }

  const bestResult = validResults.reduce((best, current) => 
    current.score > best.score ? current : best
  );

  // Calculate consensus score
  const avgScore = validResults.reduce((sum, result) => sum + result.score, 0) / validResults.length;
  
  // Generate recommendations based on results
  const recommendations = [
    `Best performing agent: ${bestResult.agent} (score: ${bestResult.score.toFixed(2)})`,
    `Average confidence: ${avgScore.toFixed(2)}`,
    `${validResults.length}/${results.length} agents completed successfully`
  ];

  // Generate next steps
  const nextSteps = [
    'Review best result for actionable insights',
    'Consider running additional analysis if needed',
    'Document findings for case records'
  ];

  if (avgScore < 0.6) {
    recommendations.push('Consider refining the prompt for better results');
    nextSteps.push('Iterate with more specific queries');
  }

  return {
    bestResult: bestResult.output,
    consensusScore: avgScore,
    recommendations,
    nextSteps
  };
}

// Health check endpoints
export const GET: RequestHandler = async () => {
  return json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    availableAgents: ['claude', 'autogen', 'crewai', 'rag'],
    context7Enabled: true,
    autoFixEnabled: true
  });
};