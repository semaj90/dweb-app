/**
 * Claude Agent Backend Implementation
 * Interfaces with Anthropic Claude API for legal AI analysis
 */

import { context7Service } from '../sveltekit-frontend/src/lib/services/context7Service.js';

export interface ClaudeAgentConfig {
  apiKey: string;
  model: string;
  maxTokens: number;
  temperature: number;
}

export interface ClaudeAgentRequest {
  prompt: string;
  context?: any;
  options?: {
    includeContext7?: boolean;
    autoFix?: boolean;
    area?: string;
  };
}

export interface ClaudeAgentResponse {
  output: string;
  score: number;
  metadata: {
    model: string;
    tokensUsed: number;
    responseTime: number;
    context7Enhanced: boolean;
    autoFixApplied?: boolean;
  };
}

export class ClaudeAgent {
  private config: ClaudeAgentConfig;
  private apiEndpoint = 'https://api.anthropic.com/v1/messages';

  constructor(config: ClaudeAgentConfig) {
    this.config = config;
  }

  async execute(request: ClaudeAgentRequest): Promise<ClaudeAgentResponse> {
    const startTime = Date.now();
    
    try {
      let enhancedPrompt = request.prompt;
      let context7Enhanced = false;
      let autoFixApplied = false;

      // Enhance with Context7 analysis if requested
      if (request.options?.includeContext7) {
        const analysis = await context7Service.analyzeComponent('sveltekit', 'legal-ai');
        enhancedPrompt = `${request.prompt}

Context7 Analysis:
${analysis.recommendations.join('\n')}
Best Practices: ${analysis.bestPractices.join('\n')}
Integration: ${analysis.integration}`;
        context7Enhanced = true;
      }

      // Apply auto-fix if requested
      if (request.options?.autoFix) {
        const autoFixResult = await context7Service.autoFixCodebase({
          area: request.options.area as any,
          dryRun: false
        });
        
        enhancedPrompt = `${enhancedPrompt}

Auto-Fix Results:
Files Fixed: ${autoFixResult.summary.filesFixed}
Issues Resolved: ${autoFixResult.summary.totalIssues}
Areas: ${Object.keys(autoFixResult.fixes).filter(key => 
          autoFixResult.fixes[key as keyof typeof autoFixResult.fixes].length > 0
        ).join(', ')}`;
        autoFixApplied = true;
      }

      // Make API call to Claude
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.config.apiKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: this.config.model,
          max_tokens: this.config.maxTokens,
          temperature: this.config.temperature,
          messages: [{
            role: 'user',
            content: enhancedPrompt
          }]
        })
      });

      if (!response.ok) {
        throw new Error(`Claude API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const responseTime = Date.now() - startTime;

      return {
        output: data.content[0].text || '',
        score: this.calculateScore(data.content[0].text, responseTime),
        metadata: {
          model: this.config.model,
          tokensUsed: data.usage?.output_tokens || 0,
          responseTime,
          context7Enhanced,
          autoFixApplied
        }
      };

    } catch (error) {
      console.error('Claude agent execution failed:', error);
      
      return {
        output: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        score: 0,
        metadata: {
          model: this.config.model,
          tokensUsed: 0,
          responseTime: Date.now() - startTime,
          context7Enhanced: false
        }
      };
    }
  }

  private calculateScore(output: string, responseTime: number): number {
    // Calculate score based on output quality and response time
    let score = 0.5; // Base score

    // Quality indicators
    if (output.length > 100) score += 0.2;
    if (output.includes('legal') || output.includes('case') || output.includes('evidence')) score += 0.1;
    if (output.includes('recommendation') || output.includes('analysis')) score += 0.1;

    // Response time bonus (faster is better, up to 5 seconds)
    const timeBonus = Math.max(0, (5000 - responseTime) / 5000) * 0.1;
    score += timeBonus;

    return Math.min(1.0, score);
  }
}

// Factory function for creating Claude agent instances
export function createClaudeAgent(config?: Partial<ClaudeAgentConfig>): ClaudeAgent {
  const defaultConfig: ClaudeAgentConfig = {
    apiKey: process.env.ANTHROPIC_API_KEY || 'your-api-key-here',
    model: 'claude-3-sonnet-20240229',
    maxTokens: 4096,
    temperature: 0.1
  };

  return new ClaudeAgent({ ...defaultConfig, ...config });
}

// Export singleton instance
export const claudeAgent = createClaudeAgent();