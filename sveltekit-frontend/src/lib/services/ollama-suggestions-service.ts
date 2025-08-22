import { dev } from '$app/environment';

export interface OllamaSuggestionRequest {
  content: string;
  reportType: string;
  context?: {
    caseId?: string;
    evidenceIds?: string[];
    previousMessages?: string[];
  };
  maxSuggestions?: number;
  temperature?: number;
}

export interface OllamaSuggestion {
  content: string;
  type: string;
  confidence: number;
  reasoning: string;
  metadata: {
    keywords?: string[];
    category: string;
    urgency?: number;
    sources?: string[];
  };
}

export interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

/**
 * Enhanced Ollama Suggestions Service
 * Provides AI-powered legal document suggestions using local Ollama models
 */
export class OllamaSuggestionsService {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly timeout: number;

  constructor(
    baseUrl: string = 'http://localhost:11434',
    model: string = 'gemma3-legal:latest',
    timeout: number = 30000
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
    this.timeout = timeout;
  }

  /**
   * Generate AI-powered suggestions for legal document content
   */
  async generateSuggestions(request: OllamaSuggestionRequest): Promise<OllamaSuggestion[]> {
    try {
      const prompt = this.buildSuggestionPrompt(request);
      const response = await this.callOllama(prompt, {
        temperature: request.temperature || 0.3,
        top_p: 0.9,
        top_k: 40,
        num_predict: 1000
      });

      return this.parseSuggestionsResponse(response, request.reportType);
    } catch (error) {
      console.error('Ollama suggestion generation failed:', error);
      throw new Error(`Failed to generate AI suggestions: ${error}`);
    }
  }

  /**
   * Generate streaming suggestions for real-time feedback
   */
  async *generateStreamingSuggestions(request: OllamaSuggestionRequest): AsyncGenerator<OllamaSuggestion> {
    try {
      const prompt = this.buildSuggestionPrompt(request);
      
      for await (const chunk of this.streamOllama(prompt, {
        temperature: request.temperature || 0.3,
        top_p: 0.9,
        top_k: 40
      })) {
        const suggestions = this.parseSuggestionsResponse(chunk, request.reportType);
        for (const suggestion of suggestions) {
          yield suggestion;
        }
      }
    } catch (error) {
      console.error('Ollama streaming suggestions failed:', error);
      throw error;
    }
  }

  /**
   * Build a comprehensive prompt for legal document suggestions
   */
  private buildSuggestionPrompt(request: OllamaSuggestionRequest): string {
    const { content, reportType, context, maxSuggestions = 5 } = request;

    let prompt = `You are an expert legal AI assistant specializing in ${reportType} documents. 

Content to analyze:
"""
${content}
"""

Document Type: ${reportType}
`;

    if (context?.caseId) {
      prompt += `Case Context: Working within case ID ${context.caseId}\n`;
    }

    if (context?.evidenceIds && context.evidenceIds.length > 0) {
      prompt += `Evidence References: ${context.evidenceIds.join(', ')}\n`;
    }

    if (context?.previousMessages && context.previousMessages.length > 0) {
      prompt += `Previous Discussion Context: ${context.previousMessages.slice(-2).join(' | ')}\n`;
    }

    prompt += `
Please provide ${maxSuggestions} specific, actionable suggestions to improve this ${reportType}. 

For each suggestion, provide:
1. The specific improvement text
2. The type of suggestion (legal_analysis, evidence_review, procedural_check, etc.)
3. Confidence level (0.0-1.0)
4. Brief reasoning
5. Relevant keywords or categories

Focus on:
- Legal accuracy and completeness
- Procedural compliance
- Evidence handling requirements
- Writing clarity and persuasiveness
- Citation needs and legal precedents
- Risk assessment and strategic considerations

Format your response as a JSON array with this structure:
[
  {
    "content": "Specific suggestion text",
    "type": "suggestion_type",
    "confidence": 0.85,
    "reasoning": "Why this suggestion is important",
    "metadata": {
      "keywords": ["keyword1", "keyword2"],
      "category": "legal_substance",
      "urgency": 3,
      "sources": ["relevant_source"]
    }
  }
]

Provide practical, implementable suggestions that would genuinely improve the legal document.`;

    return prompt;
  }

  /**
   * Call Ollama API with the given prompt
   */
  private async callOllama(prompt: string, options: any = {}): Promise<OllamaResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          prompt,
          stream: false,
          ...options
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error('Ollama request timed out');
      }
      throw error;
    }
  }

  /**
   * Stream responses from Ollama for real-time suggestions
   */
  private async *streamOllama(prompt: string, options: any = {}): AsyncGenerator<OllamaResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          prompt,
          stream: true,
          ...options
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Ollama streaming API error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response stream reader');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim()) {
              try {
                const chunk: OllamaResponse = JSON.parse(line);
                yield chunk;
              } catch (parseError) {
                console.warn('Failed to parse Ollama stream chunk:', line);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error('Ollama streaming request timed out');
      }
      throw error;
    }
  }

  /**
   * Parse suggestions from Ollama response
   */
  private parseSuggestionsResponse(response: OllamaResponse, reportType: string): OllamaSuggestion[] {
    try {
      const responseText = response.response.trim();
      
      // Try to extract JSON from the response
      let jsonMatch = responseText.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        // If no JSON array found, try to parse the entire response as JSON
        jsonMatch = [responseText];
      }

      const suggestionsData = JSON.parse(jsonMatch[0]);
      
      if (!Array.isArray(suggestionsData)) {
        throw new Error('Response is not an array of suggestions');
      }

      return suggestionsData.map((suggestion: any, index: number) => ({
        content: suggestion.content || 'No suggestion content provided',
        type: suggestion.type || 'general_improvement',
        confidence: Math.min(Math.max(suggestion.confidence || 0.7, 0.0), 1.0),
        reasoning: suggestion.reasoning || 'AI-generated suggestion',
        metadata: {
          keywords: suggestion.metadata?.keywords || [],
          category: suggestion.metadata?.category || 'general',
          urgency: suggestion.metadata?.urgency || 2,
          sources: suggestion.metadata?.sources || [],
          aiGenerated: true,
          model: this.model,
          reportType,
          index
        }
      }));
    } catch (error) {
      console.warn('Failed to parse structured suggestions, falling back to text analysis:', error);
      return this.fallbackTextParsing(response.response, reportType);
    }
  }

  /**
   * Fallback parsing when JSON parsing fails
   */
  private fallbackTextParsing(responseText: string, reportType: string): OllamaSuggestion[] {
    const suggestions: OllamaSuggestion[] = [];
    
    // Split by numbered items or bullet points
    const parts = responseText.split(/\d+[\.\)]\s+|\*\s+|\-\s+/).filter(part => part.trim().length > 20);
    
    parts.forEach((part, index) => {
      const trimmedPart = part.trim();
      if (trimmedPart.length > 20) {
        suggestions.push({
          content: trimmedPart,
          type: this.inferSuggestionType(trimmedPart, reportType),
          confidence: 0.75, // Default confidence for text-parsed suggestions
          reasoning: 'Extracted from AI response text',
          metadata: {
            category: 'ai_generated',
            urgency: 2,
            aiGenerated: true,
            model: this.model,
            reportType,
            index: index + 1,
            parseMethod: 'text_fallback'
          }
        });
      }
    });

    return suggestions.slice(0, 5); // Limit to 5 suggestions
  }

  /**
   * Infer suggestion type from content
   */
  private inferSuggestionType(content: string, reportType: string): string {
    const contentLower = content.toLowerCase();
    
    if (contentLower.includes('evidence') || contentLower.includes('proof')) {
      return 'evidence_review';
    } else if (contentLower.includes('statute') || contentLower.includes('law') || contentLower.includes('cite')) {
      return 'legal_analysis';
    } else if (contentLower.includes('procedure') || contentLower.includes('filing') || contentLower.includes('deadline')) {
      return 'procedural_check';
    } else if (contentLower.includes('witness') || contentLower.includes('testimony')) {
      return 'witness_analysis';
    } else if (contentLower.includes('conclusion') || contentLower.includes('summary')) {
      return 'content_structure';
    } else {
      return 'content_enhancement';
    }
  }

  /**
   * Check if Ollama service is available
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      return response.ok;
    } catch (error) {
      console.error('Ollama health check failed:', error);
      return false;
    }
  }

  /**
   * Get available models
   */
  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      
      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error) {
      console.error('Failed to get available models:', error);
      return [];
    }
  }

  /**
   * Get service configuration
   */
  getConfig(): { baseUrl: string; model: string; timeout: number } {
    return {
      baseUrl: this.baseUrl,
      model: this.model,
      timeout: this.timeout
    };
  }
}

// Singleton instance for the application
export const ollamaSuggestionsService = new OllamaSuggestionsService();

/**
 * Convenience function for generating suggestions
 */
export async function generateOllamaSuggestions(
  content: string,
  reportType: string = 'prosecution_memo',
  context?: OllamaSuggestionRequest['context'],
  options: Partial<OllamaSuggestionRequest> = {}
): Promise<OllamaSuggestion[]> {
  const request: OllamaSuggestionRequest = {
    content,
    reportType,
    context,
    maxSuggestions: 5,
    temperature: 0.3,
    ...options
  };

  return await ollamaSuggestionsService.generateSuggestions(request);
}

/**
 * Test function to verify Ollama integration
 */
export async function testOllamaIntegration(): Promise<{
  success: boolean;
  model: string;
  availableModels: string[];
  testSuggestion?: OllamaSuggestion;
  error?: string;
}> {
  try {
    const isHealthy = await ollamaSuggestionsService.healthCheck();
    if (!isHealthy) {
      throw new Error('Ollama service is not responding');
    }

    const availableModels = await ollamaSuggestionsService.getAvailableModels();
    const config = ollamaSuggestionsService.getConfig();

    // Test with a simple request
    const testSuggestions = await ollamaSuggestionsService.generateSuggestions({
      content: 'The defendant was arrested on suspicion of burglary. Evidence includes fingerprints found at the scene.',
      reportType: 'prosecution_memo',
      maxSuggestions: 1
    });

    return {
      success: true,
      model: config.model,
      availableModels,
      testSuggestion: testSuggestions[0]
    };
  } catch (error) {
    return {
      success: false,
      model: 'unknown',
      availableModels: [],
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}