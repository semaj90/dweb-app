// Enhanced Ollama Gemma3-Legal Service for Native Windows Integration
// Optimized for gemma3-legal:latest model with Windows-native setup

export interface OllamaResponse {
  model: string;
  response: string;
  done: boolean;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaStreamResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
}

export interface OllamaModelInfo {
  name: string;
  size: number;
  digest: string;
  modified_at: string;
  details: {
    format: string;
    family: string;
    families?: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

export class Gemma3LegalService {
  private baseUrl: string;
  private model: string;
  private timeout: number;
  private retryAttempts: number;

  constructor(options: {
    baseUrl?: string;
    model?: string;
    timeout?: number;
    retryAttempts?: number;
  } = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:11434';
    this.model = options.model || 'gemma3-legal:latest';
    this.timeout = options.timeout || 120000; // 2 minutes for complex legal queries
    this.retryAttempts = options.retryAttempts || 3;
  }

  /**
   * Check if Ollama service is running and gemma3-legal model is available
   */
  async healthCheck(): Promise<{ status: 'healthy' | 'unhealthy' | 'model-missing'; message: string; models?: string[] }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });

      if (!response.ok) {
        return {
          status: 'unhealthy',
          message: `Ollama service not responding (HTTP ${response.status})`
        };
      }

      const data = await response.json();
      const models = data.models?.map((m: any) => m.name) || [];
      
      const hasGemma3Legal = models.some((name: string) => 
        name.includes('gemma3-legal') || name.includes('gemma3') && name.includes('legal')
      );

      if (!hasGemma3Legal) {
        return {
          status: 'model-missing',
          message: 'gemma3-legal model not found. Please run: ollama pull gemma3-legal:latest',
          models
        };
      }

      return {
        status: 'healthy',
        message: 'Ollama service is healthy and gemma3-legal model is available',
        models
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        message: `Connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  /**
   * Generate legal AI response with enhanced prompting for legal accuracy
   */
  async generateLegalResponse(
    prompt: string,
    options: {
      temperature?: number;
      max_tokens?: number;
      top_p?: number;
      stream?: boolean;
      context?: string;
      legalContext?: 'contract' | 'litigation' | 'compliance' | 'research' | 'general';
    } = {}
  ): Promise<OllamaResponse> {
    const legalSystemPrompt = this.buildLegalSystemPrompt(options.legalContext || 'general');
    const enhancedPrompt = `${legalSystemPrompt}\n\nUser Query: ${prompt}`;

    const payload = {
      model: this.model,
      prompt: enhancedPrompt,
      stream: options.stream || false,
      options: {
        temperature: options.temperature || 0.3, // Lower for legal accuracy
        max_tokens: options.max_tokens || 4096,
        top_p: options.top_p || 0.9,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        // Windows-native optimization
        num_predict: options.max_tokens || 4096,
        num_ctx: 8192, // Extended context for legal documents
        num_gpu: -1, // Use all available GPU layers (Windows NVIDIA support)
        num_thread: Math.min(16, navigator.hardwareConcurrency || 8), // Optimize for Windows threading
      }
    };

    return this.makeRequest('/api/generate', payload);
  }

  /**
   * Stream legal AI response for real-time interaction
   */
  async *streamLegalResponse(
    prompt: string,
    options: {
      temperature?: number;
      max_tokens?: number;
      legalContext?: 'contract' | 'litigation' | 'compliance' | 'research' | 'general';
    } = {}
  ): AsyncGenerator<OllamaStreamResponse> {
    const legalSystemPrompt = this.buildLegalSystemPrompt(options.legalContext || 'general');
    const enhancedPrompt = `${legalSystemPrompt}\n\nUser Query: ${prompt}`;

    const payload = {
      model: this.model,
      prompt: enhancedPrompt,
      stream: true,
      options: {
        temperature: options.temperature || 0.3,
        max_tokens: options.max_tokens || 4096,
        num_predict: options.max_tokens || 4096,
        num_ctx: 8192,
        num_gpu: -1,
        num_thread: Math.min(16, navigator.hardwareConcurrency || 8),
      }
    };

    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Response body is not readable');

    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            yield parsed as OllamaStreamResponse;
          } catch (e) {
            console.warn('Failed to parse streaming response line:', line);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Get available models from Ollama
   */
  async getModels(): Promise<OllamaModelInfo[]> {
    const response = await this.makeRequest('/api/tags');
    return response.models || [];
  }

  /**
   * Pull gemma3-legal model if not available
   */
  async pullGemma3Legal(onProgress?: (progress: { status: string; total?: number; completed?: number }) => void): Promise<void> {
    const payload = { name: this.model };
    
    const response = await fetch(`${this.baseUrl}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Failed to pull model: HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const progress = JSON.parse(line);
            onProgress?.(progress);
            if (progress.status === 'success') return;
          } catch (e) {
            console.warn('Failed to parse pull progress:', line);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private buildLegalSystemPrompt(context: string): string {
    const basePrompt = `You are a professional legal AI assistant trained on legal documents, case law, and legal principles. You provide accurate, helpful, and ethical legal information while maintaining appropriate disclaimers.

IMPORTANT: Your responses should be informative but always include appropriate disclaimers that this is not legal advice and users should consult with qualified attorneys for specific legal matters.`;

    const contextPrompts = {
      contract: `Focus on contract law, terms analysis, clause interpretation, and agreement structures. Consider enforceability, terms clarity, and standard practices.`,
      litigation: `Focus on litigation strategy, case law analysis, procedural requirements, and dispute resolution. Consider evidence, precedents, and court procedures.`,
      compliance: `Focus on regulatory compliance, policy analysis, risk assessment, and legal requirements. Consider industry standards and regulatory frameworks.`,
      research: `Focus on legal research methodology, case law analysis, statute interpretation, and comprehensive legal analysis. Provide detailed citations and reasoning.`,
      general: `Provide general legal information across various areas of law while maintaining accuracy and appropriate scope.`
    };

    return `${basePrompt}\n\nContext Focus: ${contextPrompts[context as keyof typeof contextPrompts] || contextPrompts.general}`;
  }

  private async makeRequest(endpoint: string, payload?: any): Promise<any> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
          method: payload ? 'POST' : 'GET',
          headers: payload ? { 'Content-Type': 'application/json' } : {},
          body: payload ? JSON.stringify(payload) : undefined,
          signal: AbortSignal.timeout(this.timeout)
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
      } catch (error) {
        lastError = error as Error;
        console.warn(`Ollama request attempt ${attempt} failed:`, error);
        
        if (attempt < this.retryAttempts) {
          // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        }
      }
    }

    throw lastError || new Error('All retry attempts failed');
  }
}

// Singleton instance for use across the application
export const gemma3LegalService = new Gemma3LegalService({
  model: 'gemma3-legal:latest',
  timeout: 120000,
  retryAttempts: 3
});

// Utility function for quick health check
export async function checkOllamaHealth(): Promise<boolean> {
  const health = await gemma3LegalService.healthCheck();
  return health.status === 'healthy';
}

// Utility function to ensure model is available
export async function ensureGemma3LegalModel(): Promise<void> {
  const health = await gemma3LegalService.healthCheck();
  if (health.status === 'model-missing') {
    throw new Error('gemma3-legal model not found. Please run: ollama pull gemma3-legal:latest');
  }
  if (health.status === 'unhealthy') {
    throw new Error(`Ollama service is not running: ${health.message}`);
  }
}