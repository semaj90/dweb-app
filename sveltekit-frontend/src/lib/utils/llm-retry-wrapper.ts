/**
 * LLM Retry Wrapper with TODO Auto-generation
 * Handles Ollama GPU throttling, token limits, and failure logging
 */

import { getLocalOllamaUrl, LOCAL_LLM_CONFIG } from "$lib/constants/local-llm-config";

export interface LLMCallOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  timeout?: number;
  retries?: number;
  useGPU?: boolean;
}

export interface LLMResponse {
  response: string;
  tokensUsed?: number;
  model: string;
  duration?: number;
  success: boolean;
}

// Placeholder implementations for missing dependencies
const todoAutogen = {
  logPerformanceIssue: async (type: string, details: any) => {
    console.warn(`Performance issue: ${type}`, details);
  },
  logLLMMisfire: async (details: any) => {
    console.warn('LLM misfire:', details);
  }
};

const retryLLMCall = async (fn: () => Promise<any>, model: string, prompt: string, retries: number) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
};

/**
 * Enhanced LLM wrapper with retry logic and TODO generation
 */
export class OllamaRetryWrapper {
  private readonly baseUrl: string;
  private failureCount = 0;
  private lastSuccessTime = Date.now();

  constructor() {
    this.baseUrl = getLocalOllamaUrl();
  }

  /**
   * Make LLM call with automatic retry and error logging
   */
  async callLLM(
    prompt: string,
    options: LLMCallOptions = {}
  ): Promise<LLMResponse> {
    const {
      model = LOCAL_LLM_CONFIG.OLLAMA_MODELS.LEGAL_DETAILED,
      temperature = 0.2,
      maxTokens = 1024,
      timeout = 30000,
      retries = 3,
      useGPU = true
    } = options;

    // Validate local-only operation
    if (!this.baseUrl.includes('localhost') && !this.baseUrl.includes('127.0.0.1')) {
      throw new Error('❌ Only local Ollama LLMs allowed. Remote access blocked.');
    }

    const startTime = Date.now();

    return await retryLLMCall(
      async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
          const response = await fetch(`${this.baseUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
              model,
              prompt,
              stream: false,
              options: {
                temperature,
                num_predict: maxTokens,
                num_ctx: LOCAL_LLM_CONFIG.MAX_CONTEXT_LENGTH,
                num_gpu: useGPU ? -1 : 0, // -1 = use all GPU layers
                num_thread: useGPU ? 1 : 4 // Fewer threads when using GPU
              }
            })
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Ollama API error (${response.status}): ${errorText}`);
          }

          const data = await response.json();
          const duration = Date.now() - startTime;

          // Update success metrics
          this.failureCount = 0;
          this.lastSuccessTime = Date.now();

          // Log performance issues
          if (duration > 30000) { // 30 second threshold
            await todoAutogen.logPerformanceIssue('timeout', {
              model,
              duration,
              promptLength: prompt.length,
              tokensUsed: data.eval_count || 0
            });
          }

          return {
            response: data.response || '',
            tokensUsed: data.eval_count || 0,
            model,
            duration,
            success: true
          };

        } catch (error: any) {
          clearTimeout(timeoutId);

          // Track failures
          this.failureCount++;

          // Log specific error types
          if (error.name === 'AbortError') {
            await todoAutogen.logPerformanceIssue('timeout', {
              model,
              timeout,
              promptLength: prompt.length
            });
            throw new Error(`LLM call timed out after ${timeout}ms`);
          }

          if (error.message.includes('CUDA') || error.message.includes('GPU')) {
            await todoAutogen.logPerformanceIssue('gpu', {
              model,
              error: error.message,
              gpuMemoryFraction: LOCAL_LLM_CONFIG.GPU_MEMORY_FRACTION
            });
          }

          throw error;
        }
      },
      model,
      prompt,
      retries
    );
  }

  /**
   * Health check for Ollama service
   */
  async healthCheck(): Promise<{ status: 'healthy' | 'degraded' | 'critical', details: any }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      const data = await response.json();
      const models = data.models || [];

      // Check if required models are available
      const requiredModels = Object.values(LOCAL_LLM_CONFIG.OLLAMA_MODELS);
      const availableModels = models.map((m: any) => m.name);
      const missingModels = requiredModels.filter((model: string) =>
        !availableModels.some((available: string) => available.includes(model))
      );

      const status = missingModels.length === 0 ? 'healthy' : 'degraded';

      const details = {
        totalModels: models.length,
        availableModels: availableModels.slice(0, 5), // Limit for logging
        missingModels,
        lastSuccessTime: this.lastSuccessTime,
        failureCount: this.failureCount,
        url: this.baseUrl
      };

      if (status === 'degraded') {
        await todoAutogen.logPerformanceIssue('gpu', {
          issue: 'Missing required models',
          details
        });
      }

      return { status, details };

    } catch (error: any) {
      await todoAutogen.logLLMMisfire({
        model: 'health-check',
        prompt: 'health check request',
        error: error.message,
        retryCount: 0
      });

      return {
        status: 'critical',
        details: {
          error: error.message,
          url: this.baseUrl,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  /**
   * Get system performance metrics
   */
  getMetrics() {
    return {
      failureCount: this.failureCount,
      lastSuccessTime: this.lastSuccessTime,
      timeSinceLastSuccess: Date.now() - this.lastSuccessTime,
      baseUrl: this.baseUrl,
      memoryConfig: {
        maxOldSpaceSize: (import.meta as any).env?.NODE_OPTIONS?.includes('max-old-space-size'),
        gpuMemoryFraction: LOCAL_LLM_CONFIG.GPU_MEMORY_FRACTION
      }
    };
  }
}

// Singleton instance for reuse
export const ollamaWrapper = new OllamaRetryWrapper();

/**
 * Convenience function for simple LLM calls
 */
export async function promptLLM(
  prompt: string,
  model?: string,
  options?: Partial<LLMCallOptions>
): Promise<string> {
  const result = await ollamaWrapper.callLLM(prompt, {
    model,
    ...options
  });

  return result.response;
}

/**
 * Streaming LLM call with retry logic
 */
export async function* streamLLM(
  prompt: string,
  options: LLMCallOptions = {}
): AsyncGenerator<string, void, unknown> {
  const {
    model = LOCAL_LLM_CONFIG.OLLAMA_MODELS.LEGAL_DETAILED,
    temperature = 0.2,
    timeout = 60000
  } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${(ollamaWrapper as any).baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        prompt,
        stream: true,
        options: { temperature }
      })
    });

    if (!response.ok) {
      throw new Error(`Streaming failed: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter((line: string) => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            yield data.response;
          }
        } catch (error: any) {
          // Skip malformed JSON
        }
      }
    }

  } catch (error: any) {
    await todoAutogen.logLLMMisfire({
      model,
      prompt: prompt.substring(0, 200) + '...',
      error: error.message,
      retryCount: 0
    });
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}