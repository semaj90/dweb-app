// Enhanced Ollama Service with Gemma3 Legal AI Integration
// Includes streaming, fallback, and error handling

import { GEMMA3_CONFIG } from "$lib/config/gemma3-legal-config";
import type {
  AIModel,
  ChatMessage,
  OllamaResponse,
  StreamResponse,
} from "$lib/types/ai";

export interface OllamaGenerateOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  num_ctx?: number;
  stream?: boolean;
  context?: ChatMessage[];
}

export interface OllamaChatOptions extends OllamaGenerateOptions {
  system?: string;
  keepAlive?: string;
}

class EnhancedOllamaService {
  private baseUrl: string;
  private currentModel: string;
  private fallbackModel: string;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private isHealthy: boolean = false;
  private retryAttempts: number = 3;
  private retryDelay: number = 1000;

  constructor() {
    this.baseUrl = GEMMA3_CONFIG.model.endpoint;
    this.currentModel = GEMMA3_CONFIG.model.name;
    this.fallbackModel = GEMMA3_CONFIG.model.fallback;
    this.startHealthCheck();
  }

  // Health check monitoring
  private startHealthCheck() {
    this.checkHealth();
    this.healthCheckInterval = setInterval(() => {
      this.checkHealth();
    }, 30000); // Check every 30 seconds
  }

  private async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/version`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      this.isHealthy = response.ok;
      return this.isHealthy;
    } catch (error) {
      console.warn("Ollama health check failed:", error);
      this.isHealthy = false;
      return false;
    }
  }

  // Retry wrapper for API calls
  private async withRetry<T>(
    operation: () => Promise<T>,
    context: string = "operation"
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        console.warn(
          `${context} failed (attempt ${attempt}/${this.retryAttempts}):`,
          error
        );

        if (attempt < this.retryAttempts) {
          await this.delay(this.retryDelay * attempt);
        }
      }
    }

    throw lastError!;
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // List available models
  async listModels(): Promise<AIModel[]> {
    return this.withRetry(async () => {
      const response = await fetch(`${this.baseUrl}/api/tags`);

      if (!response.ok) {
        throw new Error(`Failed to list models: ${response.statusText}`);
      }

      const data = await response.json();
      return (
        data.models?.map((model: any) => ({
          name: model.name,
          size: this.formatSize(model.size),
          modified: new Date(model.modified_at),
        })) || []
      );
    }, "List models");
  }

  // Check if model exists
  async modelExists(modelName: string): Promise<boolean> {
    try {
      const models = await this.listModels();
      return models.some((model) => model.name === modelName);
    } catch (error) {
      console.warn("Failed to check model existence:", error);
      return false;
    }
  }

  // Pull model if it doesn't exist
  async ensureModel(modelName: string): Promise<boolean> {
    try {
      if (await this.modelExists(modelName)) {
        return true;
      }

      console.log(`Pulling model: ${modelName}`);
      const response = await fetch(`${this.baseUrl}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: modelName, stream: false }),
      });

      if (!response.ok) {
        throw new Error(`Failed to pull model: ${response.statusText}`);
      }

      return true;
    } catch (error) {
      console.error(`Failed to ensure model ${modelName}:`, error);
      return false;
    }
  }

  // Generate completion
  async generate(
    model: string,
    prompt: string,
    options: OllamaGenerateOptions = {}
  ): Promise<OllamaResponse> {
    return this.withRetry(async () => {
      // Ensure model exists, fallback if needed
      const modelToUse = await this.selectBestModel(model);

      const requestBody = {
        model: modelToUse,
        prompt,
        stream: false,
        options: {
          temperature:
            options.temperature ?? GEMMA3_CONFIG.parameters.temperature,
          top_p: options.top_p ?? GEMMA3_CONFIG.parameters.top_p,
          top_k: options.top_k ?? GEMMA3_CONFIG.parameters.top_k,
          repeat_penalty:
            options.repeat_penalty ?? GEMMA3_CONFIG.parameters.repeat_penalty,
          num_ctx: options.num_ctx ?? GEMMA3_CONFIG.parameters.num_ctx,
        },
      };

      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Generate request failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        response: data.response,
        model: modelToUse,
        created_at: new Date(data.created_at),
        done: data.done,
        context: data.context,
        total_duration: data.total_duration,
        load_duration: data.load_duration,
        prompt_eval_count: data.prompt_eval_count,
        prompt_eval_duration: data.prompt_eval_duration,
        eval_count: data.eval_count,
        eval_duration: data.eval_duration,
      };
    }, `Generate with ${model}`);
  }

  // Chat completion
  async chat(
    model: string,
    message: string,
    options: OllamaChatOptions = {}
  ): Promise<OllamaResponse> {
    return this.withRetry(async () => {
      const modelToUse = await this.selectBestModel(model);

      // Build messages array
      const messages: Array<{ role: string; content: string }> = [];

      // Add system message if provided
      if (options.system) {
        messages.push({
          role: "system",
          content: options.system,
        });
      }

      // Add context messages if provided
      if (options.context && options.context.length > 0) {
        messages.push(
          ...options.context.map((msg) => ({
            role: msg.role,
            content: msg.content,
          }))
        );
      }

      // Add current message
      messages.push({
        role: "user",
        content: message,
      });

      const requestBody = {
        model: modelToUse,
        messages,
        stream: false,
        options: {
          temperature:
            options.temperature ?? GEMMA3_CONFIG.parameters.temperature,
          top_p: options.top_p ?? GEMMA3_CONFIG.parameters.top_p,
          top_k: options.top_k ?? GEMMA3_CONFIG.parameters.top_k,
          repeat_penalty:
            options.repeat_penalty ?? GEMMA3_CONFIG.parameters.repeat_penalty,
          num_ctx: options.num_ctx ?? GEMMA3_CONFIG.parameters.num_ctx,
        },
        keep_alive: options.keepAlive ?? "5m",
      };

      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        response: data.message?.content || data.response,
        model: modelToUse,
        created_at: new Date(data.created_at),
        done: data.done,
        total_duration: data.total_duration,
        load_duration: data.load_duration,
        prompt_eval_count: data.prompt_eval_count,
        prompt_eval_duration: data.prompt_eval_duration,
        eval_count: data.eval_count,
        eval_duration: data.eval_duration,
      };
    }, `Chat with ${model}`);
  }

  // Stream chat completion
  async *streamChat(
    model: string,
    message: string,
    options: OllamaChatOptions = {}
  ): AsyncGenerator<StreamResponse, void, unknown> {
    const modelToUse = await this.selectBestModel(model);

    // Build messages array
    const messages: Array<{ role: string; content: string }> = [];

    if (options.system) {
      messages.push({
        role: "system",
        content: options.system,
      });
    }

    if (options.context && options.context.length > 0) {
      messages.push(
        ...options.context.map((msg) => ({
          role: msg.role,
          content: msg.content,
        }))
      );
    }

    messages.push({
      role: "user",
      content: message,
    });

    const requestBody = {
      model: modelToUse,
      messages,
      stream: true,
      options: {
        temperature:
          options.temperature ?? GEMMA3_CONFIG.parameters.temperature,
        top_p: options.top_p ?? GEMMA3_CONFIG.parameters.top_p,
        top_k: options.top_k ?? GEMMA3_CONFIG.parameters.top_k,
        repeat_penalty:
          options.repeat_penalty ?? GEMMA3_CONFIG.parameters.repeat_penalty,
        num_ctx: options.num_ctx ?? GEMMA3_CONFIG.parameters.num_ctx,
      },
      keep_alive: options.keepAlive ?? "5m",
    };

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Stream chat request failed: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              yield {
                response: data.message?.content || data.response || "",
                model: modelToUse,
                done: data.done,
                created_at: new Date(),
              };
            } catch (error) {
              console.warn("Failed to parse stream chunk:", error);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Select best available model
  private async selectBestModel(preferredModel: string): Promise<string> {
    // Check if preferred model exists
    if (await this.modelExists(preferredModel)) {
      return preferredModel;
    }

    // Try to pull the preferred model
    if (await this.ensureModel(preferredModel)) {
      return preferredModel;
    }

    // Fall back to fallback model
    console.warn(
      `Falling back to ${this.fallbackModel} as ${preferredModel} is not available`
    );

    if (await this.ensureModel(this.fallbackModel)) {
      return this.fallbackModel;
    }

    throw new Error(
      `Neither ${preferredModel} nor ${this.fallbackModel} are available`
    );
  }

  // Utility methods
  private formatSize(bytes: number): string {
    if (!bytes) return "Unknown";

    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  // Get service status
  getStatus() {
    return {
      healthy: this.isHealthy,
      baseUrl: this.baseUrl,
      currentModel: this.currentModel,
      fallbackModel: this.fallbackModel,
    };
  }

  // Update configuration
  updateConfig(config: Partial<typeof GEMMA3_CONFIG.model>) {
    if (config.name) this.currentModel = config.name;
    if (config.fallback) this.fallbackModel = config.fallback;
    if (config.endpoint) this.baseUrl = config.endpoint;
  }

  // Cleanup
  destroy() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }
}

// Export singleton instance
export const ollamaService = new EnhancedOllamaService();

// Export for testing
export { EnhancedOllamaService };
