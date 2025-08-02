/**
 * Ollama Service for local LLM integration
 */

import { logger } from "../logger";

export interface OllamaModel {
  name: string;
  size: string;
  digest: string;
  details: {
    format: string;
    family: string;
    families?: string[];
    parameter_size: string;
    quantization_level: string;
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

export class OllamaService {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl = "http://localhost:11434", timeout = 30000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  async isHealthy(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch (error) {
      logger.error("Ollama health check failed", error);
      return false;
    }
  }

  async listModels(): Promise<OllamaModel[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      return data.models || [];
    } catch (error) {
      logger.error("Failed to list Ollama models", error);
      return [];
    }
  }

  async generate(
    model: string,
    prompt: string,
    options: {
      temperature?: number;
      max_tokens?: number;
      stream?: boolean;
    } = {}
  ): Promise<string> {
    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
          options: {
            temperature: options.temperature || 0.7,
            num_predict: options.max_tokens || 1000,
          },
        }),
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!response.ok) {
        throw new Error(
          `Ollama API error: ${response.status} ${response.statusText}`
        );
      }

      const data: OllamaResponse = await response.json();
      return data.response;
    } catch (error) {
      logger.error("Failed to generate with Ollama", { model, error });
      throw error;
    }
  }

  async embeddings(model: string, prompt: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          prompt,
        }),
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!response.ok) {
        throw new Error(
          `Ollama embeddings error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      return data.embedding || [];
    } catch (error) {
      logger.error("Failed to get embeddings from Ollama", { model, error });
      throw error;
    }
  }
}

export const ollamaService = new OllamaService();
export default ollamaService;
