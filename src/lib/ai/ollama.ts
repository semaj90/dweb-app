// Ollama Service Integration for SvelteKit
// Handles local LLM inference with GPU acceleration

import type { OllamaResponse, OllamaEmbedding, ModelInfo } from './types';

export class OllamaService {
  private baseUrl: string;
  
  constructor(baseUrl = 'http://localhost:11434') {
    this.baseUrl = baseUrl;
  }

  /**
   * Generate text using Ollama models
   */
  async generate(
    model: string,
    prompt: string,
    options: {
      system?: string;
      temperature?: number;
      stream?: boolean;
      context?: number[];
    } = {}
  ): Promise<OllamaResponse> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
        system: options.system,
        stream: options.stream ?? false,
        options: {
          temperature: options.temperature ?? 0.3,
          num_ctx: 8192,
          num_gpu: -1, // Use all GPU layers
        },
        context: options.context,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Stream responses for real-time UI updates
   */
  async *generateStream(
    model: string,
    prompt: string,
    options: {
      system?: string;
      temperature?: number;
      onToken?: (token: string) => void;
    } = {}
  ): AsyncGenerator<string> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
        system: options.system,
        stream: true,
        options: {
          temperature: options.temperature ?? 0.3,
          num_ctx: 8192,
          num_gpu: -1,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const json = JSON.parse(line);
          if (json.response) {
            yield json.response;
            options.onToken?.(json.response);
          }
        } catch (e) {
          // Skip invalid JSON lines
        }
      }
    }
  }

  /**
   * Generate embeddings for vector search
   */
  async embed(model: string, prompt: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embed error: ${response.statusText}`);
    }

    const data: OllamaEmbedding = await response.json();
    return data.embedding;
  }

  /**
   * Batch embed multiple texts efficiently
   */
  async embedBatch(model: string, texts: string[]): Promise<number[][]> {
    const embeddings = await Promise.all(
      texts.map(text => this.embed(model, text))
    );
    return embeddings;
  }

  /**
   * List available models
   */
  async listModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/api/tags`);
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.models || [];
  }

  /**
   * Check if Ollama service is running
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(2000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Pull a model from Ollama library
   */
  async pullModel(modelName: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: modelName,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to pull model: ${response.statusText}`);
    }
  }
}

// Singleton instance
export const ollama = new OllamaService();

// Model presets for different use cases
export const MODELS = {
  LEGAL_DETAILED: 'gemma3-legal',
  LEGAL_QUICK: 'gemma3-quick',
  EMBEDDINGS: 'nomic-embed-text', // Or use all-minilm for smaller embeddings
} as const;

// Helper functions for common legal AI tasks
export const legalAI = {
  /**
   * Analyze a legal document
   */
  async analyzeDocument(content: string, analysisType: string) {
    return ollama.generate(MODELS.LEGAL_DETAILED, content, {
      system: `You are a legal document analyzer. Perform a ${analysisType} analysis of the provided document. 
               Be thorough and cite specific sections.`,
      temperature: 0.2,
    });
  },

  /**
   * Quick legal definition lookup
   */
  async defineTerm(term: string) {
    return ollama.generate(MODELS.LEGAL_QUICK, `Define the legal term: ${term}`, {
      system: 'Provide a concise legal definition with key points and relevant law citations.',
      temperature: 0.1,
    });
  },

  /**
   * Contract clause suggestion
   */
  async suggestClause(context: string, clauseType: string) {
    return ollama.generate(MODELS.LEGAL_DETAILED, 
      `Suggest a ${clauseType} clause for: ${context}`, {
      system: 'Draft a professional legal clause. Include standard language and considerations.',
      temperature: 0.3,
    });
  },

  /**
   * Legal research assistant
   */
  async research(query: string, jurisdiction?: string) {
    const jurisdictionContext = jurisdiction ? ` in ${jurisdiction} jurisdiction` : '';
    return ollama.generate(MODELS.LEGAL_DETAILED, 
      `Research: ${query}${jurisdictionContext}`, {
      system: 'Provide comprehensive legal research with case law, statutes, and analysis.',
      temperature: 0.2,
    });
  },
};
