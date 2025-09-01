
// lib/server/ai/ollama-local-llm.ts
// Ollama integration for local LLM inference with legal models

import { logger } from './logger';
import { streamingService } from './streaming-service';

export interface OllamaModel {
  name: string;
  size: string;
  digest: string;
  modified: string;
}

export interface OllamaGenerateOptions {
  model: string;
  prompt: string;
  system?: string;
  template?: string;
  context?: number[];
  stream?: boolean;
  raw?: boolean;
  format?: 'json';
  options?: {
    temperature?: number;
    top_k?: number;
    top_p?: number;
    num_predict?: number;
    num_ctx?: number;
    stop?: string[];
    seed?: number;
    repeat_penalty?: number;
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

class OllamaLocalLLM {
  private baseUrl: string;
  private defaultModel: string = 'gemma3:legal-latest';
  private availableModels: Map<string, OllamaModel> = new Map();
  private modelCache: Map<string, { loaded: boolean; lastUsed: number }> = new Map();
  
  constructor(baseUrl: string = 'http://localhost:11434') {
    this.baseUrl = baseUrl;
    this.initialize();
  }

  private async initialize(): Promise<void> {
    try {
      logger.info('[OllamaLLM] Initializing Ollama local LLM service...');
      
      // Check if Ollama is available
      const available = await this.checkAvailability();
      if (!available) {
        logger.warn('[OllamaLLM] Ollama is not available at', this.baseUrl);
        return;
      }
      
      // Load available models
      await this.loadAvailableModels();
      
      // Try to pull legal-specific models if not present
      await this.ensureLegalModels();
      
      logger.info('[OllamaLLM] Ollama service initialized successfully');
    } catch (error: any) {
      logger.error('[OllamaLLM] Initialization failed:', error);
    }
  }

  /**
   * Check if Ollama service is available
   */
  async checkAvailability(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      return response.ok;
    } catch (error: any) {
      return false;
    }
  }

  /**
   * Load list of available models
   */
  async loadAvailableModels(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      
      const data = await response.json();
      this.availableModels.clear();
      
      for (const model of data.models || []) {
        this.availableModels.set(model.name, model);
        logger.info(`[OllamaLLM] Available model: ${model.name} (${model.size})`);
      }
    } catch (error: any) {
      logger.error('[OllamaLLM] Failed to load models:', error);
    }
  }

  /**
   * Ensure legal-specific models are available
   */
  async ensureLegalModels(): Promise<void> {
    const legalModels = [
      'gemma3:legal-latest',
      'llama2:legal-7b',
      'mistral:legal-instruct'
    ];
    
    for (const modelName of legalModels) {
      if (!this.availableModels.has(modelName)) {
        logger.info(`[OllamaLLM] Legal model ${modelName} not found, attempting to pull...`);
        
        // Check if base model exists and can be customized
        const baseModel = modelName.split(':')[0];
        if (this.availableModels.has(baseModel)) {
          // Create legal-tuned variant with custom modelfile
          await this.createLegalModel(baseModel, modelName);
        }
      }
    }
  }

  /**
   * Create a legal-tuned model variant
   */
  async createLegalModel(baseModel: string, targetName: string): Promise<void> {
    try {
      const modelfile = `
FROM ${baseModel}

# Legal domain specialization
SYSTEM """You are a legal AI assistant with expertise in legal analysis, case law, statutes, and legal procedures. 
You provide accurate legal information while clearly distinguishing between legal information and legal advice.
You cite sources appropriately and acknowledge the limitations of AI-generated legal analysis."""

# Adjust parameters for legal text
PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Legal-specific template
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}"""
`;

      const response = await fetch(`${this.baseUrl}/api/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: targetName,
          modelfile: modelfile
        })
      });

      if (response.ok) {
        logger.info(`[OllamaLLM] Created legal model variant: ${targetName}`);
        await this.loadAvailableModels();
      }
    } catch (error: any) {
      logger.error(`[OllamaLLM] Failed to create legal model ${targetName}:`, error);
    }
  }

  /**
   * Generate completion using local LLM
   */
  async generate(options: OllamaGenerateOptions): Promise<OllamaResponse | null> {
    try {
      // Use legal model if available
      const model = this.selectBestModel(options.model);
      
      logger.info(`[OllamaLLM] Generating with model ${model}`);
      
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...options,
          model,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Update model cache
      this.modelCache.set(model, {
        loaded: true,
        lastUsed: Date.now()
      });
      
      return result;
      
    } catch (error: any) {
      logger.error('[OllamaLLM] Generation failed:', error);
      return null;
    }
  }

  /**
   * Stream generation with progressive updates
   */
  async generateStream(
    options: OllamaGenerateOptions,
    onToken: (token: string) => void,
    onComplete: (response: string) => void
  ): Promise<void> {
    try {
      const model = this.selectBestModel(options.model);
      
      logger.info(`[OllamaLLM] Streaming generation with model ${model}`);
      
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...options,
          model,
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error(`Stream generation failed: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.response) {
              fullResponse += data.response;
              onToken(data.response);
            }
            if (data.done) {
              onComplete(fullResponse);
            }
          } catch (e: any) {
            // Ignore parsing errors
          }
        }
      }
    } catch (error: any) {
      logger.error('[OllamaLLM] Stream generation failed:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings using local model
   */
  async generateEmbeddings(text: string, model?: string): Promise<number[] | null> {
    try {
      const embeddingModel = model || 'nomic-embed-text';
      
      const response = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: embeddingModel,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.embedding;
      
    } catch (error: any) {
      logger.error('[OllamaLLM] Embedding generation failed:', error);
      return null;
    }
  }

  /**
   * Chat completion with conversation history
   */
  async chat(messages: Array<{ role: string; content: string }>, model?: string): Promise<string | null> {
    try {
      const selectedModel = this.selectBestModel(model);
      
      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          messages,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Chat completion failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.message?.content || null;
      
    } catch (error: any) {
      logger.error('[OllamaLLM] Chat completion failed:', error);
      return null;
    }
  }

  /**
   * Process legal document with specialized prompting
   */
  async processLegalDocument(
    document: string,
    task: 'summarize' | 'extract' | 'analyze' | 'classify',
    options?: unknown
  ): Promise<any> {
    try {
      let prompt = '';
      let systemPrompt = 'You are a legal document analysis expert.';
      
      switch (task) {
        case 'summarize':
          prompt = `Provide a comprehensive legal summary of the following document, highlighting key legal points, parties involved, and conclusions:\n\n${document}`;
          break;
          
        case 'extract':
          prompt = `Extract the following information from this legal document:
- Case citations
- Statute references
- Legal entities and parties
- Key dates
- Monetary amounts
- Legal holdings or decisions

Document:\n${document}`;
          break;
          
        case 'analyze':
          prompt = `Perform a detailed legal analysis of this document, including:
- Legal issues presented
- Arguments from each party
- Court's reasoning
- Precedents cited
- Legal implications

Document:\n${document}`;
          systemPrompt += ' Focus on legal reasoning and precedential value.';
          break;
          
        case 'classify':
          prompt = `Classify this legal document:
- Document type (contract, pleading, opinion, statute, etc.)
- Area of law (criminal, civil, contract, tort, etc.)
- Jurisdiction
- Key legal concepts

Document:\n${document}`;
          break;
      }
      
      const result = await this.generate({
        model: this.defaultModel,
        prompt,
        system: systemPrompt,
        options: {
          temperature: 0.3,
          top_p: 0.9,
          num_predict: 2000
        }
      });
      
      if (result) {
        // Parse structured output if needed
        if ((options as any)?.format === 'json') {
          try {
            return JSON.parse(result.response);
          } catch {
            return { text: result.response };
          }
        }
        return result.response;
      }
      
      return null;
      
    } catch (error: any) {
      logger.error('[OllamaLLM] Legal document processing failed:', error);
      return null;
    }
  }

  /**
   * Select best available model for the task
   */
  private selectBestModel(requestedModel?: string): string {
    // If specific model requested and available, use it
    if (requestedModel && this.availableModels.has(requestedModel)) {
      return requestedModel;
    }
    
    // Try legal-specific models first
    const legalModels = [
      'gemma3:legal-latest',
      'llama2:legal-7b',
      'mistral:legal-instruct'
    ];
    
    for (const model of legalModels) {
      if (this.availableModels.has(model)) {
        return model;
      }
    }
    
    // Fall back to general models
    const fallbackModels = [
      'llama2:13b',
      'mistral:7b-instruct',
      'gemma:7b',
      'llama2:7b'
    ];
    
    for (const model of fallbackModels) {
      if (this.availableModels.has(model)) {
        return model;
      }
    }
    
    // Use first available model
    if (this.availableModels.size > 0) {
      return Array.from(this.availableModels.keys())[0];
    }
    
    // Default fallback
    return requestedModel || 'llama2';
  }

  /**
   * Unload model from memory
   */
  async unloadModel(model: string): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt: '',
          keep_alive: 0
        })
      });
      
      this.modelCache.delete(model);
      logger.info(`[OllamaLLM] Unloaded model ${model}`);
    } catch (error: any) {
      logger.error(`[OllamaLLM] Failed to unload model ${model}:`, error);
    }
  }

  /**
   * Get model information
   */
  async getModelInfo(model: string): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/show`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: model })
      });

      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error: any) {
      logger.error(`[OllamaLLM] Failed to get model info for ${model}:`, error);
      return null;
    }
  }

  /**
   * Pull a model from Ollama library
   */
  async pullModel(model: string): Promise<boolean> {
    try {
      logger.info(`[OllamaLLM] Pulling model ${model}...`);
      
      const response = await fetch(`${this.baseUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: model })
      });

      if (!response.ok) {
        throw new Error(`Failed to pull model: ${response.statusText}`);
      }

      // Stream the pull progress
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.status) {
              logger.info(`[OllamaLLM] Pull progress: ${data.status}`);
            }
          } catch (e: any) {
            // Ignore parsing errors
          }
        }
      }

      await this.loadAvailableModels();
      return true;
      
    } catch (error: any) {
      logger.error(`[OllamaLLM] Failed to pull model ${model}:`, error);
      return false;
    }
  }

  /**
   * Health check for Ollama service
   */
  async healthCheck(): Promise<{
    status: string;
    available: boolean;
    models: string[];
    loaded: string[];
  }> {
    const available = await this.checkAvailability();
    
    return {
      status: available ? 'healthy' : 'unavailable',
      available,
      models: Array.from(this.availableModels.keys()),
      loaded: Array.from(this.modelCache.keys()).filter(
        model => this.modelCache.get(model)?.loaded
      )
    };
  }
}

// Export singleton instance
export const ollamaLLM = new OllamaLocalLLM();
;
// Types are already exported as interfaces above

