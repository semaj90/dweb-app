// WebAssembly AI Adapter for AIAssistantManager
// Bridges XState-managed AI assistant with WebAssembly llama.cpp service

import { webLlamaService, type WebLlamaResponse, type WebLlamaConfig } from '../ai/webasm-llamacpp.js';
import { browser } from '$app/environment';
import type { ConversationEntry } from '../stores/aiAssistant.svelte.js';

export interface WebAssemblyAIConfig {
  modelPath: string;
  wasmPath: string;
  enableGPU: boolean;
  enableMultiCore: boolean;
  maxTokens: number;
  temperature: number;
  contextSize: number;
}

export interface WebAssemblyAIResponse {
  content: string;
  metadata: {
    tokensGenerated: number;
    processingTime: number;
    confidence: number;
    method: 'webassembly';
    modelUsed: string;
    fromCache: boolean;
  };
  conversationId?: string;
}

export class WebAssemblyAIAdapter {
  private initialized = false;
  private config: WebAssemblyAIConfig;
  private currentModel = 'gemma3-legal';
  
  constructor(config: Partial<WebAssemblyAIConfig> = {}) {
    this.config = {
      modelPath: '/models/gemma-3-legal-8b-q4_k_m.gguf',
      wasmPath: '/wasm/llama.wasm',
      enableGPU: true,
      enableMultiCore: true,
      maxTokens: 2048,
      temperature: 0.7,
      contextSize: 8192,
      ...config
    };
  }

  /**
   * Initialize WebAssembly AI service
   */
  async initialize(): Promise<boolean> {
    if (!browser) {
      console.warn('[WebAssembly AI] Not running in browser environment');
      return false;
    }

    if (this.initialized) {
      return true;
    }

    try {
      console.log('[WebAssembly AI] Initializing AI adapter...');
      
      // Configure WebLlama service
      const webLlamaConfig: Partial<WebLlamaConfig> = {
        modelUrl: this.config.modelPath,
        wasmUrl: this.config.wasmPath,
        threadsCount: navigator.hardwareConcurrency || 4,
        contextSize: this.config.contextSize,
        enableWebGPU: this.config.enableGPU,
        enableMultiCore: this.config.enableMultiCore,
        temperature: this.config.temperature,
        batchSize: 512
      };

      // Load model
      const modelLoaded = await webLlamaService.loadModel();
      if (!modelLoaded) {
        throw new Error('Failed to load WebAssembly model');
      }

      this.initialized = true;
      console.log('[WebAssembly AI] Adapter initialized successfully');
      return true;

    } catch (error) {
      console.error('[WebAssembly AI] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Send message to WebAssembly AI and get response
   */
  async sendMessage(
    message: string,
    options: {
      conversationHistory?: ConversationEntry[];
      useContext?: boolean;
      model?: string;
      temperature?: number;
      maxTokens?: number;
    } = {}
  ): Promise<WebAssemblyAIResponse> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.initialized) {
      throw new Error('WebAssembly AI adapter not initialized');
    }

    try {
      const startTime = performance.now();
      
      // Build prompt with conversation context
      const prompt = this.buildPromptWithContext(message, options.conversationHistory || []);
      
      // Generate response using WebAssembly
      const response: WebLlamaResponse = await webLlamaService.generate(prompt, {
        maxTokens: options.maxTokens || this.config.maxTokens,
        temperature: options.temperature || this.config.temperature,
        useCache: true
      });

      const totalTime = performance.now() - startTime;

      return {
        content: response.text,
        metadata: {
          tokensGenerated: response.tokensGenerated,
          processingTime: totalTime,
          confidence: response.confidence,
          method: 'webassembly',
          modelUsed: options.model || this.currentModel,
          fromCache: response.fromCache
        }
      };

    } catch (error: any) {
      console.error('[WebAssembly AI] Message processing failed:', error);
      throw new Error(`WebAssembly AI error: ${error.message}`);
    }
  }

  /**
   * Analyze legal document using WebAssembly Gemma 3 Legal
   */
  async analyzeLegalDocument(
    title: string,
    content: string,
    analysisType: 'comprehensive' | 'quick' | 'risk-focused' = 'comprehensive'
  ): Promise<{
    summary: string;
    keyTerms: string[];
    entities: Array<{ type: string; value: string; confidence: number }>;
    risks: Array<{ type: string; severity: string; description: string }>;
    recommendations: string[];
    confidence: number;
    processingTime: number;
    method: string;
  }> {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      const result = await webLlamaService.analyzeLegalDocument(title, content, analysisType);
      return result;
    } catch (error: any) {
      console.error('[WebAssembly AI] Legal analysis failed:', error);
      throw error;
    }
  }

  /**
   * Stream response (simulated chunked responses for WebAssembly)
   */
  async streamMessage(
    message: string,
    options: {
      conversationHistory?: ConversationEntry[];
      onChunk?: (chunk: string) => void;
      onComplete?: (response: WebAssemblyAIResponse) => void;
      onError?: (error: Error) => void;
    } = {}
  ): Promise<void> {
    try {
      const response = await this.sendMessage(message, {
        conversationHistory: options.conversationHistory
      });

      // Simulate streaming by chunking the response
      const chunks = this.chunkResponse(response.content, 50);
      
      for (const chunk of chunks) {
        if (options.onChunk) {
          options.onChunk(chunk);
        }
        // Add small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      if (options.onComplete) {
        options.onComplete(response);
      }

    } catch (error: any) {
      console.error('[WebAssembly AI] Streaming failed:', error);
      if (options.onError) {
        options.onError(error);
      }
    }
  }

  /**
   * Get available models
   */
  getAvailableModels(): string[] {
    return [
      'gemma3-legal',
      'gemma3-legal-8b',
      'gemma3-legal-4b'
    ];
  }

  /**
   * Set model configuration
   */
  setModel(model: string): void {
    if (!this.getAvailableModels().includes(model)) {
      throw new Error(`Unsupported model: ${model}`);
    }
    this.currentModel = model;
  }

  /**
   * Set temperature
   */
  setTemperature(temperature: number): void {
    if (temperature < 0 || temperature > 2) {
      throw new Error('Temperature must be between 0 and 2');
    }
    this.config.temperature = temperature;
  }

  /**
   * Get health status
   */
  getHealthStatus(): {
    initialized: boolean;
    modelLoaded: boolean;
    webgpuAvailable: boolean;
    webgpuEnabled: boolean;
    workerEnabled: boolean;
    cacheSize: number;
    threadsCount: number;
    wasmSupported: boolean;
    currentModel: string;
  } {
    const wasmHealth = webLlamaService.getHealthStatus();
    
    return {
      initialized: this.initialized,
      currentModel: this.currentModel,
      ...wasmHealth
    };
  }

  /**
   * Check if WebAssembly is supported
   */
  isSupported(): boolean {
    return browser && 
           typeof WebAssembly !== 'undefined' &&
           typeof Worker !== 'undefined' &&
           typeof performance !== 'undefined';
  }

  // Private helper methods

  private buildPromptWithContext(message: string, history: ConversationEntry[]): string {
    let prompt = '<|system|>You are a specialized legal AI assistant. Provide accurate, helpful responses about legal matters. Be concise but thorough.<|end|>\n\n';
    
    // Add conversation history (last 5 exchanges)
    const recentHistory = history.slice(-10);
    for (const entry of recentHistory) {
      if (entry.type === 'user') {
        prompt += `<|user|>${entry.content}<|end|>\n`;
      } else if (entry.type === 'assistant') {
        prompt += `<|assistant|>${entry.content}<|end|>\n`;
      }
    }
    
    // Add current message
    prompt += `<|user|>${message}<|end|>\n<|assistant|>`;
    
    return prompt;
  }

  private chunkResponse(text: string, chunkSize: number): string[] {
    const words = text.split(' ');
    const chunks: string[] = [];
    
    for (let i = 0; i < words.length; i += chunkSize) {
      const chunk = words.slice(i, i + chunkSize).join(' ');
      chunks.push(chunk + ' ');
    }
    
    return chunks;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (webLlamaService) {
      webLlamaService.dispose();
    }
    this.initialized = false;
  }
}

// Export singleton instance
export const webAssemblyAIAdapter = new WebAssemblyAIAdapter();