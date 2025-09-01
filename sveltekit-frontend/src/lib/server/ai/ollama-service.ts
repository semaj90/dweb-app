import { EventEmitter } from "events";
/**
 * Ollama Service - Main integration with local LLM models
 * Implements intelligent fallback: gemma3:legal-latest -> legal-bert
 */

import { OLLAMA_CONFIG, getModelConfig, getOptimalModel, selectBestAvailableModel, isLegalTask } from './ollama-config';
import type {
  OllamaGenerateRequest,
  OllamaResponse,
  OllamaEmbeddingRequest,
  OllamaEmbeddingResponse,
  DocumentChunk,
  LegalDocument,
  AnalysisResult,
  UserQuery
} from './types';

class OllamaService extends EventEmitter {
  private baseUrl: string;
  private cache: Map<string, any> = new Map();
  private requestQueue: Array<() => Promise<any>> = [];
  private activeRequests = 0;
  private availableModels: string[] = [];
  private modelCheckInterval: NodeJS.Timeout | null = null;
  
  constructor() {
    super();
    this.baseUrl = OLLAMA_CONFIG.baseUrl;
    this.startQueueProcessor();
    this.startModelMonitor();
  }

  /**
   * Start monitoring available models
   */
  private startModelMonitor() {
    // Check models immediately
    this.updateAvailableModels();
    
    // Then check every 30 seconds
    this.modelCheckInterval = setInterval(() => {
      this.updateAvailableModels();
    }, 30000);
  }

  /**
   * Update list of available models
   */
  private async updateAvailableModels() {
    try {
      const models = await this.listModels();
      this.availableModels = models.models?.map((m: any) => m.name) || [];
      this.emit('models-updated', this.availableModels);
    } catch (error: any) {
      console.error('Failed to update model list:', error);
    }
  }

  /**
   * Check if Ollama service is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * List available models
   */
  async listModels(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) throw new Error('Failed to list models');
      return await response.json();
    } catch (error: any) {
      console.error('Error listing models:', error);
      return { models: [] };
    }
  }

  /**
   * Check if gemma3:legal-latest is available
   */
  async checkLegalModel(): Promise<boolean> {
    const { models } = await this.listModels();
    return models?.some((m: any) => 
      m.name === 'gemma3:legal-latest' || 
      m.name.includes('gemma3') && m.name.includes('legal')
    );
  }

  /**
   * Check if legal-bert is available
   */
  async checkLegalBert(): Promise<boolean> {
    const { models } = await this.listModels();
    return models?.some((m: any) => 
      m.name === 'legal-bert' || 
      m.name.includes('legal') && m.name.includes('bert')
    );
  }

  /**
   * Select the best model for a task with intelligent fallback
   */
  private async selectModelForTask(task: 'generation' | 'legal-analysis' | 'embedding', prompt?: string): Promise<string> {
    // Ensure we have current model list
    if (this.availableModels.length === 0) {
      await this.updateAvailableModels();
    }

    // Determine if this is a legal task
    const isLegal = prompt ? isLegalTask(prompt) : task === 'legal-analysis';
    
    // Get the appropriate fallback chain
    let preferredModels: string[];
    if (isLegal || task === 'legal-analysis') {
      preferredModels = getOptimalModel('legal-analysis');
    } else {
      preferredModels = getOptimalModel(task as any);
    }

    // Select the best available model
    const selectedModel = selectBestAvailableModel(preferredModels, this.availableModels);
    
    if (selectedModel) {
      this.emit('model-selected', { task, selected: selectedModel, isLegal });
      console.log(`Selected model: ${selectedModel} for ${task} (legal: ${isLegal})`);
      return selectedModel;
    }

    // If no models available, throw error
    throw new Error(`No suitable models available for ${task}. Preferred: ${preferredModels.join(', ')}`);
  }

  /**
   * Generate text with intelligent model selection
   */
  async generate(
    prompt: string,
    options: Partial<OllamaGenerateRequest> = {}
  ): Promise<OllamaResponse> {
    // Select the best model if not specified
    let modelName = options.model;
    if (!modelName) {
      try {
        modelName = await this.selectModelForTask('generation', prompt);
      } catch (error: any) {
        console.error('Model selection failed:', error);
        // Try with legal-bert as fallback
        modelName = 'legal-bert';
      }
    }
    
    const modelConfig = getModelConfig(modelName);
    
    // Check cache if enabled
    const cacheKey = this.getCacheKey('generate', prompt, { ...options, model: modelName });
    if (OLLAMA_CONFIG.performance.cacheEnabled && this.cache.has(cacheKey)) {
      this.emit('cache-hit', cacheKey);
      return this.cache.get(cacheKey);
    }

    const request: OllamaGenerateRequest = {
      model: modelName,
      prompt,
      system: options.system || modelConfig.systemPrompt,
      stream: options.stream ?? OLLAMA_CONFIG.streamEnabled,
      options: {
        ...modelConfig.options,
        ...options.options
      }
    };

    // Queue the request if we're at capacity
    if (this.activeRequests >= OLLAMA_CONFIG.performance.parallelRequests) {
      return this.queueRequest(() => this.executeGenerate(request, cacheKey));
    }

    return this.executeGenerate(request, cacheKey);
  }

  /**
   * Execute generation request with fallback support
   */
  private async executeGenerate(
    request: OllamaGenerateRequest,
    cacheKey: string
  ): Promise<OllamaResponse> {
    this.activeRequests++;
    this.emit('request-start', { type: 'generate', model: request.model });

    // Try primary model first
    let lastError: Error | null = null;
    const fallbackChain = [request.model];
    
    // Add legal-bert as fallback if not already the primary
    if (request.model !== 'legal-bert' && !fallbackChain.includes('legal-bert')) {
      fallbackChain.push('legal-bert');
    }

    for (const model of fallbackChain) {
      try {
        const modelConfig = getModelConfig(model);
        const fallbackRequest = {
          ...request,
          model,
          system: request.system || modelConfig.systemPrompt,
          options: {
            ...modelConfig.options,
            ...request.options
          }
        };

        const response = await fetch(`${this.baseUrl}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(fallbackRequest),
          signal: AbortSignal.timeout(OLLAMA_CONFIG.timeout)
        });

        if (!response.ok) {
          throw new Error(`Model ${model} failed: ${response.statusText}`);
        }

        let result: OllamaResponse;
        
        if (fallbackRequest.stream) {
          result = await this.handleStreamResponse(response);
        } else {
          result = await response.json();
        }

        // Add metadata about which model was used
        result.model = model;
        result.fallback_used = model !== request.model;
        result.models_tried = fallbackChain.slice(0, fallbackChain.indexOf(model) + 1);
        
        // Cache the result
        if (OLLAMA_CONFIG.performance.cacheEnabled) {
          this.cache.set(cacheKey, result);
          setTimeout(() => this.cache.delete(cacheKey), OLLAMA_CONFIG.performance.cacheTTL * 1000);
        }

        this.emit('request-complete', { type: 'generate', model, fallback: model !== request.model });
        
        // Log if we used a fallback
        if (model !== request.model) {
          console.log(`Used fallback model: ${model} (original: ${request.model})`);
        }
        
        return result;
      } catch (error: any) {
        lastError = error as Error;
        console.error(`Model ${model} failed:`, error);
        
        // Continue to next model in fallback chain
        if (fallbackChain.indexOf(model) < fallbackChain.length - 1) {
          console.log(`Trying fallback model: ${fallbackChain[fallbackChain.indexOf(model) + 1]}`);
        }
      }
    }

    // All models failed
    this.emit('request-error', { type: 'generate', error: lastError });
    this.activeRequests--;
    throw new Error(`All models failed. Last error: ${lastError?.message}`);
  }

  /**
   * Generate embeddings with fallback support
   */
  async generateEmbeddings(text: string): Promise<number[]> {
    const embeddingModels = getOptimalModel('embedding');
    let lastError: Error | null = null;
    
    for (const model of embeddingModels) {
      // Check cache
      const cacheKey = this.getCacheKey('embedding', text, { model });
      if (OLLAMA_CONFIG.performance.cacheEnabled && this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey);
      }

      const request: OllamaEmbeddingRequest = {
        model,
        prompt: text
      };

      try {
        const response = await fetch(`${this.baseUrl}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
          signal: AbortSignal.timeout(OLLAMA_CONFIG.timeout)
        });

        if (!response.ok) {
          throw new Error(`Embedding generation failed with ${model}: ${response.statusText}`);
        }

        const result: OllamaEmbeddingResponse = await response.json();
        
        // Cache the embeddings
        if (OLLAMA_CONFIG.performance.cacheEnabled) {
          this.cache.set(cacheKey, result.embedding);
          setTimeout(() => this.cache.delete(cacheKey), OLLAMA_CONFIG.performance.cacheTTL * 1000);
        }

        return result.embedding;
      } catch (error: any) {
        lastError = error as Error;
        console.error(`Embedding model ${model} failed:`, error);
      }
    }

    throw new Error(`All embedding models failed. Last error: ${lastError?.message}`);
  }

  /**
   * Analyze legal document with intelligent model selection
   */
  async analyzeLegalDocument(document: LegalDocument): Promise<AnalysisResult> {
    // Always use legal-analysis task for documents
    const model = await this.selectModelForTask('legal-analysis', document.content);
    const prompt = this.buildLegalAnalysisPrompt(document);
    
    const response = await this.generate(prompt, {
      model,
      format: 'json',
      options: {
        temperature: 0.3, // Lower temperature for more consistent legal analysis
        top_p: 0.95,
        num_predict: 2048
      }
    });

    try {
      const analysis = JSON.parse(response.response || '{}');
      return this.formatAnalysisResult(document.id, analysis, response.model);
    } catch (error: any) {
      console.error('Error parsing legal analysis:', error);
      
      // If parsing fails and we haven't tried legal-bert yet
      if (response.model !== 'legal-bert') {
        console.log('Retrying with legal-bert fallback...');
        const fallbackResponse = await this.generate(prompt, {
          model: 'legal-bert',
          format: 'json',
          options: { temperature: 0.3 }
        });
        
        const fallbackAnalysis = JSON.parse(fallbackResponse.response || '{}');
        return this.formatAnalysisResult(document.id, fallbackAnalysis, 'legal-bert');
      }
      
      throw new Error('Failed to parse legal analysis response');
    }
  }

  /**
   * Process user query with context-aware responses
   */
  async processQuery(query: UserQuery, relevantDocs: DocumentChunk[]): Promise<string> {
    const context = this.buildQueryContext(relevantDocs);
    const isLegal = isLegalTask(query.query);
    const model = await this.selectModelForTask(isLegal ? 'legal-analysis' : 'generation', query.query);
    
    const prompt = `
      Legal Query: ${query.query}
      
      Relevant Context:
      ${context}
      
      Please provide a comprehensive legal analysis addressing the query based on the provided context.
      Include relevant citations and legal precedents where applicable.
    `;

    const response = await this.generate(prompt, {
      model,
      options: {
        temperature: 0.5,
        num_predict: 1500
      }
    });

    return response.response || 'Unable to generate response';
  }

  /**
   * Build legal analysis prompt
   */
  private buildLegalAnalysisPrompt(document: LegalDocument): string {
    return `
      Analyze the following legal document and provide a structured analysis in JSON format:
      
      Document Type: ${document.type}
      Title: ${document.title}
      Content: ${document.content.substring(0, 8000)} // Limit to context window
      
      Please provide:
      1. Executive summary (2-3 sentences)
      2. Key legal points (list of 3-5 main points)
      3. Identified entities (people, organizations, dates, locations, legal concepts)
      4. Overall sentiment assessment
      5. Risk factors if any
      6. Recommendations for action
      7. Relevant legal precedents or statutes (if applicable)
      
      Format the response as valid JSON.
    `;
  }

  /**
   * Build context from document chunks
   */
  private buildQueryContext(chunks: DocumentChunk[]): string {
    return chunks
      .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0))
      .slice(0, 5) // Top 5 most relevant chunks
      .map(chunk => `
        Source: ${chunk.metadata.source}
        ${chunk.metadata.section ? `Section: ${chunk.metadata.section}` : ''}
        Content: ${chunk.content}
      `)
      .join('\n---\n');
  }

  /**
   * Format analysis result with model info
   */
  private formatAnalysisResult(documentId: string, analysis: any, modelUsed?: string): AnalysisResult {
    return {
      documentId,
      summary: analysis.summary || '',
      keyPoints: analysis.keyPoints || [],
      entities: {
        people: analysis.entities?.people || [],
        organizations: analysis.entities?.organizations || [],
        dates: analysis.entities?.dates || [],
        locations: analysis.entities?.locations || [],
        legalConcepts: analysis.entities?.legalConcepts || []
      },
      sentiment: analysis.sentiment || 'neutral',
      riskFactors: analysis.riskFactors || [],
      recommendations: analysis.recommendations || [],
      citations: analysis.citations || [],
      metadata: {
        modelUsed: modelUsed || 'unknown',
        timestamp: new Date().toISOString()
      }
    };
  }

  /**
   * Handle streaming responses
   */
  private async handleStreamResponse(response: Response): Promise<OllamaResponse> {
    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let fullResponse = '';
    let lastResponse: OllamaResponse | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.response) {
            fullResponse += parsed.response;
            this.emit('stream-chunk', parsed.response);
          }
          lastResponse = parsed;
        } catch {
          // Skip non-JSON lines
        }
      }
    }

    if (lastResponse) {
      lastResponse.response = fullResponse;
      return lastResponse;
    }

    throw new Error('No valid response received');
  }

  /**
   * Queue request for later execution
   */
  private async queueRequest<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
        } catch (error: any) {
          reject(error);
        }
      });
    });
  }

  /**
   * Process queued requests
   */
  private async startQueueProcessor() {
    setInterval(async () => {
      if (this.requestQueue.length > 0 && this.activeRequests < OLLAMA_CONFIG.performance.parallelRequests) {
        const request = this.requestQueue.shift();
        if (request) {
          request();
        }
      }
    }, 100);
  }

  /**
   * Generate cache key
   */
  private getCacheKey(type: string, input: string, options: any): string {
    return `${type}:${Buffer.from(input).toString('base64').substring(0, 20)}:${JSON.stringify(options)}`;
  }

  /**
   * Get system status
   */
  async getSystemStatus() {
    const isAvailable = await this.isAvailable();
    const models = await this.listModels();
    const hasGemma = await this.checkLegalModel();
    const hasLegalBert = await this.checkLegalBert();
    
    return {
      ollamaAvailable: isAvailable,
      availableModels: this.availableModels,
      primaryModel: hasGemma ? 'gemma3:legal-latest' : 'not available',
      legalFallback: hasLegalBert ? 'legal-bert' : 'not available',
      cacheSize: this.cache.size,
      queueLength: this.requestQueue.length,
      activeRequests: this.activeRequests,
      fallbackChain: {
        legal: getOptimalModel('legal-analysis'),
        general: getOptimalModel('generation'),
        embedding: getOptimalModel('embedding')
      }
    };
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
    this.emit('cache-cleared');
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys())
    };
  }

  /**
   * Cleanup
   */
  destroy() {
    if (this.modelCheckInterval) {
      clearInterval(this.modelCheckInterval);
    }
  }

  /**
   * Embed a document (wrapper for generateEmbedding with document)
   */
  async embedDocument(document: LegalDocument): Promise<number[]> {
    const embeddings = await this.generateEmbeddings(document.content);
    return Array.isArray(embeddings) ? embeddings : [];
  }

  /**
   * Analyze a document (alias for analyzeLegalDocument) 
   */
  async analyzeDocument(document: LegalDocument): Promise<AnalysisResult> {
    return this.analyzeLegalDocument(document);
  }
}

// Export singleton instance
export const ollamaService = new OllamaService();
;
// Export for testing and extension
export default OllamaService;
