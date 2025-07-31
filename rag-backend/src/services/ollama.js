/**
 * Ollama Service for Enhanced RAG
 * Handles local LLM integration with Ollama
 */

import fetch from 'node-fetch';

export class OllamaService {
  constructor(config) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:11434',
      defaultModel: config.defaultModel || 'gemma2:9b',
      embeddingModel: config.embeddingModel || 'nomic-embed-text',
      timeout: config.timeout || 30000,
      ...config
    };
    this.isHealthy = false;
    this.availableModels = [];
  }

  async initialize() {
    try {
      // Check Ollama health
      const healthCheck = await this.healthCheck();
      if (!healthCheck) {
        throw new Error('Ollama service is not healthy');
      }

      // Load available models
      await this.loadAvailableModels();

      // Verify required models are available
      await this.verifyRequiredModels();

      console.log('✅ Ollama service initialized');
      console.log(`📋 Available models: ${this.availableModels.map(m => m.name).join(', ')}`);
      return true;
    } catch (error) {
      console.error('Failed to initialize Ollama service:', error);
      return false;
    }
  }

  /**
   * Health check for Ollama service
   */
  async healthCheck() {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`, {
        method: 'GET',
        timeout: 5000
      });

      this.isHealthy = response.ok;
      return this.isHealthy;
    } catch (error) {
      console.error('Ollama health check failed:', error);
      this.isHealthy = false;
      return false;
    }
  }

  /**
   * Load available models from Ollama
   */
  async loadAvailableModels() {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Failed to load models: ${response.statusText}`);
      }

      const data = await response.json();
      this.availableModels = data.models || [];
      
      return this.availableModels;
    } catch (error) {
      console.error('Failed to load available models:', error);
      this.availableModels = [];
      return [];
    }
  }

  /**
   * Verify required models are available
   */
  async verifyRequiredModels() {
    const requiredModels = [this.config.defaultModel, this.config.embeddingModel];
    const availableModelNames = this.availableModels.map(m => m.name);
    
    const missingModels = requiredModels.filter(model => 
      !availableModelNames.some(available => available.includes(model.split(':')[0]))
    );

    if (missingModels.length > 0) {
      console.warn(`⚠️ Missing models: ${missingModels.join(', ')}`);
      console.log('💡 Pull missing models with: ollama pull <model-name>');
    }

    return missingModels.length === 0;
  }

  /**
   * Generate completion using Ollama
   */
  async generateCompletion(prompt, options = {}) {
    const {
      model = this.config.defaultModel,
      temperature = 0.1,
      maxTokens = 2048,
      stream = false,
      context = [],
      system = null
    } = options;

    try {
      const payload = {
        model,
        prompt,
        options: {
          temperature,
          num_predict: maxTokens,
          stop: options.stop || [],
        },
        stream,
        context,
        system
      };

      const response = await fetch(`${this.config.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        timeout: this.config.timeout
      });

      if (!response.ok) {
        throw new Error(`Ollama generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      return {
        content: result.response,
        model: result.model,
        context: result.context,
        created_at: result.created_at,
        done: result.done,
        total_duration: result.total_duration,
        load_duration: result.load_duration,
        prompt_eval_count: result.prompt_eval_count,
        prompt_eval_duration: result.prompt_eval_duration,
        eval_count: result.eval_count,
        eval_duration: result.eval_duration
      };
    } catch (error) {
      console.error('Completion generation failed:', error);
      throw error;
    }
  }

  /**
   * Stream completion using Ollama
   */
  async *streamCompletion(prompt, options = {}) {
    const {
      model = this.config.defaultModel,
      temperature = 0.1,
      maxTokens = 2048,
      context = [],
      system = null
    } = options;

    try {
      const payload = {
        model,
        prompt,
        options: {
          temperature,
          num_predict: maxTokens,
        },
        stream: true,
        context,
        system
      };

      const response = await fetch(`${this.config.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        timeout: this.config.timeout
      });

      if (!response.ok) {
        throw new Error(`Ollama streaming failed: ${response.statusText}`);
      }

      const reader = response.body;
      let buffer = '';

      for await (const chunk of reader) {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              yield {
                content: data.response || '',
                done: data.done || false,
                context: data.context,
                model: data.model
              };

              if (data.done) {
                return;
              }
            } catch (error) {
              console.error('Error parsing stream chunk:', error);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream completion failed:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings using Ollama
   */
  async generateEmbedding(text, model = null) {
    const embeddingModel = model || this.config.embeddingModel;

    try {
      const response = await fetch(`${this.config.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: embeddingModel,
          prompt: text
        }),
        timeout: this.config.timeout
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.embedding;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw error;
    }
  }

  /**
   * Generate multiple embeddings
   */
  async generateEmbeddings(texts, model = null) {
    const embeddings = [];
    
    for (const text of texts) {
      try {
        const embedding = await this.generateEmbedding(text, model);
        embeddings.push(embedding);
      } catch (error) {
        console.error(`Failed to generate embedding for text: ${text.substring(0, 50)}...`);
        embeddings.push(null);
      }
    }

    return embeddings;
  }

  /**
   * Chat completion with conversation context
   */
  async chatCompletion(messages, options = {}) {
    const {
      model = this.config.defaultModel,
      temperature = 0.1,
      maxTokens = 2048,
      stream = false
    } = options;

    try {
      // Convert messages to Ollama format
      const prompt = this.formatMessagesForOllama(messages);
      
      return await this.generateCompletion(prompt, {
        model,
        temperature,
        maxTokens,
        stream
      });
    } catch (error) {
      console.error('Chat completion failed:', error);
      throw error;
    }
  }

  /**
   * Format messages for Ollama prompt
   */
  formatMessagesForOllama(messages) {
    return messages.map(msg => {
      const role = msg.role === 'assistant' ? 'Assistant' : 'User';
      return `${role}: ${msg.content}`;
    }).join('\n\n') + '\n\nAssistant:';
  }

  /**
   * Legal document analysis using specialized prompt
   */
  async analyzeLegalDocument(documentText, analysisType = 'general', options = {}) {
    const {
      model = this.config.defaultModel,
      maxTokens = 2048
    } = options;

    const systemPrompt = this.getLegalAnalysisPrompt(analysisType);
    
    const prompt = `${systemPrompt}

Document to analyze:
${documentText}

Analysis:`;

    try {
      const result = await this.generateCompletion(prompt, {
        model,
        maxTokens,
        temperature: 0.1
      });

      return {
        analysis: result.content,
        analysisType,
        processingTime: result.total_duration,
        model: result.model
      };
    } catch (error) {
      console.error('Legal document analysis failed:', error);
      throw error;
    }
  }

  /**
   * Get legal analysis prompt based on type
   */
  getLegalAnalysisPrompt(analysisType) {
    const prompts = {
      general: 'You are a legal AI assistant. Analyze the following document and provide key insights, important clauses, and potential legal issues.',
      contract: 'You are a contract analysis expert. Identify key terms, obligations, risks, and important dates in this contract.',
      evidence: 'You are a legal evidence analyst. Examine this evidence document and identify relevant facts, potential legal implications, and connections to legal proceedings.',
      compliance: 'You are a compliance expert. Review this document for regulatory compliance issues, potential violations, and recommendations.',
      litigation: 'You are a litigation support specialist. Analyze this document for relevant facts, legal theories, and evidence that could be important in litigation.'
    };

    return prompts[analysisType] || prompts.general;
  }

  /**
   * Summarize long text using Ollama
   */
  async summarizeText(text, options = {}) {
    const {
      model = this.config.defaultModel,
      maxTokens = 1024,
      summaryLength = 'medium'
    } = options;

    const lengthInstructions = {
      short: 'Provide a brief 2-3 sentence summary.',
      medium: 'Provide a comprehensive paragraph summary.',
      long: 'Provide a detailed multi-paragraph summary.'
    };

    const instruction = lengthInstructions[summaryLength] || lengthInstructions.medium;

    const prompt = `Please summarize the following text. ${instruction}

Text to summarize:
${text}

Summary:`;

    try {
      const result = await this.generateCompletion(prompt, {
        model,
        maxTokens,
        temperature: 0.1
      });

      return {
        summary: result.content,
        originalLength: text.length,
        summaryLength: result.content.length,
        compressionRatio: (result.content.length / text.length * 100).toFixed(1) + '%',
        model: result.model
      };
    } catch (error) {
      console.error('Text summarization failed:', error);
      throw error;
    }
  }

  /**
   * Extract key information from document
   */
  async extractKeyInfo(documentText, extractionType = 'entities', options = {}) {
    const {
      model = this.config.defaultModel,
      maxTokens = 1024
    } = options;

    const extractionPrompts = {
      entities: 'Extract all named entities (people, organizations, locations, dates) from this document.',
      dates: 'Extract all important dates and deadlines from this document.',
      amounts: 'Extract all monetary amounts, quantities, and measurements from this document.',
      contacts: 'Extract all contact information (names, phone numbers, emails, addresses) from this document.',
      keywords: 'Extract the most important keywords and key phrases from this document.'
    };

    const instruction = extractionPrompts[extractionType] || extractionPrompts.entities;

    const prompt = `${instruction} Format the output as a structured list.

Document:
${documentText}

Extracted information:`;

    try {
      const result = await this.generateCompletion(prompt, {
        model,
        maxTokens,
        temperature: 0.1
      });

      return {
        extractedInfo: result.content,
        extractionType,
        model: result.model
      };
    } catch (error) {
      console.error('Key information extraction failed:', error);
      throw error;
    }
  }

  /**
   * Get model information
   */
  async getModelInfo(modelName) {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/show`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName })
      });

      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get model info:', error);
      return null;
    }
  }

  /**
   * Pull model from Ollama registry
   */
  async pullModel(modelName) {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName }),
        timeout: 300000 // 5 minutes timeout for model pulling
      });

      if (!response.ok) {
        throw new Error(`Failed to pull model: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to pull model:', error);
      throw error;
    }
  }

  /**
   * Get service statistics
   */
  async getStats() {
    try {
      await this.loadAvailableModels();
      
      return {
        isHealthy: this.isHealthy,
        baseUrl: this.config.baseUrl,
        defaultModel: this.config.defaultModel,
        embeddingModel: this.config.embeddingModel,
        availableModels: this.availableModels.length,
        models: this.availableModels.map(m => ({
          name: m.name,
          size: m.size,
          modified_at: m.modified_at
        }))
      };
    } catch (error) {
      console.error('Failed to get Ollama stats:', error);
      return {
        isHealthy: false,
        error: error.message
      };
    }
  }
}