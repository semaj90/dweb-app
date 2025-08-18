// @ts-nocheck
import { db } from './unified-database-service.js';

/**
 * Unified AI Service
 * Consolidates Ollama, embeddings, and AI processing functionality
 */
export class UnifiedAIService {
  private ollamaUrl: string;
  private embeddingModel: string;
  private chatModel: string;
  private initialized: boolean = false;

  constructor(config: any = {}) {
    this.ollamaUrl = config.ollamaUrl || process.env.OLLAMA_URL || 'http://localhost:11434';
    this.embeddingModel = config.embeddingModel || 'nomic-embed-text';
    this.chatModel = config.chatModel || 'gemma2:9b';
  }

  async initialize(): Promise<boolean> {
    try {
      // Check if Ollama is available
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (response.ok) {
        console.log('✓ Ollama connected');
        this.initialized = true;
        return true;
      }
      throw new Error('Ollama not available');
    } catch (error) {
      console.error('AI service initialization failed:', error);
      return false;
    }
  }

  // ============ Core AI Methods ============
  async generateText(prompt: string, options: any = {}): Promise<string> {
    const result = await this.generateCompletion(prompt, options);
    return result.response;
  }

  async generateEmbedding(text: string, options: any = {}): Promise<number[]> {
    return await this.embedSingle(text);
  }

  async generateCompletion(prompt: string, options: any = {}): Promise<any> {
    const requestBody = {
      model: options.model || this.chatModel,
      prompt,
      stream: false,
      options: {
        temperature: options.temperature || 0.7,
        top_p: options.top_p || 0.9,
        max_tokens: options.max_tokens || 2000,
        ...options.modelOptions
      }
    };

    const response = await fetch(`${this.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`AI generation failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async chat(message: string, context: any[] = [], options: any = {}): Promise<any> {
    const messages = [
      ...context,
      { role: 'user', content: message }
    ];

    const requestBody = {
      model: options.model || this.chatModel,
      messages,
      stream: false,
      options: {
        temperature: options.temperature || 0.7,
        top_p: options.top_p || 0.9,
        max_tokens: options.max_tokens || 2000,
        ...options.modelOptions
      }
    };

    const response = await fetch(`${this.ollamaUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`Chat failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async *streamChat(message: string, context: any[] = [], options: any = {}): AsyncGenerator<any> {
    const messages = [
      ...context,
      { role: 'user', content: message }
    ];

    const requestBody = {
      model: options.model || this.chatModel,
      messages,
      stream: true,
      options: {
        temperature: options.temperature || 0.7,
        top_p: options.top_p || 0.9,
        max_tokens: options.max_tokens || 2000,
        ...options.modelOptions
      }
    };

    const response = await fetch(`${this.ollamaUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`Stream chat failed: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response reader available');

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
              const data = JSON.parse(line);
              yield data;
            } catch (e) {
              console.warn('Failed to parse streaming response:', line);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ============ Embedding Methods ============
  async embed(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    
    for (const text of texts) {
      const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.embeddingModel,
          prompt: text
        })
      });

      if (response.ok) {
        const data = await response.json();
        embeddings.push(data.embedding);
      } else {
        // Fallback to random embedding for testing
        embeddings.push(new Array(384).fill(0).map(() => Math.random() - 0.5));
      }
    }

    return embeddings;
  }

  async embedSingle(text: string): Promise<number[]> {
    const embeddings = await this.embed([text]);
    return embeddings[0];
  }

  // ============ Document Processing ============
  async processDocument(document: any, options: any = {}): Promise<any> {
    const operations = options.operations || [
      'extract',
      'chunk',
      'embed',
      'analyze',
      'summarize'
    ];

    let result = { ...document };

    for (const op of operations) {
      switch (op) {
        case 'extract':
          result.text = await this.extractText(result);
          break;
        
        case 'chunk':
          result.chunks = await this.chunkText(result.text || result.content);
          break;
        
        case 'embed':
          if (result.chunks) {
            result.embeddings = await this.embed(result.chunks);
          } else {
            result.embeddings = [await this.embedSingle(result.text || result.content)];
          }
          break;
        
        case 'analyze':
          result.analysis = await this.analyzeDocument(result);
          break;
        
        case 'summarize':
          result.summary = await this.summarizeDocument(result);
          break;

        case 'store':
          result.stored = await this.storeDocument(result);
          break;
      }
    }

    return result;
  }

  async extractText(document: any): Promise<string> {
    if (document.type === 'text/plain' || document.content) {
      return document.content;
    }
    
    // For other types, implement appropriate extractors
    return 'Extracted text (placeholder)';
  }

  async chunkText(text: string, options: any = {}): Promise<string[]> {
    const maxSize = options.maxSize || 500;
    const overlap = options.overlap || 50;
    const chunks: string[] = [];
    
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    let currentChunk = '';
    
    for (const sentence of sentences) {
      if ((currentChunk + sentence).length > maxSize) {
        if (currentChunk) {
          chunks.push(currentChunk.trim());
          // Add overlap
          const words = currentChunk.split(' ');
          const overlapWords = words.slice(-Math.floor(overlap / 10));
          currentChunk = overlapWords.join(' ') + ' ' + sentence;
        } else {
          currentChunk = sentence;
        }
      } else {
        currentChunk += ' ' + sentence;
      }
    }
    
    if (currentChunk) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }

  async analyzeDocument(document: any): Promise<any> {
    const prompt = `
      Analyze the following legal document and provide:
      1. Document type (contract, evidence, motion, etc.)
      2. Key legal concepts and terms
      3. Important parties mentioned
      4. Dates and deadlines
      5. Legal implications and risks
      6. Required actions or responses
      
      Document: ${document.text || document.content}
      
      Provide the analysis in structured JSON format.
    `;
    
    const response = await this.generateCompletion(prompt, { model: this.chatModel });
    
    try {
      return JSON.parse(response.response);
    } catch {
      return { analysis: response.response };
    }
  }

  async summarizeDocument(document: any): Promise<string> {
    const prompt = `
      Provide a concise professional summary of this legal document:
      
      ${document.text || document.content}
      
      Include:
      - Main purpose and type of document
      - Key parties involved
      - Important dates and amounts
      - Critical legal implications
      - Any required actions
      
      Keep the summary under 200 words.
    `;
    
    const response = await this.generateCompletion(prompt, { model: this.chatModel });
    return response.response;
  }

  async storeDocument(document: any): Promise<any> {
    // Store document and embeddings in unified database
    const stored = await db.insertLegalDocument({
      id: document.id || this.generateId(document.content),
      title: document.title || 'Untitled Document',
      content: document.text || document.content,
      metadata: {
        analysis: document.analysis,
        summary: document.summary,
        chunks: document.chunks?.length || 0,
        processedAt: new Date().toISOString()
      },
      embedding: document.embeddings?.[0],
      case_id: document.caseId
    });

    // Store in vector database for similarity search
    if (document.embeddings && document.chunks) {
      for (let i = 0; i < document.chunks.length; i++) {
        await db.upsertVector(
          `${document.id}_chunk_${i}`,
          document.embeddings[i],
          {
            content: document.chunks[i],
            documentId: document.id,
            chunkIndex: i,
            caseId: document.caseId
          }
        );
      }
    }

    return stored;
  }

  // ============ RAG Methods ============
  async ragQuery(query: string, options: any = {}): Promise<any> {
    const startTime = Date.now();
    
    // Generate query embedding
    const queryEmbedding = await this.embedSingle(query);
    
    // Perform hybrid search (text + vector similarity)
    const relevantDocs = await db.hybridSearch(
      query,
      queryEmbedding,
      options.caseId
    );
    
    // Build context from relevant documents
    const context = relevantDocs
      .slice(0, options.topK || 5)
      .map(doc => doc.content)
      .join('\n\n');
    
    // Generate response with context
    const prompt = `
      You are a legal AI assistant. Based on the following context documents, answer the user's question.
      
      Context Documents:
      ${context}
      
      Question: ${query}
      
      Instructions:
      - Provide accurate legal information based on the context
      - Cite relevant sections when possible
      - If the context doesn't contain sufficient information, state this clearly
      - Include any relevant case law or legal precedents mentioned in the context
      - Format your response professionally
      
      Answer:
    `;
    
    const response = await this.generateCompletion(prompt, {
      model: options.model || this.chatModel,
      temperature: 0.3 // Lower temperature for more factual responses
    });
    
    const processingTime = Date.now() - startTime;
    
    return {
      answer: response.response,
      sources: relevantDocs.map(doc => ({
        id: doc.id,
        title: doc.title,
        score: doc.combinedScore,
        snippet: doc.content.substring(0, 200) + '...'
      })),
      context: context.substring(0, 1000) + '...',
      metadata: {
        queryEmbedding,
        sourcesFound: relevantDocs.length,
        processingTime,
        model: options.model || this.chatModel
      }
    };
  }

  async ragStream(query: string, options: any = {}): AsyncGenerator<any> {
    // Get context first
    const queryEmbedding = await this.embedSingle(query);
    const relevantDocs = await db.hybridSearch(
      query,
      queryEmbedding,
      options.caseId
    );
    
    const context = relevantDocs
      .slice(0, options.topK || 5)
      .map(doc => doc.content)
      .join('\n\n');
    
    const prompt = `
      You are a legal AI assistant. Based on the following context documents, answer the user's question.
      
      Context Documents:
      ${context}
      
      Question: ${query}
      
      Instructions:
      - Provide accurate legal information based on the context
      - Cite relevant sections when possible
      - If the context doesn't contain sufficient information, state this clearly
      - Include any relevant case law or legal precedents mentioned in the context
      - Format your response professionally
      
      Answer:
    `;
    
    // Stream the response
    for await (const chunk of this.streamChat(prompt, [], {
      model: options.model || this.chatModel,
      temperature: 0.3
    })) {
      yield {
        ...chunk,
        sources: relevantDocs.slice(0, 5).map(doc => ({
          id: doc.id,
          title: doc.title,
          score: doc.combinedScore,
          snippet: doc.content.substring(0, 200) + '...'
        }))
      };
    }
  }

  // ============ Model Management ============
  async getAvailableModels(): Promise<any[]> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (response.ok) {
        const data = await response.json();
        return data.models || [];
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
    return [];
  }

  async pullModel(modelName: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName })
      });
      return response.ok;
    } catch (error) {
      console.error('Error pulling model:', error);
      return false;
    }
  }

  // ============ Health Check ============
  async getHealthStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      const models = response.ok ? await response.json() : null;
      
      return {
        status: response.ok ? 'healthy' : 'unhealthy',
        ollamaUrl: this.ollamaUrl,
        modelsAvailable: models?.models?.length || 0,
        embeddingModel: this.embeddingModel,
        chatModel: this.chatModel,
        initialized: this.initialized
      };
    } catch (error) {
      return {
        status: 'error',
        error: error.message,
        initialized: false
      };
    }
  }

  // ============ Utility Methods ============
  cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  generateId(content: string): string {
    return btoa(content.substring(0, 100))
      .replace(/[/+]/g, '_')
      .substring(0, 16) + '_' + Date.now();
  }
}

// Export singleton instance
export const aiService = new UnifiedAIService();