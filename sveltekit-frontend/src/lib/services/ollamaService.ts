// @ts-nocheck
import { env } from '$env/dynamic/private';
import type { EmbeddingResponse, GenerateResponse, OllamaModel } from '$lib/types/ollama';

export class OllamaService {
  private baseUrl: string;
  private embedModel = 'nomic-embed-text';
  private llmModel = 'gemma3-legal';

  constructor(baseUrl: string = env.OLLAMA_API_URL || 'http://localhost:11434') {
    this.baseUrl = baseUrl;
  }

  /**
   * Generate embeddings for text using nomic-embed-text
   */
  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.embedModel,
          prompt: text.slice(0, 8192) // Limit context window
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const data: EmbeddingResponse = await response.json();
      return data.embedding;
    } catch (error) {
      console.error('Ollama embedding error:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for multiple texts in batch
   */
  async generateBatchEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings = await Promise.all(
      texts.map(text: any => this.generateEmbedding(text))
    );
    return embeddings;
  }

  /**
   * Generate text completion using gemma3-legal
   */
  async generateCompletion(
    prompt: string,
    options: {
      temperature?: number;
      maxTokens?: number;
      systemPrompt?: string;
      stream?: boolean;
    } = {}
  ): Promise<string> {
    const { temperature = 0.7, maxTokens = 2000, systemPrompt, stream = false } = options;

    try {
      const fullPrompt = systemPrompt 
        ? `System: ${systemPrompt}\n\nUser: ${prompt}\n\nAssistant:`
        : prompt;

      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.llmModel,
          prompt: fullPrompt,
          temperature,
          max_tokens: maxTokens,
          stream
        })
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.statusText}`);
      }

      if (stream) {
        // Handle streaming response
        return this.handleStreamingResponse(response);
      } else {
        const data: GenerateResponse = await response.json();
        return data.response;
      }
    } catch (error) {
      console.error('Ollama generation error:', error);
      throw error;
    }
  }

  /**
   * Handle streaming response from Ollama
   */
  private async handleStreamingResponse(response: Response): Promise<string> {
    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line: any => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            fullText += data.response;
          }
        } catch (e) {
          // Skip invalid JSON lines
        }
      }
    }

    return fullText;
  }

  /**
   * Generate embeddings and metadata for document chunks
   */
  async embedDocument(
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<{
    chunks: Array<{
      content: string;
      embedding: number[];
      metadata: Record<string, any>;
    }>;
  }> {
    // Chunk the document (simple chunking for now, can be improved)
    const chunks = this.chunkText(content, 1000, 200); // 1000 chars with 200 overlap
    
    const embeddings = await this.generateBatchEmbeddings(chunks);
    
    return {
      chunks: chunks.map((chunk, index) => ({
        content: chunk,
        embedding: embeddings[index],
        metadata: {
          ...metadata,
          chunkIndex: index,
          totalChunks: chunks.length
        }
      }))
    };
  }

  /**
   * Simple text chunking with overlap
   */
  private chunkText(text: string, chunkSize: number, overlap: number): string[] {
    const chunks: string[] = [];
    let start = 0;

    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      chunks.push(text.slice(start, end));
      start = end - overlap;
    }

    return chunks;
  }

  /**
   * Analyze document with AI
   */
  async analyzeDocument(
    content: string,
    analysisType: 'summary' | 'entities' | 'sentiment' | 'classification'
  ): Promise<string> {
    const prompts = {
      summary: 'Provide a concise summary of the following document:',
      entities: 'Extract all named entities (people, organizations, locations, dates) from:',
      sentiment: 'Analyze the sentiment and tone of the following text:',
      classification: 'Classify the following document by type and main topics:'
    };

    const systemPrompt = 'You are a legal AI assistant specialized in document analysis.';
    const prompt = `${prompts[analysisType]}\n\n${content}`;

    return this.generateCompletion(prompt, { systemPrompt });
  }

  /**
   * Check if Ollama is running and models are available
   */
  async checkHealth(): Promise<{
    status: 'healthy' | 'unhealthy';
    embedModel: boolean;
    llmModel: boolean;
    models: string[];
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) {
        return { status: 'unhealthy', embedModel: false, llmModel: false, models: [] };
      }

      const data = await response.json();
      const models = data.models?.map((m: OllamaModel) => m.name) || [];
      
      return {
        status: 'healthy',
        embedModel: models.includes(this.embedModel),
        llmModel: models.includes(this.llmModel),
        models
      };
    } catch (error) {
      return { status: 'unhealthy', embedModel: false, llmModel: false, models: [] };
    }
  }

  /**
   * Generate contextual embeddings with enhanced metadata
   */
  async generateContextualEmbedding(
    text: string,
    context: {
      documentType?: string;
      caseId?: string;
      userId?: string;
      timestamp?: Date;
    }
  ): Promise<{
    embedding: number[];
    metadata: Record<string, any>;
  }> {
    // Enhance text with context for better embeddings
    const contextualText = context.documentType 
      ? `[${context.documentType}] ${text}`
      : text;

    const embedding = await this.generateEmbedding(contextualText);
    
    return {
      embedding,
      metadata: {
        ...context,
        textLength: text.length,
        embeddingDimension: embedding.length,
        model: this.embedModel,
        timestamp: context.timestamp || new Date()
      }
    };
  }
}

// Export singleton instance
export const ollamaService = new OllamaService();