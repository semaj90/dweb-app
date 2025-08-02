// ======================================================================
// ENHANCED RAG SERVICE - Production Ready
// Integrates vector search, semantic analysis, and local LLM processing
// ======================================================================

import type { Database, API } from '$lib/types';

interface RAGConfig {
  vectorStoreUrl: string;
  embeddingModel: string;
  retrievalLimit: number;
  similarityThreshold: number;
  chunkSize: number;
  chunkOverlap: number;
}

interface EmbeddingResult {
  vector: number[];
  model: string;
  tokens: number;
}

interface RetrievalResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  source: string;
}

interface RAGResponse {
  answer: string;
  sources: RetrievalResult[];
  confidence: number;
  processingTime: number;
  model: string;
  reasoning?: {
    queryIntent: string;
    retrievedContext: string[];
    synthesisStrategy: string;
  };
}

class EnhancedRAGService {
  private config: RAGConfig;
  private initialized = false;
  private vectorClient: any = null;
  private embeddingCache = new Map<string, EmbeddingResult>();
  
  constructor(config?: Partial<RAGConfig>) {
    this.config = {
      vectorStoreUrl: 'http://localhost:6333',
      embeddingModel: 'nomic-embed-text',
      retrievalLimit: 10,
      similarityThreshold: 0.7,
      chunkSize: 1000,
      chunkOverlap: 200,
      ...config
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Initialize vector store connection (Qdrant)
      await this.initializeVectorStore();
      
      // Test embedding service
      await this.testEmbeddingService();
      
      this.initialized = true;
      console.log('✅ Enhanced RAG Service initialized successfully');
      
    } catch (error) {
      console.warn('⚠️ RAG Service initialization failed:', error);
      // Continue without RAG capabilities
      this.initialized = false;
    }
  }

  private async initializeVectorStore(): Promise<void> {
    try {
      // Test Qdrant connection
      const response = await fetch(`${this.config.vectorStoreUrl}/health`, {
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error('Qdrant not available');
      }
      
      // Initialize collections if needed
      await this.ensureCollections();
      
    } catch (error) {
      // Fallback to in-memory storage for development
      console.warn('Vector store not available, using fallback mode');
      this.vectorClient = new InMemoryVectorStore();
    }
  }

  private async ensureCollections(): Promise<void> {
    const collections = ['legal-documents', 'case-files', 'evidence-chunks'];
    
    for (const collection of collections) {
      try {
        await fetch(`${this.config.vectorStoreUrl}/collections/${collection}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            vectors: {
              size: 768, // nomic-embed-text dimensions
              distance: 'Cosine'
            }
          })
        });
      } catch (error) {
        console.warn(`Failed to create collection ${collection}:`, error);
      }
    }
  }

  private async testEmbeddingService(): Promise<void> {
    try {
      await this.generateEmbedding('test');
    } catch (error) {
      throw new Error('Embedding service not available');
    }
  }

  // Main RAG query method
  async query(
    userQuery: string, 
    context?: {
      caseId?: string;
      userId?: string;
      documentTypes?: string[];
    }
  ): Promise<RAGResponse> {
    const startTime = Date.now();

    try {
      if (!this.initialized) {
        await this.initialize();
      }

      // 1. Generate query embedding
      const queryEmbedding = await this.generateEmbedding(userQuery);
      
      // 2. Retrieve relevant documents
      const retrievalResults = await this.retrieveDocuments(
        queryEmbedding.vector,
        context
      );
      
      // 3. Re-rank documents (if multiple results)
      const rankedResults = await this.rerankDocuments(userQuery, retrievalResults);
      
      // 4. Generate response using retrieved context
      const response = await this.generateResponse(userQuery, rankedResults);
      
      const processingTime = Date.now() - startTime;
      
      return {
        answer: response.answer,
        sources: rankedResults.slice(0, 5), // Top 5 sources
        confidence: response.confidence,
        processingTime,
        model: response.model,
        reasoning: response.reasoning
      };
      
    } catch (error) {
      console.error('RAG query failed:', error);
      
      // Fallback to direct LLM without context
      const fallbackResponse = await this.generateFallbackResponse(userQuery);
      
      return {
        answer: fallbackResponse,
        sources: [],
        confidence: 0.5,
        processingTime: Date.now() - startTime,
        model: 'fallback',
        reasoning: {
          queryIntent: 'Unable to retrieve context',
          retrievedContext: [],
          synthesisStrategy: 'Direct LLM response without RAG'
        }
      };
    }
  }

  // Document indexing for RAG
  async indexDocument(document: {
    id: string;
    title: string;
    content: string;
    metadata?: Record<string, any>;
    type?: string;
  }): Promise<{ success: boolean; chunks: number }> {
    try {
      // 1. Chunk the document
      const chunks = await this.chunkDocument(document.content);
      
      // 2. Generate embeddings for each chunk
      const embeddingPromises = chunks.map(async (chunk, index) => {
        const embedding = await this.generateEmbedding(chunk);
        
        return {
          id: `${document.id}_chunk_${index}`,
          vector: embedding.vector,
          payload: {
            content: chunk,
            title: document.title,
            document_id: document.id,
            chunk_index: index,
            type: document.type || 'document',
            ...document.metadata
          }
        };
      });
      
      const embeddedChunks = await Promise.all(embeddingPromises);
      
      // 3. Store in vector database
      await this.storeEmbeddings(embeddedChunks, document.type || 'legal-documents');
      
      return { success: true, chunks: chunks.length };
      
    } catch (error) {
      console.error('Document indexing failed:', error);
      return { success: false, chunks: 0 };
    }
  }

  private async generateEmbedding(text: string): Promise<EmbeddingResult> {
    // Check cache first
    const cacheKey = this.hashString(text);
    if (this.embeddingCache.has(cacheKey)) {
      return this.embeddingCache.get(cacheKey)!;
    }

    try {
      // Call Ollama embedding API
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.embeddingModel,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding API error: ${response.status}`);
      }

      const data = await response.json();
      
      const result: EmbeddingResult = {
        vector: data.embedding,
        model: this.config.embeddingModel,
        tokens: text.split(' ').length // Approximate
      };

      // Cache the result
      this.embeddingCache.set(cacheKey, result);
      
      return result;
      
    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw error;
    }
  }

  private async retrieveDocuments(
    queryVector: number[],
    context?: { caseId?: string; documentTypes?: string[] }
  ): Promise<RetrievalResult[]> {
    try {
      // Build search filter
      const filter: any = {};
      if (context?.caseId) {
        filter.case_id = context.caseId;
      }
      if (context?.documentTypes?.length) {
        filter.type = { $in: context.documentTypes };
      }

      // Search vector store
      const searchResponse = await fetch(`${this.config.vectorStoreUrl}/collections/legal-documents/points/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: queryVector,
          limit: this.config.retrievalLimit,
          score_threshold: this.config.similarityThreshold,
          with_payload: true,
          filter
        })
      });

      if (!searchResponse.ok) {
        throw new Error('Vector search failed');
      }

      const searchResults = await searchResponse.json();
      
      return searchResults.result?.map((item: any) => ({
        id: item.id,
        content: item.payload.content,
        score: item.score,
        metadata: item.payload,
        source: item.payload.title || 'Unknown'
      })) || [];
      
    } catch (error) {
      console.error('Document retrieval failed:', error);
      return [];
    }
  }

  private async rerankDocuments(
    query: string,
    documents: RetrievalResult[]
  ): Promise<RetrievalResult[]> {
    if (documents.length <= 1) return documents;

    try {
      // Use LLM to rerank documents based on relevance
      const rerankingPrompt = `
        Query: "${query}"
        
        Rank these documents by relevance (1 = most relevant):
        
        ${documents.map((doc, i) => 
          `${i + 1}. ${doc.source}: ${doc.content.substring(0, 200)}...`
        ).join('\n')}
        
        Respond with only the numbers in order of relevance (e.g., "3,1,2"):
      `;

      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: rerankingPrompt,
          stream: false,
          options: { max_tokens: 50, temperature: 0.1 }
        })
      });

      if (response.ok) {
        const data = await response.json();
        const ranking = data.response?.match(/\d+/g)?.map((n: string) => parseInt(n) - 1);
        
        if (ranking && ranking.length === documents.length) {
          return ranking.map(i => documents[i]).filter(Boolean);
        }
      }
    } catch (error) {
      console.warn('Reranking failed, using original order:', error);
    }

    return documents;
  }

  private async generateResponse(
    query: string,
    sources: RetrievalResult[]
  ): Promise<{
    answer: string;
    confidence: number;
    model: string;
    reasoning?: any;
  }> {
    const context = sources.map(s => `Source: ${s.source}\n${s.content}`).join('\n\n');
    
    const prompt = `You are a legal AI assistant. Answer the user's question based on the provided context.

Context:
${context}

User Question: ${query}

Instructions:
1. Answer based on the provided context
2. Cite specific sources when relevant
3. If the context doesn't contain enough information, state this clearly
4. Provide accurate legal information but remind users to consult legal professionals

Answer:`;

    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt,
          stream: false,
          options: {
            temperature: 0.3,
            max_tokens: 1024,
            top_p: 0.9
          }
        })
      });

      if (!response.ok) {
        throw new Error('LLM generation failed');
      }

      const data = await response.json();
      
      return {
        answer: data.response || 'Unable to generate response',
        confidence: this.calculateConfidence(sources, data.response),
        model: 'gemma3-legal',
        reasoning: {
          queryIntent: this.analyzeQueryIntent(query),
          retrievedContext: sources.map(s => s.source),
          synthesisStrategy: 'Context-aware legal response with source citation'
        }
      };
      
    } catch (error) {
      console.error('Response generation failed:', error);
      throw error;
    }
  }

  private async generateFallbackResponse(query: string): Promise<string> {
    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: `As a legal AI assistant, please answer this question: ${query}
          
          Note: I don't have access to specific case documents right now, so I'll provide general legal information. Please consult with a legal professional for specific advice.`,
          stream: false,
          options: { temperature: 0.5, max_tokens: 512 }
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.response || 'I apologize, but I cannot process your request right now.';
      }
    } catch (error) {
      console.error('Fallback response failed:', error);
    }

    return 'I apologize, but the AI service is currently unavailable. Please try again later.';
  }

  // Utility methods
  private async chunkDocument(content: string): Promise<string[]> {
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const chunks: string[] = [];
    let currentChunk = '';

    for (const sentence of sentences) {
      if (currentChunk.length + sentence.length > this.config.chunkSize) {
        if (currentChunk) {
          chunks.push(currentChunk.trim());
          currentChunk = sentence;
        }
      } else {
        currentChunk += sentence + '. ';
      }
    }

    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }

    return chunks;
  }

  private async storeEmbeddings(embeddings: any[], collection: string): Promise<void> {
    try {
      await fetch(`${this.config.vectorStoreUrl}/collections/${collection}/points`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points: embeddings
        })
      });
    } catch (error) {
      console.error('Failed to store embeddings:', error);
      throw error;
    }
  }

  private calculateConfidence(sources: RetrievalResult[], response: string): number {
    if (!sources.length) return 0.3;
    
    const avgScore = sources.reduce((sum, s) => sum + s.score, 0) / sources.length;
    const responseLength = response.length;
    const lengthFactor = Math.min(responseLength / 200, 1); // Prefer longer responses
    
    return Math.min(avgScore * lengthFactor, 0.95);
  }

  private analyzeQueryIntent(query: string): string {
    const legalKeywords = ['case', 'law', 'statute', 'precedent', 'evidence', 'court'];
    const foundKeywords = legalKeywords.filter(keyword => 
      query.toLowerCase().includes(keyword)
    );
    
    if (foundKeywords.length > 0) {
      return `Legal inquiry about: ${foundKeywords.join(', ')}`;
    }
    
    return 'General legal question';
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }

  // Health check
  async healthCheck(): Promise<{ status: string; details: any }> {
    const checks = {
      initialized: this.initialized,
      vectorStore: false,
      embedding: false,
      llm: false
    };

    try {
      // Check vector store
      const vectorResponse = await fetch(`${this.config.vectorStoreUrl}/health`, {
        signal: AbortSignal.timeout(3000)
      });
      checks.vectorStore = vectorResponse.ok;
    } catch {}

    try {
      // Check embedding service
      await this.generateEmbedding('test');
      checks.embedding = true;
    } catch {}

    try {
      // Check LLM
      const llmResponse = await fetch('http://localhost:11434/api/tags', {
        signal: AbortSignal.timeout(3000)
      });
      checks.llm = llmResponse.ok;
    } catch {}

    const status = Object.values(checks).every(Boolean) ? 'healthy' : 'degraded';
    
    return { status, details: checks };
  }
}

// Fallback in-memory vector store for development
class InMemoryVectorStore {
  private documents: Array<{ id: string; vector: number[]; payload: any }> = [];

  async search(vector: number[], limit: number, threshold: number) {
    const similarities = this.documents.map(doc => ({
      ...doc,
      score: this.cosineSimilarity(vector, doc.vector)
    }));

    return similarities
      .filter(item => item.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  async add(points: Array<{ id: string; vector: number[]; payload: any }>) {
    this.documents.push(...points);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (normA * normB);
  }
}

// Export singleton instance
export const enhancedRAGService = new EnhancedRAGService();
