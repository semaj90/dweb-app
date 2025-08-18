// Enhanced RAG Service for YoRHa Interface
import { EnhancedVectorService } from './vector.service';
import { db } from '../db';
import { documents, documentChunks, conversations, knowledgeBase } from '../db/schema';
import { eq, and, gte, sql } from 'drizzle-orm';
import { QueueService } from './queue.service';

interface RAGContext {
  query: string;
  userId?: string;
  conversationId?: string;
  previousMessages?: Array<{ role: string; content: string }>;
  filters?: any;
  maxTokens?: number;
}

interface RAGResponse {
  answer: string;
  sources: Array<{
    documentId: string;
    title: string;
    chunk: string;
    score: number;
  }>;
  confidence: number;
  metadata?: any;
}

export class EnhancedRAGService {
  private vectorService: EnhancedVectorService;
  private queueService: QueueService;
  
  // RAG configuration
  private config = {
    maxContextLength: 4096,
    maxSourceChunks: 5,
    minConfidenceScore: 0.7,
    includeSystemPrompt: true,
    temperature: 0.7,
    rerankResults: true
  };

  constructor() {
    this.vectorService = new EnhancedVectorService();
    this.queueService = new QueueService();
  }

  // Main RAG query processing
  async query(context: RAGContext): Promise<RAGResponse> {
    try {
      console.log('Processing RAG query:', context.query);

      // Step 1: Retrieve relevant documents
      const retrievedChunks = await this.retrieve(context);
      
      if (retrievedChunks.length === 0) {
        return {
          answer: 'I couldn\'t find relevant information to answer your question.',
          sources: [],
          confidence: 0,
          metadata: { noResults: true }
        };
      }

      // Step 2: Build context from retrieved chunks
      const augmentedContext = await this.buildContext(context, retrievedChunks);

      // Step 3: Generate response
      const response = await this.generate(augmentedContext, context);

      // Step 4: Post-process and validate
      const finalResponse = await this.postProcess(response, retrievedChunks);

      // Step 5: Store interaction for learning
      await this.storeInteraction(context, finalResponse);

      return finalResponse;
    } catch (error) {
      console.error('RAG query failed:', error);
      throw error;
    }
  }

  // Retrieve relevant documents using hybrid search
  private async retrieve(context: RAGContext): Promise<any[]> {
    try {
      // Perform hybrid search
      const searchResults = await this.vectorService.hybridSearch(context.query, {
        collection: 'yorha_documents_v2',
        limit: this.config.maxSourceChunks * 2,
        filter: context.filters,
        scoreThreshold: this.config.minConfidenceScore,
        rerank: this.config.rerankResults,
        searchType: 'hybrid'
      });

      // If we have conversation history, also search conversations
      if (context.conversationId) {
        const conversationResults = await this.searchConversationHistory(
          context.conversationId,
          context.query
        );
        searchResults.push(...conversationResults);
      }

      // Filter and rank results
      const rankedResults = this.rankResults(searchResults, context);

      // Get top chunks with their documents
      const topChunks = await this.enrichChunksWithDocuments(
        rankedResults.slice(0, this.config.maxSourceChunks)
      );

      return topChunks;
    } catch (error) {
      console.error('Document retrieval failed:', error);
      return [];
    }
  }

  // Build augmented context for generation
  private async buildContext(context: RAGContext, retrievedChunks: any[]): Promise<string> {
    let augmentedContext = '';

    // Add system prompt if configured
    if (this.config.includeSystemPrompt) {
      augmentedContext += this.getSystemPrompt(context) + '\n\n';
    }

    // Add conversation history if available
    if (context.previousMessages && context.previousMessages.length > 0) {
      augmentedContext += 'Previous conversation:\n';
      const recentMessages = context.previousMessages.slice(-5); // Last 5 messages
      for (const msg of recentMessages) {
        augmentedContext += `${msg.role}: ${msg.content}\n`;
      }
      augmentedContext += '\n';
    }

    // Add retrieved context
    augmentedContext += 'Relevant information:\n';
    for (const chunk of retrievedChunks) {
      augmentedContext += `[Source: ${chunk.title || 'Document'}]\n`;
      augmentedContext += `${chunk.content}\n\n`;
    }

    // Add the current query
    augmentedContext += `\nUser Query: ${context.query}\n`;
    augmentedContext += '\nBased on the provided information, please answer the query accurately and comprehensively.';

    // Trim to max context length
    if (augmentedContext.length > this.config.maxContextLength) {
      augmentedContext = this.trimContext(augmentedContext);
    }

    return augmentedContext;
  }

  // Generate response using the augmented context
  private async generate(augmentedContext: string, context: RAGContext): Promise<any> {
    try {
      // Here you would typically call your LLM API
      // For now, we'll create a structured response based on the context
      
      // Simulate LLM response generation
      const response = {
        answer: await this.generateAnswerFromContext(augmentedContext, context),
        reasoning: 'Generated based on retrieved documents',
        confidence: 0.85
      };

      return response;
    } catch (error) {
      console.error('Generation failed:', error);
      throw error;
    }
  }

  // Generate answer from context (placeholder for LLM integration)
  private async generateAnswerFromContext(augmentedContext: string, context: RAGContext): Promise<string> {
    // This is where you would integrate with your LLM
    // For now, we'll extract key information from the context
    
    const chunks = augmentedContext.split('[Source:');
    const relevantInfo = chunks.slice(1).map(chunk => {
      const lines = chunk.split('\n').filter(line => line.trim());
      return lines.slice(1, 3).join(' '); // Get first few lines of each chunk
    });

    // Create a basic answer
    const answer = `Based on the available information: ${relevantInfo.join(' ')}`;
    
    return answer;
  }

  // Post-process the generated response
  private async postProcess(
    response: any,
    retrievedChunks: any[]
  ): Promise<RAGResponse> {
    // Extract sources from retrieved chunks
    const sources = retrievedChunks.map(chunk => ({
      documentId: chunk.documentId || chunk.id,
      title: chunk.title || 'Unknown',
      chunk: chunk.content.substring(0, 200) + '...',
      score: chunk.score || 0
    }));

    // Calculate final confidence
    const avgScore = sources.reduce((acc, s) => acc + s.score, 0) / sources.length;
    const confidence = Math.min(response.confidence || avgScore, 1);

    return {
      answer: response.answer,
      sources,
      confidence,
      metadata: {
        generationTime: Date.now(),
        model: 'enhanced-rag-v1',
        chunksUsed: retrievedChunks.length
      }
    };
  }

  // Store interaction for continuous learning
  private async storeInteraction(context: RAGContext, response: RAGResponse): Promise<void> {
    try {
      // Store in conversation history
      if (context.conversationId) {
        const messages = [
          ...(context.previousMessages || []),
          { role: 'user', content: context.query },
          { role: 'assistant', content: response.answer }
        ];

        await this.vectorService.storeConversation(
          context.conversationId,
          messages,
          {
            userId: context.userId,
            confidence: response.confidence,
            sourcesUsed: response.sources.length
          }
        );
      }

      // Queue for analytics
      await this.queueService.publishMessage('analytics', {
        event: 'rag_query',
        userId: context.userId,
        query: context.query,
        confidence: response.confidence,
        sourcesCount: response.sources.length,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to store interaction:', error);
      // Don't throw - this is non-critical
    }
  }

  // Search conversation history
  private async searchConversationHistory(
    conversationId: string,
    query: string
  ): Promise<any[]> {
    try {
      const results = await this.vectorService.semanticSearch(
        query,
        'yorha_conversations_v2',
        {
          limit: 3,
          filters: {
            must: [
              { key: 'conversationId', match: { value: conversationId } }
            ]
          }
        }
      );

      return results.map(r => ({
        ...r,
        content: r.payload.messages.map((m: any) => m.content).join('\n'),
        title: 'Previous Conversation',
        score: r.score * 0.8 // Slightly lower weight for conversation history
      }));
    } catch (error) {
      console.error('Failed to search conversation history:', error);
      return [];
    }
  }

  // Rank results based on multiple factors
  private rankResults(results: any[], context: RAGContext): any[] {
    return results.sort((a, b) => {
      // Calculate composite score
      const scoreA = this.calculateCompositeScore(a, context);
      const scoreB = this.calculateCompositeScore(b, context);
      return scoreB - scoreA;
    });
  }

  // Calculate composite score for ranking
  private calculateCompositeScore(result: any, context: RAGContext): number {
    let score = result.score || 0;

    // Boost recent documents
    if (result.payload?.timestamp) {
      const age = Date.now() - new Date(result.payload.timestamp).getTime();
      const dayInMs = 24 * 60 * 60 * 1000;
      if (age < dayInMs) score *= 1.2;
      else if (age < 7 * dayInMs) score *= 1.1;
    }

    // Boost if from same user
    if (result.payload?.userId === context.userId) {
      score *= 1.15;
    }

    // Boost if matches conversation context
    if (context.conversationId && result.payload?.conversationId === context.conversationId) {
      score *= 1.1;
    }

    return score;
  }

  // Enrich chunks with document metadata
  private async enrichChunksWithDocuments(chunks: any[]): Promise<any[]> {
    const enriched = [];

    for (const chunk of chunks) {
      try {
        // Get document metadata if available
        if (chunk.payload?.documentId) {
          const doc = await db.query.documents.findFirst({
            where: eq(documents.id, chunk.payload.documentId)
          });

          enriched.push({
            ...chunk,
            documentId: chunk.payload.documentId,
            title: doc?.title || 'Unknown Document',
            content: chunk.payload?.chunkText || chunk.payload?.content || '',
            metadata: {
              ...chunk.payload,
              documentTitle: doc?.title,
              documentSource: doc?.source
            }
          });
        } else {
          enriched.push({
            ...chunk,
            content: chunk.payload?.chunkText || chunk.payload?.content || '',
            title: 'Unknown Source'
          });
        }
      } catch (error) {
        enriched.push(chunk);
      }
    }

    return enriched;
  }

  // Trim context to fit within limits
  private trimContext(context: string): string {
    const lines = context.split('\n');
    let trimmed = '';
    let length = 0;

    for (const line of lines) {
      if (length + line.length > this.config.maxContextLength - 500) {
        break;
      }
      trimmed += line + '\n';
      length += line.length + 1;
    }

    return trimmed + '\n[Context trimmed for length]';
  }

  // Get system prompt based on context
  private getSystemPrompt(context: RAGContext): string {
    let prompt = 'You are YoRHa Assistant, an AI system designed to help users with information retrieval and question answering.';
    
    if (context.userId) {
      prompt += ' You are assisting a registered YoRHa unit.';
    }

    prompt += ' Please provide accurate, helpful, and concise answers based on the provided context.';
    prompt += ' If the information is not available in the context, clearly state that.';
    prompt += ' Always cite your sources when providing information.';

    return prompt;
  }

  // Index a new document
  async indexDocument(
    title: string,
    content: string,
    source: string,
    metadata: any = {}
  ): Promise<string> {
    try {
      const documentId = `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Store document with chunking
      await this.vectorService.storeDocument(
        documentId,
        content,
        {
          title,
          source,
          ...metadata,
          indexedAt: new Date().toISOString()
        }
      );

      // Queue for processing
      await this.queueService.publishMessage('vectorProcessing', {
        type: 'document',
        documentId,
        action: 'index_complete'
      });

      console.log(`Document indexed: ${title} (${documentId})`);
      return documentId;
    } catch (error) {
      console.error('Failed to index document:', error);
      throw error;
    }
  }

  // Update document
  async updateDocument(
    documentId: string,
    content: string,
    metadata: any = {}
  ): Promise<void> {
    try {
      // Delete old chunks
      await db.delete(documentChunks)
        .where(eq(documentChunks.documentId, documentId));

      // Delete old embeddings
      await this.vectorService.deleteEmbedding('documents', documentId);

      // Re-index document
      await this.vectorService.storeDocument(
        documentId,
        content,
        metadata
      );

      console.log(`Document updated: ${documentId}`);
    } catch (error) {
      console.error('Failed to update document:', error);
      throw error;
    }
  }

  // Delete document
  async deleteDocument(documentId: string): Promise<void> {
    try {
      // Delete from PostgreSQL
      await db.delete(documents)
        .where(eq(documents.id, documentId));
      
      await db.delete(documentChunks)
        .where(eq(documentChunks.documentId, documentId));

      // Delete from vector store
      await this.vectorService.deleteEmbedding('documents', documentId);

      console.log(`Document deleted: ${documentId}`);
    } catch (error) {
      console.error('Failed to delete document:', error);
      throw error;
    }
  }

  // Search documents
  async searchDocuments(
    query: string,
    options: {
      limit?: number;
      source?: string;
      userId?: string;
    } = {}
  ): Promise<any[]> {
    const filters: any = {};
    
    if (options.source) {
      filters.must = filters.must || [];
      filters.must.push({ key: 'source', match: { value: options.source } });
    }
    
    if (options.userId) {
      filters.must = filters.must || [];
      filters.must.push({ key: 'userId', match: { value: options.userId } });
    }

    return await this.vectorService.hybridSearch(query, {
      collection: 'yorha_documents_v2',
      limit: options.limit || 10,
      filter: filters,
      rerank: true
    });
  }

  // Get conversation context
  async getConversationContext(conversationId: string): Promise<any> {
    try {
      const conversation = await db.query.conversations.findFirst({
        where: eq(conversations.id, conversationId)
      });

      if (!conversation) {
        return null;
      }

      return {
        id: conversation.id,
        messages: conversation.messages,
        metadata: conversation.metadata,
        createdAt: conversation.createdAt,
        updatedAt: conversation.updatedAt
      };
    } catch (error) {
      console.error('Failed to get conversation context:', error);
      return null;
    }
  }

  // Create knowledge base entry
  async createKnowledgeEntry(
    category: string,
    title: string,
    content: string,
    tags: string[] = []
  ): Promise<string> {
    try {
      const [entry] = await db.insert(knowledgeBase).values({
        category,
        title,
        content,
        tags,
        metadata: { version: 1 }
      }).returning();

      // Index for search
      await this.vectorService.storeDocument(
        entry.id,
        content,
        {
          title,
          category,
          tags,
          source: 'knowledge_base'
        },
        'yorha_knowledge_v2'
      );

      return entry.id;
    } catch (error) {
      console.error('Failed to create knowledge entry:', error);
      throw error;
    }
  }
}