// LangChain + Ollama Integration with CUDA Support
// Production-ready AI service for legal document processing

import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RetrievalQAChain } from 'langchain/chains';
import { PromptTemplate } from '@langchain/core/prompts';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { BaseRetriever } from '@langchain/core/retrievers';
import { CallbackManagerForRetrieverRun } from '@langchain/core/callbacks/manager';
import { db } from '$lib/server/database';
import { documents, documentChunks, aiInteractions, embeddingJobs } from '$lib/database/enhanced-schema';
import { eq, desc, and, sql } from 'drizzle-orm';
import crypto from 'crypto';

// ============================================================================
// CONFIGURATION & TYPES
// ============================================================================

export interface LangChainConfig {
  ollamaBaseUrl: string;
  model: string;
  embeddingModel: string;
  temperature: number;
  maxTokens: number;
  chunkSize: number;
  chunkOverlap: number;
  maxRetrieverResults: number;
  useCuda: boolean;
  vectorDimensions: number;
}

export interface ProcessingResult {
  documentId: string;
  chunksCreated: number;
  embeddings: number[][];
  processingTime: number;
  metadata: {
    totalTokens: number;
    avgChunkSize: number;
    model: string;
    usedCuda: boolean;
  };
}

export interface QueryResult {
  answer: string;
  sourceDocuments: Array<{
    id: string;
    content: string;
    metadata: Record<string, any>;
    similarity: number;
  }>;
  confidence: number;
  processingTime: number;
  tokensUsed: number;
}

export interface SummarizationResult {
  summary: string;
  keyPoints: string[];
  entities: Array<{
    text: string;
    type: string;
    confidence: number;
  }>;
  riskAssessment: string;
  recommendations: string[];
  confidence: number;
}

// ============================================================================
// CUSTOM POSTGRESQL VECTOR RETRIEVER
// ============================================================================

class PostgreSQLVectorRetriever extends BaseRetriever {
  lc_namespace = ["custom", "retrievers", "postgresql"];
  private embeddings: OllamaEmbeddings;
  private maxResults: number;
  private similarityThreshold: number;

  constructor(embeddings: OllamaEmbeddings, maxResults = 10, similarityThreshold = 0.7) {
    super();
    this.embeddings = embeddings;
    this.maxResults = maxResults;
    this.similarityThreshold = similarityThreshold;
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun
  ): Promise<LangChainDocument[]> {
    const startTime = Date.now();
    
    try {
      // Generate query embedding
      const queryEmbedding = await this.embeddings.embedQuery(query);
      
      // Search using pgvector
      const results = await db
        .select({
          id: documentChunks.id,
          content: documentChunks.content,
          metadata: documentChunks.metadata,
          documentId: documentChunks.documentId,
          similarity: sql<number>`1 - (${documentChunks.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`,
        })
        .from(documentChunks)
        .where(sql`1 - (${documentChunks.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector) > ${this.similarityThreshold}`)
        .orderBy(sql`${documentChunks.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector`)
        .limit(this.maxResults);

      const processingTime = Date.now() - startTime;
      console.log(`Vector search completed in ${processingTime}ms, found ${results.length} results`);

      return results.map(result => new LangChainDocument({
        pageContent: result.content,
        metadata: {
          ...result.metadata,
          id: result.id,
          documentId: result.documentId,
          similarity: result.similarity,
          retrievalTime: processingTime
        }
      }));

    } catch (error) {
      console.error('PostgreSQL vector retrieval error:', error);
      throw error;
    }
  }
}

// ============================================================================
// MAIN LANGCHAIN SERVICE
// ============================================================================

export class LangChainOllamaService {
  private config: LangChainConfig;
  private llm: ChatOllama;
  private embeddings: OllamaEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private retriever: PostgreSQLVectorRetriever;
  private qaChain: RetrievalQAChain | null = null;

  constructor(config: Partial<LangChainConfig> = {}) {
    this.config = {
      ollamaBaseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
      model: 'llama3.2',
      embeddingModel: 'nomic-embed-text',
      temperature: 0.7,
      maxTokens: 4000,
      chunkSize: 1000,
      chunkOverlap: 200,
      maxRetrieverResults: 10,
      useCuda: true,
      vectorDimensions: 768,
      ...config
    };

    this.initializeComponents();
  }

  private initializeComponents() {
    // Initialize Ollama LLM with CUDA support
    this.llm = new ChatOllama({
      baseUrl: this.config.ollamaBaseUrl,
      model: this.config.model,
      temperature: this.config.temperature,
      numCtx: this.config.maxTokens,
      // CUDA optimization parameters
      numGpu: this.config.useCuda ? -1 : 0, // -1 uses all available GPUs
      numThread: this.config.useCuda ? undefined : 4, // Let CUDA handle threading
    });

    // Initialize embeddings with CUDA
    this.embeddings = new OllamaEmbeddings({
      baseUrl: this.config.ollamaBaseUrl,
      model: this.config.embeddingModel,
      requestOptions: {
        numGpu: this.config.useCuda ? -1 : 0,
      }
    });

    // Initialize text splitter
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.config.chunkSize,
      chunkOverlap: this.config.chunkOverlap,
      separators: ['\n\n', '\n', '.', '!', '?', ',', ' ', ''],
    });

    // Initialize PostgreSQL retriever
    this.retriever = new PostgreSQLVectorRetriever(
      this.embeddings,
      this.config.maxRetrieverResults
    );
  }

  // ============================================================================
  // DOCUMENT PROCESSING
  // ============================================================================

  async processDocument(
    documentId: string,
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<ProcessingResult> {
    const startTime = Date.now();
    
    try {
      console.log(`Processing document ${documentId} with LangChain...`);

      // Create processing job
      const [job] = await db
        .insert(embeddingJobs)
        .values({
          entityType: 'document',
          entityId: documentId,
          jobType: 'embedding',
          status: 'processing',
          model: this.config.embeddingModel,
          metadata: { startedAt: new Date().toISOString() }
        })
        .returning();

      // Split document into chunks
      const chunks = await this.textSplitter.splitText(content);
      console.log(`Document split into ${chunks.length} chunks`);

      // Generate embeddings for all chunks
      const embeddings: number[][] = [];
      const processedChunks: any[] = [];

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const embedding = await this.embeddings.embedQuery(chunk);
        embeddings.push(embedding);

        // Store chunk in database
        const [savedChunk] = await db
          .insert(documentChunks)
          .values({
            documentId,
            chunkIndex: i,
            content: chunk,
            embedding: JSON.stringify(embedding),
            startIndex: content.indexOf(chunk),
            endIndex: content.indexOf(chunk) + chunk.length,
            tokenCount: Math.ceil(chunk.length / 4), // Rough token estimation
            metadata: {
              chunkType: 'paragraph',
              importance: 1.0,
              keywords: this.extractKeywords(chunk),
              ...metadata
            }
          })
          .returning();

        processedChunks.push(savedChunk);

        // Update job progress
        const progress = Math.round(((i + 1) / chunks.length) * 100);
        await db
          .update(embeddingJobs)
          .set({ 
            progress,
            metadata: { 
              ...job.metadata,
              processedChunks: i + 1,
              totalChunks: chunks.length 
            }
          })
          .where(eq(embeddingJobs.id, job.id));
      }

      // Update document with processing status
      await db
        .update(documents)
        .set({
          isIndexed: true,
          updatedAt: new Date(),
          metadata: {
            ...metadata,
            chunkCount: chunks.length,
            avgChunkSize: Math.round(chunks.reduce((acc, chunk) => acc + chunk.length, 0) / chunks.length),
            processingModel: this.config.embeddingModel,
            usedCuda: this.config.useCuda
          }
        })
        .where(eq(documents.id, documentId));

      // Complete the job
      const processingTime = Date.now() - startTime;
      await db
        .update(embeddingJobs)
        .set({
          status: 'completed',
          progress: 100,
          metadata: {
            ...job.metadata,
            completedAt: new Date().toISOString(),
            processingTime
          }
        })
        .where(eq(embeddingJobs.id, job.id));

      console.log(`Document processing completed in ${processingTime}ms`);

      return {
        documentId,
        chunksCreated: chunks.length,
        embeddings,
        processingTime,
        metadata: {
          totalTokens: Math.ceil(content.length / 4),
          avgChunkSize: Math.round(chunks.reduce((acc, chunk) => acc + chunk.length, 0) / chunks.length),
          model: this.config.embeddingModel,
          usedCuda: this.config.useCuda
        }
      };

    } catch (error) {
      console.error('Document processing error:', error);
      throw error;
    }
  }

  // ============================================================================
  // AI QUERY & RAG
  // ============================================================================

  async queryDocuments(
    query: string,
    userId: string,
    caseId?: string,
    sessionId?: string
  ): Promise<QueryResult> {
    const startTime = Date.now();
    
    try {
      console.log(`Processing query: "${query}"`);

      // Initialize QA chain if not already done
      if (!this.qaChain) {
        const prompt = PromptTemplate.fromTemplate(`
          You are a legal AI assistant specialized in document analysis and legal research.
          
          Context from relevant documents:
          {context}
          
          Question: {question}
          
          Instructions:
          1. Provide a comprehensive answer based on the context provided
          2. Cite specific documents and sections when possible
          3. If the context doesn't contain enough information, clearly state this
          4. Highlight any legal implications or considerations
          5. Use professional legal language appropriate for attorneys and legal professionals
          
          Answer:
        `);

        this.qaChain = RetrievalQAChain.fromLLM(this.llm, this.retriever, {
          prompt,
          returnSourceDocuments: true,
        });
      }

      // Execute the query
      const result = await this.qaChain.call({
        query,
        question: query
      });

      const processingTime = Date.now() - startTime;
      const tokensUsed = Math.ceil(result.text.length / 4); // Rough estimation

      // Calculate confidence based on source similarity
      const avgSimilarity = result.sourceDocuments.reduce(
        (acc: number, doc: any) => acc + (doc.metadata.similarity || 0),
        0
      ) / result.sourceDocuments.length;
      
      const confidence = Math.min(avgSimilarity * 1.2, 1.0); // Scale and cap at 1.0

      // Log interaction
      await db.insert(aiInteractions).values({
        userId,
        caseId,
        sessionId,
        prompt: query,
        response: result.text,
        model: this.config.model,
        tokensUsed,
        responseTime: processingTime,
        confidence,
        metadata: {
          temperature: this.config.temperature,
          maxTokens: this.config.maxTokens,
          sources: result.sourceDocuments.map((doc: any) => ({
            id: doc.metadata.id,
            similarity: doc.metadata.similarity,
            type: 'document_chunk'
          })),
          chainType: 'retrieval_qa',
          memoryUsed: false
        }
      }).returning();

      return {
        answer: result.text,
        sourceDocuments: result.sourceDocuments.map((doc: any) => ({
          id: doc.metadata.id,
          content: doc.pageContent,
          metadata: doc.metadata,
          similarity: doc.metadata.similarity || 0
        })),
        confidence,
        processingTime,
        tokensUsed
      };

    } catch (error) {
      console.error('Query processing error:', error);
      throw error;
    }
  }

  // ============================================================================
  // AI SUMMARIZATION
  // ============================================================================

  async summarizeDocument(
    documentId: string,
    content: string,
    options: {
      extractEntities?: boolean;
      riskAssessment?: boolean;
      generateRecommendations?: boolean;
    } = {}
  ): Promise<SummarizationResult> {
    const startTime = Date.now();
    
    try {
      console.log(`Summarizing document ${documentId}...`);

      const prompt = PromptTemplate.fromTemplate(`
        You are a legal AI assistant specializing in document analysis. Analyze the following legal document and provide:
        
        1. A comprehensive summary (2-3 paragraphs)
        2. Key points (5-7 bullet points)
        ${options.extractEntities ? '3. Named entities (people, organizations, dates, amounts, locations)' : ''}
        ${options.riskAssessment ? '4. Risk assessment and potential legal issues' : ''}
        ${options.generateRecommendations ? '5. Recommendations for legal action or review' : ''}
        
        Document Content:
        {content}
        
        Provide your analysis in JSON format:
        {{
          "summary": "...",
          "keyPoints": ["..."],
          ${options.extractEntities ? '"entities": [{"text": "...", "type": "...", "confidence": 0.9}],' : ''}
          ${options.riskAssessment ? '"riskAssessment": "...",' : ''}
          ${options.generateRecommendations ? '"recommendations": ["..."],' : ''}
          "confidence": 0.95
        }}
      `);

      const chain = prompt.pipe(this.llm);
      const result = await chain.invoke({ content });

      // Parse the JSON response
      let parsedResult: any;
      try {
        parsedResult = JSON.parse(result.content as string);
      } catch (parseError) {
        // Fallback if JSON parsing fails
        parsedResult = {
          summary: result.content as string,
          keyPoints: [],
          entities: [],
          riskAssessment: 'Unable to assess risk automatically',
          recommendations: [],
          confidence: 0.5
        };
      }

      const processingTime = Date.now() - startTime;
      console.log(`Document summarization completed in ${processingTime}ms`);

      return {
        summary: parsedResult.summary || '',
        keyPoints: parsedResult.keyPoints || [],
        entities: parsedResult.entities || [],
        riskAssessment: parsedResult.riskAssessment || '',
        recommendations: parsedResult.recommendations || [],
        confidence: parsedResult.confidence || 0.5
      };

    } catch (error) {
      console.error('Summarization error:', error);
      throw error;
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private extractKeywords(text: string): string[] {
    // Simple keyword extraction - could be enhanced with more sophisticated NLP
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3);
    
    const wordCount = new Map<string, number>();
    words.forEach(word => {
      wordCount.set(word, (wordCount.get(word) || 0) + 1);
    });
    
    return Array.from(wordCount.entries())
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  }

  async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    ollama: boolean;
    embedding: boolean;
    database: boolean;
    cuda: boolean;
    models: string[];
  }> {
    try {
      // Test Ollama connection
      const ollamaTest = await fetch(`${this.config.ollamaBaseUrl}/api/tags`)
        .then(res => res.json())
        .catch(() => null);

      // Test embedding
      let embeddingTest = false;
      try {
        await this.embeddings.embedQuery('test');
        embeddingTest = true;
      } catch {}

      // Test database
      let databaseTest = false;
      try {
        await db.select({ count: sql`count(*)` }).from(documents);
        databaseTest = true;
      } catch {}

      return {
        status: ollamaTest && embeddingTest && databaseTest ? 'healthy' : 'unhealthy',
        ollama: !!ollamaTest,
        embedding: embeddingTest,
        database: databaseTest,
        cuda: this.config.useCuda,
        models: ollamaTest?.models?.map((m: any) => m.name) || []
      };

    } catch (error) {
      console.error('Health check error:', error);
      return {
        status: 'unhealthy',
        ollama: false,
        embedding: false,
        database: false,
        cuda: false,
        models: []
      };
    }
  }

  // Get processing statistics
  async getProcessingStats(days = 7): Promise<{
    documentsProcessed: number;
    averageProcessingTime: number;
    totalQueries: number;
    averageConfidence: number;
    modelUsage: Record<string, number>;
  }> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const [jobs, interactions] = await Promise.all([
      db
        .select()
        .from(embeddingJobs)
        .where(and(
          eq(embeddingJobs.status, 'completed'),
          sql`${embeddingJobs.createdAt} >= ${startDate}`
        )),
      db
        .select()
        .from(aiInteractions)
        .where(sql`${aiInteractions.createdAt} >= ${startDate}`)
    ]);

    const avgProcessingTime = jobs.reduce((acc, job) => {
      const time = job.metadata?.processingTime || 0;
      return acc + time;
    }, 0) / (jobs.length || 1);

    const avgConfidence = interactions.reduce((acc, interaction) => {
      return acc + (interaction.confidence || 0);
    }, 0) / (interactions.length || 1);

    const modelUsage: Record<string, number> = {};
    interactions.forEach(interaction => {
      const model = interaction.model || 'unknown';
      modelUsage[model] = (modelUsage[model] || 0) + 1;
    });

    return {
      documentsProcessed: jobs.length,
      averageProcessingTime: Math.round(avgProcessingTime),
      totalQueries: interactions.length,
      averageConfidence: Math.round(avgConfidence * 100) / 100,
      modelUsage
    };
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const langChainService = new LangChainOllamaService({
  useCuda: process.env.ENABLE_CUDA === 'true',
  model: process.env.OLLAMA_MODEL || 'llama3.2',
  embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text',
  ollamaBaseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434'
});

export default langChainService;