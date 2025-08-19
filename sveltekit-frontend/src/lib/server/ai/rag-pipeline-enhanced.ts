import Redis from "ioredis";
import crypto from "crypto";
// @ts-nocheck
// lib/server/ai/rag-pipeline-enhanced.ts
// Enhanced RAG Pipeline integrating best practices while maintaining compatibility

import { Ollama } from "@langchain/community/llms/ollama";
// Orphaned content: import {

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// Orphaned content: import {

import { PromptTemplate } from "@langchain/core/prompts";
// Orphaned content: import {
RunnableSequence, RunnablePassthrough
import { StringOutputParser } from '@langchain/core/output_parsers';
// Orphaned content: import postgres from "postgres";
import {

import { eq, sql as drizzleSql, and, gte, desc  } from "drizzle-orm";
// Orphaned content: import * as schema from './db/schema-postgres.js';
import {

// === CONFIGURATION ===
const EMBEDDING_MODEL = import.meta.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text:latest';
const EMBEDDING_DIMENSIONS = 768;
const LLM_MODEL = import.meta.env.OLLAMA_LLM_MODEL || 'gemma3-legal:latest';
const OLLAMA_BASE_URL = import.meta.env.OLLAMA_URL || 'http://localhost:11434';

// Enhanced configuration with error handling
const config = {
  database: {
    host: import.meta.env.DATABASE_HOST || 'localhost',
    port: parseInt(import.meta.env.DATABASE_PORT || '5432'),
    database: import.meta.env.DATABASE_NAME || 'legal_ai_db',
    username: import.meta.env.DATABASE_USER || 'legal_admin',
    password: import.meta.env.DATABASE_PASSWORD || '123456',
    max: parseInt(import.meta.env.DATABASE_MAX_CONNECTIONS || '20'),
    idle_timeout: parseInt(import.meta.env.DATABASE_IDLE_TIMEOUT || '20'),
    ssl: import.meta.env.NODE_ENV === 'production' ? 'require' : false
  },
  redis: {
    host: import.meta.env.REDIS_HOST || 'localhost',
    port: parseInt(import.meta.env.REDIS_PORT || '6379'),
    db: parseInt(import.meta.env.REDIS_DB || '0'),
    maxRetriesPerRequest: parseInt(import.meta.env.REDIS_MAX_RETRIES || '3'),
    cacheTtl: parseInt(import.meta.env.RAG_CACHE_TTL || '86400') // 24 hours
  },
  rag: {
    chunkSize: parseInt(import.meta.env.RAG_CHUNK_SIZE || '1500'),
    chunkOverlap: parseInt(import.meta.env.RAG_CHUNK_OVERLAP || '300'),
    maxSources: parseInt(import.meta.env.RAG_MAX_SOURCES || '10'),
    similarityThreshold: parseFloat(import.meta.env.RAG_SIMILARITY_THRESHOLD || '0.5'),
    timeoutMs: parseInt(import.meta.env.RAG_TIMEOUT_MS || '30000'),
    enableMetrics: import.meta.env.RAG_ENABLE_METRICS !== 'false',
    enableAutoTagging: import.meta.env.RAG_ENABLE_AUTO_TAGGING !== 'false'
  }
};

// === VALIDATION & TYPES ===
interface DocumentIngestionParams {
  title: string;
  content: string;
  documentType: string;
  metadata?: Record<string, any>;
  caseId?: string;
  userId: string;
}

interface SearchParams {
  query: string;
  caseId?: string;
  documentType?: string;
  limit?: number;
  threshold?: number;
}

interface QuestionParams {
  question: string;
  caseId?: string;
  userId: string;
  conversationContext?: string;
}

// === UTILITY FUNCTIONS ===
function validateInput(input: string, maxLength: number = 10000): string {
  if (!input || typeof input !== 'string') {
    throw new Error('Input must be a non-empty string');
  }
  
  // Sanitize input to prevent injection attacks
  const sanitized = input
    .replace(/[<>]/g, '') // Remove HTML tags
    .replace(/[;'"`]/g, '') // Remove SQL injection chars
    .trim();
    
  if (sanitized.length > maxLength) {
    throw new Error(`Input exceeds maximum length of ${maxLength} characters`);
  }
  
  return sanitized;
}

function isValidUUID(uuid: string): boolean {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  return uuidRegex.test(uuid);
}

// === RATE LIMITING ===
class RateLimiter {
  private requests = new Map<string, number[]>();
  private windowMs = 60 * 1000; // 1 minute
  private maxRequests = parseInt(import.meta.env.RAG_RATE_LIMIT_PER_MINUTE || '60');

  isAllowed(identifier: string): boolean {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    
    let requests = this.requests.get(identifier) || [];
    requests = requests.filter(time => time > windowStart);
    
    if (requests.length >= this.maxRequests) {
      return false;
    }
    
    requests.push(now);
    this.requests.set(identifier, requests);
    
    return true;
  }
}

// === METRICS COLLECTION ===
class MetricsCollector {
  private metrics = new Map<string, number[]>();
  private counters = new Map<string, number>();

  recordTiming(operation: string, duration: number): void {
    if (!config.rag.enableMetrics) return;
    
    const timings = this.metrics.get(operation) || [];
    timings.push(duration);
    
    // Keep only last 1000 measurements
    if (timings.length > 1000) {
      timings.shift();
    }
    
    this.metrics.set(operation, timings);
  }

  incrementCounter(name: string, value: number = 1): void {
    if (!config.rag.enableMetrics) return;
    
    const current = this.counters.get(name) || 0;
    this.counters.set(name, current + value);
  }

  getMetrics(): Record<string, any> {
    const result: Record<string, any> = {};
    
    // Add timing metrics
    for (const [operation, timings] of this.metrics.entries()) {
      if (timings.length > 0) {
        result[`${operation}_avg_ms`] = timings.reduce((a, b) => a + b, 0) / timings.length;
        result[`${operation}_count`] = timings.length;
      }
    }
    
    // Add counter metrics
    for (const [name, value] of this.counters.entries()) {
      result[name] = value;
    }
    
    return result;
  }
}

// === ENHANCED RAG PIPELINE ===
export class EnhancedLegalRAGPipeline {
  private initialized = false;
  private sql: postgres.Sql;
  private db: ReturnType<typeof drizzle>;
  private redis: Redis;
  private embeddings: OllamaEmbeddings;
  private llm: Ollama;
  private textSplitter: RecursiveCharacterTextSplitter;
  private rateLimiter: RateLimiter;
  private metrics: MetricsCollector;

  constructor() {
    this.rateLimiter = new RateLimiter();
    this.metrics = new MetricsCollector();
    this.initializeComponents();
  }

  private initializeComponents(): void {
    // Initialize PostgreSQL with enhanced error handling
    this.sql = postgres({
      ...config.database,
      prepare: true,
      connect_timeout: 10,
      onnotice: (notice) => logger.debug('[DB] Notice:', notice),
      onparameter: (key, value) => logger.debug(`[DB] Parameter ${key}:`, value),
    });

    this.db = drizzle(this.sql, { schema });

    // Initialize Redis with enhanced configuration
    this.redis = new Redis({
      ...config.redis,
      enableReadyCheck: true,
      lazyConnect: false,
      retryStrategy: (times) => Math.min(times * 50, 2000),
      reconnectOnError: (err) => {
        logger.warn('Redis reconnect on error:', err.message);
        return err.message.includes('READONLY');
      },
    });

    // Initialize LangChain components with error handling
    this.embeddings = new OllamaEmbeddings({
      baseUrl: OLLAMA_BASE_URL,
      model: EMBEDDING_MODEL,
      requestOptions: {
        useMMap: true,
        numThread: 8,
        timeout: config.rag.timeoutMs,
      },
    });

    this.llm = new Ollama({
      baseUrl: OLLAMA_BASE_URL,
      model: LLM_MODEL,
      temperature: 0.3,
      numCtx: 8192,
      numPredict: 2048,
      topK: 40,
      topP: 0.9,
      repeatPenalty: 1.1,
      timeout: config.rag.timeoutMs,
      callbacks: [
        {
          handleLLMStart: async (llm, prompts) => {
            logger.debug(`[RAG] LLM Started: ${LLM_MODEL}`);
            this.metrics.incrementCounter('llm_requests');
          },
          handleLLMEnd: async (output) => {
            logger.debug('[RAG] LLM Completed');
            this.metrics.incrementCounter('llm_completions');
          },
          handleLLMError: async (err) => {
            logger.error('[RAG] LLM Error:', err);
            this.metrics.incrementCounter('llm_errors');
          },
        },
      ],
    });

    // Enhanced text splitter for legal documents
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: config.rag.chunkSize,
      chunkOverlap: config.rag.chunkOverlap,
      separators: [
        '\n\nSECTION', '\n\nARTICLE', '\n\nCLAUSE', // Legal sections
        '\n\n§', '\n\n¶', // Legal symbols
        '\n\n', '\n', '.', '!', '?', ';', ':', ' ', ''
      ],
      keepSeparator: true,
    });
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Test database connection
      const testResult = await this.sql`SELECT 1 as test`;
      if (testResult[0]?.test !== 1) {
        throw new Error('Database connection test failed');
      }
      logger.info('[RAG] Database connected successfully');

      // Test Redis connection
      await this.redis.ping();
      logger.info('[RAG] Redis connected successfully');

      // Test Ollama connection
      const testEmbedding = await this.embeddings.embedQuery('test');
      if (testEmbedding.length !== EMBEDDING_DIMENSIONS) {
        throw new Error(`Expected ${EMBEDDING_DIMENSIONS} dimensions, got ${testEmbedding.length}`);
      }
      logger.info('[RAG] Ollama embeddings working correctly');

      this.initialized = true;
      this.metrics.incrementCounter('pipeline_initializations');
      logger.info('[RAG] Pipeline initialized successfully');

    } catch (error) {
      logger.error('[RAG] Initialization failed:', error);
      throw error;
    }
  }

  // === DOCUMENT INGESTION ===
  async ingestLegalDocument(params: DocumentIngestionParams) {
    const startTime = Date.now();
    
    try {
      // Validate inputs
      const title = validateInput(params.title, 500);
      const content = validateInput(params.content, 10485760); // 10MB limit
      const documentType = validateInput(params.documentType, 50);
      const userId = params.userId;
      
      if (!isValidUUID(userId)) {
        throw new Error('Invalid user ID format');
      }

      // Rate limiting
      if (!this.rateLimiter.isAllowed(userId)) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      await this.ensureInitialized();

      const { caseId, metadata = {} } = params;

      // 1. Create main document record with transaction
      const [document] = await this.db.transaction(async (tx) => {
        const [doc] = await tx.insert(schema.legalDocuments)
          .values({
            title,
            content: content.substring(0, 10000),
            fullText: content,
            documentType,
            keywords: metadata.keywords || [],
            topics: metadata.topics || [],
            jurisdiction: metadata.jurisdiction,
            caseId,
            createdBy: userId,
          })
          .returning();

        return [doc];
      });

      logger.info(`[RAG] Created document: ${document.id}`);

      // 2. Generate document-level embedding with caching
      const docEmbedding = await this.generateEmbedding(
        `${title}\n${content.substring(0, 2000)}`
      );

      await this.db.update(schema.legalDocuments)
        .set({ embedding: JSON.stringify(docEmbedding) })
        .where(eq(schema.legalDocuments.id, document.id));

      // 3. Smart chunking based on document type
      const chunks = await this.smartLegalChunking(content, documentType);
      logger.info(`[RAG] Split into ${chunks.length} chunks`);

      // 4. Process chunks in batches with error handling
      const BATCH_SIZE = 10;
      let successfulChunks = 0;
      const errors: string[] = [];

      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const batch = chunks.slice(i, i + BATCH_SIZE);

        try {
          const chunkRecords = await Promise.all(
            batch.map(async (chunk, idx) => {
              try {
                const embedding = await this.generateEmbedding(chunk);
                successfulChunks++;

                return {
                  documentId: document.id,
                  documentType,
                  chunkIndex: i + idx,
                  content: chunk,
                  embedding: JSON.stringify(embedding),
                  metadata: {
                    title,
                    position: i + idx,
                    totalChunks: chunks.length,
                    ...metadata,
                  },
                };
              } catch (error) {
                const errorMsg = `Failed to process chunk ${i + idx}: ${error}`;
                errors.push(errorMsg);
                logger.error(errorMsg);
                return null;
              }
            })
          );

          // Filter out failed chunks
          const validChunks = chunkRecords.filter(record => record !== null);

          if (validChunks.length > 0) {
            await this.db.insert(schema.documentChunks).values(validChunks);
          }

          logger.debug(`[RAG] Processed batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(chunks.length / BATCH_SIZE)}`);
        } catch (error) {
          const errorMsg = `Failed to process batch ${Math.floor(i / BATCH_SIZE) + 1}: ${error}`;
          errors.push(errorMsg);
          logger.error(errorMsg);
        }
      }

      // 5. Auto-generate tags if enabled
      let tags: Array<{ tag: string; confidence: number }> = [];
      if (config.rag.enableAutoTagging) {
        try {
          tags = await this.generateAutoTags(content, documentType);

          for (const tag of tags) {
            await this.db.insert(schema.autoTags).values({
              entityId: document.id,
              entityType: 'document',
              tag: tag.tag,
              confidence: tag.confidence.toString(),
              source: 'ai_analysis',
              model: LLM_MODEL,
            });
          }
        } catch (error) {
          const errorMsg = `Failed to generate auto-tags: ${error}`;
          errors.push(errorMsg);
          logger.warn(errorMsg);
        }
      }

      const processingTime = Date.now() - startTime;
      const success = successfulChunks > 0;

      logger.info(`[RAG] Document ingestion completed in ${processingTime}ms (${successfulChunks}/${chunks.length} chunks successful)`);

      this.metrics.incrementCounter('documents_ingested');
      this.metrics.recordTiming('ingestion_time', processingTime);

      return {
        documentId: document.id,
        chunksCreated: successfulChunks,
        tags: tags.map(t => t.tag),
        processingTime,
        success,
        errors: errors.length > 0 ? errors : undefined,
      };

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('[RAG] Ingestion error:', error);
      this.metrics.incrementCounter('ingestion_errors');
      this.metrics.recordTiming('ingestion_error_time', processingTime);

      throw error;
    }
  }

  // === SEARCH & RETRIEVAL ===
  async hybridSearch(params: SearchParams): Promise<Document[]> {
    const startTime = Date.now();
    
    try {
      const query = validateInput(params.query, 1000);
      const { caseId, documentType, limit = 10, threshold = config.rag.similarityThreshold } = params;

      await this.ensureInitialized();

      // Generate query embedding with caching
      const queryEmbedding = await this.generateEmbedding(query);

      // Build dynamic WHERE conditions
      const vectorConditions: any[] = [
        drizzleSql`1 - (dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector) > ${threshold}`
      ];

      const keywordConditions: any[] = [
        drizzleSql`to_tsvector('english', dc.content) @@ plainto_tsquery('english', ${query})`
      ];

      if (caseId && isValidUUID(caseId)) {
        vectorConditions.push(drizzleSql`dc.metadata->>'caseId' = ${caseId}`);
        keywordConditions.push(drizzleSql`dc.metadata->>'caseId' = ${caseId}`);
      }

      if (documentType) {
        vectorConditions.push(drizzleSql`dc.document_type = ${documentType}`);
        keywordConditions.push(drizzleSql`dc.document_type = ${documentType}`);
      }

      // Perform vector similarity search
      const vectorResults = await this.sql`
        SELECT 
          dc.id,
          dc.content,
          dc.metadata,
          dc.document_id,
          ld.title,
          1 - (dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector) as similarity
        FROM document_chunks dc
        LEFT JOIN legal_documents ld ON dc.document_id = ld.id
        WHERE ${drizzleSql.join(vectorConditions, drizzleSql` AND `)}
        ORDER BY dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector
        LIMIT ${limit * 2}
      `;

      // Perform keyword search
      const keywordResults = await this.sql`
        SELECT 
          dc.id,
          dc.content,
          dc.metadata,
          dc.document_id,
          ld.title,
          ts_rank(to_tsvector('english', dc.content), 
                  plainto_tsquery('english', ${query})) as text_rank
        FROM document_chunks dc
        LEFT JOIN legal_documents ld ON dc.document_id = ld.id
        WHERE ${drizzleSql.join(keywordConditions, drizzleSql` AND `)}
        ORDER BY text_rank DESC
        LIMIT ${limit}
      `;

      // Combine and deduplicate results
      const combinedResults = new Map<string, any>();

      // Add vector results with higher weight
      vectorResults.forEach(r => {
        combinedResults.set(r.id, {
          ...r,
          score: (r.similarity as number) * 0.7,
        });
      });

      // Add or update with keyword results
      keywordResults.forEach(r => {
        const existing = combinedResults.get(r.id);
        if (existing) {
          existing.score += (r.text_rank as number) * 0.3;
        } else {
          combinedResults.set(r.id, {
            ...r,
            score: (r.text_rank as number) * 0.3,
          });
        }
      });

      // Sort by combined score and convert to Documents
      const sortedResults = Array.from(combinedResults.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);

      const searchResults = sortedResults.map(r => new Document({
        pageContent: r.content,
        metadata: {
          ...r.metadata,
          documentId: r.document_id,
          title: r.title,
          score: r.score,
          similarity: r.similarity || 0,
          textRank: r.text_rank || 0,
        },
      }));

      this.metrics.incrementCounter('searches_performed');
      this.metrics.recordTiming('search_time', Date.now() - startTime);

      return searchResults;

    } catch (error) {
      logger.error('[RAG] Search error:', error);
      this.metrics.incrementCounter('search_errors');
      throw error;
    }
  }

  // === QUESTION ANSWERING ===
  async answerLegalQuestion(params: QuestionParams) {
    const startTime = Date.now();
    
    try {
      const question = validateInput(params.question, 2000);
      const { caseId, userId, conversationContext } = params;

      if (!isValidUUID(userId)) {
        throw new Error('Invalid user ID format');
      }

      // Rate limiting
      if (!this.rateLimiter.isAllowed(userId)) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      await this.ensureInitialized();

      // 1. Retrieve relevant context
      const relevantDocs = await this.hybridSearch({
        query: question,
        caseId,
        limit: 5,
        threshold: 0.6,
      });

      if (relevantDocs.length === 0) {
        return {
          answer: "I couldn't find relevant information in the knowledge base to answer your question. Please provide more context or try rephrasing your question.",
          sources: [],
          confidence: 0,
          keyPoints: [],
          processingTime: Date.now() - startTime,
        };
      }

      // 2. Build context from retrieved documents
      const context = relevantDocs
        .map((doc, idx) => `[Source ${idx + 1}]:\n${doc.pageContent}`)
        .join('\n\n---\n\n');

      // 3. Create enhanced prompt with legal context
      const promptTemplate = PromptTemplate.fromTemplate(`
You are a legal AI assistant with expertise in legal analysis. Answer the question based ONLY on the provided context.

${conversationContext ? `Previous Conversation Context:\n${conversationContext}\n\n` : ''}

Legal Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, accurate answer based on the context
2. Cite specific sources using [Source N] notation
3. Identify any legal principles or precedents mentioned
4. Note any important caveats or limitations
5. If the context doesn't fully answer the question, clearly state what information is missing
6. Maintain a professional legal tone

Answer:
      `);

      // 4. Create chain and generate answer
      const chain = RunnableSequence.from([
        {
          context: () => context,
          question: new RunnablePassthrough(),
        },
        promptTemplate,
        this.llm,
        new StringOutputParser(),
      ]);

      const answer = await Promise.race([
        chain.invoke(question),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('LLM response timed out')), config.rag.timeoutMs)
        ),
      ]);

      // 5. Extract confidence and key points
      const analysis = await this.analyzeAnswer(answer, relevantDocs);

      // 6. Log the query for analytics
      try {
        const queryEmbedding = await this.generateEmbedding(question);

        await this.db.insert(schema.userAiQueries).values({
          userId,
          caseId,
          query: question,
          response: answer,
          model: LLM_MODEL,
          queryType: 'legal_research',
          confidence: analysis.confidence.toString(),
          processingTime: Date.now() - startTime,
          contextUsed: relevantDocs.map(d => d.metadata.documentId),
          embedding: JSON.stringify(queryEmbedding),
          metadata: {
            sourcesCount: relevantDocs.length,
            keyPoints: analysis.keyPoints,
          },
        });
      } catch (error) {
        logger.warn('Failed to log query:', error);
        // Don't fail the main operation for logging issues
      }

      const result = {
        answer,
        sources: relevantDocs.map(d => ({
          id: d.metadata.documentId,
          title: d.metadata.title,
          score: d.metadata.score,
          excerpt: d.pageContent.substring(0, 200) + '...',
        })),
        confidence: analysis.confidence,
        keyPoints: analysis.keyPoints,
        processingTime: Date.now() - startTime,
      };

      this.metrics.incrementCounter('questions_answered');
      this.metrics.recordTiming('qa_time', result.processingTime);

      return result;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('[RAG] QA error:', error);
      this.metrics.incrementCounter('qa_errors');

      // Log failed query
      try {
        await this.db.insert(schema.userAiQueries).values({
          userId: params.userId,
          caseId: params.caseId,
          query: params.question,
          response: '',
          model: LLM_MODEL,
          isSuccessful: false,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          processingTime,
        });
      } catch (logError) {
        logger.warn('Failed to log error query:', logError);
      }

      throw error;
    }
  }

  // === CONTRACT ANALYSIS ===
  async analyzeContract(contractText: string, jurisdiction?: string) {
    const startTime = Date.now();
    
    try {
      const sanitizedText = validateInput(contractText, 1048576); // 1MB limit
      
      await this.ensureInitialized();

      const contractPrompt = PromptTemplate.fromTemplate(`
You are a legal expert specializing in contract analysis. Analyze the following contract and provide a structured assessment.

${jurisdiction ? `Jurisdiction: ${jurisdiction}\n` : ''}

Contract:
{contract}

Provide your analysis in the following structured format:

1. CONTRACT TYPE & PARTIES
- Type of contract
- Parties involved
- Governing law/jurisdiction

2. KEY TERMS & OBLIGATIONS
- Primary obligations of each party
- Payment terms
- Duration and termination clauses
- Deliverables/milestones

3. RISK ASSESSMENT
- Potential risks for each party (classify as HIGH, MEDIUM, LOW)
- Liability limitations
- Indemnification clauses
- Force majeure provisions

4. LEGAL ISSUES
- Ambiguous terms requiring clarification
- Potential enforceability issues
- Missing standard clauses
- Compliance considerations

5. RECOMMENDATIONS
- Suggested modifications
- Points for negotiation
- Additional clauses to consider

Provide specific clause references where applicable.
      `);

      const chain = RunnableSequence.from([
        contractPrompt,
        this.llm,
        new StringOutputParser(),
      ]);

      const analysis = await Promise.race([
        chain.invoke({ contract: sanitizedText }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Contract analysis timed out')), config.rag.timeoutMs)
        ),
      ]);

      const parsedAnalysis = this.parseContractAnalysis(analysis);
      const processingTime = Date.now() - startTime;

      this.metrics.incrementCounter('contracts_analyzed');
      this.metrics.recordTiming('contract_analysis_time', processingTime);

      return {
        ...parsedAnalysis,
        confidence: 0.85,
        processingTime,
      };

    } catch (error) {
      logger.error('[RAG] Contract analysis error:', error);
      this.metrics.incrementCounter('contract_analysis_errors');
      throw error;
    }
  }

  // === HELPER METHODS ===
  private async generateEmbedding(text: string): Promise<number[]> {
    const textHash = this.hashText(text);

    try {
      // Check cache first
      const cached = await this.redis.get(`embedding:${textHash}`);
      if (cached) {
        this.metrics.incrementCounter('cache_hits');
        return JSON.parse(cached);
      }

      this.metrics.incrementCounter('cache_misses');

      // Generate new embedding
      const embedding = await this.embeddings.embedQuery(text);

      // Cache for configured TTL
      await this.redis.setex(`embedding:${textHash}`, config.redis.cacheTtl, JSON.stringify(embedding));

      this.metrics.incrementCounter('embeddings_generated');
      return embedding;

    } catch (error) {
      logger.error('Embedding generation failed:', error);
      this.metrics.incrementCounter('embedding_errors');
      throw error;
    }
  }

  private async smartLegalChunking(content: string, documentType: string): Promise<string[]> {
    const chunks: string[] = [];

    // Document type specific patterns
    const patterns = {
      contract: [
        /(?:^|\n)(?:WHEREAS|NOW THEREFORE|SECTION|ARTICLE|CLAUSE)\s+[^\n]*/gi,
        /(?:^|\n)\d+\.\s+[A-Z][^\n]+/g,
      ],
      statute: [
        /(?:^|\n)(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+[\d.]+[^\n]*/gi,
        /(?:^|\n)§\s*[\d.]+[^\n]*/g,
      ],
      case_law: [
        /(?:^|\n)(?:FACTS|HOLDING|ANALYSIS|CONCLUSION|DISSENT)\s*[^\n]*/gi,
        /(?:^|\n)[IVX]+\.\s+[A-Z][^\n]+/g,
      ]
    };

    // Try legal structure-based chunking first
    const docPatterns = patterns[documentType as keyof typeof patterns] || patterns.contract;
    let structuredChunks: string[] = [];

    for (const pattern of docPatterns) {
      const matches = content.match(pattern);
      if (matches && matches.length > 0) {
        const sections = content.split(pattern);
        structuredChunks = sections
          .filter(section => section.trim().length > 50)
          .map(section => section.trim());
        
        if (structuredChunks.length > 0) break;
      }
    }

    // Fallback to standard chunking
    if (structuredChunks.length === 0) {
      const docs = await this.textSplitter.createDocuments([content]);
      structuredChunks = docs.map(d => d.pageContent);
    }

    // Further split large chunks if needed
    for (const chunk of structuredChunks) {
      if (chunk.length > config.rag.chunkSize * 1.5) {
        const subDocs = await this.textSplitter.createDocuments([chunk]);
        chunks.push(...subDocs.map(d => d.pageContent));
      } else {
        chunks.push(chunk);
      }
    }

    return chunks;
  }

  private async generateAutoTags(content: string, documentType: string): Promise<Array<{ tag: string; confidence: number }>> {
    if (!config.rag.enableAutoTagging) return [];

    const tagPrompt = PromptTemplate.fromTemplate(`
Extract relevant legal tags from this {documentType} document. 
Focus on: legal concepts, parties, jurisdictions, case types, and key topics.

Document excerpt:
{content}

Return ONLY a JSON array of tags with confidence scores (0-1):
[{"tag": "contract law", "confidence": 0.95}, ...]
    `);

    const chain = RunnableSequence.from([
      tagPrompt,
      this.llm,
      new StringOutputParser(),
    ]);

    try {
      const response = await Promise.race([
        chain.invoke({
          documentType,
          content: content.substring(0, 3000),
        }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Auto-tagging timed out')), config.rag.timeoutMs / 2)
        ),
      ]);

      // Extract JSON from response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const tags = JSON.parse(jsonMatch[0]);
        return Array.isArray(tags) ? tags : [];
      }

      return [];
    } catch (error) {
      logger.warn('Auto-tagging failed:', error);
      return [];
    }
  }

  private async analyzeAnswer(answer: string, sources: Document[]) {
    // Calculate confidence based on source relevance and answer characteristics
    const avgScore = sources.reduce((sum, doc) => sum + doc.metadata.score, 0) / sources.length;
    
    // Adjust confidence based on answer length and citation count
    const citations = (answer.match(/\[Source \d+\]/g) || []).length;
    const citationBonus = Math.min(citations / sources.length, 0.3);
    
    const baseConfidence = Math.min(0.95, avgScore + citationBonus);

    // Extract key points from structured answer
    const keyPoints = answer
      .split('\n')
      .filter(line => line.match(/^[\d.•-]|^[A-Z][a-z]+:/) && line.length > 10)
      .slice(0, 5)
      .map(line => line.replace(/^[.\d•-]*\s*/, '').trim());

    return {
      confidence: Math.max(0.1, baseConfidence),
      keyPoints,
    };
  }

  private parseContractAnalysis(analysis: string) {
    const sections = {
      contractType: '',
      parties: [] as string[],
      keyTerms: [] as string[],
      risks: [] as Array<{ description: string; severity: 'low' | 'medium' | 'high' }>,
      legalIssues: [] as string[],
      recommendations: [] as string[],
    };

    const lines = analysis.split('\n');
    let currentSection = '';

    for (const line of lines) {
      const trimmed = line.trim();
      
      if (trimmed.includes('CONTRACT TYPE')) currentSection = 'type';
      else if (trimmed.includes('KEY TERMS')) currentSection = 'terms';
      else if (trimmed.includes('RISK')) currentSection = 'risks';
      else if (trimmed.includes('LEGAL ISSUES')) currentSection = 'issues';
      else if (trimmed.includes('RECOMMENDATIONS')) currentSection = 'recommendations';
      else if (trimmed && currentSection) {
        const cleanLine = trimmed.replace(/^[-•*\d.]\s*/, '');
        
        switch (currentSection) {
          case 'type':
            if (!sections.contractType && !cleanLine.includes(':')) {
              sections.contractType = cleanLine;
            }
            break;
          case 'terms':
            if (cleanLine.length > 10) sections.keyTerms.push(cleanLine);
            break;
          case 'risks':
            if (cleanLine.length > 10) {
              const severity: 'low' | 'medium' | 'high' = 
                cleanLine.toLowerCase().includes('high') ? 'high' :
                cleanLine.toLowerCase().includes('medium') ? 'medium' : 'low';
              
              sections.risks.push({
                description: cleanLine,
                severity,
              });
            }
            break;
          case 'issues':
            if (cleanLine.length > 10) sections.legalIssues.push(cleanLine);
            break;
          case 'recommendations':
            if (cleanLine.length > 10) sections.recommendations.push(cleanLine);
            break;
        }
      }
    }

    return sections;
  }

  private hashText(text: string): string {
    return crypto.createHash('sha256').update(text.trim()).digest('hex');
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
  }

  // === HEALTH & MONITORING ===
  async getHealthStatus() {
    const checks = await Promise.allSettled([
      this.checkDatabaseHealth(),
      this.checkRedisHealth(),
      this.checkOllamaHealth(),
    ]);

    return checks.map((result, index) => {
      const services = ['Database', 'Redis', 'Ollama'];
      return {
        service: services[index],
        status: result.status === 'fulfilled' ? 'healthy' : 'unhealthy',
        error: result.status === 'rejected' ? result.reason.message : undefined,
      };
    });
  }

  private async checkDatabaseHealth() {
    const result = await this.sql`SELECT 1 as test`;
    if (result[0]?.test !== 1) throw new Error('Database check failed');
  }

  private async checkRedisHealth() {
    await this.redis.ping();
  }

  private async checkOllamaHealth() {
    const testEmbedding = await this.embeddings.embedQuery('test');
    if (testEmbedding.length !== EMBEDDING_DIMENSIONS) {
      throw new Error(`Expected ${EMBEDDING_DIMENSIONS} dimensions, got ${testEmbedding.length}`);
    }
  }

  getMetrics(): Record<string, any> {
    return this.metrics.getMetrics();
  }

  // === CLEANUP ===
  async close(): Promise<void> {
    try {
      await Promise.allSettled([
        this.redis.quit(),
        this.sql.end(),
      ]);
      
      logger.info('[RAG] Pipeline closed successfully');
    } catch (error) {
      logger.error('[RAG] Error during shutdown:', error);
    }
  }
}

// Export enhanced singleton instance
export const enhancedRAGPipeline = new EnhancedLegalRAGPipeline();

// Also export the original interface for backward compatibility
export const ragPipeline = enhancedRAGPipeline;

