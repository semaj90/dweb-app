/**
 * Enhanced RAG Pipeline - Legal AI Platform
 * 
 * Advanced Retrieval-Augmented Generation system specifically designed for legal AI
 * applications with comprehensive document processing, vector search, and intelligent
 * question answering capabilities.
 * 
 * Features:
 * - Multi-modal document ingestion with legal-specific chunking
 * - Hybrid vector and keyword search with PostgreSQL pgvector
 * - Intelligent auto-tagging and metadata extraction
 * - Contract analysis and legal document processing
 * - Rate limiting and comprehensive error handling
 * - Redis caching for embeddings and search results
 * - Real-time metrics and performance monitoring
 * - Legal compliance and audit trail tracking
 * 
 * @author Legal AI Platform Team
 * @version 4.2.0
 * @lastModified 2025-01-20
 */

import fs from 'fs';
import crypto from 'crypto';
import Redis from 'ioredis';
import postgres from 'postgres';
import { drizzle } from 'drizzle-orm/postgres-js';
import { eq, sql as drizzleSql, and, gte, desc } from 'drizzle-orm';
import { Ollama } from '@langchain/community/llms/ollama';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import type { Document } from '@langchain/core/documents';
import * as schema from '$lib/db/unified-schema';
import type { LegalDocument, DocumentChunk, UserAiQuery, AutoTag } from '$lib/db/schema-types';

// ===== CONFIGURATION & CONSTANTS =====

/**
 * RAG Pipeline Configuration
 */
interface RAGConfig {
  database: DatabaseConfig;
  redis: RedisConfig;
  ollama: OllamaConfig;
  rag: RAGSettings;
  security: SecuritySettings;
}

/**
 * Database Configuration
 */
interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  max: number;
  idle_timeout: number;
  ssl: boolean | string;
  connect_timeout: number;
}

/**
 * Redis Configuration  
 */
interface RedisConfig {
  host: string;
  port: number;
  db: number;
  maxRetriesPerRequest: number;
  cacheTtl: number;
  enableReadyCheck: boolean;
  lazyConnect: boolean;
}

/**
 * Ollama Configuration
 */
interface OllamaConfig {
  baseUrl: string;
  embeddingModel: string;
  llmModel: string;
  embeddingDimensions: number;
  timeout: number;
  temperature: number;
  numCtx: number;
  numPredict: number;
}

/**
 * RAG Settings
 */
interface RAGSettings {
  chunkSize: number;
  chunkOverlap: number;
  maxSources: number;
  similarityThreshold: number;
  timeoutMs: number;
  enableMetrics: boolean;
  enableAutoTagging: boolean;
  enableCaching: boolean;
  batchSize: number;
}

/**
 * Security Settings
 */
interface SecuritySettings {
  rateLimit: {
    perMinute: number;
    windowMs: number;
  };
  validation: {
    maxInputLength: number;
    maxDocumentSize: number;
    allowedDocumentTypes: string[];
  };
  sanitization: {
    removeHtmlTags: boolean;
    removeSqlChars: boolean;
    maxLineLength: number;
  };
}

/**
 * Default configuration with environment variable overrides
 */
const createDefaultConfig = (): RAGConfig => ({
  database: {
    host: process.env.DATABASE_HOST || 'localhost',
    port: parseInt(process.env.DATABASE_PORT || '5432'),
    database: process.env.DATABASE_NAME || 'legal_ai_db', 
    username: process.env.DATABASE_USER || 'legal_admin',
    password: process.env.DATABASE_PASSWORD || '123456',
    max: parseInt(process.env.DATABASE_MAX_CONNECTIONS || '20'),
    idle_timeout: parseInt(process.env.DATABASE_IDLE_TIMEOUT || '20'),
    ssl: process.env.NODE_ENV === 'production' ? 'require' : false,
    connect_timeout: parseInt(process.env.DATABASE_CONNECT_TIMEOUT || '10')
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    db: parseInt(process.env.REDIS_DB || '0'),
    maxRetriesPerRequest: parseInt(process.env.REDIS_MAX_RETRIES || '3'),
    cacheTtl: parseInt(process.env.RAG_CACHE_TTL || '86400'), // 24 hours
    enableReadyCheck: true,
    lazyConnect: false
  },
  ollama: {
    baseUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
    embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text:latest',
    llmModel: process.env.OLLAMA_LLM_MODEL || 'gemma3-legal:latest',
    embeddingDimensions: parseInt(process.env.OLLAMA_EMBEDDING_DIMENSIONS || '768'),
    timeout: parseInt(process.env.OLLAMA_TIMEOUT || '30000'),
    temperature: parseFloat(process.env.OLLAMA_TEMPERATURE || '0.3'),
    numCtx: parseInt(process.env.OLLAMA_NUM_CTX || '8192'),
    numPredict: parseInt(process.env.OLLAMA_NUM_PREDICT || '2048')
  },
  rag: {
    chunkSize: parseInt(process.env.RAG_CHUNK_SIZE || '1500'),
    chunkOverlap: parseInt(process.env.RAG_CHUNK_OVERLAP || '300'),
    maxSources: parseInt(process.env.RAG_MAX_SOURCES || '10'),
    similarityThreshold: parseFloat(process.env.RAG_SIMILARITY_THRESHOLD || '0.5'),
    timeoutMs: parseInt(process.env.RAG_TIMEOUT_MS || '30000'),
    enableMetrics: process.env.RAG_ENABLE_METRICS !== 'false',
    enableAutoTagging: process.env.RAG_ENABLE_AUTO_TAGGING !== 'false',
    enableCaching: process.env.RAG_ENABLE_CACHING !== 'false',
    batchSize: parseInt(process.env.RAG_BATCH_SIZE || '10')
  },
  security: {
    rateLimit: {
      perMinute: parseInt(process.env.RAG_RATE_LIMIT_PER_MINUTE || '60'),
      windowMs: parseInt(process.env.RAG_RATE_LIMIT_WINDOW_MS || '60000')
    },
    validation: {
      maxInputLength: parseInt(process.env.RAG_MAX_INPUT_LENGTH || '10000'),
      maxDocumentSize: parseInt(process.env.RAG_MAX_DOCUMENT_SIZE || '10485760'), // 10MB
      allowedDocumentTypes: (process.env.RAG_ALLOWED_DOC_TYPES || 'contract,statute,case_law,brief,memo').split(',')
    },
    sanitization: {
      removeHtmlTags: process.env.RAG_REMOVE_HTML_TAGS !== 'false',
      removeSqlChars: process.env.RAG_REMOVE_SQL_CHARS !== 'false',
      maxLineLength: parseInt(process.env.RAG_MAX_LINE_LENGTH || '2000')
    }
  }
});

// ===== INTERFACES & TYPES =====

/**
 * Document Ingestion Parameters
 */
export interface DocumentIngestionParams {
  title: string;
  content: string;
  documentType: string;
  metadata?: Record<string, any>;
  caseId?: string;
  userId: string;
  confidentialityLevel?: 'public' | 'confidential' | 'privileged' | 'attorney_client';
  jurisdiction?: string;
  clientId?: string;
}

/**
 * Search Parameters
 */
export interface SearchParams {
  query: string;
  caseId?: string;
  documentType?: string;
  limit?: number;
  threshold?: number;
  userId?: string;
  includeMetadata?: boolean;
  sortBy?: 'relevance' | 'date' | 'score';
}

/**
 * Question Answering Parameters
 */
export interface QuestionParams {
  question: string;
  caseId?: string;
  userId: string;
  conversationContext?: string;
  confidentialityLevel?: string;
  requireSources?: boolean;
  maxSources?: number;
}

/**
 * Search Result Document
 */
export interface SearchResult {
  id: string;
  content: string;
  title: string;
  documentId: string;
  score: number;
  similarity: number;
  textRank: number;
  metadata: Record<string, any>;
  confidentialityLevel?: string;
  highlights?: string[];
}

/**
 * Answer Result
 */
export interface AnswerResult {
  answer: string;
  sources: Array<{
    id: string;
    title: string;
    score: number;
    excerpt: string;
    confidentialityLevel?: string;
  }>;
  confidence: number;
  keyPoints: string[];
  processingTime: number;
  citations?: string[];
  legalPrecedents?: string[];
  riskAssessment?: {
    level: 'low' | 'medium' | 'high';
    factors: string[];
  };
}

/**
 * Contract Analysis Result
 */
export interface ContractAnalysisResult {
  contractType: string;
  parties: string[];
  keyTerms: string[];
  risks: Array<{
    description: string;
    severity: 'low' | 'medium' | 'high';
    category: string;
  }>;
  legalIssues: string[];
  recommendations: string[];
  confidence: number;
  processingTime: number;
  complianceFlags?: string[];
  jurisdiction?: string;
}

/**
 * Ingestion Result
 */
export interface IngestionResult {
  documentId: string;
  chunksCreated: number;
  tags: string[];
  processingTime: number;
  success: boolean;
  errors?: string[];
  metadata?: Record<string, any>;
  confidentialityLevel?: string;
}

// ===== UTILITY CLASSES =====

/**
 * Advanced Input Validation and Sanitization
 */
class InputValidator {
  private config: SecuritySettings;

  constructor(config: SecuritySettings) {
    this.config = config;
  }

  validateAndSanitize(input: string, maxLength?: number): string {
    if (!input || typeof input !== 'string') {
      throw new Error('Input must be a non-empty string');
    }

    let sanitized = input.trim();

    // Remove HTML tags if enabled
    if (this.config.sanitization.removeHtmlTags) {
      sanitized = sanitized.replace(/<[^>]*>/g, '');
    }

    // Remove SQL injection characters if enabled
    if (this.config.sanitization.removeSqlChars) {
      sanitized = sanitized.replace(/[;'"`]/g, '');
    }

    // Truncate excessively long lines
    if (this.config.sanitization.maxLineLength > 0) {
      sanitized = sanitized
        .split('\n')
        .map(line => line.length > this.config.sanitization.maxLineLength ? 
          line.substring(0, this.config.sanitization.maxLineLength) + '...' : line)
        .join('\n');
    }

    const effectiveMaxLength = maxLength || this.config.validation.maxInputLength;
    if (sanitized.length > effectiveMaxLength) {
      throw new Error(`Input exceeds maximum length of ${effectiveMaxLength} characters`);
    }

    return sanitized;
  }

  validateUUID(uuid: string): boolean {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    return uuidRegex.test(uuid);
  }

  validateDocumentType(documentType: string): boolean {
    return this.config.validation.allowedDocumentTypes.includes(documentType.toLowerCase());
  }

  validateConfidentialityLevel(level: string): boolean {
    const validLevels = ['public', 'confidential', 'privileged', 'attorney_client'];
    return validLevels.includes(level.toLowerCase());
  }
}

/**
 * Advanced Rate Limiting
 */
class RateLimiter {
  private requests = new Map<string, number[]>();
  private config: SecuritySettings['rateLimit'];

  constructor(config: SecuritySettings['rateLimit']) {
    this.config = config;
  }

  isAllowed(identifier: string): boolean {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;
    
    let requests = this.requests.get(identifier) || [];
    requests = requests.filter(time => time > windowStart);
    
    if (requests.length >= this.config.perMinute) {
      return false;
    }
    
    requests.push(now);
    this.requests.set(identifier, requests);
    
    return true;
  }

  getRemainingRequests(identifier: string): number {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;
    
    const requests = this.requests.get(identifier) || [];
    const validRequests = requests.filter(time => time > windowStart);
    
    return Math.max(0, this.config.perMinute - validRequests.length);
  }

  getTimeUntilReset(identifier: string): number {
    const requests = this.requests.get(identifier) || [];
    if (requests.length === 0) return 0;
    
    const oldestRequest = Math.min(...requests);
    const resetTime = oldestRequest + this.config.windowMs;
    
    return Math.max(0, resetTime - Date.now());
  }
}

/**
 * Comprehensive Metrics Collection
 */
class MetricsCollector {
  private metrics = new Map<string, number[]>();
  private counters = new Map<string, number>();
  private labels = new Map<string, Map<string, number>>();

  recordTiming(operation: string, duration: number, labels?: Record<string, string>): void {
    const timings = this.metrics.get(operation) || [];
    timings.push(duration);
    
    // Keep only last 1000 measurements
    if (timings.length > 1000) {
      timings.shift();
    }
    
    this.metrics.set(operation, timings);

    // Record labeled metrics
    if (labels) {
      for (const [key, value] of Object.entries(labels)) {
        const labelKey = `${operation}_${key}_${value}`;
        this.incrementCounter(labelKey);
      }
    }
  }

  incrementCounter(name: string, value: number = 1, labels?: Record<string, string>): void {
    const current = this.counters.get(name) || 0;
    this.counters.set(name, current + value);

    // Record labeled counters
    if (labels) {
      for (const [key, labelValue] of Object.entries(labels)) {
        const labelKey = `${name}_${key}`;
        if (!this.labels.has(labelKey)) {
          this.labels.set(labelKey, new Map());
        }
        const labelMap = this.labels.get(labelKey)!;
        labelMap.set(labelValue, (labelMap.get(labelValue) || 0) + value);
      }
    }
  }

  getMetrics(): Record<string, any> {
    const result: Record<string, any> = {};
    
    // Add timing metrics with percentiles
    for (const [operation, timings] of this.metrics.entries()) {
      if (timings.length > 0) {
        const sorted = [...timings].sort((a, b) => a - b);
        result[`${operation}_avg_ms`] = timings.reduce((a, b) => a + b, 0) / timings.length;
        result[`${operation}_count`] = timings.length;
        result[`${operation}_p50_ms`] = sorted[Math.floor(sorted.length * 0.5)];
        result[`${operation}_p95_ms`] = sorted[Math.floor(sorted.length * 0.95)];
        result[`${operation}_p99_ms`] = sorted[Math.floor(sorted.length * 0.99)];
        result[`${operation}_min_ms`] = sorted[0];
        result[`${operation}_max_ms`] = sorted[sorted.length - 1];
      }
    }
    
    // Add counter metrics
    for (const [name, value] of this.counters.entries()) {
      result[name] = value;
    }

    // Add labeled metrics
    for (const [labelKey, labelMap] of this.labels.entries()) {
      result[labelKey] = Object.fromEntries(labelMap);
    }
    
    return result;
  }

  reset(): void {
    this.metrics.clear();
    this.counters.clear();
    this.labels.clear();
  }
}

/**
 * Legal Document Chunking Strategies
 */
class LegalChunker {
  private textSplitter: RecursiveCharacterTextSplitter;
  private config: RAGSettings;

  constructor(config: RAGSettings) {
    this.config = config;
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: config.chunkSize,
      chunkOverlap: config.chunkOverlap,
      separators: [
        '\n\nSECTION', '\n\nARTICLE', '\n\nCLAUSE', // Legal sections
        '\n\n§', '\n\n¶', // Legal symbols
        '\n\nWHEREAS', '\n\nNOW THEREFORE', // Contract terms
        '\n\nFACTS', '\n\nHOLDING', '\n\nANALYSIS', // Case law
        '\n\n', '\n', '.', '!', '?', ';', ':', ' ', ''
      ],
      keepSeparator: true,
    });
  }

  async chunkDocument(content: string, documentType: string): Promise<string[]> {
    const chunks: string[] = [];

    // Document type specific patterns
    const patterns = {
      contract: [
        /(?:^|\n)(?:WHEREAS|NOW THEREFORE|SECTION|ARTICLE|CLAUSE)\s+[^\n]*/gi,
        /(?:^|\n)\d+\.\s+[A-Z][^\n]+/g,
        /(?:^|\n)(?:Party|Parties|Agreement|Terms|Conditions)/gi
      ],
      statute: [
        /(?:^|\n)(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+[\d.]+[^\n]*/gi,
        /(?:^|\n)§\s*[\d.]+[^\n]*/g,
        /(?:^|\n)(?:TITLE|CHAPTER|PART)\s+[IVX\d]+/gi
      ],
      case_law: [
        /(?:^|\n)(?:FACTS|HOLDING|ANALYSIS|CONCLUSION|DISSENT|MAJORITY|CONCURRENCE)\s*[^\n]*/gi,
        /(?:^|\n)[IVX]+\.\s+[A-Z][^\n]+/g,
        /(?:^|\n)(?:Plaintiff|Defendant|Appellant|Appellee)/gi
      ],
      brief: [
        /(?:^|\n)(?:ARGUMENT|STATEMENT|CONCLUSION|INTRODUCTION)\s*[^\n]*/gi,
        /(?:^|\n)[IVX]+\.\s+[A-Z][^\n]+/g,
        /(?:^|\n)(?:Issue|Question|Standard of Review)/gi
      ],
      memo: [
        /(?:^|\n)(?:TO|FROM|RE|DATE|MEMORANDUM)\s*[^\n]*/gi,
        /(?:^|\n)(?:BACKGROUND|ANALYSIS|RECOMMENDATION|CONCLUSION)\s*[^\n]*/gi,
        /(?:^|\n)\d+\.\s+[A-Z][^\n]+/g
      ]
    };

    // Try legal structure-based chunking first
    const docPatterns = patterns[documentType as keyof typeof patterns] || patterns.contract;
    let structuredChunks: string[] = [];

    for (const pattern of docPatterns) {
      try {
        const matches = content.match(pattern);
        if (matches && matches.length > 1) {
          // Split content by pattern matches
          const sections = content.split(pattern).filter(Boolean);
          
          structuredChunks = sections
            .filter(section => section.trim().length > 50)
            .map(section => section.trim());
          
          if (structuredChunks.length > 1) break;
        }
      } catch (error) {
        console.warn(`Pattern matching failed for ${documentType}:`, error);
      }
    }

    // Fallback to standard recursive chunking
    if (structuredChunks.length === 0) {
      const docs = await this.textSplitter.createDocuments([content]);
      structuredChunks = docs.map(d => d.pageContent);
    }

    // Further split large chunks if needed
    for (const chunk of structuredChunks) {
      if (chunk.length > this.config.chunkSize * 1.5) {
        const subDocs = await this.textSplitter.createDocuments([chunk]);
        chunks.push(...subDocs.map(d => d.pageContent));
      } else if (chunk.trim().length > 0) {
        chunks.push(chunk.trim());
      }
    }

    return chunks.filter(chunk => chunk.length > 10); // Filter out tiny chunks
  }

  // Enhanced chunking for specific legal document sections
  extractLegalSections(content: string, documentType: string): Record<string, string> {
    const sections: Record<string, string> = {};

    const sectionPatterns = {
      contract: {
        parties: /(?:PARTIES|Party|Parties to this Agreement)[^\n]*\n([\s\S]*?)(?=\n(?:RECITALS|WHEREAS|BACKGROUND|TERMS|$))/i,
        recitals: /(?:RECITALS|WHEREAS)[^\n]*\n([\s\S]*?)(?=\n(?:NOW THEREFORE|TERMS|AGREEMENT|$))/i,
        terms: /(?:TERMS|AGREEMENT|NOW THEREFORE)[^\n]*\n([\s\S]*?)(?=\n(?:SIGNATURES|EXECUTION|$))/i,
        signatures: /(?:SIGNATURES|EXECUTION|IN WITNESS WHEREOF)[^\n]*\n([\s\S]*?)$/i
      },
      case_law: {
        facts: /(?:FACTS|BACKGROUND|PROCEDURAL HISTORY)[^\n]*\n([\s\S]*?)(?=\n(?:ISSUE|HOLDING|ANALYSIS|$))/i,
        issues: /(?:ISSUE|ISSUES|QUESTION)[^\n]*\n([\s\S]*?)(?=\n(?:HOLDING|ANALYSIS|RULE|$))/i,
        holding: /(?:HOLDING|RULE|RULING)[^\n]*\n([\s\S]*?)(?=\n(?:ANALYSIS|REASONING|CONCLUSION|$))/i,
        analysis: /(?:ANALYSIS|REASONING|DISCUSSION)[^\n]*\n([\s\S]*?)(?=\n(?:CONCLUSION|DISSENT|$))/i,
        conclusion: /(?:CONCLUSION|DISPOSITION)[^\n]*\n([\s\S]*?)$/i
      },
      statute: {
        title: /(?:TITLE|CHAPTER|ACT)[^\n]*\n([\s\S]*?)(?=\n(?:SECTION|§|DEFINITIONS|$))/i,
        definitions: /(?:DEFINITIONS|TERMS)[^\n]*\n([\s\S]*?)(?=\n(?:SECTION|§|PROVISIONS|$))/i,
        provisions: /(?:PROVISIONS|REQUIREMENTS)[^\n]*\n([\s\S]*?)(?=\n(?:PENALTIES|ENFORCEMENT|$))/i,
        enforcement: /(?:ENFORCEMENT|PENALTIES|SANCTIONS)[^\n]*\n([\s\S]*?)$/i
      }
    };

    const patterns = sectionPatterns[documentType as keyof typeof sectionPatterns];
    if (patterns) {
      for (const [sectionName, pattern] of Object.entries(patterns)) {
        const match = content.match(pattern);
        if (match && match[1]) {
          sections[sectionName] = match[1].trim();
        }
      }
    }

    return sections;
  }
}

// ===== MAIN RAG PIPELINE CLASS =====

/**
 * Enhanced Legal RAG Pipeline
 * 
 * Comprehensive RAG system for legal AI applications with advanced features
 * for document processing, vector search, and intelligent question answering.
 */
export class EnhancedLegalRAGPipeline {
  private config: RAGConfig;
  private initialized = false;
  private sql?: postgres.Sql;
  private db?: ReturnType<typeof drizzle>;
  private redis?: Redis;
  private embeddings?: OllamaEmbeddings;
  private llm?: Ollama;
  private validator: InputValidator;
  private rateLimiter: RateLimiter;
  private metrics: MetricsCollector;
  private chunker: LegalChunker;

  constructor(config?: Partial<RAGConfig>) {
    this.config = { ...createDefaultConfig(), ...config };
    this.validator = new InputValidator(this.config.security);
    this.rateLimiter = new RateLimiter(this.config.security.rateLimit);
    this.metrics = new MetricsCollector();
    this.chunker = new LegalChunker(this.config.rag);
  }

  /**
   * Initialize all pipeline components
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    const startTime = Date.now();

    try {
      console.log('[RAG] Initializing Enhanced Legal RAG Pipeline...');

      // Initialize PostgreSQL connection
      await this.initializeDatabase();

      // Initialize Redis connection
      await this.initializeRedis();

      // Initialize Ollama LLM and embeddings
      await this.initializeOllama();

      // Verify all connections
      await this.verifyConnections();

      this.initialized = true;
      this.metrics.incrementCounter('pipeline_initializations');
      this.metrics.recordTiming('initialization_time', Date.now() - startTime);

      console.log(`[RAG] Pipeline initialized successfully in ${Date.now() - startTime}ms`);

    } catch (error) {
      console.error('[RAG] Initialization failed:', error);
      this.metrics.incrementCounter('initialization_errors');
      throw new Error(`RAG Pipeline initialization failed: ${error}`);
    }
  }

  /**
   * Initialize database connection
   */
  private async initializeDatabase(): Promise<void> {
    try {
      this.sql = postgres({
        host: this.config.database.host,
        port: this.config.database.port,
        database: this.config.database.database,
        username: this.config.database.username,
        password: this.config.database.password,
        max: this.config.database.max,
        idle_timeout: this.config.database.idle_timeout,
        ssl: this.config.database.ssl,
        prepare: true,
        connect_timeout: this.config.database.connect_timeout,
        onnotice: (notice) => console.debug('[DB] Notice:', notice),
        onparameter: (key, value) => console.debug(`[DB] Parameter ${key}:`, value),
        onconnect: async () => {
          console.log('[DB] PostgreSQL connected successfully');
        }
      });

      this.db = drizzle(this.sql, { schema });

      // Test connection
      const testResult = await this.sql`SELECT 1 as test`;
      if (testResult[0]?.test !== 1) {
        throw new Error('Database connection test failed');
      }

      console.log('[RAG] Database initialized successfully');
    } catch (error) {
      throw new Error(`Database initialization failed: ${error}`);
    }
  }

  /**
   * Initialize Redis connection
   */
  private async initializeRedis(): Promise<void> {
    try {
      this.redis = new Redis({
        ...this.config.redis,
        retryStrategy: (times) => Math.min(times * 50, 2000),
        reconnectOnError: (err) => {
          console.warn('Redis reconnect on error:', err.message);
          return err.message.includes('READONLY');
        },
        lazyConnect: this.config.redis.lazyConnect
      });

      // Test connection
      await this.redis.set('health-check', 'ok');
      console.log('[RAG] Redis initialized successfully');
    } catch (error) {
      throw new Error(`Redis initialization failed: ${error}`);
    }
  }

  /**
   * Initialize Ollama components
   */
  private async initializeOllama(): Promise<void> {
    try {
      // Initialize embeddings
      this.embeddings = new OllamaEmbeddings({
        baseUrl: this.config.ollama.baseUrl,
        model: this.config.ollama.embeddingModel,
        requestOptions: {
          useMMap: true,
          numThread: 8,
        },
      });

      // Initialize LLM
      this.llm = new Ollama({
        baseUrl: this.config.ollama.baseUrl,
        model: this.config.ollama.llmModel,
        temperature: this.config.ollama.temperature,
        numCtx: this.config.ollama.numCtx,
        numPredict: this.config.ollama.numPredict,
        topK: 40,
        topP: 0.9,
        repeatPenalty: 1.1,
        callbacks: [
          {
            handleLLMStart: async () => {
              console.debug(`[RAG] LLM Started: ${this.config.ollama.llmModel}`);
              this.metrics.incrementCounter('llm_requests');
            },
            handleLLMEnd: async () => {
              console.debug('[RAG] LLM Completed');
              this.metrics.incrementCounter('llm_completions');
            },
            handleLLMError: async (err) => {
              console.error('[RAG] LLM Error:', err);
              this.metrics.incrementCounter('llm_errors');
            },
          },
        ],
      });

      console.log('[RAG] Ollama components initialized successfully');
    } catch (error) {
      throw new Error(`Ollama initialization failed: ${error}`);
    }
  }

  /**
   * Verify all connections are working
   */
  private async verifyConnections(): Promise<void> {
    try {
      // Test database
      await this.sql!`SELECT 1 as test`;

      // Test Redis
      await this.redis!.set('health-check', 'ok');

      // Test Ollama embeddings
      const testEmbedding = await this.embeddings!.embedQuery('test');
      if (testEmbedding.length !== this.config.ollama.embeddingDimensions) {
        throw new Error(`Expected ${this.config.ollama.embeddingDimensions} dimensions, got ${testEmbedding.length}`);
      }

      console.log('[RAG] All connections verified successfully');
    } catch (error) {
      throw new Error(`Connection verification failed: ${error}`);
    }
  }

  /**
   * Ensure pipeline is initialized
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
  }

  // ===== DOCUMENT INGESTION =====

  /**
   * Ingest a legal document with comprehensive processing
   */
  async ingestLegalDocument(params: DocumentIngestionParams): Promise<IngestionResult> {
    const startTime = Date.now();
    
    try {
      // Validate and sanitize inputs
      const title = this.validator.validateAndSanitize(params.title, 500);
      const content = this.validator.validateAndSanitize(params.content, this.config.security.validation.maxDocumentSize);
      const documentType = this.validator.validateAndSanitize(params.documentType, 50);
      const userId = params.userId;
      
      if (!this.validator.validateUUID(userId)) {
        throw new Error('Invalid user ID format');
      }

      if (!this.validator.validateDocumentType(documentType)) {
        throw new Error(`Invalid document type: ${documentType}`);
      }

      if (params.confidentialityLevel && !this.validator.validateConfidentialityLevel(params.confidentialityLevel)) {
        throw new Error(`Invalid confidentiality level: ${params.confidentialityLevel}`);
      }

      // Rate limiting
      if (!this.rateLimiter.isAllowed(userId)) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      await this.ensureInitialized();

      const { caseId, metadata = {}, confidentialityLevel = 'public', jurisdiction, clientId } = params;

      // Start transaction for document creation
      const [document] = await this.db!.transaction(async (tx) => {
        const [doc] = await tx.insert(schema.legalDocuments)
          .values({
            title,
            content: content.substring(0, 10000), // Preview content
            fullText: content,
            documentType,
            keywords: metadata.keywords || [],
            topics: metadata.topics || [],
            jurisdiction: jurisdiction || metadata.jurisdiction,
            caseId,
            createdBy: userId,
            confidentialityLevel,
            clientId,
            metadata: {
              ...metadata,
              ingestionDate: new Date().toISOString(),
              version: '1.0',
              source: 'rag_pipeline'
            }
          })
          .returning();

        return [doc];
      });

      console.log(`[RAG] Created document: ${document.id}`);

      // Generate document-level embedding
      const docEmbedding = await this.generateEmbedding(
        `${title}\n${content.substring(0, 2000)}`
      );

      await this.db!.update(schema.legalDocuments)
        .set({ embedding: JSON.stringify(docEmbedding) })
        .where(eq(schema.legalDocuments.id, document.id));

      // Smart legal chunking
      const chunks = await this.chunker.chunkDocument(content, documentType);
      console.log(`[RAG] Split into ${chunks.length} chunks`);

      // Extract legal sections for enhanced metadata
      const legalSections = this.chunker.extractLegalSections(content, documentType);

      // Process chunks in batches
      let successfulChunks = 0;
      const errors: string[] = [];

      for (let i = 0; i < chunks.length; i += this.config.rag.batchSize) {
        const batch = chunks.slice(i, i + this.config.rag.batchSize);

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
                    confidentialityLevel,
                    legalSections: Object.keys(legalSections),
                    ...metadata,
                  },
                };
              } catch (error) {
                const errorMsg = `Failed to process chunk ${i + idx}: ${error}`;
                errors.push(errorMsg);
                console.error(errorMsg);
                return null;
              }
            })
          );

          // Filter out failed chunks and insert valid ones
          const validChunks = chunkRecords.filter(record => record !== null);

          if (validChunks.length > 0) {
            await this.db!.insert(schema.documentChunks).values(validChunks as unknown[]);
          }

          console.debug(`[RAG] Processed batch ${Math.floor(i / this.config.rag.batchSize) + 1}/${Math.ceil(chunks.length / this.config.rag.batchSize)}`);
        } catch (error) {
          const errorMsg = `Failed to process batch ${Math.floor(i / this.config.rag.batchSize) + 1}: ${error}`;
          errors.push(errorMsg);
          console.error(errorMsg);
        }
      }

      // Auto-generate tags if enabled
      let tags: Array<{ tag: string; confidence: number }> = [];
      if (this.config.rag.enableAutoTagging) {
        try {
          tags = await this.generateAutoTags(content, documentType);

          for (const tag of tags) {
            await this.db!.insert(schema.autoTags).values({
              entityId: document.id,
              entityType: 'document',
              tag: tag.tag,
              confidence: tag.confidence.toString(),
              source: 'ai_analysis',
              model: this.config.ollama.llmModel,
            });
          }
        } catch (error) {
          const errorMsg = `Failed to generate auto-tags: ${error}`;
          errors.push(errorMsg);
          console.warn(errorMsg);
        }
      }

      const processingTime = Date.now() - startTime;
      const success = successfulChunks > 0;

      console.log(`[RAG] Document ingestion completed in ${processingTime}ms (${successfulChunks}/${chunks.length} chunks successful)`);

      this.metrics.incrementCounter('documents_ingested');
      this.metrics.recordTiming('ingestion_time', processingTime, { 
        document_type: documentType,
        confidentiality_level: confidentialityLevel 
      });

      return {
        documentId: document.id,
        chunksCreated: successfulChunks,
        tags: tags.map(t => t.tag),
        processingTime,
        success,
        errors: errors.length > 0 ? errors : undefined,
        metadata: {
          documentType,
          confidentialityLevel,
          legalSections: Object.keys(legalSections),
          totalChunks: chunks.length
        },
        confidentialityLevel
      };

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      console.error('[RAG] Ingestion error:', error);
      this.metrics.incrementCounter('ingestion_errors');
      this.metrics.recordTiming('ingestion_error_time', processingTime);

      throw error;
    }
  }

  // ===== SEARCH & RETRIEVAL =====

  /**
   * Perform hybrid vector and keyword search
   */
  async hybridSearch(params: SearchParams): Promise<SearchResult[]> {
    const startTime = Date.now();
    
    try {
      const query = this.validator.validateAndSanitize(params.query, 1000);
      const { 
        caseId, 
        documentType, 
        limit = this.config.rag.maxSources, 
        threshold = this.config.rag.similarityThreshold,
        userId,
        includeMetadata = true,
        sortBy = 'relevance'
      } = params;

      // Rate limiting if userId provided
      if (userId && !this.rateLimiter.isAllowed(userId)) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      await this.ensureInitialized();

      // Generate query embedding with caching
      const queryEmbedding = await this.generateEmbedding(query);

      // Build SQL conditions using template literals
      let vectorWhereClause = `1 - (dc.embedding::vector <=> '${JSON.stringify(queryEmbedding)}'::vector) > ${threshold}`;
      let keywordWhereClause = `to_tsvector('english', dc.content) @@ plainto_tsquery('english', '${query.replace(/'/g, "''")}')`;
      
      if (caseId && this.validator.validateUUID(caseId)) {
        vectorWhereClause += ` AND dc.metadata->>'caseId' = '${caseId}'`;
        keywordWhereClause += ` AND dc.metadata->>'caseId' = '${caseId}'`;
      }

      if (documentType) {
        vectorWhereClause += ` AND dc.document_type = '${documentType}'`;
        keywordWhereClause += ` AND dc.document_type = '${documentType}'`;
      }

      // Perform vector similarity search
      const vectorResults = await this.sql!`
        SELECT 
          dc.id,
          dc.content,
          dc.metadata,
          dc.document_id,
          ld.title,
          ld.confidentiality_level,
          1 - (dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector) as similarity
        FROM document_chunks dc
        LEFT JOIN legal_documents ld ON dc.document_id = ld.id
        WHERE ${drizzleSql.raw(vectorWhereClause)}
        ORDER BY dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector
        LIMIT ${limit * 2}
      `;

      // Perform keyword search
      const keywordResults = await this.sql!`
        SELECT 
          dc.id,
          dc.content,
          dc.metadata,
          dc.document_id,
          ld.title,
          ld.confidentiality_level,
          ts_rank(to_tsvector('english', dc.content), 
                  plainto_tsquery('english', ${query})) as text_rank
        FROM document_chunks dc
        LEFT JOIN legal_documents ld ON dc.document_id = ld.id
        WHERE ${drizzleSql.raw(keywordWhereClause)}
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
          highlights: this.extractHighlights(r.content, query)
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
            highlights: this.extractHighlights(r.content, query)
          });
        }
      });

      // Sort by combined score or other criteria
      let sortedResults = Array.from(combinedResults.values());
      
      switch (sortBy) {
        case 'date':
          sortedResults.sort((a, b) => 
            new Date(b.metadata?.ingestionDate || 0).getTime() - 
            new Date(a.metadata?.ingestionDate || 0).getTime()
          );
          break;
        case 'score':
          sortedResults.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
          break;
        default: // relevance
          sortedResults.sort((a, b) => b.score - a.score);
      }

      sortedResults = sortedResults.slice(0, limit);

      // Convert to SearchResult format
      const searchResults: SearchResult[] = sortedResults.map(r => ({
        id: r.id,
        content: r.content,
        title: r.title || 'Untitled',
        documentId: r.document_id,
        score: r.score,
        similarity: r.similarity || 0,
        textRank: r.text_rank || 0,
        metadata: includeMetadata ? r.metadata : {},
        confidentialityLevel: r.confidentiality_level,
        highlights: r.highlights
      }));

      this.metrics.incrementCounter('searches_performed');
      this.metrics.recordTiming('search_time', Date.now() - startTime, {
        document_type: documentType || 'all',
        sort_by: sortBy
      });

      return searchResults;

    } catch (error) {
      console.error('[RAG] Search error:', error);
      this.metrics.incrementCounter('search_errors');
      throw error;
    }
  }

  // ===== QUESTION ANSWERING =====

  /**
   * Answer legal questions with comprehensive context
   */
  async answerLegalQuestion(params: QuestionParams): Promise<AnswerResult> {
    const startTime = Date.now();
    
    try {
      const question = this.validator.validateAndSanitize(params.question, 2000);
      const { 
        caseId, 
        userId, 
        conversationContext, 
        confidentialityLevel,
        requireSources = true,
        maxSources = 5
      } = params;

      if (!this.validator.validateUUID(userId)) {
        throw new Error('Invalid user ID format');
      }

      // Rate limiting
      if (!this.rateLimiter.isAllowed(userId)) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      await this.ensureInitialized();

      // Retrieve relevant context
      const relevantDocs = await this.hybridSearch({
        query: question,
        caseId,
        limit: maxSources,
        threshold: 0.6,
        userId,
        sortBy: 'relevance'
      });

      if (requireSources && relevantDocs.length === 0) {
        return {
          answer: "I couldn't find relevant information in the knowledge base to answer your question. Please provide more context or try rephrasing your question.",
          sources: [],
          confidence: 0,
          keyPoints: [],
          processingTime: Date.now() - startTime,
        };
      }

      // Build context from retrieved documents
      const context = relevantDocs
        .map((doc, idx) => `[Source ${idx + 1}]:\nTitle: ${doc.title}\nContent: ${doc.content}\nConfidentiality: ${doc.confidentialityLevel || 'public'}`)
        .join('\n\n---\n\n');

      // Create enhanced legal prompt
      const promptTemplate = PromptTemplate.fromTemplate(`
You are an expert legal AI assistant specializing in legal analysis and research. Answer the question based ONLY on the provided context.

${conversationContext ? `Previous Conversation Context:\n${conversationContext}\n\n` : ''}

Legal Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, accurate answer based solely on the context provided
2. Cite specific sources using [Source N] notation when referencing information
3. Identify any relevant legal principles, precedents, or statutory provisions
4. Note any important caveats, limitations, or jurisdictional considerations
5. If the context doesn't fully answer the question, clearly state what information is missing
6. Maintain a professional legal tone appropriate for ${confidentialityLevel || 'general'} matters
7. Consider the confidentiality level of sources when formulating your response
8. Highlight any potential legal risks or compliance issues
9. Provide actionable recommendations where appropriate

Answer:
      `);

      // Create chain and generate answer
      const chain = RunnableSequence.from([
        {
          context: () => context,
          question: new RunnablePassthrough(),
        },
        promptTemplate,
        this.llm!,
        new StringOutputParser(),
      ]);

      const llmResponse = await Promise.race([
        chain.invoke(question),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('LLM response timed out')), this.config.rag.timeoutMs)
        ),
      ]);

      // Handle streaming response or direct string
      const answer = typeof llmResponse === 'string' 
        ? llmResponse 
        : (llmResponse as any).parse || (llmResponse as any).content || String(llmResponse);

      // Analyze answer quality and extract insights
      const analysis = await this.analyzeAnswer(answer, relevantDocs);

      // Extract legal citations and precedents
      const citations = this.extractCitations(answer);
      const legalPrecedents = this.extractLegalPrecedents(answer);

      // Assess legal risks mentioned in the answer
      const riskAssessment = this.assessLegalRisks(answer);

      // Log the query for analytics and compliance
      try {
        const queryEmbedding = await this.generateEmbedding(question);

        await this.db!.insert(schema.userAiQueries).values({
          userId,
          caseId,
          query: question,
          response: answer,
          model: this.config.ollama.llmModel,
          queryType: 'legal_research',
          confidence: analysis.confidence.toString(),
          processingTime: Date.now() - startTime,
          contextUsed: relevantDocs.map(d => d.documentId),
          embedding: JSON.stringify(queryEmbedding),
          metadata: {
            sourcesCount: relevantDocs.length,
            keyPoints: analysis.keyPoints,
            confidentialityLevel,
            citations: citations.length,
            legalPrecedents: legalPrecedents.length,
            riskLevel: riskAssessment.level
          },
        });
      } catch (error) {
        console.warn('Failed to log query:', error);
        // Don't fail the main operation for logging issues
      }

      const result: AnswerResult = {
        answer,
        sources: relevantDocs.map(d => ({
          id: d.documentId,
          title: d.title,
          score: d.score,
          excerpt: d.content.substring(0, 200) + '...',
          confidentialityLevel: d.confidentialityLevel
        })),
        confidence: analysis.confidence,
        keyPoints: analysis.keyPoints,
        processingTime: Date.now() - startTime,
        citations,
        legalPrecedents,
        riskAssessment
      };

      this.metrics.incrementCounter('questions_answered');
      this.metrics.recordTiming('qa_time', result.processingTime, {
        confidentiality_level: confidentialityLevel || 'general',
        sources_count: relevantDocs.length.toString()
      });

      return result;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      console.error('[RAG] QA error:', error);
      this.metrics.incrementCounter('qa_errors');

      // Log failed query
      try {
        await this.db!.insert(schema.userAiQueries).values({
          userId: params.userId,
          caseId: params.caseId,
          query: params.question,
          response: '',
          model: this.config.ollama.llmModel,
          isSuccessful: false,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          processingTime,
        });
      } catch (logError) {
        console.warn('Failed to log error query:', logError);
      }

      throw error;
    }
  }

  // ===== CONTRACT ANALYSIS =====

  /**
   * Analyze contracts with detailed legal assessment
   */
  async analyzeContract(contractText: string, jurisdiction?: string): Promise<ContractAnalysisResult> {
    const startTime = Date.now();
    
    try {
      const sanitizedText = this.validator.validateAndSanitize(contractText, 1048576); // 1MB limit
      
      await this.ensureInitialized();

      const contractPrompt = PromptTemplate.fromTemplate(`
You are a legal expert specializing in contract analysis with extensive experience in ${jurisdiction || 'various jurisdictions'}. Analyze the following contract and provide a comprehensive structured assessment.

${jurisdiction ? `Jurisdiction: ${jurisdiction}\n` : ''}

Contract:
{contract}

Provide your analysis in the following structured format:

1. CONTRACT TYPE & PARTIES
- Type of contract (e.g., Service Agreement, NDA, Employment Contract)
- Parties involved (identify each party and their role)
- Governing law/jurisdiction
- Effective date and term

2. KEY TERMS & OBLIGATIONS
- Primary obligations of each party
- Payment terms and conditions
- Performance standards and deliverables
- Duration, renewal, and termination clauses
- Notice requirements

3. RISK ASSESSMENT
- Potential risks for each party (classify as HIGH, MEDIUM, LOW)
- Liability limitations and caps
- Indemnification clauses and scope
- Insurance requirements
- Force majeure provisions
- Intellectual property considerations

4. LEGAL ISSUES & COMPLIANCE
- Ambiguous terms requiring clarification
- Potential enforceability issues
- Missing standard clauses or protections
- Regulatory compliance considerations
- Dispute resolution mechanisms

5. RECOMMENDATIONS
- Suggested modifications to reduce risk
- Key points for negotiation
- Additional clauses to consider
- Compliance requirements to address

6. COMPLIANCE FLAGS
- Identify any potential regulatory issues
- Data privacy and security considerations
- Industry-specific compliance requirements

Provide specific clause references and line numbers where applicable. Focus on practical legal advice.
      `);

      const chain = RunnableSequence.from([
        contractPrompt,
        this.llm!,
        new StringOutputParser(),
      ]);

      const llmResponse = await Promise.race([
        chain.invoke({ contract: sanitizedText }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Contract analysis timed out')), this.config.rag.timeoutMs)
        ),
      ]);

      // Handle streaming response or direct string
      const analysis = typeof llmResponse === 'string' 
        ? llmResponse 
        : (llmResponse as any).parse || (llmResponse as any).content || String(llmResponse);

      const parsedAnalysis = this.parseContractAnalysis(analysis);
      const complianceFlags = this.extractComplianceFlags(analysis);
      const processingTime = Date.now() - startTime;

      this.metrics.incrementCounter('contracts_analyzed');
      this.metrics.recordTiming('contract_analysis_time', processingTime, {
        jurisdiction: jurisdiction || 'general'
      });

      return {
        ...parsedAnalysis,
        confidence: 0.85,
        processingTime,
        complianceFlags,
        jurisdiction
      };

    } catch (error) {
      console.error('[RAG] Contract analysis error:', error);
      this.metrics.incrementCounter('contract_analysis_errors');
      throw error;
    }
  }

  // ===== HELPER METHODS =====

  /**
   * Generate embeddings with caching
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    const textHash = this.hashText(text);

    try {
      // Check cache first if enabled
      if (this.config.rag.enableCaching && this.redis) {
        const cached = await this.redis.get(`embedding:${textHash}`);
        if (cached) {
          this.metrics.incrementCounter('cache_hits');
          return JSON.parse(cached);
        }
      }

      this.metrics.incrementCounter('cache_misses');

      // Generate new embedding
      const embedding = await this.embeddings!.embedQuery(text);

      // Cache for configured TTL if enabled
      if (this.config.rag.enableCaching && this.redis) {
        await this.redis.set(`embedding:${textHash}`, JSON.stringify(embedding));
        // Set expiration using pexpire (milliseconds) or just use temporary storage
      }

      this.metrics.incrementCounter('embeddings_generated');
      return embedding;

    } catch (error) {
      console.error('Embedding generation failed:', error);
      this.metrics.incrementCounter('embedding_errors');
      throw error;
    }
  }

  /**
   * Generate auto-tags for documents
   */
  private async generateAutoTags(content: string, documentType: string): Promise<Array<{ tag: string; confidence: number }>> {
    if (!this.config.rag.enableAutoTagging) return [];

    const tagPrompt = PromptTemplate.fromTemplate(`
Extract relevant legal tags from this {documentType} document. 
Focus on: legal concepts, practice areas, jurisdictions, case types, parties, and key legal topics.

Document excerpt:
{content}

Return ONLY a JSON array of tags with confidence scores (0-1):
[{"tag": "contract law", "confidence": 0.95}, {"tag": "intellectual property", "confidence": 0.87}, ...]

Limit to 10 most relevant tags.
    `);

    const chain = RunnableSequence.from([
      tagPrompt,
      this.llm!,
      new StringOutputParser(),
    ]);

    try {
      const llmResponse = await Promise.race([
        chain.invoke({
          documentType,
          content: content.substring(0, 3000),
        }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Auto-tagging timed out')), this.config.rag.timeoutMs / 2)
        ),
      ]);

      // Handle streaming response or direct string
      const response = typeof llmResponse === 'string' 
        ? llmResponse 
        : (llmResponse as any).parse || (llmResponse as any).content || String(llmResponse);

      // Extract JSON from response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const tags = JSON.parse(jsonMatch[0]);
        return Array.isArray(tags) ? tags.filter(t => t.tag && typeof t.confidence === 'number') : [];
      }

      return [];
    } catch (error) {
      console.warn('Auto-tagging failed:', error);
      return [];
    }
  }

  /**
   * Analyze answer quality and extract key points
   */
  private async analyzeAnswer(answer: string, sources: SearchResult[]) {
    // Calculate confidence based on source relevance and answer characteristics
    const avgScore = sources.length > 0 ? sources.reduce((sum, doc) => sum + doc.score, 0) / sources.length : 0;
    
    // Adjust confidence based on answer length and citation count
    const citations = (answer.match(/\[Source \d+\]/g) || []).length;
    const citationBonus = sources.length > 0 ? Math.min(citations / sources.length, 0.3) : 0;
    
    const baseConfidence = Math.min(0.95, avgScore + citationBonus);

    // Extract key points from structured answer
    const keyPoints = answer
      .split('\n')
      .filter(line => line.match(/^[\d.•-]|^[A-Z][a-z]+:/) && line.length > 10)
      .slice(0, 5)
      .map(line => line.replace(/^[.\d•-]*\s*/, '').trim())
      .filter(point => point.length > 0);

    return {
      confidence: Math.max(0.1, baseConfidence),
      keyPoints,
    };
  }

  /**
   * Extract highlights from content based on query
   */
  private extractHighlights(content: string, query: string): string[] {
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    const highlights: string[] = [];

    for (const word of words) {
      const regex = new RegExp(`\\b\\w*${word}\\w*\\b`, 'gi');
      const matches = content.match(regex);
      if (matches) {
        highlights.push(...matches.slice(0, 3)); // Limit highlights per word
      }
    }

    return [...new Set(highlights)].slice(0, 10); // Remove duplicates and limit total
  }

  /**
   * Extract legal citations from text
   */
  private extractCitations(text: string): string[] {
    const citationPatterns = [
      /\d+\s+[A-Z][a-z]+\.?\s+\d+/g, // Basic citation pattern
      /\d+\s+U\.S\.\s+\d+/g, // US Supreme Court
      /\d+\s+F\.\d*d?\s+\d+/g, // Federal courts
      /\d+\s+S\.Ct\.\s+\d+/g, // Supreme Court Reporter
      /\d+\s+[A-Z][a-z]+\.?\s+App\.?\s+\d+/g // State appellate
    ];

    const citations: string[] = [];
    for (const pattern of citationPatterns) {
      const matches = text.match(pattern);
      if (matches) {
        citations.push(...matches);
      }
    }

    return [...new Set(citations)].slice(0, 10); // Remove duplicates and limit
  }

  /**
   * Extract legal precedents from text
   */
  private extractLegalPrecedents(text: string): string[] {
    const precedentPatterns = [
      /(?:In|in)\s+([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)/g,
      /([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)(?:\s+holding|held|ruled)/gi,
      /(?:case|decision|ruling)\s+(?:of|in)\s+([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)/gi
    ];

    const precedents: string[] = [];
    for (const pattern of precedentPatterns) {
      const matches = text.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          precedents.push(match[1]);
        }
      }
    }

    return [...new Set(precedents)].slice(0, 5);
  }

  /**
   * Assess legal risks mentioned in text
   */
  private assessLegalRisks(text: string): { level: 'low' | 'medium' | 'high'; factors: string[] } {
    const highRiskTerms = ['breach', 'violation', 'penalty', 'criminal', 'fraud', 'negligence'];
    const mediumRiskTerms = ['liability', 'compliance', 'regulation', 'obligation', 'duty'];
    const lowRiskTerms = ['notice', 'disclosure', 'review', 'standard'];

    const lowerText = text.toLowerCase();
    const factors: string[] = [];

    let riskScore = 0;

    for (const term of highRiskTerms) {
      if (lowerText.includes(term)) {
        riskScore += 3;
        factors.push(`High risk: ${term} mentioned`);
      }
    }

    for (const term of mediumRiskTerms) {
      if (lowerText.includes(term)) {
        riskScore += 2;
        factors.push(`Medium risk: ${term} mentioned`);
      }
    }

    for (const term of lowRiskTerms) {
      if (lowerText.includes(term)) {
        riskScore += 1;
        factors.push(`Low risk: ${term} mentioned`);
      }
    }

    const level = riskScore >= 6 ? 'high' : riskScore >= 3 ? 'medium' : 'low';

    return {
      level,
      factors: factors.slice(0, 5) // Limit factors
    };
  }

  /**
   * Parse contract analysis results
   */
  private parseContractAnalysis(analysis: string): Omit<ContractAnalysisResult, 'confidence' | 'processingTime' | 'complianceFlags' | 'jurisdiction'> {
    const sections = {
      contractType: '',
      parties: [] as string[],
      keyTerms: [] as string[],
      risks: [] as Array<{ description: string; severity: 'low' | 'medium' | 'high'; category: string }>,
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
            if (!sections.contractType && !cleanLine.includes(':') && cleanLine.length > 3) {
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
              
              const category = cleanLine.toLowerCase().includes('liability') ? 'liability' :
                             cleanLine.toLowerCase().includes('compliance') ? 'compliance' :
                             cleanLine.toLowerCase().includes('financial') ? 'financial' : 'general';
              
              sections.risks.push({
                description: cleanLine,
                severity,
                category
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

  /**
   * Extract compliance flags from analysis
   */
  private extractComplianceFlags(analysis: string): string[] {
    const flags: string[] = [];
    const lowerAnalysis = analysis.toLowerCase();

    const flagPatterns = {
      'data_privacy': ['gdpr', 'privacy', 'personal data', 'data protection'],
      'securities': ['sec', 'securities', 'insider trading', 'disclosure'],
      'employment': ['employment law', 'labor', 'discrimination', 'wage'],
      'intellectual_property': ['ip', 'patent', 'trademark', 'copyright'],
      'anti_trust': ['antitrust', 'monopoly', 'competition', 'market'],
      'international': ['export', 'import', 'sanctions', 'foreign']
    };

    for (const [flag, terms] of Object.entries(flagPatterns)) {
      if (terms.some(term => lowerAnalysis.includes(term))) {
        flags.push(flag);
      }
    }

    return flags;
  }

  /**
   * Hash text for caching
   */
  private hashText(text: string): string {
    return crypto.createHash('sha256').update(text.trim()).digest('hex');
  }

  // ===== HEALTH & MONITORING =====

  /**
   * Get comprehensive health status
   */
  async getHealthStatus() {
    const checks = await Promise.allSettled([
      this.checkDatabaseHealth(),
      this.checkRedisHealth(),
      this.checkOllamaHealth(),
    ]);

    const services = ['Database', 'Redis', 'Ollama'];
    return checks.map((result, index) => ({
      service: services[index],
      status: result.status === 'fulfilled' ? 'healthy' : 'unhealthy',
      error: result.status === 'rejected' ? result.reason?.message : undefined,
      timestamp: new Date().toISOString()
    }));
  }

  private async checkDatabaseHealth() {
    if (!this.sql) throw new Error('Database not initialized');
    const result = await this.sql`SELECT 1 as test`;
    if (result[0]?.test !== 1) throw new Error('Database check failed');
  }

  private async checkRedisHealth() {
    if (!this.redis) throw new Error('Redis not initialized');
    await this.redis.set('health-check', 'ok');
  }

  private async checkOllamaHealth() {
    if (!this.embeddings) throw new Error('Ollama embeddings not initialized');
    const testEmbedding = await this.embeddings.embedQuery('test');
    if (testEmbedding.length !== this.config.ollama.embeddingDimensions) {
      throw new Error(`Expected ${this.config.ollama.embeddingDimensions} dimensions, got ${testEmbedding.length}`);
    }
  }

  /**
   * Get comprehensive metrics
   */
  getMetrics(): Record<string, any> {
    return {
      ...this.metrics.getMetrics(),
      config: {
        chunkSize: this.config.rag.chunkSize,
        maxSources: this.config.rag.maxSources,
        enableCaching: this.config.rag.enableCaching,
        enableAutoTagging: this.config.rag.enableAutoTagging
      },
      rateLimiting: {
        perMinute: this.config.security.rateLimit.perMinute,
        windowMs: this.config.security.rateLimit.windowMs
      }
    };
  }

  /**
   * Get rate limiting status for user
   */
  getRateLimitStatus(userId: string) {
    return {
      remaining: this.rateLimiter.getRemainingRequests(userId),
      resetTime: this.rateLimiter.getTimeUntilReset(userId),
      limit: this.config.security.rateLimit.perMinute
    };
  }

  // ===== CLEANUP =====

  /**
   * Clean shutdown of all connections
   */
  async close(): Promise<void> {
    try {
      await Promise.allSettled([
        this.redis ? new Promise(resolve => this.redis!.on('end', resolve)) : Promise.resolve(),
        this.sql?.end(),
      ]);
      
      this.initialized = false;
      console.log('[RAG] Pipeline closed successfully');
    } catch (error) {
      console.error('[RAG] Error during shutdown:', error);
    }
  }
}

// ===== EXPORTS =====

/**
 * Export enhanced singleton instance
 */
export const enhancedRAGPipeline = new EnhancedLegalRAGPipeline();

/**
 * Export the original interface for backward compatibility
 */
export const ragPipeline = enhancedRAGPipeline;

/**
 * Export configuration creator for custom instances
 */
export { createDefaultConfig };

/**
 * Export all interfaces for external use
 */
// Types already exported inline above - duplicate export removed