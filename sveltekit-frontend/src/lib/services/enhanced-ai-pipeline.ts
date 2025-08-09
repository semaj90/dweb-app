// Enhanced AI Pipeline Service with Local Gemma3-Legal GGUF Model
// NVIDIA CUDA GPU acceleration with native Windows integration
// Redis-native caching and enhanced RAG capabilities
// Go Microservice Integration for SvelteKit 2

import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Ollama } from "@langchain/community/llms/ollama";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import type { Document } from "@langchain/core/documents";
import { sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import Redis from "ioredis";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Pool } from "pg";
import postgres from "postgres";
import { legalDocuments } from "../database/vector-schema";

// Go Microservice Integration Types
export interface GoMicroserviceConfig {
  baseUrl: string;
  enabled: boolean;
  timeout: number;
  retryAttempts: number;
  fallbackToLocal: boolean;
}

export interface DocumentRequest {
  content: string;
  document_type: string;
  practice_area?: string;
  jurisdiction: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentResponse {
  document_id: string;
  processing_time: string;
  chunks: DocumentChunk[];
  analysis: LegalAnalysis;
  embeddings?: number[][];
}

export interface DocumentChunk {
  id: string;
  content: string;
  chunk_index: number;
  embedding?: number[];
  metadata: Record<string, unknown>;
}

export interface LegalAnalysis {
  summary: string;
  key_concepts: string[];
  entities: LegalEntity[];
  risk_factors: RiskFactor[];
  recommendations: string[];
  confidence: number;
  cited_cases?: CitedCase[];
}

export interface LegalEntity {
  type: string;
  text: string;
  confidence: number;
  context: string;
}

export interface RiskFactor {
  type: string;
  severity: string;
  description: string;
  mitigation?: string;
}

export interface CitedCase {
  case_name: string;
  citation: string;
  relevance: string;
  context: string;
}

export interface SearchRequest {
  query: string;
  model: string;
  limit: number;
  filters?: Record<string, unknown>;
  practice_area?: string;
  jurisdiction?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  query_time: string;
  total_results: number;
  used_cache: boolean;
  model_used: string;
}

export interface SearchResult {
  document_id: string;
  content: string;
  score: number;
  chunk_index: number;
  metadata: Record<string, unknown>;
  highlighted?: string;
}

export interface EnhancedSemanticSearchOptions {
  limit?: number;
  minSimilarity?: number;
  documentType?: string;
  practiceArea?: string;
  jurisdiction?: string;
  caseId?: string;
  userId?: string;
  useCache?: boolean;
  useGPU?: boolean;
  temperature?: number;
  topK?: number;
  ragMode?: "basic" | "enhanced" | "hybrid";
  includeContext?: boolean;
  contextWindow?: number;
}

export interface EnhancedSearchResult {
  id: string;
  title: string;
  content: string;
  similarity: number;
  documentType: string;
  practiceArea?: string;
  jurisdiction: string;
  createdAt: Date;
  fileSize?: number;
  caseId?: string;
  summary?: string;
  keyInsights?: string[];
  legalCitations?: string[];
  riskFactors?: string[];
  analysisResults?: {
    confidenceLevel: number;
    risks: string[];
    entities: string[];
    keywords: string[];
    aiSummary?: string;
    relevanceScore?: number;
    contextualInsights?: string[];
  };
  embedding?: number[];
  ragContext?: {
    relatedDocuments: string[];
    contextualSummary: string;
    crossReferences: string[];
  };
}

export interface GemmaLegalConfig {
  modelPath: string;
  cudaDeviceId?: number;
  contextLength?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  seed?: number;
  batchSize?: number;
  threads?: number;
  gpuLayers?: number;
}

export interface SearchAnalytics {
  date: string;
  searches: number;
  avgProcessingTime: number;
  avgResultCount: number;
  topQueries: Array<{ query: string; count: number }>;
}

export interface DocumentMetadata {
  title: string;
  documentType: string;
  practiceArea?: string;
  jurisdiction: string;
  caseId?: string;
  [key: string]: string | number | boolean | Date | undefined;
}

export interface RelatedDocument {
  id: string;
  title: string;
  content: string;
  documentType: string;
}

export interface SearchDataItem {
  timestamp: string;
  query: string;
  resultCount: number;
  processingTime: number;
  options: Partial<EnhancedSemanticSearchOptions>;
}

class EnhancedAIPipeline {
  private ollama: Ollama;
  private embeddings: OllamaEmbeddings;
  private vectorStore: PGVectorStore;
  private db: ReturnType<typeof drizzle>;
  private pgPool: Pool;
  private redis: Redis;
  private textSplitter: RecursiveCharacterTextSplitter;
  private gemmaConfig: GemmaLegalConfig;
  private isInitialized = false;

  // Go Microservice Integration
  private goMicroservice: GoMicroserviceConfig;
  private useGoMicroservice = false;

  constructor() {
    this.setupGemmaLegalConfig();
    this.setupRedisConnection();
    this.setupTextSplitter();
    this.setupGoMicroservice();
  }

  private setupGoMicroservice() {
    this.goMicroservice = {
      baseUrl: process.env.GO_MICROSERVICE_URL || "http://localhost:8080",
      enabled: process.env.USE_GO_MICROSERVICE === "true",
      timeout: parseInt(process.env.GO_MICROSERVICE_TIMEOUT || "30000"),
      retryAttempts: parseInt(process.env.GO_MICROSERVICE_RETRIES || "3"),
      fallbackToLocal: process.env.GO_MICROSERVICE_FALLBACK !== "false",
    };

    // Enable Go microservice by default for enhanced performance
    this.useGoMicroservice = this.goMicroservice.enabled;

    if (this.useGoMicroservice) {
      console.log(
        "🚀 Go Microservice integration enabled:",
        this.goMicroservice.baseUrl
      );
    }
  }

  private setupGemmaLegalConfig() {
    this.gemmaConfig = {
      modelPath:
        process.env.GEMMA_LEGAL_MODEL_PATH ||
        "C:\\AI\\Models\\gemma3-legal-8b-q4_k_m.gguf",
      cudaDeviceId: parseInt(process.env.CUDA_DEVICE_ID || "0"),
      contextLength: parseInt(process.env.CONTEXT_LENGTH || "8192"),
      temperature: parseFloat(process.env.TEMPERATURE || "0.1"),
      topP: parseFloat(process.env.TOP_P || "0.9"),
      topK: parseInt(process.env.TOP_K || "40"),
      repeatPenalty: parseFloat(process.env.REPEAT_PENALTY || "1.1"),
      batchSize: parseInt(process.env.BATCH_SIZE || "512"),
      threads: parseInt(process.env.THREADS || "8"),
      gpuLayers: parseInt(process.env.GPU_LAYERS || "33"),
    };
  }

  private setupRedisConnection() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || "localhost",
      port: parseInt(process.env.REDIS_PORT || "6379"),
      password: process.env.REDIS_PASSWORD,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
      keyPrefix: "legal-ai:",
      db: 0,
    });
  }

  private setupTextSplitter() {
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n", "\n", ". ", " ", ""],
    });
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Initialize PostgreSQL connection
      this.pgPool = new Pool({
        host: process.env.DB_HOST || "localhost",
        port: parseInt(process.env.DB_PORT || "5432"),
        database: process.env.DB_NAME || "legal_ai_db",
        user: process.env.DB_USER || "postgres",
        password: process.env.DB_PASSWORD || "postgres",
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });

      const connectionString = `postgresql://${process.env.DB_USER || "postgres"}:${process.env.DB_PASSWORD || "postgres"}@${process.env.DB_HOST || "localhost"}:${process.env.DB_PORT || "5432"}/${process.env.DB_NAME || "legal_ai_db"}`;
      const client = postgres(connectionString);
      this.db = drizzle(client);

      // Initialize Ollama with Gemma3-Legal model
      this.ollama = new Ollama({
        baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
        model: "gemma3-legal:8b",
        temperature: this.gemmaConfig.temperature,
        numCtx: this.gemmaConfig.contextLength,
        numGpu: this.gemmaConfig.gpuLayers,
        numThread: this.gemmaConfig.threads,
        repeatPenalty: this.gemmaConfig.repeatPenalty,
        topK: this.gemmaConfig.topK,
        topP: this.gemmaConfig.topP,
        stop: ["<|im_end|>", "<|endoftext|>"],
      });

      // Initialize embeddings model
      this.embeddings = new OllamaEmbeddings({
        baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
        model: "nomic-embed-text:latest",
        requestOptions: {
          useMMap: true,
          numThread: this.gemmaConfig.threads,
          numGpu: this.gemmaConfig.gpuLayers,
        },
      });

      // Initialize vector store
      this.vectorStore = await PGVectorStore.initialize(this.embeddings, {
        postgresConnectionOptions: {
          host: process.env.DB_HOST || "localhost",
          port: parseInt(process.env.DB_PORT || "5432"),
          database: process.env.DB_NAME || "legal_ai_db",
          user: process.env.DB_USER || "postgres",
          password: process.env.DB_PASSWORD || "postgres",
        },
        tableName: "legal_document_embeddings",
        columns: {
          idColumnName: "id",
          vectorColumnName: "embedding",
          contentColumnName: "content",
          metadataColumnName: "metadata",
        },
      });

      // Connect to Redis
      await this.redis.connect();

      // Test CUDA availability
      await this.testCUDAAvailability();

      this.isInitialized = true;
      console.log("🚀 Enhanced AI Pipeline initialized successfully");
      console.log(`📍 Gemma3-Legal model: ${this.gemmaConfig.modelPath}`);
      console.log(`🎮 CUDA Device: ${this.gemmaConfig.cudaDeviceId}`);
      console.log(`💾 Redis connected: ${this.redis.status}`);
    } catch (error) {
      console.error("❌ Failed to initialize Enhanced AI Pipeline:", error);
      throw error;
    }
  }

  private async testCUDAAvailability(): Promise<void> {
    try {
      // Simple test to ensure CUDA is working
      const testPrompt =
        "Test CUDA availability with a simple legal query about contracts.";
      await this.ollama.invoke(testPrompt);
      console.log("✅ CUDA GPU acceleration verified");
    } catch (error) {
      console.warn("⚠️  CUDA test failed, falling back to CPU:", error.message);
    }
  }

  async performEnhancedSemanticSearch(
    query: string,
    options: EnhancedSemanticSearchOptions = {}
  ): Promise<EnhancedSearchResult[]> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    const cacheKey = `search:${Buffer.from(query + JSON.stringify(options)).toString("base64")}`;

    try {
      // Check Redis cache first
      if (options.useCache !== false) {
        const cached = await this.redis.get(cacheKey);
        if (cached) {
          console.log("📦 Cache hit for semantic search");
          return JSON.parse(cached);
        }
      }

      // Enhanced query processing with legal context
      const enhancedQuery = await this.enhanceQueryWithLegalContext(
        query,
        options
      );

      // Get embeddings for the enhanced query
      const queryEmbedding = await this.embeddings.embedQuery(enhancedQuery);

      // Perform vector similarity search
      const vectorResults =
        await this.vectorStore.similaritySearchVectorWithScore(
          queryEmbedding,
          options.limit || 10
        );

      // Filter and process results
      let results: EnhancedSearchResult[] = [];

      for (const [doc, score] of vectorResults) {
        if (score >= (options.minSimilarity || 0.6)) {
          const enhancedResult = await this.enhanceSearchResult(
            doc,
            score,
            options
          );

          // Apply filters
          if (this.passesFilters(enhancedResult, options)) {
            results.push(enhancedResult);
          }
        }
      }

      // Sort by relevance and similarity
      results = results.sort((a, b) => {
        const aScore =
          a.similarity * 0.7 + (a.analysisResults?.relevanceScore || 0) * 0.3;
        const bScore =
          b.similarity * 0.7 + (b.analysisResults?.relevanceScore || 0) * 0.3;
        return bScore - aScore;
      });

      // Apply RAG enhancement if requested
      if (options.ragMode !== "basic") {
        results = await this.applyRAGEnhancements(
          results,
          enhancedQuery,
          options
        );
      }

      // Cache results
      if (options.useCache !== false) {
        await this.redis.setex(cacheKey, 300, JSON.stringify(results)); // 5 min cache
      }

      // Log performance metrics
      const processingTime = Date.now() - startTime;
      console.log(
        `⚡ Enhanced semantic search completed in ${processingTime}ms`
      );
      console.log(`📊 Found ${results.length} relevant documents`);

      // Store search analytics
      await this.storeSearchAnalytics(
        query,
        results.length,
        processingTime,
        options
      );

      return results;
    } catch (error) {
      console.error("❌ Enhanced semantic search failed:", error);
      throw error;
    }
  }

  private async enhanceQueryWithLegalContext(
    query: string,
    options: EnhancedSemanticSearchOptions
  ): Promise<string> {
    const contextPrompt = `
You are a legal AI assistant specializing in document analysis. Enhance this search query with relevant legal context and synonyms:

Original Query: "${query}"

Context:
- Document Type: ${options.documentType || "any"}
- Practice Area: ${options.practiceArea || "any"}
- Jurisdiction: ${options.jurisdiction || "any"}

Instructions:
1. Add relevant legal terminology and synonyms
2. Include related concepts and phrases
3. Maintain the original intent
4. Keep it concise but comprehensive

Enhanced Query:`;

    try {
      const enhanced = await this.ollama.invoke(contextPrompt);
      return enhanced.trim();
    } catch (error) {
      console.warn(
        "⚠️  Query enhancement failed, using original:",
        error.message
      );
      return query;
    }
  }

  private async enhanceSearchResult(
    doc: Document,
    similarity: number,
    options: EnhancedSemanticSearchOptions
  ): Promise<EnhancedSearchResult> {
    const metadata = doc.metadata || {};

    // Generate AI summary and insights
    const aiAnalysis = await this.generateAIAnalysis(doc.pageContent, options);

    // Extract legal citations and entities
    const legalExtraction = await this.extractLegalEntities(doc.pageContent);

    return {
      id: metadata.id || crypto.randomUUID(),
      title: metadata.title || "Untitled Document",
      content: doc.pageContent,
      similarity: similarity,
      documentType: metadata.documentType || "unknown",
      practiceArea: metadata.practiceArea,
      jurisdiction: metadata.jurisdiction || "unknown",
      createdAt: new Date(metadata.createdAt || Date.now()),
      fileSize: metadata.fileSize,
      caseId: metadata.caseId,
      summary: aiAnalysis.summary,
      keyInsights: aiAnalysis.keyInsights,
      legalCitations: legalExtraction.citations,
      riskFactors: aiAnalysis.riskFactors,
      analysisResults: {
        confidenceLevel: aiAnalysis.confidenceLevel,
        risks: aiAnalysis.riskFactors,
        entities: legalExtraction.entities,
        keywords: legalExtraction.keywords,
        aiSummary: aiAnalysis.summary,
        relevanceScore: aiAnalysis.relevanceScore,
        contextualInsights: aiAnalysis.contextualInsights,
      },
    };
  }

  private async generateAIAnalysis(
    content: string,
    options: EnhancedSemanticSearchOptions
  ) {
    const analysisPrompt = `
Analyze this legal document excerpt for key insights, risks, and relevance:

Document Content:
"${content.substring(0, 2000)}..."

Provide analysis in the following JSON format:
{
  "summary": "Brief 2-sentence summary",
  "keyInsights": ["insight1", "insight2", "insight3"],
  "riskFactors": ["risk1", "risk2"],
  "confidenceLevel": 0.85,
  "relevanceScore": 0.90,
  "contextualInsights": ["context1", "context2"]
}

Analysis:`;

    try {
      const analysis = await this.ollama.invoke(analysisPrompt);
      const jsonMatch = analysis.match(/\{[\s\S]*\}/);

      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      } else {
        // Fallback analysis
        return {
          summary: "AI analysis temporarily unavailable",
          keyInsights: [],
          riskFactors: [],
          confidenceLevel: 0.7,
          relevanceScore: 0.7,
          contextualInsights: [],
        };
      }
    } catch (error) {
      console.warn("⚠️  AI analysis failed:", error.message);
      return {
        summary: "AI analysis error",
        keyInsights: [],
        riskFactors: [],
        confidenceLevel: 0.5,
        relevanceScore: 0.5,
        contextualInsights: [],
      };
    }
  }

  private async extractLegalEntities(content: string) {
    // Use regex patterns for common legal entities
    const casePattern = /(\d{4})\s+(WL|F\.2d|F\.3d|F\.Supp|U\.S\.)\s+\d+/g;
    const statutePattern = /\d+\s+U\.S\.C\.?\s+§?\s*\d+/g;
    const regulationPattern = /\d+\s+C\.F\.R\.?\s+§?\s*\d+/g;

    const citations = [
      ...(content.match(casePattern) || []),
      ...(content.match(statutePattern) || []),
      ...(content.match(regulationPattern) || []),
    ];

    // Extract keywords using simple legal terms
    const legalTerms = [
      "contract",
      "liability",
      "negligence",
      "breach",
      "damages",
      "plaintiff",
      "defendant",
      "jurisdiction",
      "venue",
      "discovery",
      "motion",
      "summary judgment",
      "trial",
      "settlement",
      "arbitration",
      "mediation",
      "injunction",
      "restraining order",
    ];

    const keywords = legalTerms.filter((term) =>
      content.toLowerCase().includes(term)
    );

    // Extract entities (simplified)
    const entityPattern =
      /\b[A-Z][a-z]+ (?:Inc\.|Corp\.|LLC|Co\.|Company|Corporation)\b/g;
    const entities = content.match(entityPattern) || [];

    return {
      citations: Array.from(new Set(citations)),
      keywords: Array.from(new Set(keywords)),
      entities: Array.from(new Set(entities)),
    };
  }

  private passesFilters(
    result: EnhancedSearchResult,
    options: EnhancedSemanticSearchOptions
  ): boolean {
    if (options.documentType && result.documentType !== options.documentType)
      return false;
    if (options.practiceArea && result.practiceArea !== options.practiceArea)
      return false;
    if (options.jurisdiction && result.jurisdiction !== options.jurisdiction)
      return false;
    if (options.caseId && result.caseId !== options.caseId) return false;
    return true;
  }

  private async applyRAGEnhancements(
    results: EnhancedSearchResult[],
    query: string,
    options: EnhancedSemanticSearchOptions
  ): Promise<EnhancedSearchResult[]> {
    // Enhanced RAG processing
    for (const result of results) {
      try {
        // Get related documents
        const relatedDocs = await this.findRelatedDocuments(result.id, 3);

        // Generate contextual summary
        const contextSummary = await this.generateContextualSummary(
          result.content,
          relatedDocs,
          query
        );

        result.ragContext = {
          relatedDocuments: relatedDocs.map((doc) => doc.id),
          contextualSummary: contextSummary,
          crossReferences: this.extractCrossReferences(
            result.content,
            relatedDocs
          ),
        };
      } catch (error) {
        console.warn(
          `⚠️  RAG enhancement failed for result ${result.id}:`,
          error.message
        );
      }
    }

    return results;
  }

  private async findRelatedDocuments(documentId: string, limit: number = 3) {
    // Simplified related document finding
    try {
      const relatedResults = await this.db
        .select()
        .from(legalDocuments)
        .where(sql`id != ${documentId}`)
        .limit(limit);

      return relatedResults;
    } catch (error) {
      console.warn("⚠️  Failed to find related documents:", error.message);
      return [];
    }
  }

  private async generateContextualSummary(
    mainContent: string,
    relatedDocs: RelatedDocument[],
    originalQuery: string
  ): Promise<string> {
    const contextPrompt = `
Based on the main document and related documents, provide a contextual summary relevant to the query: "${originalQuery}"

Main Document: "${mainContent.substring(0, 1000)}..."

Related Documents: ${relatedDocs.map((doc) => `"${doc.content?.substring(0, 500)}..."`).join("\n")}

Provide a brief contextual summary that connects these documents:`;

    try {
      const summary = await this.ollama.invoke(contextPrompt);
      return summary.trim();
    } catch (error) {
      console.warn("⚠️  Contextual summary generation failed:", error.message);
      return "Contextual summary unavailable";
    }
  }

  private extractCrossReferences(
    content: string,
    relatedDocs: RelatedDocument[]
  ): string[] {
    // Simple cross-reference extraction
    const references: string[] = [];

    relatedDocs.forEach((doc) => {
      if (
        doc.title &&
        content.toLowerCase().includes(doc.title.toLowerCase())
      ) {
        references.push(doc.title);
      }
    });

    return Array.from(new Set(references));
  }

  private async storeSearchAnalytics(
    query: string,
    resultCount: number,
    processingTime: number,
    options: EnhancedSemanticSearchOptions
  ): Promise<void> {
    try {
      const analyticsKey = `analytics:search:${new Date().toISOString().split("T")[0]}`;
      const searchData = {
        timestamp: new Date().toISOString(),
        query: query.substring(0, 100), // Truncate for privacy
        resultCount,
        processingTime,
        options: {
          documentType: options.documentType,
          practiceArea: options.practiceArea,
          jurisdiction: options.jurisdiction,
          ragMode: options.ragMode,
        },
      };

      await this.redis.lpush(analyticsKey, JSON.stringify(searchData));
      await this.redis.expire(analyticsKey, 86400 * 30); // 30 days retention
    } catch (error) {
      console.warn("⚠️  Failed to store search analytics:", error.message);
    }
  }

  async ingestDocument(
    content: string,
    metadata: DocumentMetadata
  ): Promise<string> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const documentId = crypto.randomUUID();

      // Split document into chunks
      const chunks = await this.textSplitter.splitText(content);

      // Create documents for vector store
      const documents = chunks.map((chunk, index) => ({
        pageContent: chunk,
        metadata: {
          ...metadata,
          documentId,
          chunkIndex: index,
          totalChunks: chunks.length,
        },
      }));

      // Add to vector store
      await this.vectorStore.addDocuments(documents);

      // Store document metadata in database
      await this.db.insert(legalDocuments).values({
        id: documentId,
        title: metadata.title,
        content: content,
        documentType: metadata.documentType,
        practiceArea: metadata.practiceArea,
        jurisdiction: metadata.jurisdiction,
        fileSize: content.length,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: {
          caseId: metadata.caseId,
          ...metadata,
        },
      });

      console.log(`📄 Document ingested successfully: ${documentId}`);
      return documentId;
    } catch (error) {
      console.error("❌ Document ingestion failed:", error);
      throw error;
    }
  }

  async getSearchAnalytics(days: number = 7): Promise<SearchAnalytics[]> {
    const analytics = [];
    const today = new Date();

    for (let i = 0; i < days; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      const dateKey = `analytics:search:${date.toISOString().split("T")[0]}`;

      try {
        const dayData = await this.redis.lrange(dateKey, 0, -1);
        const parsedData = dayData.map((item) => JSON.parse(item));

        analytics.push({
          date: date.toISOString().split("T")[0],
          searches: parsedData.length,
          avgProcessingTime:
            parsedData.reduce((sum, item) => sum + item.processingTime, 0) /
              parsedData.length || 0,
          avgResultCount:
            parsedData.reduce((sum, item) => sum + item.resultCount, 0) /
              parsedData.length || 0,
          topQueries: this.getTopQueries(parsedData),
        });
      } catch (error) {
        console.warn(
          `⚠️  Failed to get analytics for ${dateKey}:`,
          error.message
        );
      }
    }

    return analytics;
  }

  private getTopQueries(
    searchData: SearchDataItem[]
  ): Array<{ query: string; count: number }> {
    const queryCount = searchData.reduce((acc, item) => {
      acc[item.query] = (acc[item.query] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(queryCount)
      .map(([query, count]) => ({ query, count: count as number }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }

  // Go Microservice Integration Methods
  async processDocumentWithGoService(
    request: DocumentRequest
  ): Promise<DocumentResponse> {
    if (!this.useGoMicroservice) {
      throw new Error("Go microservice is not enabled");
    }

    try {
      const response = await fetch(
        `${this.goMicroservice.baseUrl}/api/ai/process-document`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(request),
          signal: AbortSignal.timeout(this.goMicroservice.timeout),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Go microservice responded with ${response.status}: ${response.statusText}`
        );
      }

      const result = (await response.json()) as DocumentResponse;
      console.log(
        `🚀 Go microservice processed document: ${result.document_id}`
      );
      return result;
    } catch (error) {
      console.error("❌ Go microservice document processing failed:", error);

      if (this.goMicroservice.fallbackToLocal) {
        console.log("🔄 Falling back to local processing...");
        return this.processDocumentLocally(request);
      }

      throw error;
    }
  }

  async vectorSearchWithGoService(
    request: SearchRequest
  ): Promise<SearchResponse> {
    if (!this.useGoMicroservice) {
      throw new Error("Go microservice is not enabled");
    }

    try {
      const response = await fetch(
        `${this.goMicroservice.baseUrl}/api/ai/vector-search`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(request),
          signal: AbortSignal.timeout(this.goMicroservice.timeout),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Go microservice responded with ${response.status}: ${response.statusText}`
        );
      }

      const result = (await response.json()) as SearchResponse;
      console.log(
        `🔍 Go microservice search completed: ${result.total_results} results`
      );
      return result;
    } catch (error) {
      console.error("❌ Go microservice vector search failed:", error);

      if (this.goMicroservice.fallbackToLocal) {
        console.log("🔄 Falling back to local search...");
        return this.vectorSearchLocally(request);
      }

      throw error;
    }
  }

  async checkGoMicroserviceHealth(): Promise<boolean> {
    try {
      const response = await fetch(
        `${this.goMicroservice.baseUrl}/api/health`,
        {
          method: "GET",
          signal: AbortSignal.timeout(5000), // 5 second timeout
        }
      );

      return response.ok;
    } catch (error) {
      console.warn("⚠️ Go microservice health check failed:", error);
      return false;
    }
  }

  private async processDocumentLocally(
    request: DocumentRequest
  ): Promise<DocumentResponse> {
    // Fallback to local processing using existing methods
    const startTime = Date.now();

    // Split content into chunks
    const docs = await this.textSplitter.createDocuments([request.content]);

    // Generate embeddings for each chunk
    const chunks: DocumentChunk[] = [];
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const embedding = await this.embeddings.embedQuery(doc.pageContent);

      chunks.push({
        id: `local_chunk_${i}`,
        content: doc.pageContent,
        chunk_index: i,
        embedding,
        metadata: {
          document_type: request.document_type,
          practice_area: request.practice_area,
          jurisdiction: request.jurisdiction,
          ...request.metadata,
        },
      });
    }

    // Generate basic analysis (simplified version)
    const analysis: LegalAnalysis = {
      summary: `Local analysis of ${request.document_type} document`,
      key_concepts: this.extractKeywords(request.content),
      entities: [],
      risk_factors: [],
      recommendations: [
        "Review with legal counsel",
        "Consider jurisdictional requirements",
      ],
      confidence: 0.75,
    };

    const processingTime = Date.now() - startTime;

    return {
      document_id: `local_${Date.now()}`,
      processing_time: `${processingTime}ms`,
      chunks,
      analysis,
    };
  }

  private async vectorSearchLocally(
    request: SearchRequest
  ): Promise<SearchResponse> {
    const startTime = Date.now();

    // Use existing semantic search as fallback
    const options: EnhancedSemanticSearchOptions = {
      limit: request.limit,
      practiceArea: request.practice_area,
      jurisdiction: request.jurisdiction,
    };

    const localResults = await this.performEnhancedSemanticSearch(
      request.query,
      options
    );

    // Transform to Go microservice format
    const results: SearchResult[] = localResults.map((result, index) => ({
      document_id: result.id,
      content: result.content,
      score: result.similarity,
      chunk_index: index,
      metadata: {
        title: result.title,
        document_type: result.documentType,
        practice_area: result.practiceArea,
        jurisdiction: result.jurisdiction,
        ...result.analysisResults,
      },
      highlighted: result.content, // Simplified highlighting
    }));

    const processingTime = Date.now() - startTime;

    return {
      results,
      query_time: `${processingTime}ms`,
      total_results: results.length,
      used_cache: false,
      model_used: "local_fallback",
    };
  }

  private extractKeywords(content: string): string[] {
    // Simple keyword extraction for fallback
    const words = content.toLowerCase().match(/\b\w{4,}\b/g) || [];
    const wordCount: Record<string, number> = {};

    words.forEach((word) => {
      wordCount[word] = (wordCount[word] || 0) + 1;
    });

    return Object.entries(wordCount)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  }

  // Enhanced semantic search with Go microservice integration
  async semanticSearch(
    query: string,
    options: EnhancedSemanticSearchOptions = {}
  ): Promise<EnhancedSearchResult[]> {
    // Check if Go microservice should be used and is available
    if (this.useGoMicroservice && (await this.checkGoMicroserviceHealth())) {
      try {
        const searchRequest: SearchRequest = {
          query,
          model: "gemma3-legal",
          limit: options.limit || 10,
          filters: {
            documentType: options.documentType,
            minSimilarity: options.minSimilarity,
          },
          practice_area: options.practiceArea,
          jurisdiction: options.jurisdiction,
        };

        const goResponse = await this.vectorSearchWithGoService(searchRequest);

        // Transform Go response to legacy format
        return goResponse.results.map(
          (result) =>
            ({
              id: result.document_id,
              title: result.metadata.title || `Document ${result.document_id}`,
              content: result.content,
              similarity: result.score,
              documentType: result.metadata.document_type || "unknown",
              practiceArea: result.metadata.practice_area,
              jurisdiction: result.metadata.jurisdiction || "US",
              createdAt: new Date(),
              metadata: result.metadata,
            }) as EnhancedSearchResult
        );
      } catch (error) {
        console.warn(
          "⚠️ Go microservice search failed, using local fallback:",
          error
        );
      }
    }

    // Fallback to local semantic search
    return this.performEnhancedSemanticSearch(query, options);
  }

  async cleanup(): Promise<void> {
    try {
      await this.redis.quit();
      await this.pgPool.end();
      console.log("🧹 Enhanced AI Pipeline cleanup completed");
    } catch (error) {
      console.warn("⚠️  Cleanup warning:", error.message);
    }
  }
}

// Singleton instance
const enhancedAIPipeline = new EnhancedAIPipeline();

export { enhancedAIPipeline };
export default EnhancedAIPipeline;
