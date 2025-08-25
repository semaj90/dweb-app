import crypto from "crypto";

// Enhanced AI Pipeline Service - Real Integration
// Connects Ollama models with PostgreSQL pgvector for semantic search
// Production-ready legal document processing

import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Ollama } from "@langchain/community/llms/ollama";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { Pool } from "pg";
import type { Document } from "@langchain/core/documents";
import { userEmbeddings } from "../server/db/schema-postgres.js";
import { OllamaService } from "./OllamaService";

export interface SemanticSearchOptions {
  limit?: number;
  minSimilarity?: number;
  documentType?: string;
  practiceArea?: string;
  jurisdiction?: string;
  caseId?: string;
  userId?: string;
  useCache?: boolean;
}

export interface SearchResult {
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
  analysisResults?: {
    confidenceLevel: number;
    risks: string[];
    entities: string[];
    keywords: string[];
  };
}

export interface EmbeddingResult {
  documentId: string;
  embedding: number[];
  processingTime: number;
  metadata: Record<string, any>;
}

export class EnhancedAIPipeline {
  private embeddings: OllamaEmbeddings;
  private llm: Ollama;
  private pgPool: Pool;
  private db: any;
  private vectorStore: PGVectorStore | null = null;
  private ollamaService: OllamaService;
  private isInitialized = false;

  constructor() {
    // Initialize Ollama embeddings with nomic-embed-text
    this.embeddings = new OllamaEmbeddings({
      baseUrl: import.meta.env.OLLAMA_BASE_URL || "http://localhost:11434",
      model: "nomic-embed-text", // 768-dimensional embeddings
    });

    // Initialize Ollama LLM
    this.llm = new Ollama({
      baseUrl: import.meta.env.OLLAMA_BASE_URL || "http://localhost:11434",
      model: "gemma2:2b", // Fast model for legal analysis
    });

    // PostgreSQL connection
    this.pgPool = new Pool({
      host: import.meta.env.POSTGRES_HOST || "localhost",
      port: parseInt(import.meta.env.POSTGRES_PORT || "5432"),
      database: import.meta.env.POSTGRES_DB || "prosecutor_db",
      user: import.meta.env.POSTGRES_USER || "postgres",
      password: import.meta.env.POSTGRES_PASSWORD || "postgres",
    });

    // Initialize Drizzle
    const client = postgres(
      import.meta.env.DATABASE_URL ||
        "postgresql://postgres:postgres@localhost:5432/prosecutor_db"
    );
    this.db = drizzle(client);

    this.ollamaService = new OllamaService();
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Test database connection
      await this.pgPool.query("SELECT 1");
      console.log("✅ PostgreSQL connection established");

      // Ensure pgvector extension is enabled
      await this.pgPool.query("CREATE EXTENSION IF NOT EXISTS vector");
      console.log("✅ pgvector extension enabled");

      // Initialize PGVectorStore
      this.vectorStore = await PGVectorStore.initialize(this.embeddings, {
        pool: this.pgPool,
        tableName: "legal_document_embeddings",
        columns: {
          idColumnName: "id",
          vectorColumnName: "embedding",
          contentColumnName: "content",
          metadataColumnName: "metadata",
        },
      });
      console.log("✅ PGVectorStore initialized");

      // Test Ollama connection
      await this.llm.invoke("Test connection");
      console.log("✅ Ollama LLM connection established");

      this.isInitialized = true;
      console.log("🚀 Enhanced AI Pipeline initialized successfully");
    } catch (error) {
      console.error("❌ AI Pipeline initialization failed:", error);
      throw error;
    }
  }

  /**
   * Generate embeddings for legal document content
   */
  async generateEmbeddings(
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<EmbeddingResult> {
    const startTime = Date.now();

    try {
      const embedding = await this.embeddings.embedQuery(content);
      const processingTime = Date.now() - startTime;

      return {
        documentId: metadata.id || crypto.randomUUID(),
        embedding,
        processingTime,
        metadata: {
          ...metadata,
          embeddingModel: "nomic-embed-text",
          dimensions: embedding.length,
          generatedAt: new Date().toISOString(),
        },
      };
    } catch (error) {
      console.error("Embedding generation failed:", error);
      throw new Error(`Failed to generate embeddings: ${error.message}`);
    }
  }

  /**
   * Store document with embeddings in PostgreSQL
   */
  async storeDocument(
    content: string,
    metadata: Record<string, any>
  ): Promise<string> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    try {
      const document: Document = {
        pageContent: content,
        metadata: {
          ...metadata,
          storedAt: new Date().toISOString(),
          source: "enhanced_ai_pipeline",
        },
      };

      const ids = await this.vectorStore.addDocuments([document]);
      console.log(`✅ Document stored with ID: ${ids[0]}`);
      return ids[0];
    } catch (error) {
      console.error("Document storage failed:", error);
      throw new Error(`Failed to store document: ${error.message}`);
    }
  }

  /**
   * Perform semantic search with real embeddings
   */
  async semanticSearch(
    query: string,
    options: SemanticSearchOptions = {}
  ): Promise<SearchResult[]> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    const {
      limit = 10,
      minSimilarity = 0.6,
      documentType,
      practiceArea,
      jurisdiction,
      caseId,
      userId,
    } = options;

    try {
      // Build filter conditions
      const filter: Record<string, any> = {};
      if (documentType) filter.documentType = documentType;
      if (practiceArea) filter.practiceArea = practiceArea;
      if (jurisdiction) filter.jurisdiction = jurisdiction;
      if (caseId) filter.caseId = caseId;
      if (userId) filter.userId = userId;

      // Perform vector similarity search
      const results = await this.vectorStore.similaritySearchWithScore(
        query,
        limit,
        filter
      );

      // Transform results to our format
      const searchResults: SearchResult[] = results
        .filter(([, score]) => score >= minSimilarity)
        .map(([document, score]) => ({
          id: document.metadata.id || crypto.randomUUID(),
          title: document.metadata.title || "Untitled Document",
          content: document.pageContent,
          similarity: score,
          documentType: document.metadata.documentType || "unknown",
          practiceArea: document.metadata.practiceArea,
          jurisdiction: document.metadata.jurisdiction || "federal",
          createdAt: new Date(document.metadata.createdAt || Date.now()),
          fileSize: document.metadata.fileSize,
          caseId: document.metadata.caseId,
          analysisResults: document.metadata.analysisResults,
        }));

      console.log(
        `🔍 Found ${searchResults.length} results for query: "${query}"`
      );
      return searchResults;
    } catch (error) {
      console.error("Semantic search failed:", error);
      throw new Error(`Semantic search failed: ${error.message}`);
    }
  }

  /**
   * Analyze legal document with Ollama
   */
  async analyzeLegalDocument(
    content: string,
    documentType: string = "unknown"
  ): Promise<{
    summary: string;
    risks: string[];
    entities: string[];
    keywords: string[];
    confidenceLevel: number;
    recommendations: string[];
  }> {
    try {
      const analysisPrompt = `Analyze this ${documentType} legal document:

${content.substring(0, 2000)}...

Provide analysis in the following JSON format:
{
  "summary": "Brief summary of the document",
  "risks": ["Risk 1", "Risk 2"],
  "entities": ["Entity 1", "Entity 2"],
  "keywords": ["Keyword 1", "Keyword 2"],
  "confidenceLevel": 0.85,
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}

Focus on legal implications, potential risks, and actionable insights.`;

      const response = await this.llm.invoke(analysisPrompt);

      try {
        const analysis = JSON.parse(response);
        return {
          summary: analysis.summary || "Analysis unavailable",
          risks: analysis.risks || [],
          entities: analysis.entities || [],
          keywords: analysis.keywords || [],
          confidenceLevel: analysis.confidenceLevel || 0.7,
          recommendations: analysis.recommendations || [],
        };
      } catch (parseError) {
        // Fallback if JSON parsing fails
        return {
          summary: response.substring(0, 200) + "...",
          risks: ["Analysis format error"],
          entities: [],
          keywords: [],
          confidenceLevel: 0.5,
          recommendations: ["Review document manually"],
        };
      }
    } catch (error) {
      console.error("Legal document analysis failed:", error);
      throw new Error(`Document analysis failed: ${error.message}`);
    }
  }

  /**
   * Process and ingest legal document with full pipeline
   */
  async ingestLegalDocument(
    content: string,
    metadata: {
      title: string;
      documentType: string;
      practiceArea?: string;
      jurisdiction?: string;
      caseId?: string;
      userId?: string;
      fileSize?: number;
    }
  ): Promise<{
    documentId: string;
    analysis: any;
    embeddingId: string;
    processingTime: number;
  }> {
    const startTime = Date.now();

    try {
      // 1. Analyze document content
      const analysis = await this.analyzeLegalDocument(
        content,
        metadata.documentType
      );

      // 2. Generate embeddings
      const embeddingResult = await this.generateEmbeddings(content, {
        ...metadata,
        analysisResults: analysis,
      });

      // 3. Store in vector database
      const embeddingId = await this.storeDocument(content, {
        ...metadata,
        analysisResults: analysis,
        embeddingDimensions: embeddingResult.embedding.length,
      });

      const processingTime = Date.now() - startTime;

      console.log(`📄 Document ingested successfully in ${processingTime}ms`);

      return {
        documentId: embeddingResult.documentId,
        analysis,
        embeddingId,
        processingTime,
      };
    } catch (error) {
      console.error("Document ingestion failed:", error);
      throw new Error(`Document ingestion failed: ${error.message}`);
    }
  }

  /**
   * Get embedding statistics
   */
  async getEmbeddingStats(): Promise<{
    totalDocuments: number;
    averageEmbeddingTime: number;
    documentTypes: Record<string, number>;
    recentActivity: number;
  }> {
    try {
      const result = await this.pgPool.query(`
        SELECT
          COUNT(*) as total,
          AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_time,
          COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as recent
        FROM legal_document_embeddings
      `);

      const typeResult = await this.pgPool.query(`
        SELECT
          metadata->>'documentType' as doc_type,
          COUNT(*) as count
        FROM legal_document_embeddings
        WHERE metadata->>'documentType' IS NOT NULL
        GROUP BY metadata->>'documentType'
      `);

      const documentTypes: Record<string, number> = {};
      typeResult.rows.forEach((row) => {
        documentTypes[row.doc_type] = parseInt(row.count);
      });

      return {
        totalDocuments: parseInt(result.rows[0].total),
        averageEmbeddingTime: parseFloat(result.rows[0].avg_time) || 0,
        documentTypes,
        recentActivity: parseInt(result.rows[0].recent),
      };
    } catch (error) {
      console.error("Failed to get embedding stats:", error);
      return {
        totalDocuments: 0,
        averageEmbeddingTime: 0,
        documentTypes: {},
        recentActivity: 0,
      };
    }
  }

  /**
   * Health check for all services
   */
  async healthCheck(): Promise<{
    ollama: boolean;
    postgres: boolean;
    vectorStore: boolean;
    embeddings: boolean;
  }> {
    const health = {
      ollama: false,
      postgres: false,
      vectorStore: false,
      embeddings: false,
    };

    try {
      // Test Ollama
      await this.llm.invoke("test");
      health.ollama = true;
    } catch (error) {
      console.error("Ollama health check failed:", error);
    }

    try {
      // Test PostgreSQL
      await this.pgPool.query("SELECT 1");
      health.postgres = true;
    } catch (error) {
      console.error("PostgreSQL health check failed:", error);
    }

    try {
      // Test vector store
      if (this.vectorStore) {
        health.vectorStore = true;
      }
    } catch (error) {
      console.error("Vector store health check failed:", error);
    }

    try {
      // Test embeddings
      await this.embeddings.embedQuery("test");
      health.embeddings = true;
    } catch (error) {
      console.error("Embeddings health check failed:", error);
    }

    return health;
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    try {
      await this.pgPool.end();
      console.log("✅ AI Pipeline cleanup completed");
    } catch (error) {
      console.error("Cleanup failed:", error);
    }
  }
}

// Singleton instance
export const aiPipeline = new EnhancedAIPipeline();
