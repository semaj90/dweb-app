// LangChain.js Integration with Ollama and pgvector
// Production-ready AI pipeline for legal document processing

import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { Document } from "@langchain/core/documents";
import postgres from 'postgres';

interface LangChainConfig {
  ollamaBaseUrl: string;
  llmModel: string;
  embeddingModel: string;
  temperature: number;
  maxTokens: number;
  vectorStoreTable: string;
  similarityThreshold: number;
}

interface AnalysisResult {
  summary: string;
  entities: Array<{ type: string; value: string; confidence: number }>;
  keyTerms: string[];
  risks: Array<{ type: string; severity: string; description: string }>;
  recommendations: string[];
  confidence: number;
  processingTime: number;
  model: string;
  tokensUsed: number;
}

interface SearchResult {
  documents: Document[];
  scores: number[];
  query: string;
  totalResults: number;
  processingTime: number;
}

class LegalAILangChainService {
  private llm: Ollama;
  private embeddings: OllamaEmbeddings;
  private vectorStore: PGVectorStore | null = null;
  private config: LangChainConfig;
  private isInitialized = false;

  constructor(config: Partial<LangChainConfig> = {}) {
    this.config = {
      ollamaBaseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
      llmModel: process.env.OLLAMA_DEFAULT_MODEL || "llama3.2:3b",
      embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text",
      temperature: 0.1,
      maxTokens: 4096,
      vectorStoreTable: "legal_document_vectors",
      similarityThreshold: 0.7,
      ...config
    };

    this.initializeModels();
  }

  private initializeModels(): void {
    // Initialize LLM
    this.llm = new Ollama({
      baseUrl: this.config.ollamaBaseUrl,
      model: this.config.llmModel,
      temperature: this.config.temperature,
      numPredict: this.config.maxTokens,
      format: "json", // Force JSON output for structured responses
    });

    // Initialize embeddings
    this.embeddings = new OllamaEmbeddings({
      baseUrl: this.config.ollamaBaseUrl,
      model: this.config.embeddingModel,
    });
  }

  /**
   * Initialize vector store connection
   */
  async initializeVectorStore(): Promise<void> {
    try {
      const connectionConfig = {
        host: process.env.DB_HOST || "localhost",
        port: parseInt(process.env.DB_PORT || "5432"),
        user: process.env.DB_USER || "postgres",
        password: process.env.DB_PASSWORD || "postgres",
        database: process.env.DB_NAME || "legal_ai",
      };

      this.vectorStore = await PGVectorStore.initialize(this.embeddings, {
        postgresConnectionOptions: connectionConfig,
        tableName: this.config.vectorStoreTable,
        columns: {
          idColumnName: "id",
          vectorColumnName: "embedding",
          contentColumnName: "content",
          metadataColumnName: "metadata",
        },
        distanceStrategy: "cosine",
      });

      this.isInitialized = true;
      console.log("✅ LangChain vector store initialized");

    } catch (error) {
      console.error("❌ Failed to initialize vector store:", error);
      throw error;
    }
  }

  /**
   * Comprehensive legal document analysis
   */
  async analyzeLegalDocument(
    title: string,
    content: string,
    documentType: string = "general"
  ): Promise<AnalysisResult> {
    const startTime = Date.now();

    try {
      const analysisPrompt = PromptTemplate.fromTemplate(`
You are a legal AI assistant specializing in document analysis. Analyze the following legal document and provide a comprehensive analysis in JSON format.

Document Title: {title}
Document Type: {documentType}
Document Content: {content}

Provide your analysis in the following JSON structure:
{{
  "summary": "Brief summary of the document (2-3 sentences)",
  "entities": [
    {{ "type": "person|organization|location|date|amount", "value": "extracted entity", "confidence": 0.0-1.0 }}
  ],
  "keyTerms": ["legal term 1", "legal term 2", ...],
  "risks": [
    {{ "type": "liability|compliance|financial|operational", "severity": "low|medium|high", "description": "risk description" }}
  ],
  "recommendations": ["recommendation 1", "recommendation 2", ...],
  "confidence": 0.0-1.0
}}

Focus on:
- Legal entities (parties, organizations, jurisdictions)
- Key legal terms and concepts
- Potential legal risks and liabilities
- Compliance considerations
- Contractual obligations
- Financial terms and amounts
- Important dates and deadlines

Respond only with valid JSON.`);

      const chain = RunnableSequence.from([
        analysisPrompt,
        this.llm,
        new StringOutputParser(),
      ]);

      const result = await chain.invoke({
        title,
        content: content.substring(0, 8000), // Limit content for token management
        documentType,
      });

      const processingTime = Date.now() - startTime;

      // Parse JSON response
      let analysis: any;
      try {
        analysis = JSON.parse(result);
      } catch (parseError) {
        // Fallback parsing for non-JSON responses
        analysis = this.parseNonJsonResponse(result);
      }

      // Estimate tokens used (rough calculation)
      const tokensUsed = Math.ceil((title.length + content.length + result.length) / 4);

      return {
        summary: analysis.summary || "Analysis completed",
        entities: analysis.entities || [],
        keyTerms: analysis.keyTerms || [],
        risks: analysis.risks || [],
        recommendations: analysis.recommendations || [],
        confidence: analysis.confidence || 0.8,
        processingTime,
        model: this.config.llmModel,
        tokensUsed,
      };

    } catch (error) {
      console.error("Document analysis error:", error);
      throw new Error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate embeddings for text
   */
  async generateEmbeddings(text: string): Promise<number[]> {
    try {
      const embeddings = await this.embeddings.embedQuery(text);
      return embeddings;
    } catch (error) {
      console.error("Embedding generation error:", error);
      throw new Error(`Embedding generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Store document with embeddings in vector store
   */
  async storeDocument(
    id: string,
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    if (!this.vectorStore) {
      await this.initializeVectorStore();
    }

    try {
      const document = new Document({
        pageContent: content,
        metadata: {
          id,
          ...metadata,
          timestamp: new Date().toISOString(),
        },
      });

      await this.vectorStore!.addDocuments([document], { ids: [id] });
      console.log(`✅ Document ${id} stored in vector database`);

    } catch (error) {
      console.error("Document storage error:", error);
      throw new Error(`Document storage failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Semantic search using pgvector
   */
  async semanticSearch(
    query: string,
    limit: number = 10,
    filter?: Record<string, any>
  ): Promise<SearchResult> {
    const startTime = Date.now();

    if (!this.vectorStore) {
      await this.initializeVectorStore();
    }

    try {
      const results = await this.vectorStore!.similaritySearchWithScore(
        query,
        limit,
        filter
      );

      const documents = results.map(([doc]) => doc);
      const scores = results.map(([, score]) => score);

      return {
        documents,
        scores,
        query,
        totalResults: results.length,
        processingTime: Date.now() - startTime,
      };

    } catch (error) {
      console.error("Semantic search error:", error);
      throw new Error(`Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Legal research with context-aware analysis
   */
  async legalResearch(
    query: string,
    jurisdiction: string = "federal",
    practiceArea: string = "general",
    limit: number = 10
  ): Promise<{
    results: SearchResult;
    synthesis: {
      summary: string;
      keyFindings: string[];
      recommendations: string[];
      confidence: number;
    };
    metadata: {
      query: string;
      jurisdiction: string;
      practiceArea: string;
      processingTime: number;
    };
  }> {
    const startTime = Date.now();

    try {
      // Perform semantic search
      const searchFilter = {
        jurisdiction,
        practiceArea,
      };

      const searchResults = await this.semanticSearch(query, limit, searchFilter);

      // Synthesize findings using LLM
      const researchPrompt = PromptTemplate.fromTemplate(`
You are a legal research assistant. Analyze the following search results for the query "{query}" in the {jurisdiction} jurisdiction, focusing on {practiceArea} law.

Search Results:
{searchResults}

Provide a synthesis in JSON format:
{{
  "summary": "Overall summary of findings",
  "keyFindings": ["finding 1", "finding 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...],
  "confidence": 0.0-1.0
}}

Focus on:
- Relevant legal precedents
- Applicable statutes and regulations
- Key legal principles
- Practical implications
- Compliance requirements

Respond only with valid JSON.`);

      const synthesisChain = RunnableSequence.from([
        researchPrompt,
        this.llm,
        new StringOutputParser(),
      ]);

      const formattedResults = formatDocumentsAsString(searchResults.documents);
      const synthesisResult = await synthesisChain.invoke({
        query,
        jurisdiction,
        practiceArea,
        searchResults: formattedResults.substring(0, 6000), // Limit for token management
      });

      let synthesis: any;
      try {
        synthesis = JSON.parse(synthesisResult);
      } catch (parseError) {
        synthesis = {
          summary: "Research completed with relevant findings",
          keyFindings: ["Multiple relevant documents found"],
          recommendations: ["Review identified documents for detailed analysis"],
          confidence: 0.7,
        };
      }

      return {
        results: searchResults,
        synthesis: {
          summary: synthesis.summary || "Research completed",
          keyFindings: synthesis.keyFindings || [],
          recommendations: synthesis.recommendations || [],
          confidence: synthesis.confidence || 0.7,
        },
        metadata: {
          query,
          jurisdiction,
          practiceArea,
          processingTime: Date.now() - startTime,
        },
      };

    } catch (error) {
      console.error("Legal research error:", error);
      throw new Error(`Research failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Question answering with document context
   */
  async answerQuestion(
    question: string,
    contextDocuments: Document[],
    options: {
      includeSourceQuotes?: boolean;
      maxLength?: number;
    } = {}
  ): Promise<{
    answer: string;
    confidence: number;
    sources: Array<{ content: string; metadata: any; relevance: number }>;
    processingTime: number;
  }> {
    const startTime = Date.now();

    try {
      const qaPrompt = PromptTemplate.fromTemplate(`
Answer the following question based on the provided legal documents. Be precise and cite relevant information.

Question: {question}

Context Documents:
{context}

Provide your answer in JSON format:
{{
  "answer": "Detailed answer to the question",
  "confidence": 0.0-1.0,
  "keyPoints": ["point 1", "point 2", ...],
  "citations": ["relevant quote 1", "relevant quote 2", ...]
}}

Guidelines:
- Base your answer only on the provided context
- Be specific and cite relevant legal provisions
- Indicate confidence level based on available information
- Include relevant quotes if helpful
- If the context doesn't contain sufficient information, indicate this clearly

Respond only with valid JSON.`);

      const chain = RunnableSequence.from([
        qaPrompt,
        this.llm,
        new StringOutputParser(),
      ]);

      const contextText = formatDocumentsAsString(contextDocuments);
      const result = await chain.invoke({
        question,
        context: contextText.substring(0, 6000), // Limit for token management
      });

      let parsed: any;
      try {
        parsed = JSON.parse(result);
      } catch (parseError) {
        parsed = {
          answer: result,
          confidence: 0.6,
          keyPoints: [],
          citations: [],
        };
      }

      // Calculate source relevance (simplified)
      const sources = contextDocuments.map((doc, index) => ({
        content: doc.pageContent.substring(0, 200) + "...",
        metadata: doc.metadata,
        relevance: Math.max(0.1, 1.0 - index * 0.1), // Simple relevance scoring
      }));

      return {
        answer: parsed.answer || "Unable to provide answer based on available context",
        confidence: parsed.confidence || 0.6,
        sources,
        processingTime: Date.now() - startTime,
      };

    } catch (error) {
      console.error("Question answering error:", error);
      throw new Error(`QA failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get service health status
   */
  async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    services: {
      ollama: { status: string; models: string[]; responseTime?: number };
      vectorStore: { status: string; connection: boolean; responseTime?: number };
      embeddings: { status: string; model: string; dimensions?: number };
    };
    timestamp: string;
  }> {
    const health: {
      status: 'healthy' | 'degraded' | 'unhealthy';
      services: {
        ollama: { status: string; models: string[]; responseTime?: number };
        vectorStore: { status: string; connection: boolean; responseTime?: number };
        embeddings: { status: string; model: string; dimensions?: number };
      };
      timestamp: string;
    } = {
      status: 'healthy',
      services: {
        ollama: { status: 'unknown', models: [] as string[] },
        vectorStore: { status: 'unknown', connection: false },
        embeddings: { status: 'unknown', model: this.config.embeddingModel },
      },
      timestamp: new Date().toISOString(),
    };

    // Check Ollama
    try {
      const startTime = Date.now();
      const testResponse = await this.llm.invoke("Hello");
      health.services.ollama = {
        status: 'healthy',
        models: [this.config.llmModel],
        responseTime: Date.now() - startTime,
      };
    } catch (error) {
      health.services.ollama.status = 'unhealthy';
      health.status = 'degraded';
    }

    // Check Vector Store
    try {
      if (this.vectorStore) {
        const startTime = Date.now();
        await this.vectorStore.similaritySearch("test", 1);
        health.services.vectorStore = {
          status: 'healthy',
          connection: true,
          responseTime: Date.now() - startTime,
        };
      } else {
        health.services.vectorStore.status = 'not_initialized';
      }
    } catch (error) {
      health.services.vectorStore.status = 'unhealthy';
      health.status = 'degraded';
    }

    // Check Embeddings
    try {
      const testEmbedding = await this.generateEmbeddings("test");
      health.services.embeddings = {
        status: 'healthy',
        model: this.config.embeddingModel,
        dimensions: testEmbedding.length,
      };
    } catch (error) {
      health.services.embeddings.status = 'unhealthy';
      health.status = 'degraded';
    }

    // Overall status
    const unhealthyServices = Object.values(health.services).filter(s => s.status === 'unhealthy').length;
    if (unhealthyServices > 1) {
      health.status = 'unhealthy';
    } else if (unhealthyServices === 1) {
      health.status = 'degraded';
    }

    return health;
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<LangChainConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.initializeModels();
    
    // Reset vector store if connection params changed
    if (newConfig.vectorStoreTable) {
      this.vectorStore = null;
      this.isInitialized = false;
    }
  }

  // Private helper methods

  private parseNonJsonResponse(response: string): any {
    // Fallback parser for non-JSON responses
    return {
      summary: response.substring(0, 200) + "...",
      entities: [],
      keyTerms: [],
      risks: [],
      recommendations: [],
      confidence: 0.6,
    };
  }

  /**
   * Clean up resources
   */
  async dispose(): Promise<void> {
    try {
      // Close vector store connections if needed
      if (this.vectorStore) {
        // PGVectorStore doesn't have explicit close method in current version
        this.vectorStore = null;
      }
      
      this.isInitialized = false;
      console.log("🧹 LangChain service disposed");
    } catch (error) {
      console.error("Error disposing LangChain service:", error);
    }
  }
}

// Export singleton instance
export const legalAIService = new LegalAILangChainService();

// Export class for testing
export { LegalAILangChainService, type LangChainConfig, type AnalysisResult, type SearchResult };

// Auto-initialize in production
if (process.env.NODE_ENV === 'production') {
  legalAIService.initializeVectorStore().catch(console.error);
}
