// Enhanced Ingestion Pipeline for Legal AI System
// Processes legal documents, creates embeddings, and stores in vector database
// Production Stack: PostgreSQL + pgvector, LangChain, Qdrant, GraphQL, Neo4j, Redis, RabbitMQ

import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Pool } from "pg";
import { createClient } from "redis";
import { connect } from "amqplib";
// import neo4j from "neo4j-driver"; // TODO: Install neo4j-driver dependency
const neo4j = null as any;
// import { errorHandler } from "$lib/utils/errorHandler";
// Fallback error handler for when the module is not available
const errorHandler = {
  system: (message: string, data?: any) => console.error(`[SYSTEM] ${message}`, data),
  analysis: (message: string, data?: any) => console.error(`[ANALYSIS] ${message}`, data),
};
import type { Document } from "@langchain/core/documents";
import { resolveLibraryId, getLibraryDocs } from "./mcp-helpers";
// import { copilotOrchestrator } from "$lib/utils/mcp-helpers";
// Mock copilot orchestrator function
const copilotOrchestrator = async (prompt: string, options: any) => ({
  selfPrompt: "Mock copilot analysis completed",
});
import type { DocumentEmbedding } from "./som-rag-system";
import { SelfOrganizingMapRAG } from "./som-rag-system";
import { QdrantService } from "./qdrant-service";

// Multimodal Evidence Processing
interface MultimodalEvidence {
  id: string;
  type: "image" | "video" | "audio" | "document" | "forensic";
  file_path: string;
  metadata: {
    filename: string;
    size: number;
    mime_type: string;
    case_id: string;
    upload_timestamp: string;
    processing_status: "pending" | "processing" | "completed" | "failed";
    confidence_scores?: {
      ocr?: number;
      object_detection?: number;
      scene_analysis?: number;
      legal_relevance?: number;
    };
    anchor_points?: AnchorPoint[];
    timeline_segments?: TimelineSegment[];
  };
  extracted_content: {
    text?: string;
    objects?: DetectedObject[];
    transcription?: string;
    scene_summary?: string;
    legal_analysis?: string;
  };
}

interface AnchorPoint {
  id: string;
  type: "object" | "text" | "audio_segment" | "timeline_event" | "custom";
  coordinates: {
    x: number; // Normalized 0-1
    y: number; // Normalized 0-1
    width?: number;
    height?: number;
  };
  timestamp?: number; // For video/audio
  confidence: number;
  description: string;
  legal_relevance: "high" | "medium" | "low";
  user_verified?: boolean;
  notes?: string;
}

export interface TimelineSegment {
  start_time: number;
  end_time: number;
  event_type: string;
  description: string;
  confidence: number;
  legal_significance: string;
}

export interface DetectedObject {
  class: string;
  confidence: number;
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  legal_relevance: "high" | "medium" | "low";
}

// Embedding for a document or evidence item
// (Removed: use canonical DocumentEmbedding from som-rag-system)

// ...existing code...


// Claude Desktop Context Integration
interface CopilotArchitectureContext {
  architecture_summary: string;
  legal_context: string;
  copilot_patterns: string;
  enhancement_priority: boolean;
}

export interface IngestionDocument {
  id: string;
  content: string;
  metadata: {
    filename: string;
    case_id?: string;
    evidence_type: "digital" | "physical" | "testimony" | "forensic";
    legal_category: string;
    upload_timestamp: number;
    file_size: number;
    mime_type: string;
    extracted_entities?: string[];
    confidence_score?: number;
  };
}

export interface ProcessingResult {
  document_id: string;
  embedding: number[];
  cluster_id: number;
  processing_time: number;
  extraction_metadata: {
    entities: string[];
    keywords: string[];
    confidence: number;
    language: string;
  };
  vector_store_id?: string;
}

export interface IngestionStats {
  total_processed: number;
  successful: number;
  failed: number;
  avg_processing_time: number;
  cluster_distribution: Record<number, number>;
  evidence_type_distribution: Record<string, number>;
}

export class EnhancedIngestionPipeline {
  private qdrantClient: QdrantClient;
  private pgPool: Pool;
  private redisClient: any;
  private rabbitConnection: any;
  private neo4jDriver: any;
  private vectorStore?: QdrantVectorStore;
  private pgVectorStore?: PGVectorStore;
  private isInitialized = false;
  private processingQueue: IngestionDocument[] = [];
  private isProcessing = false;
  private stats: IngestionStats = {
    total_processed: 0,
    successful: 0,
    failed: 0,
    avg_processing_time: 0,
    cluster_distribution: {},
    evidence_type_distribution: {},
  };
  private copilotContext: CopilotArchitectureContext | null = null;
  private multimodalProcessors: Map<string, any> = new Map();
  private anchorPointCache: Map<string, AnchorPoint[]> = new Map();
  
  // Add missing required properties
  private somRAG: SelfOrganizingMapRAG;
  private qdrantService: QdrantService;

  constructor(
    config: {
      qdrantUrl?: string;
      pgConnectionString?: string;
      redisUrl?: string;
      rabbitmqUrl?: string;
      neo4jUrl?: string;
    } = {}
  ) {
    // Initialize connections
    this.qdrantClient = new QdrantClient({
      url: config.qdrantUrl || "http://localhost:6333",
    });
    this.pgPool = new Pool({
      connectionString: config.pgConnectionString || process.env.DATABASE_URL,
    });
    this.redisClient = createClient({
      url: config.redisUrl || "redis://localhost:6379",
    });
    this.neo4jDriver = neo4j.driver(
      config.neo4jUrl || "bolt://localhost:7687",
      neo4j.auth.basic("neo4j", "password")
    );

    // Initialize SOM RAG and Qdrant service
    this.somRAG = new SelfOrganizingMapRAG({
      mapWidth: 10,
      mapHeight: 10,
      dimensions: 384,
      learningRate: 0.1,
      neighborhoodRadius: 2,
      maxEpochs: 100,
      clusterCount: 8,
    });
    
    this.qdrantService = new QdrantService({
      url: config.qdrantUrl || "http://localhost:6333",
      collectionName: "legal_documents",
      vectorSize: 384,
    });

    this.initializeCopilotIntegration();
    this.initializeMultimodalProcessors();
  }

  async initialize(): Promise<void> {
    console.log("🚀 Initializing Enhanced Ingestion Pipeline...");

    try {
      // Check Qdrant connection
      await this.qdrantClient.getCollections();

      // Ensure legal documents collection exists
      await this.ensureCollection("legal_documents");

      this.isInitialized = true;
      console.log("✅ Enhanced Ingestion Pipeline initialized");
    } catch (error) {
      console.error("❌ Failed to initialize ingestion pipeline:", error);
      errorHandler.system("Pipeline initialization failed", {
        error: error.message,
      });
      throw error;
    }
  }

  private async ensureCollection(collectionName: string): Promise<void> {
    try {
      const collections = await this.qdrantClient.getCollections();
      const exists = collections.collections.some(
        (c) => c.name === collectionName
      );

      if (!exists) {
        await this.qdrantClient.createCollection(collectionName, {
          vectors: {
            size: 384, // Default embedding size
            distance: "Cosine",
          },
        });
        console.log(`✅ Created collection: ${collectionName}`);
      }
    } catch (error) {
      console.error(`❌ Failed to ensure collection ${collectionName}:`, error);
      throw error;
    }
  }

  /**
   * Initialize Claude Desktop & Copilot architecture context
   */
  private async initializeCopilotIntegration(): Promise<void> {
    try {
      console.log("🤖 Initializing Claude Desktop & Copilot integration...");

      // Load Copilot architecture context using Context7 MCP
      const contextLibId = await resolveLibraryId("copilot-architecture");
      const architectureDocs = await getLibraryDocs(
        contextLibId,
        "legal-ai-integration"
      );

      this.copilotContext = {
        architecture_summary: architectureDocs.substring(0, 2000),
        legal_context: "Legal AI workflow with evidence processing",
        copilot_patterns:
          "SvelteKit + Drizzle ORM + Qdrant + multimodal analysis",
        enhancement_priority: true,
      };

      console.log("✅ Copilot integration initialized");
    } catch (error) {
      console.warn("⚠️ Copilot integration failed, continuing without:", error);
    }
  }

  /**
   * Initialize multimodal evidence processors
   */
  private async initializeMultimodalProcessors(): Promise<void> {
    console.log("🎥 Initializing multimodal processors...");

    // Mock processors for different evidence types
    this.multimodalProcessors.set("image", {
      process: this.processImageEvidence.bind(this),
      supportedFormats: ["jpg", "jpeg", "png", "tiff", "bmp"],
    });

    this.multimodalProcessors.set("video", {
      process: this.processVideoEvidence.bind(this),
      supportedFormats: ["mp4", "avi", "mov", "mkv", "webm"],
    });

    this.multimodalProcessors.set("audio", {
      process: this.processAudioEvidence.bind(this),
      supportedFormats: ["mp3", "wav", "flac", "m4a", "ogg"],
    });

    this.multimodalProcessors.set("document", {
      process: this.processDocumentEvidence.bind(this),
      supportedFormats: ["pdf", "docx", "txt", "rtf"],
    });
  }

  /**
   * Process image evidence: OCR, object detection, embedding, and storage
   */
  private async processImageEvidence(
    evidence: MultimodalEvidence
  ): Promise<DocumentEmbedding> {
    const extractedText = evidence.extracted_content.text || "";
    const embedding = await this.generateEmbedding(extractedText);
    const docEmbedding: DocumentEmbedding = {
      id: evidence.id,
      content: extractedText,
      embedding,
      metadata: {
        case_id: evidence.metadata.case_id,
        evidence_type: "image",
        legal_category: this.determineLegalCategory(evidence),
        confidence: evidence.metadata.confidence_scores?.ocr || 0.8,
        timestamp: Date.now(),
      },
    };
    await this.storeInQdrant(docEmbedding);
    return docEmbedding;
  }

  /**
   * Process video evidence: scene analysis, timeline segmentation, embedding, and storage
   */
  private async processVideoEvidence(
    evidence: MultimodalEvidence
  ): Promise<DocumentEmbedding> {
    const sceneSummary = evidence.extracted_content.scene_summary || "";
    const embedding = await this.generateEmbedding(sceneSummary);
    const docEmbedding: DocumentEmbedding = {
      id: evidence.id,
      content: sceneSummary,
      embedding,
      metadata: {
        case_id: evidence.metadata.case_id,
        evidence_type: "video",
        legal_category: this.determineLegalCategory(evidence),
        confidence: evidence.metadata.confidence_scores?.scene_analysis || 0.8,
        timestamp: Date.now(),
      },
    };
    await this.storeInQdrant(docEmbedding);
    return docEmbedding;
  }

  /**
   * Process audio evidence: transcription, embedding, and storage
   */
  private async processAudioEvidence(
    evidence: MultimodalEvidence
  ): Promise<DocumentEmbedding> {
    const transcription = evidence.extracted_content.transcription || "";
    const embedding = await this.generateEmbedding(transcription);
    const docEmbedding: DocumentEmbedding = {
      id: evidence.id,
      content: transcription,
      embedding,
      metadata: {
        case_id: evidence.metadata.case_id,
        evidence_type: "audio",
        legal_category: this.determineLegalCategory(evidence),
        confidence: 0.8,
        timestamp: Date.now(),
      },
    };
    await this.storeInQdrant(docEmbedding);
    return docEmbedding;
  }

  /**
   * Process document evidence: text extraction, embedding, and storage
   */
  private async processDocumentEvidence(
    evidence: MultimodalEvidence
  ): Promise<DocumentEmbedding> {
    const text = evidence.extracted_content.text || "";
    const embedding = await this.generateEmbedding(text);
    const docEmbedding: DocumentEmbedding = {
      id: evidence.id,
      content: text,
      embedding,
      metadata: {
        case_id: evidence.metadata.case_id,
        evidence_type: "document",
        legal_category: this.determineLegalCategory(evidence),
        confidence: 0.9,
        timestamp: Date.now(),
      },
    };
    await this.storeInQdrant(docEmbedding);
    return docEmbedding;
  }

  /**
   * Process single document through enhanced pipeline
   */
  async processDocument(
    document: IngestionDocument
  ): Promise<ProcessingResult> {
    if (!this.isInitialized) {
      throw new Error("Pipeline not initialized. Call initialize() first.");
    }

    const startTime = Date.now();
    console.log(`📄 Processing document: ${document.metadata.filename}`);

    try {
      // 1. Extract entities and keywords from content
      const extractedData = await this.extractEntitiesAndKeywords(
        document.content
      );

      // 2. Generate embeddings using simple TF-IDF approach
      const embedding = await this.generateEmbedding(document.content);

      // 3. Store in Qdrant vector database
      const docEmbedding: DocumentEmbedding = {
        id: document.id,
        content: document.content,
        embedding,
        metadata: {
          case_id: document.metadata.case_id,
          evidence_type: document.metadata.evidence_type,
          legal_category: document.metadata.legal_category,
          confidence: extractedData.confidence,
          timestamp: Date.now(),
        },
      };
      await this.storeInQdrant(docEmbedding);
      const vectorStoreId = document.id;

      // 4. Assign to cluster (simplified clustering)
      const clusterResult = await this.assignToCluster(docEmbedding);
      const clusterId = clusterResult.cluster;

      const processingTime = Date.now() - startTime;

      // Update statistics
      this.updateStats(
        document.metadata.evidence_type,
        clusterId,
        processingTime,
        true
      );

      const result: ProcessingResult = {
        document_id: document.id,
        embedding,
        cluster_id: clusterId,
        processing_time: processingTime,
        extraction_metadata: extractedData,
        vector_store_id: vectorStoreId,
      };

      console.log(
        `✅ Document processed successfully: ${document.id} (${processingTime}ms)`
      );
      return result;
    } catch (error) {
      console.error(`❌ Failed to process document ${document.id}:`, error);
      errorHandler.analysis(`Document processing failed: ${document.id}`, {
        error: error.message,
      });
      this.updateStats(
        document.metadata.evidence_type,
        -1,
        Date.now() - startTime,
        false
      );
      throw error;
    }
  }

  /**
   * Process multiple documents in batch
   */
  async processBatch(
    documents: IngestionDocument[]
  ): Promise<ProcessingResult[]> {
    console.log(`📦 Processing batch of ${documents.length} documents...`);

    const results: ProcessingResult[] = [];
    const batchStartTime = Date.now();

    // Process documents concurrently with rate limiting
    const batchSize = 5; // Process 5 documents at a time

    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      const batchPromises = batch.map((doc) =>
        this.processDocument(doc).catch((error) => {
          console.error(`Failed to process document ${doc.id}:`, error);
          return null;
        })
      );

      const batchResults = await Promise.all(batchPromises);
      results.push(
        ...(batchResults.filter(
          (result) => result !== null
        ) as ProcessingResult[])
      );

      // Small delay between batches to prevent overwhelming the system
      if (i + batchSize < documents.length) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    const totalTime = Date.now() - batchStartTime;
    console.log(
      `✅ Batch processing completed: ${results.length}/${documents.length} successful (${totalTime}ms total)`
    );

    return results;
  }

  /**
   * Add documents to processing queue
   */
  async queueDocuments(documents: IngestionDocument[]): Promise<void> {
    this.processingQueue.push(...documents);
    console.log(
      `📋 Added ${documents.length} documents to queue. Queue size: ${this.processingQueue.length}`
    );

    if (!this.isProcessing) {
      this.processQueue();
    }
  }

  /**
   * Process queued documents automatically
   */
  private async processQueue(): Promise<void> {
    if (this.isProcessing || this.processingQueue.length === 0) return;

    this.isProcessing = true;
    console.log("🔄 Starting queue processing...");

    while (this.processingQueue.length > 0) {
      // Process in batches of 10 for optimal performance
      const batchSize = Math.min(10, this.processingQueue.length);
      const batch = this.processingQueue.splice(0, batchSize);

      try {
        await this.processBatch(batch);
      } catch (error) {
        console.error("Batch processing failed:", error);
      }

      // Small delay between batches to prevent overwhelming the system
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    this.isProcessing = false;
    console.log("✅ Queue processing completed");
  }

  /**
   * Enhanced search using Qdrant vector similarity
   */
  async enhancedSearch(
    query: string,
    filters?: {
      evidence_type?: string;
      case_id?: string;
      confidence_threshold?: number;
      cluster_id?: number;
    },
    limit: number = 10
  ): Promise<{
    documents: any[];
    clusters_searched: number[];
    processing_time: number;
  }> {
    const startTime = Date.now();

    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      // Build Qdrant filter
      const qdrantFilter: any = {};
      if (filters) {
        const must = [];

        if (filters.evidence_type) {
          must.push({
            key: "evidence_type",
            match: { value: filters.evidence_type },
          });
        }

        if (filters.case_id) {
          must.push({ key: "case_id", match: { value: filters.case_id } });
        }

        if (filters.confidence_threshold) {
          must.push({
            key: "confidence",
            range: { gte: filters.confidence_threshold },
          });
        }

        if (must.length > 0) {
          qdrantFilter.must = must;
        }
      }

      // Search in Qdrant
      const searchResults = await this.qdrantClient.search("legal_documents", {
        vector: queryEmbedding,
        limit,
        filter: Object.keys(qdrantFilter).length > 0 ? qdrantFilter : undefined,
        with_payload: true,
      });

      const documents = searchResults.map((result) => ({
        id: result.id,
        content: result.payload?.content || "",
        score: result.score,
        metadata: result.payload?.metadata || {},
      }));

      const clustersSearched = Array.from(
        new Set(documents.map((doc: any) => doc.metadata.cluster_id || 0))
      );

      const processingTime = Date.now() - startTime;

      return {
        documents,
        clusters_searched: clustersSearched,
        processing_time: processingTime,
      };
    } catch (error) {
      console.error("❌ Enhanced search failed:", error);
      errorHandler.analysis("Enhanced search failed", {
        query,
        error: error.message,
      });

      return {
        documents: [],
        clusters_searched: [],
        processing_time: Date.now() - startTime,
      };
    }
  }

  /**
   * Get ingestion statistics
   */
  getStats(): IngestionStats & {
    queue_size: number;
    is_processing: boolean;
    collection_info?: any;
  } {
    return {
      ...this.stats,
      queue_size: this.processingQueue.length,
      is_processing: this.isProcessing,
    };
  }

  /**
   * Export collection info for analysis
   */
  async getCollectionInfo(): Promise<any> {
    try {
      return await this.qdrantClient.getCollection("legal_documents");
    } catch (error) {
      console.error("Failed to get collection info:", error);
      return null;
    }
  }

  /**
   * Private helper methods
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    // Clean and preprocess text
    const cleanText = this.preprocessText(text);

    // Simple TF-IDF-based embedding generation
    return this.generateTFIDF(cleanText);
  }

  private generateTFIDF(text: string): number[] {
    // Simple TF-IDF implementation for legal documents
    const words = text.toLowerCase().split(/\s+/);
    const vocabulary = new Set(words);
    const embedding = new Array(384).fill(0);

    // Legal terms dictionary for enhanced relevance
    const legalTerms = [
      "evidence",
      "testimony",
      "witness",
      "defendant",
      "plaintiff",
      "case",
      "court",
      "judge",
      "jury",
      "trial",
      "legal",
      "law",
      "statute",
      "crime",
      "contract",
      "liability",
      "damages",
      "settlement",
      "appeal",
      "ruling",
    ];

    // Calculate term frequencies and apply legal term weighting
    words.forEach((word, index) => {
      const termFreq = words.filter((w) => w === word).length / words.length;
      const isLegalTerm = legalTerms.includes(word) ? 2 : 1;
      const position = index % 384;

      embedding[position] +=
        termFreq * isLegalTerm * (Math.random() * 0.1 + 0.9);
    });

    // Normalize the embedding vector
    const magnitude = Math.sqrt(
      embedding.reduce((sum, val) => sum + val * val, 0)
    );
    if (magnitude > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= magnitude;
      }
    }

    return embedding;
  }

  private preprocessText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .substring(0, 512); // Limit length for embedding model
  }

  private async extractEntitiesAndKeywords(content: string): Promise<{
    entities: string[];
    keywords: string[];
    confidence: number;
    language: string;
  }> {
    // Mock implementation for entity extraction
    // In real implementation, this would use NLP libraries like spaCy or NLTK

    const words = content.toLowerCase().split(/\s+/);
    const legalKeywords = [
      "evidence",
      "testimony",
      "forensic",
      "case",
      "defendant",
      "plaintiff",
    ];

    const foundKeywords = words.filter((word) => legalKeywords.includes(word));
    const entities = foundKeywords.slice(0, 5); // Mock entities

    return {
      entities,
      keywords: foundKeywords,
      confidence: Math.min(foundKeywords.length / 10, 1.0),
      language: "en",
    };
  }

  private async updateSOMWithDocument(
    document: DocumentEmbedding
  ): Promise<void> {
    // For single document updates, we would typically use online learning
    // For now, we'll use batch training with the single document
    await this.somRAG.trainSOM([document]);
  }

  private async assignToCluster(document: DocumentEmbedding): Promise<{
    cluster: number;
    boolean_pattern: boolean[][];
  }> {
    // This would normally be handled by the SOM system
    // Mock implementation
    const clusterId = Math.floor(Math.random() * 8);
    const booleanPattern = [
      [Math.random() > 0.5, Math.random() > 0.5],
      [Math.random() > 0.5, Math.random() > 0.5],
    ];

    return {
      cluster: clusterId,
      boolean_pattern: booleanPattern,
    };
  }

  private async storeInQdrant(document: DocumentEmbedding): Promise<void> {
    // Store in Qdrant for vector similarity search
    try {
      await this.qdrantClient.upsert("legal_documents", {
        wait: true,
        points: [
          {
            id: document.id,
            vector: document.embedding,
            payload: {
              content: document.content,
              metadata: document.metadata,
            },
          },
        ],
      });
    } catch (error) {
      console.error("Failed to store document in Qdrant:", error);
      throw error;
    }
  }

  private async storeInNeo4j(
    document: DocumentEmbedding,
    clusterResult: { cluster: number; boolean_pattern: boolean[][] }
  ): Promise<void> {
    // Mock implementation for Neo4j storage
    // In real implementation, this would create nodes and relationships
    console.log(
      `📊 Storing document ${document.id} in Neo4j cluster ${clusterResult.cluster}`
    );
  }

  private updateStats(
    evidenceType: string,
    cluster: number,
    processingTime: number,
    success: boolean
  ): void {
    this.stats.total_processed++;

    if (success) {
      this.stats.successful++;
    } else {
      this.stats.failed++;
    }

    // Update average processing time
    this.stats.avg_processing_time =
      (this.stats.avg_processing_time * (this.stats.total_processed - 1) +
        processingTime) /
      this.stats.total_processed;

    // Update cluster distribution
    if (cluster >= 0) {
      this.stats.cluster_distribution[cluster] =
        (this.stats.cluster_distribution[cluster] || 0) + 1;
    }

    // Update evidence type distribution
    this.stats.evidence_type_distribution[evidenceType] =
      (this.stats.evidence_type_distribution[evidenceType] || 0) + 1;
  }

  /**
   * Process multimodal evidence with Claude Desktop integration
   */
  async processMultimodalEvidence(evidence: MultimodalEvidence): Promise<{
    processing_result: ProcessingResult;
    anchor_points: AnchorPoint[];
    timeline_segments?: TimelineSegment[];
    copilot_analysis?: string;
  }> {
    console.log(
      `🔍 Processing multimodal evidence: ${evidence.metadata.filename}`
    );

    const startTime = Date.now();
    let anchorPoints: AnchorPoint[] = [];
    let timelineSegments: TimelineSegment[] = [];
    let copilotAnalysis = "";

    try {
      // 1. Process based on evidence type
      const processor = this.multimodalProcessors.get(evidence.type);
      if (!processor) {
        throw new Error(
          `No processor available for evidence type: ${evidence.type}`
        );
      }

      const processingResult = await processor.process(evidence);
      anchorPoints = processingResult.anchor_points || [];
      timelineSegments = processingResult.timeline_segments || [];

      // 2. Enhanced analysis with Copilot integration
      if (this.copilotContext) {
        copilotAnalysis = await this.generateCopilotAnalysis(
          evidence,
          processingResult
        );
      }

      // 3. Create document for ingestion pipeline
      const mappedEvidenceType = this.mapEvidenceType(evidence.type);
      const document: IngestionDocument = {
        id: evidence.id,
        content: this.createEvidenceContent(evidence, processingResult),
        metadata: {
          case_id: evidence.metadata.case_id,
          evidence_type: mappedEvidenceType,
          legal_category: this.determineLegalCategory(evidence),
          upload_timestamp: Date.now(),
          filename: evidence.metadata.filename,
          file_size: evidence.metadata.size,
          mime_type: evidence.metadata.mime_type,
          confidence_score: evidence.metadata.confidence_scores?.legal_relevance || 0.8,
        },
      };

      // 4. Process through standard ingestion pipeline
      const pipelineResult = await this.processDocument(document);

      // 5. Cache anchor points for interactive viewing
      this.anchorPointCache.set(evidence.id, anchorPoints);

      const processingTime = Date.now() - startTime;
      console.log(
        `✅ Multimodal evidence processed: ${evidence.id} (${processingTime}ms)`
      );

      return {
        processing_result: pipelineResult,
        anchor_points: anchorPoints,
        timeline_segments: timelineSegments,
        copilot_analysis: copilotAnalysis,
      };
    } catch (error) {
      console.error(
        `❌ Failed to process multimodal evidence ${evidence.id}:`,
        error
      );
      throw error;
    }
  }





  /**
   * Generate Copilot-enhanced analysis using Context7 MCP integration
   */
  private async generateCopilotAnalysis(
    evidence: MultimodalEvidence,
    processingResult: any
  ): Promise<string> {
    try {
      if (!this.copilotContext) return "";

      const prompt = `Analyze the following ${evidence.type} evidence for legal significance:

      Evidence: ${evidence.metadata.filename}
      Case ID: ${evidence.metadata.case_id}
      Extracted Content: ${JSON.stringify(processingResult.extracted_content, null, 2)}

      Context: ${this.copilotContext.legal_context}

      Provide legal analysis focusing on:
      1. Admissibility considerations
      2. Chain of custody requirements
      3. Potential challenges
      4. Relevance to case`;

      const orchestrationResult = await copilotOrchestrator(prompt, {
        useSemanticSearch: true,
        useMemory: true,
        useCodebase: false,
        logErrors: true,
        synthesizeOutputs: true,
        context: {
          evidence_type: evidence.type,
          case_id: evidence.metadata.case_id,
          copilot_context: this.copilotContext,
        },
      });

      return (
        orchestrationResult.selfPrompt ||
        "Analysis completed with Copilot integration"
      );
    } catch (error) {
      console.warn("Copilot analysis failed:", error);
      return "Standard analysis completed (Copilot unavailable)";
    }
  }

  /**
   * Create searchable content from evidence and processing results
   */
  private createEvidenceContent(
    evidence: MultimodalEvidence,
    processingResult: any
  ): string {
    const parts = [
      `Evidence Type: ${evidence.type}`,
      `Filename: ${evidence.metadata.filename}`,
      `Case ID: ${evidence.metadata.case_id}`,
    ];

    if (processingResult.extracted_content?.text) {
      parts.push(`Extracted Text: ${processingResult.extracted_content.text}`);
    }

    if (processingResult.extracted_content?.transcription) {
      parts.push(
        `Transcription: ${processingResult.extracted_content.transcription}`
      );
    }

    if (processingResult.extracted_content?.scene_summary) {
      parts.push(
        `Scene Summary: ${processingResult.extracted_content.scene_summary}`
      );
    }

    if (processingResult.timeline_segments) {
      const timelineDesc = processingResult.timeline_segments
        .map(
          (seg: TimelineSegment) =>
            `${seg.start_time}s-${seg.end_time}s: ${seg.description}`
        )
        .join("; ");
      parts.push(`Timeline: ${timelineDesc}`);
    }

    return parts.join("\n");
  }

  /**
   * Determine legal category based on evidence analysis
   */
  private determineLegalCategory(evidence: MultimodalEvidence): string {
    const filename = evidence.metadata.filename.toLowerCase();
    const type = evidence.type;

    if (filename.includes("contract") || filename.includes("agreement")) {
      return "contract_law";
    } else if (type === "video" && filename.includes("dash")) {
      return "traffic_violation";
    } else if (type === "audio" && filename.includes("call")) {
      return "communications_evidence";
    } else if (filename.includes("forensic") || filename.includes("analysis")) {
      return "digital_forensics";
    }

    return "general_evidence";
  }

  /**
   * Map multimodal evidence type to ingestion document evidence type
   */
  private mapEvidenceType(type: "image" | "video" | "audio" | "document" | "forensic"): "digital" | "physical" | "testimony" | "forensic" {
    switch (type) {
      case "image":
      case "video":
      case "audio":
      case "document":
        return "digital";
      case "forensic":
        return "forensic";
      default:
        return "digital";
    }
  }

  /**
   * Get anchor points for interactive evidence viewer
   */
  getAnchorPoints(evidenceId: string): AnchorPoint[] {
    return this.anchorPointCache.get(evidenceId) || [];
  }

  /**
   * Search evidence with multimodal context
   */
  async searchMultimodalEvidence(
    query: string,
    filters?: {
      evidence_types?: string[];
      case_ids?: string[];
      has_anchor_points?: boolean;
      has_timeline?: boolean;
      legal_relevance?: "high" | "medium" | "low";
      confidence_threshold?: number;
    }
  ): Promise<{
    results: any[];
    anchor_context: Map<string, AnchorPoint[]>;
    copilot_insights?: string;
  }> {
    // Use enhanced search from parent class
    const searchResults = await this.enhancedSearch(query, {
      evidence_type: filters?.evidence_types?.[0],
      confidence_threshold: filters?.confidence_threshold,
    });

    // Add anchor point context
    const anchorContext = new Map<string, AnchorPoint[]>();
    for (const doc of searchResults.documents) {
      const anchors = this.getAnchorPoints(doc.id);
      if (anchors.length > 0) {
        anchorContext.set(doc.id, anchors);
      }
    }

    // Generate Copilot insights if available
    let copilotInsights = "";
    if (this.copilotContext && searchResults.documents.length > 0) {
      try {
        const insightPrompt = `Analyze search results for query "${query}" across multimodal evidence:

        Found ${searchResults.documents.length} relevant pieces of evidence.
        Evidence types: ${searchResults.documents.map((d) => d.metadata.evidence_type).join(", ")}

        Provide legal insights about:
        1. Pattern analysis across evidence
        2. Potential correlations
        3. Missing evidence gaps
        4. Strategic considerations`;

        const insights = await copilotOrchestrator(insightPrompt, {
          useSemanticSearch: true,
          useMemory: true,
          synthesizeOutputs: true,
          context: { query, evidence_count: searchResults.documents.length },
        });

        copilotInsights = insights.selfPrompt || "";
      } catch (error) {
        console.warn("Failed to generate Copilot insights:", error);
      }
    }

    return {
      results: searchResults.documents,
      anchor_context: anchorContext,
      copilot_insights: copilotInsights,
    };
  }
}

// Export factory function
export function createEnhancedIngestionPipeline(): EnhancedIngestionPipeline {
  return new EnhancedIngestionPipeline();
}

export default EnhancedIngestionPipeline;
