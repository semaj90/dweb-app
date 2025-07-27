// Enhanced Ingestion Pipeline with SOM Integration
// Processes legal documents through SOM, creates embeddings, and stores in boolean clusters

import { createSOMRAGSystem, type DocumentEmbedding, type SOMConfig } from './som-rag-system';
import { QdrantService } from './qdrant-service';

export interface IngestionDocument {
  id: string;
  content: string;
  metadata: {
    filename: string;
    case_id?: string;
    evidence_type: 'digital' | 'physical' | 'testimony' | 'forensic';
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
  som_cluster: number;
  boolean_pattern: boolean[][];
  processing_time: number;
  extraction_metadata: {
    entities: string[];
    keywords: string[];
    confidence: number;
    language: string;
  };
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
  private somRAG = createSOMRAGSystem();
  private qdrantService = new QdrantService();
  private embeddingModel: any; // Will be initialized with actual embedding model
  private isInitialized = false;
  private processingQueue: IngestionDocument[] = [];
  private isProcessing = false;
  private stats: IngestionStats = {
    total_processed: 0,
    successful: 0,
    failed: 0,
    avg_processing_time: 0,
    cluster_distribution: {},
    evidence_type_distribution: {}
  };

  async initialize(): Promise<void> {
    console.log('🚀 Initializing Enhanced Ingestion Pipeline...');
    
    try {
      // Initialize embedding model (placeholder for actual model)
      await this.initializeEmbeddingModel();
      
      // Initialize Qdrant service
      await this.qdrantService.initialize();
      
      this.isInitialized = true;
      console.log('✅ Enhanced Ingestion Pipeline initialized');
    } catch (error) {
      console.error('❌ Failed to initialize ingestion pipeline:', error);
      throw error;
    }
  }

  /**
   * Process single document through enhanced pipeline
   */
  async processDocument(document: IngestionDocument): Promise<ProcessingResult> {
    if (!this.isInitialized) {
      throw new Error('Pipeline not initialized. Call initialize() first.');
    }

    const startTime = Date.now();
    console.log(`📄 Processing document: ${document.metadata.filename}`);

    try {
      // 1. Extract entities and keywords from content
      const extractedData = await this.extractEntitiesAndKeywords(document.content);
      
      // 2. Generate embeddings
      const embedding = await this.generateEmbedding(document.content);
      
      // 3. Create document embedding object
      const docEmbedding: DocumentEmbedding = {
        id: document.id,
        content: document.content,
        embedding,
        metadata: {
          case_id: document.metadata.case_id,
          evidence_type: document.metadata.evidence_type,
          legal_category: document.metadata.legal_category,
          confidence: extractedData.confidence,
          timestamp: document.metadata.upload_timestamp
        }
      };
      
      // 4. Train/update SOM with new document
      await this.updateSOMWithDocument(docEmbedding);
      
      // 5. Find SOM cluster and boolean pattern
      const clusterResult = await this.assignToCluster(docEmbedding);
      
      // 6. Store in Qdrant for vector search
      await this.storeInQdrant(docEmbedding);
      
      // 7. Store in Neo4j for graph relationships
      await this.storeInNeo4j(docEmbedding, clusterResult);
      
      const processingTime = Date.now() - startTime;
      
      // Update statistics
      this.updateStats(document.metadata.evidence_type, clusterResult.cluster, processingTime, true);
      
      const result: ProcessingResult = {
        document_id: document.id,
        embedding,
        som_cluster: clusterResult.cluster,
        boolean_pattern: clusterResult.boolean_pattern,
        processing_time: processingTime,
        extraction_metadata: extractedData
      };
      
      console.log(`✅ Document processed successfully: ${document.id} (${processingTime}ms)`);
      return result;
      
    } catch (error) {
      console.error(`❌ Failed to process document ${document.id}:`, error);
      this.updateStats(document.metadata.evidence_type, -1, Date.now() - startTime, false);
      throw error;
    }
  }

  /**
   * Process multiple documents in batch with optimized SOM training
   */
  async processBatch(documents: IngestionDocument[]): Promise<ProcessingResult[]> {
    console.log(`📦 Processing batch of ${documents.length} documents...`);
    
    const results: ProcessingResult[] = [];
    const documentEmbeddings: DocumentEmbedding[] = [];
    
    // Phase 1: Generate embeddings for all documents
    console.log('⚡ Phase 1: Generating embeddings...');
    for (const document of documents) {
      try {
        const extractedData = await this.extractEntitiesAndKeywords(document.content);
        const embedding = await this.generateEmbedding(document.content);
        
        const docEmbedding: DocumentEmbedding = {
          id: document.id,
          content: document.content,
          embedding,
          metadata: {
            case_id: document.metadata.case_id,
            evidence_type: document.metadata.evidence_type,
            legal_category: document.metadata.legal_category,
            confidence: extractedData.confidence,
            timestamp: document.metadata.upload_timestamp
          }
        };
        
        documentEmbeddings.push(docEmbedding);
      } catch (error) {
        console.error(`Failed to process document ${document.id}:`, error);
      }
    }
    
    // Phase 2: Batch train SOM with all embeddings
    console.log('🧠 Phase 2: Training SOM with batch data...');
    await this.somRAG.trainSOM(documentEmbeddings);
    
    // Phase 3: Assign clusters and store results
    console.log('🗃️ Phase 3: Storing processed documents...');
    for (let i = 0; i < documentEmbeddings.length; i++) {
      const docEmbedding = documentEmbeddings[i];
      const originalDoc = documents[i];
      
      try {
        const clusterResult = await this.assignToCluster(docEmbedding);
        await this.storeInQdrant(docEmbedding);
        await this.storeInNeo4j(docEmbedding, clusterResult);
        
        const extractedData = await this.extractEntitiesAndKeywords(docEmbedding.content);
        
        results.push({
          document_id: docEmbedding.id,
          embedding: docEmbedding.embedding,
          som_cluster: clusterResult.cluster,
          boolean_pattern: clusterResult.boolean_pattern,
          processing_time: 0, // Batch processing time not tracked per document
          extraction_metadata: extractedData
        });
        
        this.updateStats(originalDoc.metadata.evidence_type, clusterResult.cluster, 0, true);
        
      } catch (error) {
        console.error(`Failed to finalize document ${docEmbedding.id}:`, error);
        this.updateStats(originalDoc.metadata.evidence_type, -1, 0, false);
      }
    }
    
    console.log(`✅ Batch processing completed: ${results.length}/${documents.length} successful`);
    return results;
  }

  /**
   * Add documents to processing queue
   */
  async queueDocuments(documents: IngestionDocument[]): Promise<void> {
    this.processingQueue.push(...documents);
    console.log(`📋 Added ${documents.length} documents to queue. Queue size: ${this.processingQueue.length}`);
    
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
    console.log('🔄 Starting queue processing...');
    
    while (this.processingQueue.length > 0) {
      // Process in batches of 10 for optimal performance
      const batchSize = Math.min(10, this.processingQueue.length);
      const batch = this.processingQueue.splice(0, batchSize);
      
      try {
        await this.processBatch(batch);
      } catch (error) {
        console.error('Batch processing failed:', error);
      }
      
      // Small delay between batches to prevent overwhelming the system
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    this.isProcessing = false;
    console.log('✅ Queue processing completed');
  }

  /**
   * Enhanced search combining SOM clusters and traditional vector search
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
    documents: DocumentEmbedding[];
    clusters_searched: number[];
    processing_time: number;
  }> {
    const startTime = Date.now();
    
    // Generate query embedding
    const queryEmbedding = await this.generateEmbedding(query);
    
    // Use SOM for initial retrieval
    const somResults = await this.somRAG.semanticSearch(query, queryEmbedding, limit * 2);
    
    // Apply filters
    let filteredResults = somResults;
    
    if (filters) {
      filteredResults = somResults.filter(doc => {
        if (filters.evidence_type && doc.metadata.evidence_type !== filters.evidence_type) {
          return false;
        }
        if (filters.case_id && doc.metadata.case_id !== filters.case_id) {
          return false;
        }
        if (filters.confidence_threshold && doc.metadata.confidence < filters.confidence_threshold) {
          return false;
        }
        return true;
      });
    }
    
    // Get clusters that were searched
    const clustersSearched = Array.from(new Set(
      filteredResults.map(doc => {
        // This would normally come from the SOM system
        return 0; // Placeholder
      })
    ));
    
    const processingTime = Date.now() - startTime;
    
    return {
      documents: filteredResults.slice(0, limit),
      clusters_searched: clustersSearched,
      processing_time: processingTime
    };
  }

  /**
   * Get ingestion statistics
   */
  getStats(): IngestionStats & {
    queue_size: number;
    is_processing: boolean;
    som_visualization: any;
  } {
    return {
      ...this.stats,
      queue_size: this.processingQueue.length,
      is_processing: this.isProcessing,
      som_visualization: this.somRAG.getVisualizationData()
    };
  }

  /**
   * Export SOM data for analysis
   */
  exportSOMData(): string {
    return this.somRAG.exportRapidJSON();
  }

  /**
   * Private helper methods
   */
  private async initializeEmbeddingModel(): Promise<void> {
    // Placeholder for actual embedding model initialization
    // This would typically initialize a model like sentence-transformers
    console.log('🤖 Initializing embedding model...');
    
    // Mock implementation
    this.embeddingModel = {
      encode: async (text: string) => {
        // Generate mock 384-dimensional embedding
        return Array.from({ length: 384 }, () => Math.random() * 2 - 1);
      }
    };
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    // Clean and preprocess text
    const cleanText = this.preprocessText(text);
    
    // Generate embedding using the model
    return await this.embeddingModel.encode(cleanText);
  }

  private preprocessText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
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
    const legalKeywords = ['evidence', 'testimony', 'forensic', 'case', 'defendant', 'plaintiff'];
    
    const foundKeywords = words.filter(word => legalKeywords.includes(word));
    const entities = foundKeywords.slice(0, 5); // Mock entities
    
    return {
      entities,
      keywords: foundKeywords,
      confidence: Math.min(foundKeywords.length / 10, 1.0),
      language: 'en'
    };
  }

  private async updateSOMWithDocument(document: DocumentEmbedding): Promise<void> {
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
      [Math.random() > 0.5, Math.random() > 0.5]
    ];
    
    return {
      cluster: clusterId,
      boolean_pattern: booleanPattern
    };
  }

  private async storeInQdrant(document: DocumentEmbedding): Promise<void> {
    // Store in Qdrant for vector similarity search
    await this.qdrantService.upsertDocument({
      id: document.id,
      vector: document.embedding,
      payload: {
        content: document.content,
        metadata: document.metadata
      }
    });
  }

  private async storeInNeo4j(
    document: DocumentEmbedding, 
    clusterResult: { cluster: number; boolean_pattern: boolean[][] }
  ): Promise<void> {
    // Mock implementation for Neo4j storage
    // In real implementation, this would create nodes and relationships
    console.log(`📊 Storing document ${document.id} in Neo4j cluster ${clusterResult.cluster}`);
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
      (this.stats.avg_processing_time * (this.stats.total_processed - 1) + processingTime) / 
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
}

// Export factory function
export function createEnhancedIngestionPipeline(): EnhancedIngestionPipeline {
  return new EnhancedIngestionPipeline();
}

export default EnhancedIngestionPipeline;