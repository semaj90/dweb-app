// MinIO Storage + Neo4j Graph + pgvector Integration Service
import { writable, derived } from 'svelte/store';
import type { 
  Document, 
  Evidence, 
  Case,
  GraphNode,
  GraphRelationship,
  VectorDocument,
  StorageMetadata
} from '$lib/types/legal-ai';

// ==========================================
// STORAGE INTEGRATION CONFIGURATION
// ==========================================

interface IntegrationConfig {
  minioEndpoint: string;
  minioAccessKey: string;
  minioSecretKey: string;
  minioBucket: string;
  neo4jUri: string;
  neo4jUsername: string;
  neo4jPassword: string;
  postgresUri: string;
  vectorDimensions: number;
  encryptionEnabled: boolean;
  compressionEnabled: boolean;
}

const defaultConfig: IntegrationConfig = {
  minioEndpoint: 'http://localhost:9000',
  minioAccessKey: 'minioadmin',
  minioSecretKey: 'minioadmin',
  minioBucket: 'legal-documents',
  neo4jUri: 'bolt://localhost:7687',
  neo4jUsername: 'neo4j',
  neo4jPassword: 'password',
  postgresUri: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  vectorDimensions: 384,
  encryptionEnabled: true,
  compressionEnabled: true
};

// ==========================================
// DOCUMENT PROCESSING WORKFLOW
// ==========================================

export interface DocumentWorkflow {
  documentId: string;
  caseId: string;
  userId: string;
  filename: string;
  content: ArrayBuffer;
  metadata: DocumentMetadata;
  stages: WorkflowStage[];
  currentStage: number;
  startTime: Date;
  endTime?: Date;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
}

export interface WorkflowStage {
  name: string;
  description: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'skipped';
  startTime?: Date;
  endTime?: Date;
  progress: number;
  metadata: Record<string, any>;
  error?: string;
}

export interface DocumentMetadata {
  filename: string;
  mimeType: string;
  fileSize: number;
  uploadDate: Date;
  caseId: string;
  userId: string;
  tags: string[];
  classification: string;
  confidentialityLevel: 'public' | 'internal' | 'confidential' | 'restricted';
  retentionPolicy: string;
  customFields: Record<string, any>;
}

// ==========================================
// INTEGRATED STORAGE SERVICE
// ==========================================

export class IntegratedStorageService {
  private config: IntegrationConfig;
  private minioClient: any; // Would be S3 client or MinIO client
  private neo4jDriver: any; // Would be Neo4j driver
  private pgPool: any;      // Would be PostgreSQL connection pool

  constructor(config: Partial<IntegrationConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
    this.initializeClients();
  }

  private async initializeClients(): Promise<void> {
    // Initialize MinIO client
    this.minioClient = {
      // Mock MinIO client - would be actual MinIO SDK
      putObject: async (bucket: string, key: string, data: ArrayBuffer, metadata: any) => {
        console.log(`Storing object ${key} in bucket ${bucket}`);
        return { etag: 'mock-etag', versionId: 'mock-version' };
      },
      getObject: async (bucket: string, key: string) => {
        console.log(`Retrieving object ${key} from bucket ${bucket}`);
        return new ArrayBuffer(0);
      },
      deleteObject: async (bucket: string, key: string) => {
        console.log(`Deleting object ${key} from bucket ${bucket}`);
        return true;
      }
    };

    // Initialize Neo4j driver
    this.neo4jDriver = {
      // Mock Neo4j driver - would be actual Neo4j driver
      session: () => ({
        run: async (query: string, params: any) => {
          console.log(`Neo4j query: ${query}`, params);
          return { records: [] };
        },
        close: () => {}
      }),
      close: () => {}
    };

    // Initialize PostgreSQL pool
    this.pgPool = {
      // Mock PostgreSQL pool - would be actual pg pool
      query: async (text: string, params: any[]) => {
        console.log(`PostgreSQL query: ${text}`, params);
        return { rows: [] };
      }
    };
  }

  // ==========================================
  // DOCUMENT WORKFLOW PROCESSING
  // ==========================================

  async processDocument(workflow: DocumentWorkflow): Promise<DocumentWorkflow> {
    workflow.status = 'processing';
    workflow.startTime = new Date();

    try {
      // Stage 1: Upload to MinIO
      await this.executeStage(workflow, 0, () => this.uploadToMinio(workflow));

      // Stage 2: Extract text and metadata
      await this.executeStage(workflow, 1, () => this.extractTextAndMetadata(workflow));

      // Stage 3: Generate embeddings
      await this.executeStage(workflow, 2, () => this.generateEmbeddings(workflow));

      // Stage 4: Store in PostgreSQL with pgvector
      await this.executeStage(workflow, 3, () => this.storeInPostgres(workflow));

      // Stage 5: Create graph nodes and relationships in Neo4j
      await this.executeStage(workflow, 4, () => this.createGraphStructure(workflow));

      // Stage 6: Index for search
      await this.executeStage(workflow, 5, () => this.indexForSearch(workflow));

      workflow.status = 'completed';
      workflow.endTime = new Date();

    } catch (error) {
      workflow.status = 'failed';
      workflow.error = error instanceof Error ? error.message : 'Unknown error';
      workflow.endTime = new Date();
    }

    return workflow;
  }

  private async executeStage(
    workflow: DocumentWorkflow, 
    stageIndex: number, 
    stageFunction: () => Promise<void>
  ): Promise<void> {
    const stage = workflow.stages[stageIndex];
    stage.status = 'processing';
    stage.startTime = new Date();
    stage.progress = 0;

    try {
      await stageFunction();
      stage.status = 'completed';
      stage.progress = 100;
      stage.endTime = new Date();
      workflow.currentStage = stageIndex + 1;
    } catch (error) {
      stage.status = 'failed';
      stage.error = error instanceof Error ? error.message : 'Unknown error';
      stage.endTime = new Date();
      throw error;
    }
  }

  // ==========================================
  // MINIO STORAGE OPERATIONS
  // ==========================================

  private async uploadToMinio(workflow: DocumentWorkflow): Promise<void> {
    const objectKey = this.generateObjectKey(workflow.documentId, workflow.metadata.filename);
    
    // Prepare metadata for MinIO
    const minioMetadata = {
      'Content-Type': workflow.metadata.mimeType,
      'X-Case-ID': workflow.caseId,
      'X-User-ID': workflow.userId,
      'X-Document-ID': workflow.documentId,
      'X-Classification': workflow.metadata.classification,
      'X-Confidentiality': workflow.metadata.confidentialityLevel,
      'X-Upload-Date': workflow.metadata.uploadDate.toISOString(),
      'X-File-Size': workflow.metadata.fileSize.toString(),
      'X-Tags': workflow.metadata.tags.join(',')
    };

    // Optional: Encrypt content before storage
    let content = workflow.content;
    if (this.config.encryptionEnabled) {
      content = await this.encryptContent(content);
      minioMetadata['X-Encrypted'] = 'true';
    }

    // Optional: Compress content
    if (this.config.compressionEnabled) {
      content = await this.compressContent(content);
      minioMetadata['X-Compressed'] = 'true';
    }

    // Upload to MinIO
    const result = await this.minioClient.putObject(
      this.config.minioBucket,
      objectKey,
      content,
      minioMetadata
    );

    // Store storage metadata in workflow
    workflow.stages[0].metadata = {
      objectKey,
      etag: result.etag,
      versionId: result.versionId,
      storagePath: `${this.config.minioBucket}/${objectKey}`,
      encrypted: this.config.encryptionEnabled,
      compressed: this.config.compressionEnabled
    };
  }

  private generateObjectKey(documentId: string, filename: string): string {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    // Organize by date hierarchy for efficient storage
    return `documents/${year}/${month}/${day}/${documentId}/${filename}`;
  }

  private async encryptContent(content: ArrayBuffer): Promise<ArrayBuffer> {
    // Implement AES encryption
    // For demo, return original content
    return content;
  }

  private async compressContent(content: ArrayBuffer): Promise<ArrayBuffer> {
    // Implement compression (gzip, brotli, etc.)
    // For demo, return original content
    return content;
  }

  // ==========================================
  // TEXT EXTRACTION AND METADATA
  // ==========================================

  private async extractTextAndMetadata(workflow: DocumentWorkflow): Promise<void> {
    const mimeType = workflow.metadata.mimeType;
    let extractedText = '';
    let additionalMetadata: Record<string, any> = {};

    switch (mimeType) {
      case 'application/pdf':
        ({ text: extractedText, metadata: additionalMetadata } = await this.extractFromPdf(workflow.content));
        break;
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        ({ text: extractedText, metadata: additionalMetadata } = await this.extractFromDocx(workflow.content));
        break;
      case 'text/plain':
        extractedText = new TextDecoder().decode(workflow.content);
        break;
      case 'application/json':
        const jsonContent = JSON.parse(new TextDecoder().decode(workflow.content));
        extractedText = JSON.stringify(jsonContent, null, 2);
        additionalMetadata = { structuredData: jsonContent };
        break;
      default:
        throw new Error(`Unsupported MIME type: ${mimeType}`);
    }

    // Perform OCR if needed for image-based documents
    if (mimeType.startsWith('image/') || additionalMetadata.hasImages) {
      const ocrText = await this.performOcr(workflow.content);
      extractedText += '\n\n[OCR Content]\n' + ocrText;
    }

    // Extract legal entities and terms
    const legalEntities = await this.extractLegalEntities(extractedText);
    const keyTerms = await this.extractKeyTerms(extractedText);

    // Store extracted data in workflow
    workflow.stages[1].metadata = {
      extractedText,
      textLength: extractedText.length,
      wordCount: extractedText.split(/\s+/).length,
      legalEntities,
      keyTerms,
      ...additionalMetadata
    };
  }

  private async extractFromPdf(content: ArrayBuffer): Promise<{ text: string; metadata: Record<string, any> }> {
    // PDF extraction logic (would use pdf-parse or similar)
    return {
      text: 'Mock extracted PDF text',
      metadata: {
        pageCount: 1,
        hasImages: false,
        fonts: [],
        creator: 'Unknown'
      }
    };
  }

  private async extractFromDocx(content: ArrayBuffer): Promise<{ text: string; metadata: Record<string, any> }> {
    // DOCX extraction logic (would use mammoth or similar)
    return {
      text: 'Mock extracted DOCX text',
      metadata: {
        wordCount: 100,
        paragraphCount: 5,
        hasImages: false,
        author: 'Unknown'
      }
    };
  }

  private async performOcr(content: ArrayBuffer): Promise<string> {
    // OCR logic (would use Tesseract.js or similar)
    return 'Mock OCR extracted text';
  }

  private async extractLegalEntities(text: string): Promise<any[]> {
    // Legal entity extraction (would use NER models)
    return [
      { name: 'Contract', type: 'LEGAL_DOCUMENT', confidence: 0.95 },
      { name: 'John Doe', type: 'PERSON', confidence: 0.92 },
      { name: 'Acme Corp', type: 'ORGANIZATION', confidence: 0.89 }
    ];
  }

  private async extractKeyTerms(text: string): Promise<string[]> {
    // Key term extraction (would use TF-IDF or similar)
    return ['liability', 'indemnification', 'termination', 'confidentiality'];
  }

  // ==========================================
  // EMBEDDING GENERATION
  // ==========================================

  private async generateEmbeddings(workflow: DocumentWorkflow): Promise<void> {
    const extractedText = workflow.stages[1].metadata.extractedText;
    const chunks = await this.chunkText(extractedText);
    const embeddings: number[][] = [];

    for (const chunk of chunks) {
      const embedding = await this.generateEmbeddingForChunk(chunk.content);
      embeddings.push(embedding);
    }

    workflow.stages[2].metadata = {
      chunks,
      embeddings,
      chunkCount: chunks.length,
      embeddingDimensions: this.config.vectorDimensions
    };
  }

  private async chunkText(text: string): Promise<Array<{ id: string; content: string; metadata: any }>> {
    const chunks = [];
    const chunkSize = 512;
    const overlap = 64;

    for (let i = 0; i < text.length; i += chunkSize - overlap) {
      const chunk = text.slice(i, i + chunkSize);
      chunks.push({
        id: `chunk_${chunks.length}`,
        content: chunk,
        metadata: {
          startIndex: i,
          endIndex: i + chunk.length,
          length: chunk.length
        }
      });
    }

    return chunks;
  }

  private async generateEmbeddingForChunk(text: string): Promise<number[]> {
    // Generate embedding using embedding service
    // For demo, return random vector
    return new Array(this.config.vectorDimensions)
      .fill(0)
      .map(() => Math.random() - 0.5);
  }

  // ==========================================
  // POSTGRESQL + PGVECTOR STORAGE
  // ==========================================

  private async storeInPostgres(workflow: DocumentWorkflow): Promise<void> {
    const documentData = {
      id: workflow.documentId,
      case_id: workflow.caseId,
      user_id: workflow.userId,
      filename: workflow.metadata.filename,
      mime_type: workflow.metadata.mimeType,
      file_size: workflow.metadata.fileSize,
      extracted_text: workflow.stages[1].metadata.extractedText,
      storage_path: workflow.stages[0].metadata.storagePath,
      classification: workflow.metadata.classification,
      confidentiality_level: workflow.metadata.confidentialityLevel,
      tags: workflow.metadata.tags,
      metadata: JSON.stringify({
        ...workflow.metadata.customFields,
        extraction: workflow.stages[1].metadata,
        storage: workflow.stages[0].metadata
      }),
      created_at: new Date(),
      updated_at: new Date()
    };

    // Insert document record
    const documentQuery = `
      INSERT INTO legal_documents (
        id, case_id, user_id, filename, mime_type, file_size,
        extracted_text, storage_path, classification, confidentiality_level,
        tags, metadata, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
      RETURNING id
    `;

    await this.pgPool.query(documentQuery, [
      documentData.id, documentData.case_id, documentData.user_id,
      documentData.filename, documentData.mime_type, documentData.file_size,
      documentData.extracted_text, documentData.storage_path,
      documentData.classification, documentData.confidentiality_level,
      documentData.tags, documentData.metadata,
      documentData.created_at, documentData.updated_at
    ]);

    // Insert document chunks with embeddings
    const chunks = workflow.stages[2].metadata.chunks;
    const embeddings = workflow.stages[2].metadata.embeddings;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = embeddings[i];

      const chunkQuery = `
        INSERT INTO document_chunks (
          id, document_id, chunk_index, content, embedding,
          start_index, end_index, metadata, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      `;

      await this.pgPool.query(chunkQuery, [
        chunk.id,
        workflow.documentId,
        i,
        chunk.content,
        `[${embedding.join(',')}]`, // pgvector format
        chunk.metadata.startIndex,
        chunk.metadata.endIndex,
        JSON.stringify(chunk.metadata),
        new Date()
      ]);
    }

    workflow.stages[3].metadata = {
      documentRecordCreated: true,
      chunksStored: chunks.length,
      vectorsStored: embeddings.length
    };
  }

  // ==========================================
  // NEO4J GRAPH STRUCTURE
  // ==========================================

  private async createGraphStructure(workflow: DocumentWorkflow): Promise<void> {
    const session = this.neo4jDriver.session();

    try {
      // Create document node
      await this.createDocumentNode(session, workflow);

      // Create case relationship
      await this.createCaseRelationship(session, workflow);

      // Create entity nodes and relationships
      await this.createEntityGraph(session, workflow);

      // Create user relationship
      await this.createUserRelationship(session, workflow);

      // Create temporal relationships
      await this.createTemporalRelationships(session, workflow);

      workflow.stages[4].metadata = {
        graphNodesCreated: true,
        relationshipsCreated: true
      };

    } finally {
      await session.close();
    }
  }

  private async createDocumentNode(session: any, workflow: DocumentWorkflow): Promise<void> {
    const query = `
      CREATE (d:Document {
        id: $documentId,
        filename: $filename,
        mimeType: $mimeType,
        fileSize: $fileSize,
        classification: $classification,
        confidentialityLevel: $confidentialityLevel,
        extractedText: $extractedText,
        wordCount: $wordCount,
        tags: $tags,
        uploadDate: $uploadDate,
        storagePath: $storagePath
      })
      RETURN d
    `;

    await session.run(query, {
      documentId: workflow.documentId,
      filename: workflow.metadata.filename,
      mimeType: workflow.metadata.mimeType,
      fileSize: workflow.metadata.fileSize,
      classification: workflow.metadata.classification,
      confidentialityLevel: workflow.metadata.confidentialityLevel,
      extractedText: workflow.stages[1].metadata.extractedText.substring(0, 1000), // Truncate for Neo4j
      wordCount: workflow.stages[1].metadata.wordCount,
      tags: workflow.metadata.tags,
      uploadDate: workflow.metadata.uploadDate.toISOString(),
      storagePath: workflow.stages[0].metadata.storagePath
    });
  }

  private async createCaseRelationship(session: any, workflow: DocumentWorkflow): Promise<void> {
    const query = `
      MATCH (c:Case {id: $caseId})
      MATCH (d:Document {id: $documentId})
      CREATE (c)-[:CONTAINS]->(d)
    `;

    await session.run(query, {
      caseId: workflow.caseId,
      documentId: workflow.documentId
    });
  }

  private async createEntityGraph(session: any, workflow: DocumentWorkflow): Promise<void> {
    const entities = workflow.stages[1].metadata.legalEntities;

    for (const entity of entities) {
      // Create entity node
      const entityQuery = `
        MERGE (e:Entity {name: $name, type: $type})
        SET e.confidence = $confidence
        RETURN e
      `;

      await session.run(entityQuery, {
        name: entity.name,
        type: entity.type,
        confidence: entity.confidence
      });

      // Create relationship between document and entity
      const relationQuery = `
        MATCH (d:Document {id: $documentId})
        MATCH (e:Entity {name: $entityName})
        CREATE (d)-[:MENTIONS {confidence: $confidence}]->(e)
      `;

      await session.run(relationQuery, {
        documentId: workflow.documentId,
        entityName: entity.name,
        confidence: entity.confidence
      });
    }
  }

  private async createUserRelationship(session: any, workflow: DocumentWorkflow): Promise<void> {
    const query = `
      MATCH (u:User {id: $userId})
      MATCH (d:Document {id: $documentId})
      CREATE (u)-[:UPLOADED {timestamp: $timestamp}]->(d)
    `;

    await session.run(query, {
      userId: workflow.userId,
      documentId: workflow.documentId,
      timestamp: workflow.metadata.uploadDate.toISOString()
    });
  }

  private async createTemporalRelationships(session: any, workflow: DocumentWorkflow): Promise<void> {
    // Create relationships based on temporal proximity of uploads
    const query = `
      MATCH (d1:Document {id: $documentId})
      MATCH (d2:Document)
      WHERE d2.id <> $documentId
        AND d2.uploadDate > $startRange
        AND d2.uploadDate < $endRange
        AND EXISTS((d1)<-[:CONTAINS]-(:Case)-[:CONTAINS]->(d2))
      CREATE (d1)-[:TEMPORAL_RELATED {
        relationship: 'uploaded_around_same_time',
        confidence: 0.7
      }]->(d2)
    `;

    const uploadDate = workflow.metadata.uploadDate;
    const oneDay = 24 * 60 * 60 * 1000;

    await session.run(query, {
      documentId: workflow.documentId,
      startRange: new Date(uploadDate.getTime() - oneDay).toISOString(),
      endRange: new Date(uploadDate.getTime() + oneDay).toISOString()
    });
  }

  // ==========================================
  // SEARCH INDEXING
  // ==========================================

  private async indexForSearch(workflow: DocumentWorkflow): Promise<void> {
    // Create full-text search indexes
    await this.createFullTextIndex(workflow);

    // Create vector similarity indexes
    await this.createVectorIndex(workflow);

    // Create metadata indexes
    await this.createMetadataIndexes(workflow);

    workflow.stages[5].metadata = {
      fullTextIndexed: true,
      vectorIndexed: true,
      metadataIndexed: true
    };
  }

  private async createFullTextIndex(workflow: DocumentWorkflow): Promise<void> {
    const query = `
      CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.extractedText)
    `;
    
    const session = this.neo4jDriver.session();
    try {
      await session.run(query);
    } finally {
      await session.close();
    }
  }

  private async createVectorIndex(workflow: DocumentWorkflow): Promise<void> {
    // Create pgvector index for similarity search
    const indexQuery = `
      CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
      ON document_chunks USING ivfflat (embedding vector_cosine_ops)
      WITH (lists = 100)
    `;

    await this.pgPool.query(indexQuery);
  }

  private async createMetadataIndexes(workflow: DocumentWorkflow): Promise<void> {
    const indexes = [
      'CREATE INDEX IF NOT EXISTS idx_documents_case_id ON legal_documents(case_id)',
      'CREATE INDEX IF NOT EXISTS idx_documents_user_id ON legal_documents(user_id)',
      'CREATE INDEX IF NOT EXISTS idx_documents_classification ON legal_documents(classification)',
      'CREATE INDEX IF NOT EXISTS idx_documents_confidentiality ON legal_documents(confidentiality_level)',
      'CREATE INDEX IF NOT EXISTS idx_documents_tags ON legal_documents USING GIN(tags)',
      'CREATE INDEX IF NOT EXISTS idx_documents_created_at ON legal_documents(created_at)',
      'CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)'
    ];

    for (const indexQuery of indexes) {
      try {
        await this.pgPool.query(indexQuery);
      } catch (error) {
        console.warn(`Failed to create index: ${indexQuery}`, error);
      }
    }
  }

  // ==========================================
  // RETRIEVAL OPERATIONS
  // ==========================================

  async retrieveDocument(documentId: string): Promise<{
    content: ArrayBuffer;
    metadata: DocumentMetadata;
    graphData: any;
    vectorData: any;
  } | null> {
    // Get document metadata from PostgreSQL
    const docQuery = `
      SELECT * FROM legal_documents WHERE id = $1
    `;
    const docResult = await this.pgPool.query(docQuery, [documentId]);
    
    if (docResult.rows.length === 0) {
      return null;
    }

    const doc = docResult.rows[0];

    // Retrieve content from MinIO
    const content = await this.minioClient.getObject(
      this.config.minioBucket,
      this.extractObjectKeyFromPath(doc.storage_path)
    );

    // Get graph data from Neo4j
    const session = this.neo4jDriver.session();
    const graphQuery = `
      MATCH (d:Document {id: $documentId})
      OPTIONAL MATCH (d)-[r]->(e)
      RETURN d, collect({relationship: type(r), entity: e}) as relationships
    `;
    const graphResult = await session.run(graphQuery, { documentId });
    await session.close();

    // Get vector chunks from PostgreSQL
    const chunkQuery = `
      SELECT * FROM document_chunks WHERE document_id = $1 ORDER BY chunk_index
    `;
    const chunkResult = await this.pgPool.query(chunkQuery, [documentId]);

    return {
      content,
      metadata: {
        filename: doc.filename,
        mimeType: doc.mime_type,
        fileSize: doc.file_size,
        uploadDate: doc.created_at,
        caseId: doc.case_id,
        userId: doc.user_id,
        tags: doc.tags,
        classification: doc.classification,
        confidentialityLevel: doc.confidentiality_level,
        retentionPolicy: '',
        customFields: JSON.parse(doc.metadata || '{}')
      },
      graphData: graphResult.records[0]?.toObject() || null,
      vectorData: chunkResult.rows
    };
  }

  async searchSimilarDocuments(
    queryVector: number[], 
    options: {
      caseId?: string;
      topK?: number;
      threshold?: number;
      filters?: Record<string, any>;
    } = {}
  ): Promise<any[]> {
    const topK = options.topK || 10;
    const threshold = options.threshold || 0.7;

    let whereClause = `embedding <=> $1 < $2`;
    const params: any[] = [`[${queryVector.join(',')}]`, 1 - threshold];
    let paramIndex = 3;

    if (options.caseId) {
      whereClause += ` AND dc.document_id IN (
        SELECT id FROM legal_documents WHERE case_id = $${paramIndex}
      )`;
      params.push(options.caseId);
      paramIndex++;
    }

    const query = `
      SELECT 
        dc.*,
        ld.filename,
        ld.case_id,
        ld.classification,
        1 - (dc.embedding <=> $1) as similarity
      FROM document_chunks dc
      JOIN legal_documents ld ON dc.document_id = ld.id
      WHERE ${whereClause}
      ORDER BY dc.embedding <=> $1
      LIMIT $${paramIndex}
    `;

    params.push(topK);

    const result = await this.pgPool.query(query, params);
    return result.rows;
  }

  private extractObjectKeyFromPath(storagePath: string): string {
    // Extract object key from storage path (bucket/key format)
    return storagePath.split('/').slice(1).join('/');
  }

  // ==========================================
  // CLEANUP AND MAINTENANCE
  // ==========================================

  async deleteDocument(documentId: string): Promise<void> {
    // Delete from MinIO
    const docQuery = `SELECT storage_path FROM legal_documents WHERE id = $1`;
    const docResult = await this.pgPool.query(docQuery, [documentId]);
    
    if (docResult.rows.length > 0) {
      const objectKey = this.extractObjectKeyFromPath(docResult.rows[0].storage_path);
      await this.minioClient.deleteObject(this.config.minioBucket, objectKey);
    }

    // Delete from Neo4j
    const session = this.neo4jDriver.session();
    const neo4jQuery = `
      MATCH (d:Document {id: $documentId})
      DETACH DELETE d
    `;
    await session.run(neo4jQuery, { documentId });
    await session.close();

    // Delete from PostgreSQL (cascading deletes chunks)
    const pgQuery = `DELETE FROM legal_documents WHERE id = $1`;
    await this.pgPool.query(pgQuery, [documentId]);
  }

  async cleanup(): Promise<void> {
    // Close all connections
    if (this.neo4jDriver) {
      await this.neo4jDriver.close();
    }
    if (this.pgPool) {
      await this.pgPool.end();
    }
  }
}

// ==========================================
// SVELTE STORES
// ==========================================

export const storageService = new IntegratedStorageService();

export const documentWorkflows = writable<Map<string, DocumentWorkflow>>(new Map());

export const storageStats = writable({
  totalDocuments: 0,
  totalStorage: 0,
  totalCases: 0,
  totalUsers: 0,
  indexHealth: 'healthy' as 'healthy' | 'degraded' | 'error'
});

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

export function createDocumentWorkflow(
  documentId: string,
  caseId: string,
  userId: string,
  filename: string,
  content: ArrayBuffer,
  metadata: Partial<DocumentMetadata>
): DocumentWorkflow {
  const fullMetadata: DocumentMetadata = {
    filename,
    mimeType: metadata.mimeType || 'application/octet-stream',
    fileSize: content.byteLength,
    uploadDate: new Date(),
    caseId,
    userId,
    tags: metadata.tags || [],
    classification: metadata.classification || 'general',
    confidentialityLevel: metadata.confidentialityLevel || 'internal',
    retentionPolicy: metadata.retentionPolicy || 'standard',
    customFields: metadata.customFields || {}
  };

  return {
    documentId,
    caseId,
    userId,
    filename,
    content,
    metadata: fullMetadata,
    stages: [
      { name: 'Upload to MinIO', description: 'Store document in object storage', status: 'pending', progress: 0, metadata: {} },
      { name: 'Extract Text', description: 'Extract text content and metadata', status: 'pending', progress: 0, metadata: {} },
      { name: 'Generate Embeddings', description: 'Create vector embeddings for search', status: 'pending', progress: 0, metadata: {} },
      { name: 'Store in PostgreSQL', description: 'Save to relational database with vectors', status: 'pending', progress: 0, metadata: {} },
      { name: 'Create Graph Structure', description: 'Build relationships in Neo4j', status: 'pending', progress: 0, metadata: {} },
      { name: 'Index for Search', description: 'Create search indexes', status: 'pending', progress: 0, metadata: {} }
    ],
    currentStage: 0,
    startTime: new Date(),
    status: 'pending'
  };
}

export async function processDocumentWorkflow(workflow: DocumentWorkflow): Promise<void> {
  documentWorkflows.update(workflows => {
    workflows.set(workflow.documentId, workflow);
    return workflows;
  });

  try {
    const processedWorkflow = await storageService.processDocument(workflow);
    
    documentWorkflows.update(workflows => {
      workflows.set(workflow.documentId, processedWorkflow);
      return workflows;
    });

    // Update storage stats
    if (processedWorkflow.status === 'completed') {
      storageStats.update(stats => ({
        ...stats,
        totalDocuments: stats.totalDocuments + 1,
        totalStorage: stats.totalStorage + workflow.metadata.fileSize
      }));
    }

  } catch (error) {
    console.error('Document workflow failed:', error);
    
    documentWorkflows.update(workflows => {
      const failedWorkflow = workflows.get(workflow.documentId);
      if (failedWorkflow) {
        failedWorkflow.status = 'failed';
        failedWorkflow.error = error instanceof Error ? error.message : 'Unknown error';
        failedWorkflow.endTime = new Date();
      }
      return workflows;
    });
  }
}

export default storageService;