/**
 * YoRHa Document Ingestion Pipeline
 * MinIO → OCR → Chunk → Embed → pgvector + Neo4j
 */

import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { createReadStream, createWriteStream, unlinkSync } from 'fs';
import { pipeline } from 'stream/promises';
import { join } from 'path';
import { createHash } from 'crypto';
import * as pdf from 'pdf-parse';
import * as tesseract from 'tesseract.js';
import amqp from 'amqplib';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { documentChunks, legalDocuments, documentSections } from './schema';
import pino from 'pino';
import neo4j from 'neo4j-driver';

// Configuration
const config = {
  s3: {
    endpoint: process.env.MINIO_ENDPOINT || 'http://localhost:9000',
    region: 'us-east-1',
    credentials: {
      accessKeyId: process.env.MINIO_ACCESS_KEY || 'minioadmin',
      secretAccessKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
    },
    forcePathStyle: true
  },
  database: {
    url: process.env.DATABASE_URL || 'postgresql://localhost:5432/legal_ai_db'
  },
  neo4j: {
    url: process.env.NEO4J_URL || 'neo4j://localhost:7687',
    username: process.env.NEO4J_USERNAME || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'password'
  },
  embedding: {
    service: process.env.EMBEDDING_SERVICE_URL || 'http://localhost:8092',
    model: 'nomic-embed-text',
    dimensions: 384
  },
  rabbitmq: {
    url: process.env.AMQP_URL || 'amqp://localhost'
  },
  chunking: {
    maxChunkSize: 500, // tokens
    overlap: 50,       // token overlap between chunks
    minChunkSize: 50   // minimum viable chunk size
  }
};

// Logger
const logger = pino({
  transport: { target: 'pino-pretty' }
});

// Text chunking with overlap
interface TextChunk {
  index: number;
  content: string;
  tokens: number;
  startOffset: number;
  endOffset: number;
  metadata: Record<string, any>;
}

class TextChunker {
  private tokenize(text: string): string[] {
    // Simple word-based tokenization (could be enhanced with tiktoken)
    return text.split(/\s+/).filter(word => word.length > 0);
  }

  chunk(text: string, metadata: Record<string, any> = {}): TextChunk[] {
    const tokens = this.tokenize(text);
    const chunks: TextChunk[] = [];
    const { maxChunkSize, overlap, minChunkSize } = config.chunking;

    let index = 0;
    let startIdx = 0;

    while (startIdx < tokens.length) {
      const endIdx = Math.min(startIdx + maxChunkSize, tokens.length);
      const chunkTokens = tokens.slice(startIdx, endIdx);
      
      if (chunkTokens.length >= minChunkSize || index === 0) {
        const content = chunkTokens.join(' ');
        const startOffset = tokens.slice(0, startIdx).join(' ').length;
        const endOffset = startOffset + content.length;

        chunks.push({
          index,
          content,
          tokens: chunkTokens.length,
          startOffset,
          endOffset,
          metadata: {
            ...metadata,
            totalTokens: tokens.length,
            chunkRatio: chunkTokens.length / tokens.length
          }
        });

        index++;
      }

      // Move window with overlap
      const step = Math.max(maxChunkSize - overlap, 1);
      startIdx += step;

      // Prevent infinite loop
      if (step <= 0) break;
    }

    logger.debug(`Chunked ${tokens.length} tokens into ${chunks.length} chunks`);
    return chunks;
  }
}

// Document processing with OCR
class DocumentProcessor {
  private s3: S3Client;
  private chunker: TextChunker;

  constructor() {
    this.s3 = new S3Client(config.s3);
    this.chunker = new TextChunker();
  }

  async downloadFromMinio(bucket: string, key: string): Promise<{ path: string; metadata: any }> {
    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const response = await this.s3.send(command);
    
    const tempPath = join('/tmp', `ingest_${Date.now()}_${Math.random().toString(36)}`);
    const writeStream = createWriteStream(tempPath);
    
    await pipeline(response.Body as any, writeStream);
    
    return {
      path: tempPath,
      metadata: {
        contentType: response.ContentType,
        lastModified: response.LastModified,
        size: response.ContentLength
      }
    };
  }

  async extractText(filePath: string, contentType?: string): Promise<string> {
    try {
      if (contentType?.includes('pdf')) {
        return await this.extractPdfText(filePath);
      } else if (contentType?.startsWith('image/')) {
        return await this.extractImageText(filePath);
      } else {
        // Try reading as plain text
        const fs = await import('fs/promises');
        return await fs.readFile(filePath, 'utf-8');
      }
    } catch (error) {
      logger.error(`Text extraction failed: ${error.message}`);
      throw error;
    }
  }

  private async extractPdfText(filePath: string): Promise<string> {
    const fs = await import('fs/promises');
    const dataBuffer = await fs.readFile(filePath);
    const pdfData = await pdf(dataBuffer);
    return pdfData.text;
  }

  private async extractImageText(filePath: string): Promise<string> {
    const { data: { text } } = await tesseract.recognize(filePath, 'eng', {
      logger: m => logger.debug(`Tesseract: ${m.status} - ${m.progress}`)
    });
    return text;
  }

  async processDocument(
    documentId: string, 
    bucket: string, 
    key: string,
    documentType: 'evidence' | 'legal_document' | 'case_summary' = 'legal_document'
  ): Promise<TextChunk[]> {
    const startTime = Date.now();
    let tempPath: string | null = null;

    try {
      // Download from MinIO
      logger.info(`Downloading document ${documentId} from ${bucket}/${key}`);
      const { path, metadata: fileMetadata } = await this.downloadFromMinio(bucket, key);
      tempPath = path;

      // Extract text with OCR support
      const extractedText = await this.extractText(path, fileMetadata.contentType);
      
      if (!extractedText || extractedText.trim().length === 0) {
        throw new Error('No text could be extracted from document');
      }

      // Create document metadata
      const docMetadata = {
        documentId,
        documentType,
        sourceUri: `minio://${bucket}/${key}`,
        fileMetadata,
        processedAt: new Date().toISOString(),
        extractedLength: extractedText.length
      };

      // Chunk the text
      const chunks = this.chunker.chunk(extractedText, docMetadata);
      
      const processingTime = Date.now() - startTime;
      logger.info(
        `Processed document ${documentId}: ${extractedText.length} chars → ${chunks.length} chunks (${processingTime}ms)`
      );

      return chunks;
    } catch (error) {
      logger.error(`Document processing failed for ${documentId}:`, error);
      throw error;
    } finally {
      // Cleanup temporary file
      if (tempPath) {
        try {
          unlinkSync(tempPath);
        } catch (cleanupError) {
          logger.warn(`Failed to cleanup temp file ${tempPath}:`, cleanupError);
        }
      }
    }
  }
}

// Embedding service client
class EmbeddingClient {
  private serviceUrl: string;

  constructor() {
    this.serviceUrl = config.embedding.service;
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.serviceUrl}/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model: config.embedding.model
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding service error: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (!result.success || !result.vector) {
        throw new Error('Invalid embedding response');
      }

      return result.vector;
    } catch (error) {
      logger.error('Embedding generation failed:', error);
      throw error;
    }
  }

  async batchGenerateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      const response = await fetch(`${this.serviceUrl}/embed/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texts,
          model: config.embedding.model
        })
      });

      if (!response.ok) {
        throw new Error(`Batch embedding service error: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (!result.success || !result.embeddings) {
        throw new Error('Invalid batch embedding response');
      }

      return result.embeddings.map((emb: any) => emb.vector);
    } catch (error) {
      logger.error('Batch embedding generation failed:', error);
      throw error;
    }
  }
}

// Entity extraction for Neo4j
class EntityExtractor {
  extractLegalEntities(text: string): Array<{ name: string; type: string; confidence: number }> {
    const entities: Array<{ name: string; type: string; confidence: number }> = [];
    
    // Legal entity patterns (simplified - could use NER model)
    const patterns = [
      { regex: /\b(?:v\.|vs?\.|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/gi, type: 'case_party' },
      { regex: /\b(?:USC|U\.S\.C\.|CFR|C\.F\.R\.)\s+§?\s*(\d+[\w\-\.]*)/gi, type: 'statute' },
      { regex: /\b(\d+)\s+(?:F\.|F\.2d|F\.3d|S\.Ct\.|U\.S\.)\s+(\d+)/gi, type: 'citation' },
      { regex: /\b([A-Z][a-z]+\s+(?:County|District|Circuit|Court))/gi, type: 'jurisdiction' },
      { regex: /\b(Judge|Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/gi, type: 'judge' }
    ];

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.regex.exec(text)) !== null) {
        entities.push({
          name: match[1] || match[0],
          type: pattern.type,
          confidence: 0.8 // Static confidence - could be enhanced with ML
        });
      }
    }

    return entities;
  }
}

// Neo4j knowledge graph integration
class KnowledgeGraphManager {
  private driver: neo4j.Driver;

  constructor() {
    this.driver = neo4j.driver(
      config.neo4j.url,
      neo4j.auth.basic(config.neo4j.username, config.neo4j.password)
    );
  }

  async createDocumentNodes(
    documentId: string,
    chunks: TextChunk[],
    entities: Array<{ name: string; type: string; confidence: number }>
  ): Promise<void> {
    const session = this.driver.session();
    
    try {
      await session.writeTransaction(async (tx) => {
        // Create document node
        await tx.run(`
          MERGE (d:Document {id: $documentId})
          SET d.chunksCount = $chunksCount,
              d.entitiesCount = $entitiesCount,
              d.processedAt = datetime()
        `, {
          documentId,
          chunksCount: chunks.length,
          entitiesCount: entities.length
        });

        // Create entity nodes and relationships
        for (const entity of entities) {
          await tx.run(`
            MERGE (e:Entity {name: $name, type: $type})
            SET e.confidence = $confidence
            WITH e
            MATCH (d:Document {id: $documentId})
            MERGE (d)-[:CONTAINS_ENTITY {confidence: $confidence}]->(e)
          `, {
            name: entity.name,
            type: entity.type,
            confidence: entity.confidence,
            documentId
          });
        }
      });

      logger.info(`Created Neo4j nodes for document ${documentId}: ${entities.length} entities`);
    } catch (error) {
      logger.error(`Neo4j node creation failed for ${documentId}:`, error);
    } finally {
      await session.close();
    }
  }

  async disconnect(): Promise<void> {
    await this.driver.close();
  }
}

// Main ingestion service
export class DocumentIngestService {
  private db: ReturnType<typeof drizzle>;
  private processor: DocumentProcessor;
  private embedding: EmbeddingClient;
  private extractor: EntityExtractor;
  private knowledgeGraph: KnowledgeGraphManager;

  constructor() {
    const client = postgres(config.database.url);
    this.db = drizzle(client);
    this.processor = new DocumentProcessor();
    this.embedding = new EmbeddingClient();
    this.extractor = new EntityExtractor();
    this.knowledgeGraph = new KnowledgeGraphManager();
  }

  async ingestDocument(job: {
    documentId: string;
    bucket: string;
    key: string;
    documentType?: 'evidence' | 'legal_document' | 'case_summary';
    caseId?: string;
  }): Promise<void> {
    const { documentId, bucket, key, documentType = 'legal_document', caseId } = job;
    
    try {
      logger.info(`Starting ingestion for document ${documentId}`);

      // Step 1: Process document (download, OCR, chunk)
      const chunks = await this.processor.processDocument(documentId, bucket, key, documentType);
      
      if (chunks.length === 0) {
        throw new Error('No chunks generated from document');
      }

      // Step 2: Generate embeddings for all chunks
      const texts = chunks.map(chunk => chunk.content);
      const embeddings = await this.embedding.batchGenerateEmbeddings(texts);

      if (embeddings.length !== chunks.length) {
        throw new Error(`Embedding count mismatch: ${embeddings.length} vs ${chunks.length}`);
      }

      // Step 3: Store chunks with embeddings in pgvector
      const chunkInserts = chunks.map((chunk, index) => ({
        documentId,
        documentType,
        chunkIndex: chunk.index,
        content: chunk.content,
        embedding: embeddings[index],
        metadata: {
          ...chunk.metadata,
          tokens: chunk.tokens,
          startOffset: chunk.startOffset,
          endOffset: chunk.endOffset,
          caseId
        }
      }));

      await this.db.insert(documentChunks).values(chunkInserts);
      
      logger.info(`Stored ${chunks.length} chunks for document ${documentId}`);

      // Step 4: Extract entities for knowledge graph
      const fullText = chunks.map(c => c.content).join(' ');
      const entities = this.extractor.extractLegalEntities(fullText);

      // Step 5: Create Neo4j knowledge graph nodes
      await this.knowledgeGraph.createDocumentNodes(documentId, chunks, entities);

      // Step 6: Update document metadata if exists
      try {
        await this.db
          .update(legalDocuments)
          .set({
            updatedAt: new Date(),
            metadata: {
              chunksCount: chunks.length,
              entitiesCount: entities.length,
              processedAt: new Date().toISOString()
            } as any
          })
          .where({
            id: documentId as any
          });
      } catch (updateError) {
        logger.warn(`Could not update document metadata: ${updateError.message}`);
      }

      logger.info(`✅ Document ingestion completed for ${documentId}`);

    } catch (error) {
      logger.error(`❌ Document ingestion failed for ${documentId}:`, error);
      throw error;
    }
  }

  async searchSimilarChunks(
    queryText: string, 
    options: {
      limit?: number;
      threshold?: number;
      documentType?: string;
      caseId?: string;
    } = {}
  ): Promise<Array<{ content: string; similarity: number; metadata: any }>> {
    const { limit = 10, threshold = 0.7, documentType, caseId } = options;

    try {
      // Generate embedding for query
      const queryEmbedding = await this.embedding.generateEmbedding(queryText);

      // Build SQL query with pgvector similarity
      let whereClause = 'WHERE embedding IS NOT NULL';
      const params: any[] = [queryEmbedding, limit];

      if (documentType) {
        whereClause += ` AND document_type = $${params.length + 1}`;
        params.push(documentType);
      }

      if (caseId) {
        whereClause += ` AND metadata->>'caseId' = $${params.length + 1}`;
        params.push(caseId);
      }

      const query = `
        SELECT 
          content, 
          metadata,
          1 - (embedding <=> $1) as similarity
        FROM document_chunks 
        ${whereClause}
        AND 1 - (embedding <=> $1) > ${threshold}
        ORDER BY embedding <=> $1
        LIMIT $2
      `;

      const result = await this.db.execute({
        sql: query,
        args: params
      } as any);

      return result.rows as any[];
    } catch (error) {
      logger.error('Similar chunks search failed:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    await this.knowledgeGraph.disconnect();
  }
}

// RabbitMQ worker
export class IngestWorker {
  private service: DocumentIngestService;
  private connection: amqp.Connection | null = null;

  constructor() {
    this.service = new DocumentIngestService();
  }

  async start(): Promise<void> {
    try {
      this.connection = await amqp.connect(config.rabbitmq.url);
      const channel = await this.connection.createChannel();

      await channel.assertQueue('document.ingest', { durable: true });
      await channel.prefetch(1); // Process one document at a time

      logger.info('🚀 Document ingestion worker started');

      channel.consume('document.ingest', async (msg) => {
        if (!msg) return;

        try {
          const job = JSON.parse(msg.content.toString());
          logger.info(`Processing ingestion job: ${JSON.stringify(job)}`);

          await this.service.ingestDocument(job);
          
          channel.ack(msg);
          logger.info(`✅ Ingestion job completed: ${job.documentId}`);
        } catch (error) {
          logger.error(`❌ Ingestion job failed:`, error);
          channel.nack(msg, false, false); // Send to DLQ
        }
      });

    } catch (error) {
      logger.error('Failed to start ingest worker:', error);
      throw error;
    }
  }

  async stop(): Promise<void> {
    if (this.connection) {
      await this.connection.close();
    }
    await this.service.disconnect();
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('🛑 Shutting down ingestion worker...');
  process.exit(0);
});

// Start worker if run directly
if (require.main === module) {
  const worker = new IngestWorker();
  worker.start().catch((error) => {
    logger.error('Worker startup failed:', error);
    process.exit(1);
  });
}