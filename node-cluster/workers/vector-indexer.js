const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { Pool } = require('pg');

/**
 * Vector Indexing Service Worker
 * Handles document embedding generation and vector database operations
 */
class VectorIndexerWorker {
  constructor(data) {
    this.workerId = data.workerId;
    this.services = data.services;
    this.indexedCount = 0;
    this.startTime = Date.now();
    
    // Vector processing configuration
    this.config = {
      embeddingModel: 'nomic-embed-text',
      embeddingDimensions: 768,
      batchSize: 10,
      similarityThreshold: 0.7,
      maxRetries: 3,
      retryDelay: 1000
    };
    
    // Connection pools
    this.pgPool = null;
    this.qdrantClient = null;
    this.ollamaClient = null;
    
    this.init();
  }
  
  async init() {
    console.log(`[VECTOR-INDEXER-${this.workerId}] Vector indexer worker starting`);
    
    // Setup message handling
    this.setupMessageHandling();
    
    // Initialize connections
    await this.initializeConnections();
    
    // Setup periodic maintenance
    setInterval(() => {
      this.performMaintenance();
    }, 600000); // 10 minutes
    
    this.sendMessage({
      type: 'worker-ready',
      worker: 'vector-indexer',
      pid: process.pid
    });
  }
  
  setupMessageHandling() {
    parentPort.on('message', async (message) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        console.error(`[VECTOR-INDEXER-${this.workerId}] Message handling error:`, error);
        this.sendMessage({
          type: 'error',
          worker: 'vector-indexer',
          error: error.message,
          timestamp: Date.now()
        });
      }
    });
  }
  
  async handleMessage(message) {
    switch (message.type) {
      case 'index-document':
        await this.indexDocument(message.data);
        break;
        
      case 'index-chunks':
        await this.indexChunks(message.data);
        break;
        
      case 'search-similar':
        await this.searchSimilar(message.data);
        break;
        
      case 'update-embedding':
        await this.updateEmbedding(message.data);
        break;
        
      case 'delete-vectors':
        await this.deleteVectors(message.data);
        break;
        
      case 'optimize-index':
        await this.optimizeIndex();
        break;
        
      case 'health-check':
        this.sendHealthReport();
        break;
        
      case 'memory-cleanup':
        await this.performCleanup();
        break;
        
      default:
        console.log(`[VECTOR-INDEXER-${this.workerId}] Unknown message type: ${message.type}`);
    }
  }
  
  async initializeConnections() {
    try {
      // Initialize PostgreSQL connection
      this.pgPool = new Pool({
        connectionString: this.services.postgresDb,
        max: 5,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 5000
      });
      
      // Test PostgreSQL connection
      const pgClient = await this.pgPool.connect();
      await pgClient.query('SELECT 1');
      pgClient.release();
      
      console.log(`[VECTOR-INDEXER-${this.workerId}] PostgreSQL connected`);
      
      // Initialize Qdrant client
      const { QdrantClient } = require('@qdrant/js-client-rest');
      this.qdrantClient = new QdrantClient({
        url: this.services.qdrantVector
      });
      
      // Test Qdrant connection
      await this.qdrantClient.getCollections();
      console.log(`[VECTOR-INDEXER-${this.workerId}] Qdrant connected`);
      
      // Initialize Ollama client for embeddings
      this.ollamaClient = require('ollama');
      
      console.log(`[VECTOR-INDEXER-${this.workerId}] All connections initialized`);
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Connection initialization error:`, error);
      throw error;
    }
  }
  
  async indexDocument(documentData) {
    const startTime = Date.now();
    
    try {
      console.log(`[VECTOR-INDEXER-${this.workerId}] Indexing document: ${documentData.documentId}`);
      
      // Generate document-level embedding
      const documentEmbedding = await this.generateEmbedding(documentData.textContent);
      
      // Store document embedding in PostgreSQL
      await this.storeDocumentEmbedding(documentData, documentEmbedding);
      
      // Index chunks if available
      if (documentData.chunks && documentData.chunks.length > 0) {
        await this.indexChunks({
          documentId: documentData.documentId,
          chunks: documentData.chunks,
          metadata: documentData.metadata
        });
      }
      
      // Store in Qdrant for similarity search
      await this.storeInQdrant(documentData, documentEmbedding);
      
      const result = {
        documentId: documentData.documentId,
        embeddingDimensions: documentEmbedding.length,
        chunksIndexed: documentData.chunks ? documentData.chunks.length : 0,
        processingTime: Date.now() - startTime,
        indexedBy: this.workerId,
        timestamp: Date.now()
      };
      
      this.indexedCount++;
      
      this.sendMessage({
        type: 'document-indexed',
        data: result
      });
      
      console.log(`[VECTOR-INDEXER-${this.workerId}] Document indexed in ${result.processingTime}ms`);
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Document indexing error:`, error);
      
      this.sendMessage({
        type: 'indexing-error',
        data: {
          documentId: documentData.documentId,
          error: error.message,
          processingTime: Date.now() - startTime
        }
      });
    }
  }
  
  async generateEmbedding(text, retryCount = 0) {
    try {
      const response = await this.ollamaClient.embeddings({
        model: this.config.embeddingModel,
        prompt: text
      });
      
      if (!response.embedding || response.embedding.length !== this.config.embeddingDimensions) {
        throw new Error(`Invalid embedding response: expected ${this.config.embeddingDimensions} dimensions`);
      }
      
      return response.embedding;
      
    } catch (error) {
      if (retryCount < this.config.maxRetries) {
        console.warn(`[VECTOR-INDEXER-${this.workerId}] Embedding generation retry ${retryCount + 1}:`, error.message);
        await this.sleep(this.config.retryDelay * (retryCount + 1));
        return this.generateEmbedding(text, retryCount + 1);
      }
      
      throw new Error(`Embedding generation failed after ${this.config.maxRetries} retries: ${error.message}`);
    }
  }
  
  async storeDocumentEmbedding(documentData, embedding) {
    const client = await this.pgPool.connect();
    
    try {
      // Insert or update document embedding
      const query = `
        INSERT INTO document_embeddings (
          document_id, 
          filename, 
          content, 
          embedding, 
          metadata, 
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $6)
        ON CONFLICT (document_id) 
        DO UPDATE SET 
          embedding = EXCLUDED.embedding,
          metadata = EXCLUDED.metadata,
          updated_at = EXCLUDED.updated_at
      `;
      
      const values = [
        documentData.documentId,
        documentData.filename,
        documentData.textContent.substring(0, 10000), // Limit content size
        JSON.stringify(embedding),
        JSON.stringify(documentData.metadata),
        new Date()
      ];
      
      await client.query(query, values);
      
    } finally {
      client.release();
    }
  }
  
  async indexChunks(chunkData) {
    const { documentId, chunks, metadata } = chunkData;
    
    console.log(`[VECTOR-INDEXER-${this.workerId}] Indexing ${chunks.length} chunks for document: ${documentId}`);
    
    // Process chunks in batches
    for (let i = 0; i < chunks.length; i += this.config.batchSize) {
      const batch = chunks.slice(i, i + this.config.batchSize);
      await this.processBatch(documentId, batch, metadata);
    }
  }
  
  async processBatch(documentId, chunks, metadata) {
    const client = await this.pgPool.connect();
    
    try {
      await client.query('BEGIN');
      
      for (const chunk of chunks) {
        // Generate embedding for chunk
        const embedding = await this.generateEmbedding(chunk.text);
        
        // Store chunk embedding
        const query = `
          INSERT INTO chunk_embeddings (
            document_id,
            chunk_index,
            chunk_text,
            embedding,
            start_char,
            end_char,
            word_count,
            metadata,
            created_at
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
          ON CONFLICT (document_id, chunk_index)
          DO UPDATE SET 
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            updated_at = NOW()
        `;
        
        const values = [
          documentId,
          chunk.index,
          chunk.text,
          JSON.stringify(embedding),
          chunk.startChar,
          chunk.endChar,
          chunk.wordCount,
          JSON.stringify({ ...metadata, ...chunk.metadata }),
          new Date()
        ];
        
        await client.query(query, values);
      }
      
      await client.query('COMMIT');
      
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
  
  async storeInQdrant(documentData, embedding) {
    try {
      const collectionName = 'legal_documents';
      
      // Ensure collection exists
      await this.ensureQdrantCollection(collectionName);
      
      // Prepare point data
      const point = {
        id: documentData.documentId,
        vector: embedding,
        payload: {
          filename: documentData.filename,
          fileType: documentData.metadata?.fileType || 'unknown',
          wordCount: documentData.metadata?.wordCount || 0,
          complexity: documentData.metadata?.complexity || 0,
          legalTerms: documentData.metadata?.legalTerms || [],
          documentType: documentData.legalAnalysis?.documentType?.primary || 'unknown',
          priority: documentData.legalAnalysis?.priority || 5,
          createdAt: new Date().toISOString(),
          indexedBy: this.workerId
        }
      };
      
      // Upsert point
      await this.qdrantClient.upsert(collectionName, {
        wait: true,
        points: [point]
      });
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Qdrant storage error:`, error);
      throw error;
    }
  }
  
  async ensureQdrantCollection(collectionName) {
    try {
      // Check if collection exists
      const collections = await this.qdrantClient.getCollections();
      const exists = collections.collections.some(col => col.name === collectionName);
      
      if (!exists) {
        // Create collection
        await this.qdrantClient.createCollection(collectionName, {
          vectors: {
            size: this.config.embeddingDimensions,
            distance: 'Cosine'
          },
          optimizers_config: {
            default_segment_number: 2,
            max_segment_size: 20000,
            memmap_threshold: 50000
          },
          replication_factor: 1
        });
        
        console.log(`[VECTOR-INDEXER-${this.workerId}] Created Qdrant collection: ${collectionName}`);
      }
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Qdrant collection setup error:`, error);
      throw error;
    }
  }
  
  async searchSimilar(searchData) {
    const { query, limit = 10, threshold = 0.7, filters = {} } = searchData;
    
    try {
      // Generate embedding for search query
      const queryEmbedding = await this.generateEmbedding(query);
      
      // Search in Qdrant
      const qdrantResults = await this.qdrantClient.search('legal_documents', {
        vector: queryEmbedding,
        limit: limit,
        score_threshold: threshold,
        with_payload: true,
        filter: this.buildQdrantFilter(filters)
      });
      
      // Search in PostgreSQL for chunk-level results
      const chunkResults = await this.searchSimilarChunks(queryEmbedding, limit, threshold);
      
      const results = {
        query: query,
        documentResults: qdrantResults.map(result => ({
          documentId: result.id,
          score: result.score,
          metadata: result.payload
        })),
        chunkResults: chunkResults,
        searchTime: Date.now(),
        searchedBy: this.workerId
      };
      
      this.sendMessage({
        type: 'search-results',
        data: results
      });
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Search error:`, error);
      
      this.sendMessage({
        type: 'search-error',
        data: {
          query: query,
          error: error.message
        }
      });
    }
  }
  
  async searchSimilarChunks(queryEmbedding, limit, threshold) {
    const client = await this.pgPool.connect();
    
    try {
      const query = `
        SELECT 
          document_id,
          chunk_index,
          chunk_text,
          1 - (embedding::vector <=> $1::vector) as similarity,
          metadata
        FROM chunk_embeddings
        WHERE 1 - (embedding::vector <=> $1::vector) > $2
        ORDER BY similarity DESC
        LIMIT $3
      `;
      
      const result = await client.query(query, [
        JSON.stringify(queryEmbedding),
        threshold,
        limit
      ]);
      
      return result.rows.map(row => ({
        documentId: row.document_id,
        chunkIndex: row.chunk_index,
        text: row.chunk_text,
        similarity: row.similarity,
        metadata: row.metadata
      }));
      
    } finally {
      client.release();
    }
  }
  
  buildQdrantFilter(filters) {
    const qdrantFilter = { must: [] };
    
    if (filters.documentType) {
      qdrantFilter.must.push({
        key: 'documentType',
        match: { value: filters.documentType }
      });
    }
    
    if (filters.minComplexity) {
      qdrantFilter.must.push({
        key: 'complexity',
        range: { gte: filters.minComplexity }
      });
    }
    
    if (filters.fileType) {
      qdrantFilter.must.push({
        key: 'fileType',
        match: { value: filters.fileType }
      });
    }
    
    return qdrantFilter.must.length > 0 ? qdrantFilter : undefined;
  }
  
  async updateEmbedding(updateData) {
    const { documentId, newContent } = updateData;
    
    try {
      // Generate new embedding
      const newEmbedding = await this.generateEmbedding(newContent);
      
      // Update PostgreSQL
      const client = await this.pgPool.connect();
      try {
        await client.query(
          'UPDATE document_embeddings SET embedding = $1, updated_at = NOW() WHERE document_id = $2',
          [JSON.stringify(newEmbedding), documentId]
        );
      } finally {
        client.release();
      }
      
      // Update Qdrant
      await this.qdrantClient.upsert('legal_documents', {
        points: [{
          id: documentId,
          vector: newEmbedding
        }]
      });
      
      this.sendMessage({
        type: 'embedding-updated',
        data: { documentId, timestamp: Date.now() }
      });
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Update embedding error:`, error);
      
      this.sendMessage({
        type: 'update-error',
        data: { documentId, error: error.message }
      });
    }
  }
  
  async deleteVectors(deleteData) {
    const { documentIds } = deleteData;
    
    try {
      // Delete from PostgreSQL
      const client = await this.pgPool.connect();
      try {
        await client.query('BEGIN');
        
        // Delete document embeddings
        await client.query(
          'DELETE FROM document_embeddings WHERE document_id = ANY($1)',
          [documentIds]
        );
        
        // Delete chunk embeddings
        await client.query(
          'DELETE FROM chunk_embeddings WHERE document_id = ANY($1)',
          [documentIds]
        );
        
        await client.query('COMMIT');
      } catch (error) {
        await client.query('ROLLBACK');
        throw error;
      } finally {
        client.release();
      }
      
      // Delete from Qdrant
      await this.qdrantClient.delete('legal_documents', {
        points: documentIds
      });
      
      this.sendMessage({
        type: 'vectors-deleted',
        data: { documentIds, timestamp: Date.now() }
      });
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Delete vectors error:`, error);
      
      this.sendMessage({
        type: 'delete-error',
        data: { documentIds, error: error.message }
      });
    }
  }
  
  async optimizeIndex() {
    try {
      console.log(`[VECTOR-INDEXER-${this.workerId}] Starting index optimization`);
      
      // Optimize Qdrant collection
      await this.qdrantClient.updateCollection('legal_documents', {
        optimizers_config: {
          default_segment_number: 4,
          max_segment_size: 50000
        }
      });
      
      // Run PostgreSQL VACUUM and ANALYZE
      const client = await this.pgPool.connect();
      try {
        await client.query('VACUUM ANALYZE document_embeddings');
        await client.query('VACUUM ANALYZE chunk_embeddings');
      } finally {
        client.release();
      }
      
      this.sendMessage({
        type: 'index-optimized',
        data: { timestamp: Date.now() }
      });
      
      console.log(`[VECTOR-INDEXER-${this.workerId}] Index optimization completed`);
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Index optimization error:`, error);
    }
  }
  
  async performMaintenance() {
    try {
      // Clean up old temporary data
      const client = await this.pgPool.connect();
      try {
        // Remove embeddings for documents older than 30 days without recent access
        await client.query(`
          DELETE FROM chunk_embeddings 
          WHERE document_id IN (
            SELECT document_id FROM document_embeddings 
            WHERE created_at < NOW() - INTERVAL '30 days'
            AND updated_at < NOW() - INTERVAL '7 days'
          )
        `);
      } finally {
        client.release();
      }
      
      console.log(`[VECTOR-INDEXER-${this.workerId}] Maintenance completed`);
      
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Maintenance error:`, error);
    }
  }
  
  async performCleanup() {
    // Force garbage collection
    if (global.gc) {
      global.gc();
    }
    
    console.log(`[VECTOR-INDEXER-${this.workerId}] Cleanup completed. Memory usage:`, process.memoryUsage());
  }
  
  sendHealthReport() {
    const health = {
      worker: 'vector-indexer',
      workerId: this.workerId,
      pid: process.pid,
      uptime: Date.now() - this.startTime,
      indexedCount: this.indexedCount,
      memoryUsage: process.memoryUsage(),
      connections: {
        postgres: this.pgPool ? 'connected' : 'disconnected',
        qdrant: this.qdrantClient ? 'connected' : 'disconnected',
        ollama: this.ollamaClient ? 'connected' : 'disconnected'
      },
      timestamp: Date.now()
    };
    
    this.sendMessage({
      type: 'health-report',
      data: health
    });
  }
  
  sendMessage(message) {
    try {
      parentPort.postMessage(message);
    } catch (error) {
      console.error(`[VECTOR-INDEXER-${this.workerId}] Failed to send message:`, error);
    }
  }
  
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize worker if running in worker thread
if (!isMainThread) {
  new VectorIndexerWorker(workerData);
}

module.exports = VectorIndexerWorker;