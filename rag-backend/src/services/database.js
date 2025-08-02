/**
 * Database Service with PostgreSQL + pgvector
 * Handles vector operations, document storage, and query optimization
 */

import pkg from "pg";
const { Pool } = pkg;

// Import pgvector properly for CommonJS compatibility
import pgvectorPkg from "pgvector/pg";
const { toSql } = pgvectorPkg;

export class DatabaseService {
  constructor(config) {
    this.config = config;
    this.pool = null;
  }

  async initialize() {
    this.pool = new Pool({
      host: this.config.host,
      port: this.config.port,
      database: this.config.database,
      user: this.config.username,
      password: this.config.password,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    // Register pgvector extension
    await this.pool.query("CREATE EXTENSION IF NOT EXISTS vector");

    // Create tables if they don't exist
    await this.createTables();

    console.log("✅ Database connection established with pgvector support");
    return true;
  }

  async createTables() {
    const createDocumentsTable = `
      CREATE TABLE IF NOT EXISTS rag_documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title VARCHAR(255) NOT NULL,
        content TEXT NOT NULL,
        file_path VARCHAR(500),
        file_type VARCHAR(50),
        file_size INTEGER,
        document_type VARCHAR(100) DEFAULT 'general',
        case_id VARCHAR(100),
        metadata JSONB DEFAULT '{}',
        embedding vector(384), -- Nomic embed text dimensions
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        indexed_at TIMESTAMP,
        processing_status VARCHAR(50) DEFAULT 'pending'
      );
    `;

    const createChunksTable = `
      CREATE TABLE IF NOT EXISTS rag_chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector(384),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMP DEFAULT NOW()
      );
    `;

    const createQueriesTable = `
      CREATE TABLE IF NOT EXISTS rag_queries (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        query_text TEXT NOT NULL,
        query_embedding vector(384),
        response TEXT,
        confidence_score DECIMAL(3,2),
        processing_time_ms INTEGER,
        sources JSONB DEFAULT '[]',
        user_id VARCHAR(100),
        case_id VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW()
      );
    `;

    const createJobsTable = `
      CREATE TABLE IF NOT EXISTS rag_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_type VARCHAR(50) NOT NULL,
        status VARCHAR(50) DEFAULT 'pending',
        input_data JSONB,
        output_data JSONB,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        started_at TIMESTAMP,
        completed_at TIMESTAMP
      );
    `;

    // Create indexes
    const createIndexes = `
      CREATE INDEX IF NOT EXISTS idx_documents_embedding ON rag_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
      CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
      CREATE INDEX IF NOT EXISTS idx_queries_embedding ON rag_queries USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);
      CREATE INDEX IF NOT EXISTS idx_documents_case_id ON rag_documents(case_id);
      CREATE INDEX IF NOT EXISTS idx_documents_type ON rag_documents(document_type);
      CREATE INDEX IF NOT EXISTS idx_documents_status ON rag_documents(processing_status);
      CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag_chunks(document_id);
      CREATE INDEX IF NOT EXISTS idx_queries_user_id ON rag_queries(user_id);
      CREATE INDEX IF NOT EXISTS idx_queries_case_id ON rag_queries(case_id);
      CREATE INDEX IF NOT EXISTS idx_jobs_status ON rag_jobs(status);
      CREATE INDEX IF NOT EXISTS idx_jobs_type ON rag_jobs(job_type);
    `;

    await this.pool.query(createDocumentsTable);
    await this.pool.query(createChunksTable);
    await this.pool.query(createQueriesTable);
    await this.pool.query(createJobsTable);
    await this.pool.query(createIndexes);

    console.log("✅ Database tables and indexes created");
  }

  // Document operations
  async insertDocument(document) {
    const query = `
      INSERT INTO rag_documents (title, content, file_path, file_type, file_size, document_type, case_id, metadata, embedding)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING *
    `;

    const values = [
      document.title,
      document.content,
      document.filePath,
      document.fileType,
      document.fileSize,
      document.documentType || "general",
      document.caseId,
      JSON.stringify(document.metadata || {}),
      document.embedding ? toSql(document.embedding) : null,
    ];

    const result = await this.pool.query(query, values);
    return result.rows[0];
  }

  async updateDocumentEmbedding(documentId, embedding) {
    const query = `
      UPDATE rag_documents
      SET embedding = $1, indexed_at = NOW(), processing_status = 'completed', updated_at = NOW()
      WHERE id = $2
      RETURNING *
    `;

    const result = await this.pool.query(query, [toSql(embedding), documentId]);
    return result.rows[0];
  }

  async getDocument(documentId) {
    const query = "SELECT * FROM rag_documents WHERE id = $1";
    const result = await this.pool.query(query, [documentId]);
    return result.rows[0];
  }

  async getDocuments(filters = {}) {
    let query = "SELECT * FROM rag_documents WHERE 1=1";
    const values = [];
    let paramCount = 1;

    if (filters.caseId) {
      query += ` AND case_id = $${paramCount}`;
      values.push(filters.caseId);
      paramCount++;
    }

    if (filters.documentType) {
      query += ` AND document_type = $${paramCount}`;
      values.push(filters.documentType);
      paramCount++;
    }

    if (filters.status) {
      query += ` AND processing_status = $${paramCount}`;
      values.push(filters.status);
      paramCount++;
    }

    query += " ORDER BY created_at DESC";

    if (filters.limit) {
      query += ` LIMIT $${paramCount}`;
      values.push(filters.limit);
      paramCount++;
    }

    const result = await this.pool.query(query, values);
    return result.rows;
  }

  // Vector similarity search
  async vectorSearch(queryEmbedding, options = {}) {
    const {
      limit = 10,
      threshold = 0.7,
      caseId,
      documentTypes = [],
      includeContent = true,
    } = options;

    let query = `
      SELECT d.id, d.title, d.document_type, d.case_id, d.metadata, d.created_at,
             1 - (d.embedding <=> $1) as similarity_score
             ${includeContent ? ", d.content" : ""}
      FROM rag_documents d
      WHERE d.embedding IS NOT NULL
        AND 1 - (d.embedding <=> $1) > $2
    `;

    const values = [toSql(queryEmbedding), threshold];
    let paramCount = 3;

    if (caseId) {
      query += ` AND d.case_id = $${paramCount}`;
      values.push(caseId);
      paramCount++;
    }

    if (documentTypes.length > 0) {
      query += ` AND d.document_type = ANY($${paramCount})`;
      values.push(documentTypes);
      paramCount++;
    }

    query += ` ORDER BY d.embedding <=> $1 LIMIT $${paramCount}`;
    values.push(limit);

    const result = await this.pool.query(query, values);
    return result.rows;
  }

  // Chunk operations
  async insertChunk(chunk) {
    const query = `
      INSERT INTO rag_chunks (document_id, chunk_index, content, embedding, metadata)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;

    const values = [
      chunk.documentId,
      chunk.chunkIndex,
      chunk.content,
      chunk.embedding ? toSql(chunk.embedding) : null,
      JSON.stringify(chunk.metadata || {}),
    ];

    const result = await this.pool.query(query, values);
    return result.rows[0];
  }

  async vectorSearchChunks(queryEmbedding, options = {}) {
    const { limit = 20, threshold = 0.7, documentIds = [] } = options;

    let query = `
      SELECT c.id, c.document_id, c.chunk_index, c.content, c.metadata,
             d.title as document_title, d.document_type, d.case_id,
             1 - (c.embedding <=> $1) as similarity_score
      FROM rag_chunks c
      JOIN rag_documents d ON c.document_id = d.id
      WHERE c.embedding IS NOT NULL
        AND 1 - (c.embedding <=> $1) > $2
    `;

    const values = [toSql(queryEmbedding), threshold];
    let paramCount = 3;

    if (documentIds.length > 0) {
      query += ` AND c.document_id = ANY($${paramCount})`;
      values.push(documentIds);
      paramCount++;
    }

    query += ` ORDER BY c.embedding <=> $1 LIMIT $${paramCount}`;
    values.push(limit);

    const result = await this.pool.query(query, values);
    return result.rows;
  }

  // Query logging
  async logQuery(queryData) {
    const query = `
      INSERT INTO rag_queries (query_text, query_embedding, response, confidence_score, processing_time_ms, sources, user_id, case_id)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
      RETURNING *
    `;

    const values = [
      queryData.queryText,
      queryData.queryEmbedding ? toSql(queryData.queryEmbedding) : null,
      queryData.response,
      queryData.confidenceScore,
      queryData.processingTimeMs,
      JSON.stringify(queryData.sources || []),
      queryData.userId,
      queryData.caseId,
    ];

    const result = await this.pool.query(query, values);
    return result.rows[0];
  }

  // Job management
  async createJob(jobType, inputData) {
    const query = `
      INSERT INTO rag_jobs (job_type, input_data)
      VALUES ($1, $2)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      jobType,
      JSON.stringify(inputData),
    ]);
    return result.rows[0];
  }

  async updateJobStatus(jobId, status, outputData = null, errorMessage = null) {
    let query = `
      UPDATE rag_jobs
      SET status = $1, updated_at = NOW()
    `;
    const values = [status, jobId];
    let paramCount = 3;

    if (status === "running") {
      query += `, started_at = NOW()`;
    } else if (status === "completed" || status === "failed") {
      query += `, completed_at = NOW()`;
    }

    if (outputData) {
      query += `, output_data = $${paramCount}`;
      values.push(JSON.stringify(outputData));
      paramCount++;
    }

    if (errorMessage) {
      query += `, error_message = $${paramCount}`;
      values.push(errorMessage);
      paramCount++;
    }

    query += ` WHERE id = $2 RETURNING *`;

    const result = await this.pool.query(query, values);
    return result.rows[0];
  }

  async getJob(jobId) {
    const query = "SELECT * FROM rag_jobs WHERE id = $1";
    const result = await this.pool.query(query, [jobId]);
    return result.rows[0];
  }

  // Health and statistics
  async getHealthStats() {
    const queries = [
      "SELECT COUNT(*) as total_documents FROM rag_documents",
      "SELECT COUNT(*) as indexed_documents FROM rag_documents WHERE embedding IS NOT NULL",
      "SELECT COUNT(*) as total_chunks FROM rag_chunks",
      "SELECT COUNT(*) as total_queries FROM rag_queries WHERE created_at > NOW() - INTERVAL '24 hours'",
      "SELECT AVG(processing_time_ms) as avg_processing_time FROM rag_queries WHERE created_at > NOW() - INTERVAL '24 hours'",
      "SELECT COUNT(*) as pending_jobs FROM rag_jobs WHERE status = 'pending'",
      "SELECT COUNT(*) as running_jobs FROM rag_jobs WHERE status = 'running'",
    ];

    const results = await Promise.all(
      queries.map((query) => this.pool.query(query))
    );

    return {
      totalDocuments: parseInt(results[0].rows[0].total_documents),
      indexedDocuments: parseInt(results[1].rows[0].indexed_documents),
      totalChunks: parseInt(results[2].rows[0].total_chunks),
      queriesLast24h: parseInt(results[3].rows[0].total_queries),
      avgProcessingTime:
        parseFloat(results[4].rows[0].avg_processing_time) || 0,
      pendingJobs: parseInt(results[5].rows[0].pending_jobs),
      runningJobs: parseInt(results[6].rows[0].running_jobs),
    };
  }

  async close() {
    if (this.pool) {
      await this.pool.end();
      console.log("✅ Database connection closed");
    }
  }
}
