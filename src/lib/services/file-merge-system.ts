// File Merging System with MinIO, PostgreSQL, RAG, and Vector Storage Integration
// Native Windows Implementation

import { writable } from 'svelte/store';
import type { Readable } from 'stream';

// Types and Interfaces
interface FileMetadata {
  id: string;
  filename: string;
  originalPath: string;
  size: number;
  mimeType: string;
  checksum: string;
  uploadedAt: Date;
  tags: Record<string, any>;
  caseId?: string;
  userId: string;
  embedding?: number[];
  vectorId?: string;
}

interface MergeOperation {
  id: string;
  sourceFiles: string[];
  targetFilename: string;
  mergeType: 'concatenate' | 'overlay' | 'archive' | 'legal-discovery';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  createdAt: Date;
  completedAt?: Date;
  result?: {
    fileId: string;
    path: string;
    metadata: FileMetadata;
  };
}

interface S3Config {
  endpoint: string;
  accessKey: string;
  secretKey: string;
  bucket: string;
  region?: string;
}

interface PostgresConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
}

interface VectorConfig {
  qdrantUrl: string;
  pgVectorConnection: string;
  collectionName: string;
}

interface RAGConfig {
  embeddingService: string;
  modelName: string;
  apiEndpoint: string;
}

// Store for reactive state management
export const fileMergeStore = writable<{
  operations: MergeOperation[];
  uploading: boolean;
  progress: number;
  error: string | null;
}>({
  operations: [],
  uploading: false,
  progress: 0,
  error: null
});

// Main File Merging Service Class
export class FileMergeSystem {
  private s3Config: S3Config;
  private postgresConfig: PostgresConfig;
  private vectorConfig: VectorConfig;
  private ragConfig: RAGConfig;
  private s3Client: any;
  private pgClient: any;
  private qdrantClient: any;

  constructor(
    s3Config: S3Config,
    postgresConfig: PostgresConfig,
    vectorConfig: VectorConfig,
    ragConfig: RAGConfig
  ) {
    this.s3Config = s3Config;
    this.postgresConfig = postgresConfig;
    this.vectorConfig = vectorConfig;
    this.ragConfig = ragConfig;

    this.initializeClients();
  }

  private async initializeClients() {
    try {
      // Initialize MinIO S3 Client
      const { Client } = await import('minio');
      this.s3Client = new Client({
        endPoint: this.s3Config.endpoint.replace(/^https?:\/\//, ''),
        port: this.s3Config.endpoint.includes('https') ? 443 : 9000,
        useSSL: this.s3Config.endpoint.includes('https'),
        accessKey: this.s3Config.accessKey,
        secretKey: this.s3Config.secretKey
      });

      // Initialize PostgreSQL Client
      const { Client: PgClient } = await import('pg');
      this.pgClient = new PgClient(this.postgresConfig);
      await this.pgClient.connect();

      // Initialize Qdrant Client
      const { QdrantClient } = await import('@qdrant/js-client-rest');
      this.qdrantClient = new QdrantClient({
        url: this.vectorConfig.qdrantUrl
      });

      console.log('✅ All clients initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize clients:', error);
      throw error;
    }
  }

  // 1. File Upload to MinIO with metadata extraction
  async uploadFile(
    file: File,
    metadata: Partial<FileMetadata>
  ): Promise<FileMetadata> {
    try {
      fileMergeStore.update(state => ({ ...state, uploading: true, progress: 0 }));

      // Generate unique filename
      const fileId = this.generateFileId();
      const filename = `${fileId}_${file.name}`;

      // Calculate checksum
      const checksum = await this.calculateChecksum(file);

      // Upload to MinIO
      const uploadStream = file.stream();
      await this.s3Client.putObject(
        this.s3Config.bucket,
        filename,
        uploadStream,
        file.size,
        {
          'Content-Type': file.type,
          'X-File-Checksum': checksum,
          'X-Upload-Time': new Date().toISOString(),
          ...metadata.tags
        }
      );

      // Create file metadata
      const fileMetadata: FileMetadata = {
        id: fileId,
        filename,
        originalPath: file.name,
        size: file.size,
        mimeType: file.type,
        checksum,
        uploadedAt: new Date(),
        tags: metadata.tags || {},
        caseId: metadata.caseId,
        userId: metadata.userId!,
        ...metadata
      };

      // Save metadata to PostgreSQL
      await this.saveFileMetadata(fileMetadata);

      // Generate embeddings and store vectors
      if (this.isTextFile(file.type)) {
        const content = await this.extractTextContent(file);
        const embedding = await this.generateEmbedding(content);

        // Store in pgVector
        await this.storeInPgVector(fileMetadata, embedding, content);

        // Store in Qdrant
        const vectorId = await this.storeInQdrant(fileMetadata, embedding, content);

        fileMetadata.embedding = embedding;
        fileMetadata.vectorId = vectorId;

        // Update metadata with vector information
        await this.updateFileMetadata(fileMetadata);
      }

      fileMergeStore.update(state => ({
        ...state,
        uploading: false,
        progress: 100,
        error: null
      }));

      return fileMetadata;
    } catch (error) {
      fileMergeStore.update(state => ({
        ...state,
        uploading: false,
        error: error.message
      }));
      throw error;
    }
  }

  // 2. File Merging Operations
  async mergeFiles(
    sourceFileIds: string[],
    targetFilename: string,
    mergeType: MergeOperation['mergeType'],
    options: {
      caseId?: string;
      userId: string;
      tags?: Record<string, any>;
    }
  ): Promise<MergeOperation> {
    const operationId = this.generateFileId();

    const operation: MergeOperation = {
      id: operationId,
      sourceFiles: sourceFileIds,
      targetFilename,
      mergeType,
      status: 'pending',
      progress: 0,
      createdAt: new Date()
    };

    fileMergeStore.update(state => ({
      ...state,
      operations: [...state.operations, operation]
    }));

    try {
      // Update status to processing
      operation.status = 'processing';
      this.updateOperationStatus(operation);

      // Get source files metadata
      const sourceFiles = await this.getFilesMetadata(sourceFileIds);

      let mergedContent: Buffer;
      let mergedMetadata: Partial<FileMetadata>;

      switch (mergeType) {
        case 'concatenate':
          ({ content: mergedContent, metadata: mergedMetadata } =
            await this.concatenateFiles(sourceFiles));
          break;

        case 'overlay':
          ({ content: mergedContent, metadata: mergedMetadata } =
            await this.overlayFiles(sourceFiles));
          break;

        case 'archive':
          ({ content: mergedContent, metadata: mergedMetadata } =
            await this.archiveFiles(sourceFiles));
          break;

        case 'legal-discovery':
          ({ content: mergedContent, metadata: mergedMetadata } =
            await this.legalDiscoveryMerge(sourceFiles));
          break;

        default:
          throw new Error(`Unsupported merge type: ${mergeType}`);
      }

      // Upload merged file
      const mergedFileId = this.generateFileId();
      const mergedFilename = `${mergedFileId}_${targetFilename}`;

      await this.s3Client.putObject(
        this.s3Config.bucket,
        mergedFilename,
        mergedContent,
        mergedContent.length,
        {
          'Content-Type': mergedMetadata.mimeType || 'application/octet-stream',
          'X-Merge-Type': mergeType,
          'X-Source-Files': sourceFileIds.join(','),
          'X-Merge-Time': new Date().toISOString()
        }
      );

      // Create merged file metadata
      const finalMetadata: FileMetadata = {
        id: mergedFileId,
        filename: mergedFilename,
        originalPath: targetFilename,
        size: mergedContent.length,
        mimeType: mergedMetadata.mimeType || 'application/octet-stream',
        checksum: await this.calculateChecksum(new Blob([mergedContent.buffer])),
        uploadedAt: new Date(),
        tags: {
          ...options.tags,
          mergeType,
          sourceFiles: sourceFileIds,
          operation: operationId
        },
        caseId: options.caseId,
        userId: options.userId
      };

      // Save merged file metadata
      await this.saveFileMetadata(finalMetadata);

      // Generate embeddings for merged content if text
      if (this.isTextFile(finalMetadata.mimeType)) {
        const textContent = mergedContent.toString('utf-8');
        const embedding = await this.generateEmbedding(textContent);

        await this.storeInPgVector(finalMetadata, embedding, textContent);
        const vectorId = await this.storeInQdrant(finalMetadata, embedding, textContent);

        finalMetadata.embedding = embedding;
        finalMetadata.vectorId = vectorId;
        await this.updateFileMetadata(finalMetadata);
      }

      // Complete operation
      operation.status = 'completed';
      operation.progress = 100;
      operation.completedAt = new Date();
      operation.result = {
        fileId: mergedFileId,
        path: mergedFilename,
        metadata: finalMetadata
      };

      this.updateOperationStatus(operation);

      return operation;
    } catch (error) {
      operation.status = 'failed';
      operation.progress = 0;
      this.updateOperationStatus(operation);
      throw error;
    }
  }

  // 3. Similarity Search across Documents
  async similaritySearch(
    query: string,
    options: {
      limit?: number;
      threshold?: number;
      caseId?: string;
      fileTypes?: string[];
    } = {}
  ): Promise<Array<FileMetadata & { similarity: number }>> {
    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      // Search in Qdrant
      const qdrantResults = await this.qdrantClient.search(this.vectorConfig.collectionName, {
        vector: queryEmbedding,
        limit: options.limit || 10,
        score_threshold: options.threshold || 0.7,
        filter: options.caseId ? {
          must: [{ key: 'case_id', match: { value: options.caseId } }]
        } : undefined
      });

      // Get full metadata for results
      const results = await Promise.all(
        qdrantResults.points.map(async (point: { payload?: { file_id?: string }, score?: number }) => {
          const metadata = await this.getFileMetadata(point.payload!.file_id as string);
          return {
            ...metadata,
            similarity: point.score!
          };
        })
      );

      // Also search in pgVector for comparison
      const pgResults = await this.searchInPgVector(queryEmbedding, options);

      // Merge and deduplicate results
      const mergedResults = this.mergeSearchResults(results, pgResults);

      return mergedResults.slice(0, options.limit || 10);
    } catch (error) {
      console.error('Similarity search failed:', error);
      throw error;
    }
  }

  // 4. File Operations
  async downloadFile(fileId: string): Promise<Readable> {
    const metadata = await this.getFileMetadata(fileId);
    return this.s3Client.getObject(this.s3Config.bucket, metadata.filename);
  }

  async deleteFile(fileId: string): Promise<void> {
    const metadata = await this.getFileMetadata(fileId);

    // Delete from MinIO
    await this.s3Client.removeObject(this.s3Config.bucket, metadata.filename);

    // Delete from vector stores
    if (metadata.vectorId) {
      await this.qdrantClient.delete(this.vectorConfig.collectionName, {
        points: [metadata.vectorId]
      });
    }

    await this.deleteFromPgVector(fileId);

    // Delete metadata
    await this.pgClient.query('DELETE FROM file_metadata WHERE id = $1', [fileId]);
  }

  // 5. Merge Operation Implementations
  private async concatenateFiles(files: FileMetadata[]): Promise<{
    content: Buffer;
    metadata: Partial<FileMetadata>;
  }> {
    const contents: Buffer[] = [];
    let totalSize = 0;

    for (const file of files) {
      const stream = await this.s3Client.getObject(this.s3Config.bucket, file.filename);
      const chunks: Buffer[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      const content = Buffer.concat(chunks);
      contents.push(content);
      totalSize += content.length;
    }

    const mergedContent = Buffer.concat(contents);

    return {
      content: mergedContent,
      metadata: {
        mimeType: files[0].mimeType,
        size: totalSize,
        tags: {
          sourceFiles: files.map(f => f.id),
          mergeType: 'concatenate'
        }
      }
    };
  }

  private async overlayFiles(files: FileMetadata[]): Promise<{
    content: Buffer;
    metadata: Partial<FileMetadata>;
  }> {
    // Implementation for overlay merge (e.g., for images or PDFs)
    // This is a simplified version - real implementation would depend on file types
    return this.concatenateFiles(files);
  }

  private async archiveFiles(files: FileMetadata[]): Promise<{
    content: Buffer;
    metadata: Partial<FileMetadata>;
  }> {
    const archiver = await import('archiver');
    const archive = archiver.default('zip', { zlib: { level: 9 } });

    const chunks: Buffer[] = [];

    archive.on('data', (chunk: Buffer) => chunks.push(chunk));

    for (const file of files) {
      const stream = await this.s3Client.getObject(this.s3Config.bucket, file.filename);
      archive.append(stream, { name: file.originalPath });
    }

    await archive.finalize();

    const content = Buffer.concat(chunks);

    return {
      content,
      metadata: {
        mimeType: 'application/zip',
        size: content.length,
        tags: {
          sourceFiles: files.map(f => f.id),
          mergeType: 'archive'
        }
      }
    };
  }

  private async legalDiscoveryMerge(files: FileMetadata[]): Promise<{
    content: Buffer;
    metadata: Partial<FileMetadata>;
  }> {
    // Legal-specific merge with metadata preservation and chain of custody
    const discoveryPackage = {
      metadata: {
        created: new Date().toISOString(),
        source: 'FileMergeSystem',
        chainOfCustody: files.map(f => ({
          fileId: f.id,
          originalName: f.originalPath,
          checksum: f.checksum,
          uploadedAt: f.uploadedAt,
          size: f.size
        }))
      },
      files: []
    };

    // Create structured legal discovery package
    const packageContent = JSON.stringify(discoveryPackage, null, 2);
    const content = Buffer.from(packageContent, 'utf-8');

    return {
      content,
      metadata: {
        mimeType: 'application/json',
        size: content.length,
        tags: {
          sourceFiles: files.map(f => f.id),
          mergeType: 'legal-discovery',
          chainOfCustody: true
        }
      }
    };
  }

  // 6. Utility Methods
  private generateFileId(): string {
    return `file_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  private async calculateChecksum(file: File | Blob): Promise<string> {
    const buffer = await file.arrayBuffer();
    const crypto = await import('crypto');
    return crypto.createHash('sha256').update(new Uint8Array(buffer)).digest('hex');
  }

  private isTextFile(mimeType: string): boolean {
    return mimeType.startsWith('text/') ||
           mimeType === 'application/json' ||
           mimeType === 'application/pdf' ||
           mimeType.includes('document');
  }

  private async extractTextContent(file: File): Promise<string> {
    if (file.type.startsWith('text/')) {
      return await file.text();
    } else if (file.type === 'application/pdf') {
      // Use pdf-parse or similar library
      const pdfParse = await import('pdf-parse');
      const buffer = await file.arrayBuffer();
      const result = await pdfParse.default(new Uint8Array(buffer));
      return result.text;
    }
    return '';
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    // Call your RAG service for embeddings
    const response = await fetch(`${this.ragConfig.apiEndpoint}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.ragConfig.modelName,
        prompt: text.substring(0, 8192) // Truncate for token limits
      })
    });

    const result = await response.json();
    return result.embedding;
  }

  private async saveFileMetadata(metadata: FileMetadata): Promise<void> {
    await this.pgClient.query(`
      INSERT INTO file_metadata (
        id, filename, original_path, size, mime_type, checksum,
        uploaded_at, tags, case_id, user_id, embedding, vector_id
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    `, [
      metadata.id, metadata.filename, metadata.originalPath, metadata.size,
      metadata.mimeType, metadata.checksum, metadata.uploadedAt,
      JSON.stringify(metadata.tags), metadata.caseId, metadata.userId,
      metadata.embedding ? JSON.stringify(metadata.embedding) : null,
      metadata.vectorId
    ]);
  }

  private async getFileMetadata(fileId: string): Promise<FileMetadata> {
    const result = await this.pgClient.query(
      'SELECT * FROM file_metadata WHERE id = $1',
      [fileId]
    );

    if (result.rows.length === 0) {
      throw new Error(`File not found: ${fileId}`);
    }

    const row = result.rows[0];
    return {
      id: row.id,
      filename: row.filename,
      originalPath: row.original_path,
      size: row.size,
      mimeType: row.mime_type,
      checksum: row.checksum,
      uploadedAt: row.uploaded_at,
      tags: row.tags,
      caseId: row.case_id,
      userId: row.user_id,
      embedding: row.embedding ? JSON.parse(row.embedding) : undefined,
      vectorId: row.vector_id
    };
  }

  private async getFilesMetadata(fileIds: string[]): Promise<FileMetadata[]> {
    const placeholders = fileIds.map((_, i) => `$${i + 1}`).join(',');
    const result = await this.pgClient.query(
      `SELECT * FROM file_metadata WHERE id IN (${placeholders})`,
      fileIds
    );

    return result.rows.map((row: any) => ({
      id: row.id,
      filename: row.filename,
      originalPath: row.original_path,
      size: row.size,
      mimeType: row.mime_type,
      checksum: row.checksum,
      uploadedAt: row.uploaded_at,
      tags: row.tags,
      caseId: row.case_id,
      userId: row.user_id,
      embedding: row.embedding ? JSON.parse(row.embedding) : undefined,
      vectorId: row.vector_id
    }));
  }

  private async updateFileMetadata(metadata: FileMetadata): Promise<void> {
    await this.pgClient.query(`
      UPDATE file_metadata SET
        embedding = $1, vector_id = $2, tags = $3
      WHERE id = $4
    `, [
      metadata.embedding ? JSON.stringify(metadata.embedding) : null,
      metadata.vectorId,
      JSON.stringify(metadata.tags),
      metadata.id
    ]);
  }

  private async storeInPgVector(
    metadata: FileMetadata,
    embedding: number[],
    content: string
  ): Promise<void> {
    await this.pgClient.query(`
      INSERT INTO document_embeddings (file_id, content, embedding, metadata)
      VALUES ($1, $2, $3, $4)
      ON CONFLICT (file_id) DO UPDATE SET
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata
    `, [
      metadata.id,
      content,
      JSON.stringify(embedding),
      JSON.stringify({
        filename: metadata.filename,
        mimeType: metadata.mimeType,
        caseId: metadata.caseId,
        tags: metadata.tags
      })
    ]);
  }

  private async storeInQdrant(
    metadata: FileMetadata,
    embedding: number[],
    content: string
  ): Promise<string> {
    const vectorId = `${metadata.id}_${Date.now()}`;

    await this.qdrantClient.upsert(this.vectorConfig.collectionName, {
      points: [{
        id: vectorId,
        vector: embedding,
        payload: {
          file_id: metadata.id,
          filename: metadata.filename,
          content: content.substring(0, 1000), // Store excerpt
          case_id: metadata.caseId,
          mime_type: metadata.mimeType,
          uploaded_at: metadata.uploadedAt.toISOString(),
          tags: metadata.tags
        }
      }]
    });

    return vectorId;
  }

  private async searchInPgVector(
    queryEmbedding: number[],
    options: any
  ): Promise<Array<FileMetadata & { similarity: number }>> {
    const result = await this.pgClient.query(`
      SELECT
        fm.*,
        de.content,
        1 - (de.embedding <-> $1::vector) as similarity
      FROM file_metadata fm
      JOIN document_embeddings de ON fm.id = de.file_id
      WHERE 1 - (de.embedding <-> $1::vector) > $2
      ${options.caseId ? 'AND fm.case_id = $3' : ''}
      ORDER BY similarity DESC
      LIMIT $4
    `, [
      JSON.stringify(queryEmbedding),
      options.threshold || 0.7,
      ...(options.caseId ? [options.caseId] : []),
      options.limit || 10
    ]);

    return result.rows.map((row: any) => ({
      id: row.id,
      filename: row.filename,
      originalPath: row.original_path,
      size: row.size,
      mimeType: row.mime_type,
      checksum: row.checksum,
      uploadedAt: row.uploaded_at,
      tags: row.tags,
      caseId: row.case_id,
      userId: row.user_id,
      similarity: row.similarity
    }));
  }

  private async deleteFromPgVector(fileId: string): Promise<void> {
    await this.pgClient.query('DELETE FROM document_embeddings WHERE file_id = $1', [fileId]);
  }

  private mergeSearchResults(
    qdrantResults: Array<FileMetadata & { similarity: number }>,
    pgResults: Array<FileMetadata & { similarity: number }>
  ): Array<FileMetadata & { similarity: number }> {
    const merged = new Map<string, FileMetadata & { similarity: number }>();

    // Add Qdrant results
    qdrantResults.forEach(result => {
      merged.set(result.id, result);
    });

    // Add pgVector results, taking higher similarity if duplicate
    pgResults.forEach(result => {
      const existing = merged.get(result.id);
      if (!existing || result.similarity > existing.similarity) {
        merged.set(result.id, result);
      }
    });

    return Array.from(merged.values()).sort((a, b) => b.similarity - a.similarity);
  }

  private updateOperationStatus(operation: MergeOperation): void {
    fileMergeStore.update(state => ({
      ...state,
      operations: state.operations.map(op =>
        op.id === operation.id ? operation : op
      )
    }));
  }
}

// Factory function to create configured instance
export function createFileMergeSystem(): FileMergeSystem {
  const s3Config: S3Config = {
    endpoint: process.env.MINIO_ENDPOINT || 'http://localhost:9000',
    accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
    secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin',
    bucket: process.env.MINIO_BUCKET || 'legal-documents'
  };

  const postgresConfig: PostgresConfig = {
    host: process.env.POSTGRES_HOST || 'localhost',
    port: parseInt(process.env.POSTGRES_PORT || '5432'),
    database: process.env.POSTGRES_DB || 'legal_ai',
    username: process.env.POSTGRES_USER || 'postgres',
    password: process.env.POSTGRES_PASSWORD || 'password'
  };

  const vectorConfig: VectorConfig = {
    qdrantUrl: process.env.QDRANT_URL || 'http://localhost:6333',
    pgVectorConnection: `postgresql://${postgresConfig.username}:${postgresConfig.password}@${postgresConfig.host}:${postgresConfig.port}/${postgresConfig.database}`,
    collectionName: 'legal_documents'
  };

  const ragConfig: RAGConfig = {
    embeddingService: 'ollama',
    modelName: 'nomic-embed-text',
    apiEndpoint: process.env.OLLAMA_URL || 'http://localhost:11434'
  };

  return new FileMergeSystem(s3Config, postgresConfig, vectorConfig, ragConfig);
}