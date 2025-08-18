// document-ingestion-pipeline.js
// Complete Document Ingestion Pipeline with RAG, MinIO, Neo4j, PGVector

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Client } = require('pg');
const neo4j = require('neo4j-driver');
const Minio = require('minio');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { OllamaEmbeddings } = require('langchain/embeddings/ollama');
const fetch = require('node-fetch');
const protobuf = require('protobufjs');
const QuicTransport = require('./quic-transport');

// Configuration
const CONFIG = {
    minio: {
        endPoint: 'localhost',
        port: 9000,
        useSSL: false,
        accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
        secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin',
        bucketName: 'legal-documents'
    },
    postgres: {
        host: 'localhost',
        port: 5432,
        database: 'legal_ai_rag',
        user: 'postgres',
        password: 'postgres'
    },
    neo4j: {
        uri: 'bolt://localhost:7687',
        user: 'neo4j',
        password: 'password'
    },
    ollama: {
        baseUrl: 'http://localhost:11434',
        model: 'gemma3-legal:latest',
        embeddingModel: 'nomic-embed-text'
    },
    legalBert: {
        endpoint: 'http://localhost:8089/api/legal-bert'
    },
    rag: {
        endpoint: 'http://localhost:8097',
        chunkSize: 1000,
        chunkOverlap: 200
    }
};

// Initialize clients
class DocumentIngestionPipeline {
    constructor() {
        this.minioClient = null;
        this.pgClient = null;
        this.neo4jDriver = null;
        this.textSplitter = null;
        this.embeddings = null;
        this.quicTransport = null;
        this.eventLoop = new Map(); // Event loop for conditional processing
    }

    async initialize() {
        console.log('🚀 Initializing Document Ingestion Pipeline...');
        
        // Initialize MinIO
        this.minioClient = new Minio.Client(CONFIG.minio);
        await this.ensureBucket();
        
        // Initialize PostgreSQL with pgvector
        this.pgClient = new Client(CONFIG.postgres);
        await this.pgClient.connect();
        await this.ensurePgVectorTables();
        
        // Initialize Neo4j
        this.neo4jDriver = neo4j.driver(
            CONFIG.neo4j.uri,
            neo4j.auth.basic(CONFIG.neo4j.user, CONFIG.neo4j.password)
        );
        await this.ensureNeo4jSchema();
        
        // Initialize LangChain components
        this.textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: CONFIG.rag.chunkSize,
            chunkOverlap: CONFIG.rag.chunkOverlap,
            separators: ['\n\n', '\n', '. ', ' ', '']
        });
        
        this.embeddings = new OllamaEmbeddings({
            baseUrl: CONFIG.ollama.baseUrl,
            model: CONFIG.ollama.embeddingModel
        });
        
        // Initialize QUIC transport for optimized communication
        this.quicTransport = new QuicTransport({
            port: 8443,
            cert: 'certs/server.crt',
            key: 'certs/server.key'
        });
        
        console.log('✅ Pipeline initialized successfully');
    }

    async ensureBucket() {
        const bucketExists = await this.minioClient.bucketExists(CONFIG.minio.bucketName);
        if (!bucketExists) {
            await this.minioClient.makeBucket(CONFIG.minio.bucketName, 'us-east-1');
            console.log(`✅ Created MinIO bucket: ${CONFIG.minio.bucketName}`);
        }
    }

    async ensurePgVectorTables() {
        // Create tables with pgvector extension
        const createTableQuery = `
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id VARCHAR(255) NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename VARCHAR(255) NOT NULL,
                minio_object_name VARCHAR(255) UNIQUE NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                file_size BIGINT NOT NULL,
                mime_type VARCHAR(100),
                status VARCHAR(50) DEFAULT 'pending',
                neo4j_node_id VARCHAR(255),
                metadata JSONB DEFAULT '{}',
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
        `;
        
        await this.pgClient.query(createTableQuery);
        console.log('✅ PostgreSQL tables with pgvector ready');
    }

    async ensureNeo4jSchema() {
        const session = this.neo4jDriver.session();
        try {
            // Create constraints and indexes
            await session.run(`
                CREATE CONSTRAINT document_id IF NOT EXISTS 
                FOR (d:Document) REQUIRE d.id IS UNIQUE
            `);
            
            await session.run(`
                CREATE INDEX document_filename IF NOT EXISTS 
                FOR (d:Document) ON (d.filename)
            `);
            
            console.log('✅ Neo4j schema ready');
        } finally {
            await session.close();
        }
    }

    // Main document ingestion function
    async ingestDocument(filePath, userId, metadata = {}) {
        console.log(`\n📄 Ingesting document: ${filePath}`);
        
        const startTime = Date.now();
        const fileBuffer = fs.readFileSync(filePath);
        const fileName = path.basename(filePath);
        const fileHash = crypto.createHash('sha256').update(fileBuffer).digest('hex');
        
        try {
            // Step 1: Check for duplicate
            const duplicate = await this.checkDuplicate(fileHash);
            if (duplicate) {
                console.log('⚠️  Duplicate document detected, skipping upload');
                return duplicate;
            }
            
            // Step 2: Upload to MinIO
            const minioObjectName = await this.uploadToMinIO(fileBuffer, fileName, fileHash);
            
            // Step 3: Create document record in PostgreSQL
            const documentId = await this.createDocumentRecord({
                filename: fileName,
                minioObjectName,
                fileHash,
                fileSize: fileBuffer.length,
                mimeType: this.getMimeType(fileName),
                metadata
            });
            
            // Step 4: Extract text content
            const textContent = await this.extractText(fileBuffer, fileName);
            
            // Step 5: Split into chunks
            const chunks = await this.textSplitter.splitText(textContent);
            console.log(`📊 Split into ${chunks.length} chunks`);
            
            // Step 6: Generate embeddings using Ollama
            const embeddings = await this.generateEmbeddings(chunks);
            
            // Step 7: Store chunks with embeddings in PostgreSQL
            await this.storeChunksWithEmbeddings(documentId, chunks, embeddings);
            
            // Step 8: Create Neo4j graph node and relationships
            const neo4jNodeId = await this.createNeo4jNode({
                documentId,
                fileName,
                userId,
                metadata,
                chunkCount: chunks.length
            });
            
            // Step 9: Update document status
            await this.updateDocumentStatus(documentId, 'processed', neo4jNodeId);
            
            // Step 10: Process with Legal-BERT for initial analysis
            const legalAnalysis = await this.analyzeLegalDocument(textContent);
            
            // Step 11: Trigger RAG enhancement
            await this.triggerRAGEnhancement(documentId, chunks, embeddings, legalAnalysis);
            
            // Step 12: Send to event loop for conditional processing
            this.scheduleConditionalProcessing(documentId, metadata);
            
            const processingTime = Date.now() - startTime;
            console.log(`✅ Document ingested successfully in ${processingTime}ms`);
            
            return {
                success: true,
                documentId,
                neo4jNodeId,
                chunksCount: chunks.length,
                processingTime,
                legalAnalysis
            };
            
        } catch (error) {
            console.error('❌ Ingestion failed:', error);
            await this.handleIngestionError(documentId, error);
            throw error;
        }
    }

    async checkDuplicate(fileHash) {
        const result = await this.pgClient.query(
            'SELECT id FROM documents WHERE file_hash = $1',
            [fileHash]
        );
        return result.rows.length > 0 ? result.rows[0] : null;
    }

    async uploadToMinIO(fileBuffer, fileName, fileHash) {
        const objectName = `${fileHash}-${Date.now()}-${fileName}`;
        
        await this.minioClient.putObject(
            CONFIG.minio.bucketName,
            objectName,
            fileBuffer,
            fileBuffer.length
        );
        
        console.log(`☁️  Uploaded to MinIO: ${objectName}`);
        return objectName;
    }

    async createDocumentRecord(data) {
        const query = `
            INSERT INTO documents (
                filename, minio_object_name, file_hash, 
                file_size, mime_type, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        `;
        
        const result = await this.pgClient.query(query, [
            data.filename,
            data.minioObjectName,
            data.fileHash,
            data.fileSize,
            data.mimeType,
            data.metadata
        ]);
        
        return result.rows[0].id;
    }

    async extractText(fileBuffer, fileName) {
        // Use appropriate extractor based on file type
        const ext = path.extname(fileName).toLowerCase();
        
        if (ext === '.pdf') {
            const pdfParse = require('pdf-parse');
            const data = await pdfParse(fileBuffer);
            return data.text;
        } else if (['.txt', '.md'].includes(ext)) {
            return fileBuffer.toString('utf8');
        } else if (['.doc', '.docx'].includes(ext)) {
            // Use mammoth for Word documents
            const mammoth = require('mammoth');
            const result = await mammoth.extractRawText({ buffer: fileBuffer });
            return result.value;
        } else {
            throw new Error(`Unsupported file type: ${ext}`);
        }
    }

    async generateEmbeddings(chunks) {
        console.log('🧮 Generating embeddings with Ollama...');
        const embeddings = [];
        
        // Batch process for efficiency
        const batchSize = 10;
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            const batchEmbeddings = await Promise.all(
                batch.map(chunk => this.embeddings.embedQuery(chunk))
            );
            embeddings.push(...batchEmbeddings);
            
            // Progress indicator
            console.log(`  Processed ${Math.min(i + batchSize, chunks.length)}/${chunks.length} chunks`);
        }
        
        return embeddings;
    }

    async storeChunksWithEmbeddings(documentId, chunks, embeddings) {
        const query = `
            INSERT INTO document_chunks (
                document_id, chunk_index, content, embedding, metadata
            ) VALUES ($1, $2, $3, $4, $5)
        `;
        
        for (let i = 0; i < chunks.length; i++) {
            await this.pgClient.query(query, [
                documentId,
                i,
                chunks[i],
                `[${embeddings[i].join(',')}]`, // Format embedding for pgvector
                JSON.stringify({ 
                    length: chunks[i].length,
                    position: i,
                    total: chunks.length 
                })
            ]);
        }
        
        console.log(`💾 Stored ${chunks.length} chunks with embeddings`);
    }

    async createNeo4jNode(data) {
        const session = this.neo4jDriver.session();
        try {
            const result = await session.run(
                `
                CREATE (d:Document {
                    id: $documentId,
                    filename: $fileName,
                    userId: $userId,
                    chunkCount: $chunkCount,
                    createdAt: datetime(),
                    metadata: $metadata
                })
                
                WITH d
                MATCH (u:User {id: $userId})
                MERGE (u)-[:UPLOADED]->(d)
                
                RETURN d.id as nodeId
                `,
                {
                    documentId: data.documentId,
                    fileName: data.fileName,
                    userId: data.userId,
                    chunkCount: data.chunkCount,
                    metadata: JSON.stringify(data.metadata)
                }
            );
            
            const nodeId = result.records[0].get('nodeId');
            console.log(`🔗 Created Neo4j node: ${nodeId}`);
            return nodeId;
            
        } finally {
            await session.close();
        }
    }

    async updateDocumentStatus(documentId, status, neo4jNodeId = null) {
        const query = neo4jNodeId
            ? 'UPDATE documents SET status = $1, neo4j_node_id = $2, processed_at = NOW() WHERE id = $3'
            : 'UPDATE documents SET status = $1, processed_at = NOW() WHERE id = $2';
        
        const params = neo4jNodeId
            ? [status, neo4jNodeId, documentId]
            : [status, documentId];
        
        await this.pgClient.query(query, params);
    }

    async analyzeLegalDocument(textContent) {
        console.log('⚖️  Analyzing with Legal-BERT...');
        
        try {
            // Call Legal-BERT service
            const response = await fetch(CONFIG.legalBert.endpoint + '/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: textContent.substring(0, 5000), // Limit for initial analysis
                    analysisType: 'comprehensive'
                })
            });
            
            if (response.ok) {
                const analysis = await response.json();
                console.log('✅ Legal analysis complete');
                return analysis;
            }
        } catch (error) {
            console.error('Legal-BERT analysis failed:', error);
        }
        
        // Fallback to Ollama
        return this.analyzeWithOllama(textContent);
    }

    async analyzeWithOllama(textContent) {
        const prompt = `
            Analyze this legal document and provide:
            1. Document type (contract, agreement, brief, etc.)
            2. Key parties involved
            3. Main legal issues or clauses
            4. Potential risks or concerns
            5. Compliance requirements
            
            Document excerpt:
            ${textContent.substring(0, 3000)}
        `;
        
        const response = await fetch(`${CONFIG.ollama.baseUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.model,
                prompt,
                stream: false
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            return {
                source: 'ollama',
                analysis: data.response
            };
        }
        
        return null;
    }

    async triggerRAGEnhancement(documentId, chunks, embeddings, legalAnalysis) {
        console.log('🔄 Triggering RAG enhancement...');
        
        // Send to Enhanced RAG V2 for processing
        const response = await fetch(`${CONFIG.rag.endpoint}/api/documents/enhance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                documentId,
                chunks: chunks.slice(0, 10), // Send sample chunks
                legalAnalysis,
                requestEnhancements: [
                    'citation_extraction',
                    'entity_recognition',
                    'clause_classification',
                    'risk_assessment'
                ]
            })
        });
        
        if (response.ok) {
            console.log('✅ RAG enhancement triggered');
        }
    }

    scheduleConditionalProcessing(documentId, metadata) {
        // Add to event loop for conditional processing
        this.eventLoop.set(documentId, {
            status: 'pending',
            metadata,
            scheduledAt: Date.now(),
            processor: async () => {
                // Service worker optimized processing
                if (metadata.priority === 'high') {
                    await this.processPriorityDocument(documentId);
                } else {
                    await this.processStandardDocument(documentId);
                }
            }
        });
        
        // Start processing if not already running
        if (!this.processingActive) {
            this.startEventLoop();
        }
    }

    async startEventLoop() {
        this.processingActive = true;
        
        while (this.eventLoop.size > 0) {
            for (const [documentId, task] of this.eventLoop) {
                if (task.status === 'pending') {
                    task.status = 'processing';
                    
                    try {
                        await task.processor();
                        task.status = 'completed';
                        this.eventLoop.delete(documentId);
                    } catch (error) {
                        console.error(`Processing failed for ${documentId}:`, error);
                        task.status = 'failed';
                        task.retries = (task.retries || 0) + 1;
                        
                        if (task.retries >= 3) {
                            this.eventLoop.delete(documentId);
                        }
                    }
                }
            }
            
            // Wait before next iteration
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        this.processingActive = false;
    }

    async processPriorityDocument(documentId) {
        // Use QUIC for faster processing
        if (this.quicTransport && this.quicTransport.isReady()) {
            const stream = await this.quicTransport.createStream();
            
            // Send document for priority processing
            const message = {
                type: 'PRIORITY_PROCESS',
                documentId,
                timestamp: Date.now()
            };
            
            await stream.write(Buffer.from(JSON.stringify(message)));
            
            // Wait for response
            const response = await stream.read();
            console.log('Priority processing complete:', response);
        }
    }

    async processStandardDocument(documentId) {
        // Standard processing through HTTP/WebSocket
        console.log(`Standard processing for document: ${documentId}`);
    }

    async searchSimilarDocuments(query, limit = 10) {
        console.log('🔍 Searching similar documents...');
        
        // Generate embedding for query
        const queryEmbedding = await this.embeddings.embedQuery(query);
        
        // Search using pgvector
        const searchQuery = `
            SELECT 
                dc.document_id,
                dc.content,
                dc.chunk_index,
                d.filename,
                d.metadata,
                dc.embedding <=> $1::vector as distance
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE d.status = 'processed'
            ORDER BY dc.embedding <=> $1::vector
            LIMIT $2
        `;
        
        const result = await this.pgClient.query(searchQuery, [
            `[${queryEmbedding.join(',')}]`,
            limit
        ]);
        
        return result.rows;
    }

    async generateChatResponse(query, context = []) {
        console.log('💬 Generating chat response...');
        
        // Search for relevant documents
        const relevantDocs = await this.searchSimilarDocuments(query, 5);
        
        // Build context from relevant documents
        const contextText = relevantDocs
            .map(doc => doc.content)
            .join('\n\n---\n\n');
        
        // Generate response using Ollama with Gemma3-legal
        const prompt = `
            You are a legal AI assistant. Use the following context to answer the question.
            
            Context:
            ${contextText}
            
            Question: ${query}
            
            Provide a detailed, legally accurate response:
        `;
        
        const response = await fetch(`${CONFIG.ollama.baseUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.model,
                prompt,
                stream: false,
                options: {
                    temperature: 0.3, // Lower temperature for legal accuracy
                    top_p: 0.9
                }
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Store interaction in Neo4j
            await this.storeInteraction(query, data.response, relevantDocs);
            
            return {
                response: data.response,
                sources: relevantDocs.map(doc => ({
                    filename: doc.filename,
                    chunkIndex: doc.chunk_index,
                    relevance: 1 - doc.distance
                }))
            };
        }
        
        throw new Error('Failed to generate response');
    }

    async storeInteraction(query, response, sources) {
        const session = this.neo4jDriver.session();
        try {
            await session.run(
                `
                CREATE (i:Interaction {
                    id: randomUUID(),
                    query: $query,
                    response: $response,
                    timestamp: datetime(),
                    sourceCount: $sourceCount
                })
                `,
                {
                    query,
                    response,
                    sourceCount: sources.length
                }
            );
        } finally {
            await session.close();
        }
    }

    getMimeType(fileName) {
        const ext = path.extname(fileName).toLowerCase();
        const mimeTypes = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.md': 'text/markdown'
        };
        return mimeTypes[ext] || 'application/octet-stream';
    }

    async handleIngestionError(documentId, error) {
        if (documentId) {
            await this.updateDocumentStatus(documentId, 'failed');
        }
        
        // Log error for monitoring
        console.error('Ingestion error:', {
            documentId,
            error: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString()
        });
    }

    async cleanup() {
        console.log('🧹 Cleaning up resources...');
        
        if (this.pgClient) {
            await this.pgClient.end();
        }
        
        if (this.neo4jDriver) {
            await this.neo4jDriver.close();
        }
        
        if (this.quicTransport) {
            await this.quicTransport.close();
        }
        
        console.log('✅ Cleanup complete');
    }
}

// Export for use in other modules
module.exports = DocumentIngestionPipeline;

// Run if executed directly
if (require.main === module) {
    const pipeline = new DocumentIngestionPipeline();
    
    pipeline.initialize().then(async () => {
        // Example usage
        const testFile = process.argv[2];
        
        if (testFile && fs.existsSync(testFile)) {
            const result = await pipeline.ingestDocument(testFile, 'user-123', {
                caseId: 'case-456',
                priority: 'high',
                category: 'contract'
            });
            
            console.log('Ingestion result:', result);
            
            // Test search
            const searchResults = await pipeline.searchSimilarDocuments('liability clause');
            console.log('Search results:', searchResults);
            
            // Test chat
            const chatResponse = await pipeline.generateChatResponse(
                'What are the liability limitations in this contract?'
            );
            console.log('Chat response:', chatResponse);
        } else {
            console.log('Usage: node document-ingestion-pipeline.js <path-to-document>');
        }
        
        await pipeline.cleanup();
    }).catch(error => {
        console.error('Pipeline initialization failed:', error);
        process.exit(1);
    });
}
