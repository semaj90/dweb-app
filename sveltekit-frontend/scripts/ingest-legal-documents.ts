#!/usr/bin/env tsx
/**
 * Legal Document Ingestion Pipeline - Command Line Interface
 * 
 * Advanced LangChain-powered batch processing system for legal documents:
 * 1. Loads documents from directories (PDF, TXT, DOCX, MD)
 * 2. Splits into semantic chunks optimized for legal content
 * 3. Generates embeddings using your Ollama nomic-embed-text model
 * 4. Performs AI analysis using gemma3-legal model  
 * 5. Stores results in PostgreSQL with pgvector embeddings
 * 
 * Integrates with your existing "tricubic tensor" vector operations system
 */

import { Command } from 'commander';
import { readFileSync, existsSync, mkdirSync, readdirSync } from 'fs';
import { join, basename, extname } from 'path';
import { createHash } from 'crypto';
import { Pool } from 'pg';

// LangChain imports for document processing
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';
import type { Document } from '@langchain/core/documents';

// Configuration
const OLLAMA_BASE_URL = 'http://localhost:11434/v1';
const LEGAL_MODEL = 'gemma3-legal';
const EMBEDDING_MODEL = 'nomic-embed-text';

// Database configuration
const DB_CONFIG = {
  host: 'localhost',
  port: 5432,
  database: 'legal_ai_db',
  user: 'legal_admin',
  password: process.env.POSTGRES_PASSWORD || '123456'
};

// Processing statistics
export interface ProcessingStats {
  documentsProcessed: number;
  chunksCreated: number;
  embeddingsGenerated: number;
  errors: number;
  startTime: Date;
  processingTime: number;
  avgChunksPerDoc: number;
  totalTokens: number;
}

// Enhanced document metadata
export interface EnhancedDocumentMetadata {
  source: string;
  fileName: string;
  fileType: string;
  fileSize: number;
  documentHash: string;
  processedAt: string;
  aiSummary?: string;
  legalEntities?: string;
  keyTerms?: string[];
  riskLevel?: string;
  jurisdiction?: string;
  documentType?: string;
  confidence?: number;
  chunkIndex: number;
  totalChunks: number;
}

class LegalDocumentIngestion {
  private llm: ChatOpenAI;
  private embeddings: OpenAIEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private dbPool: Pool;
  private stats: ProcessingStats;

  constructor() {
    // Initialize AI models
    this.llm = new ChatOpenAI({
      modelName: LEGAL_MODEL,
      temperature: 0.1, // Low temperature for consistent legal analysis
      openAIApiKey: 'not-needed',
      configuration: {
        baseURL: OLLAMA_BASE_URL,
      },
      maxTokens: 500
    });

    this.embeddings = new OpenAIEmbeddings({
      modelName: EMBEDDING_MODEL,
      openAIApiKey: 'not-needed',
      configuration: {
        baseURL: OLLAMA_BASE_URL,
      }
    });

    // Legal-optimized text splitter
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1200,           // Optimal for legal context
      chunkOverlap: 200,         // Maintain legal context continuity
      separators: [
        '\n\n',                 // Paragraph breaks (highest priority)
        '\n',                   // Line breaks
        '. ',                   // Sentence endings
        '? ',                   // Question endings
        '! ',                   // Exclamation endings
        '; ',                   // Semicolons (common in legal text)
        ', ',                   // Commas
        ' ',                    // Spaces (lowest priority)
      ]
    });

    // Database connection
    this.dbPool = new Pool(DB_CONFIG);

    // Initialize stats
    this.stats = {
      documentsProcessed: 0,
      chunksCreated: 0,
      embeddingsGenerated: 0,
      errors: 0,
      startTime: new Date(),
      processingTime: 0,
      avgChunksPerDoc: 0,
      totalTokens: 0
    };
  }

  /**
   * Main ingestion pipeline
   */
  async ingestDocuments(
    inputDir: string,
    options: {
      recursive?: boolean;
      includeAI?: boolean;
      batchSize?: number;
      skipExisting?: boolean;
      dryRun?: boolean;
    } = {}
  ): Promise<ProcessingStats> {
    const {
      recursive = true,
      includeAI = true,
      batchSize = 10,
      skipExisting = true,
      dryRun = false
    } = options;

    console.log('🚀 Legal Document Ingestion Pipeline Starting...');
    console.log(`📁 Input Directory: ${inputDir}`);
    console.log(`🔧 Options: ${JSON.stringify(options, null, 2)}`);

    this.stats.startTime = new Date();

    try {
      // 1. Load documents from directory
      console.log('\n📄 Step 1: Loading documents...');
      const documents = await this.loadDocuments(inputDir);
      console.log(`✅ Loaded ${documents.length} documents`);

      if (documents.length === 0) {
        console.log('❌ No documents found. Check the input directory and file types.');
        return this.stats;
      }

      // 2. Process documents in batches
      console.log('\n✂️ Step 2: Processing documents...');
      const allChunks: Document[] = [];
      
      for (let i = 0; i < documents.length; i += batchSize) {
        const batch = documents.slice(i, i + batchSize);
        console.log(`📦 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(documents.length / batchSize)} (${batch.length} docs)`);
        
        const batchChunks = await this.processBatch(batch, includeAI, skipExisting);
        allChunks.push(...batchChunks);
        
        // Show progress
        this.stats.documentsProcessed = i + batch.length;
        console.log(`📊 Progress: ${this.stats.documentsProcessed}/${documents.length} docs, ${allChunks.length} chunks`);
      }

      this.stats.chunksCreated = allChunks.length;
      this.stats.avgChunksPerDoc = Math.round(allChunks.length / documents.length);

      if (dryRun) {
        console.log('\n🔍 DRY RUN - Would process:', {
          documents: documents.length,
          chunks: allChunks.length,
          avgChunksPerDoc: this.stats.avgChunksPerDoc
        });
        return this.stats;
      }

      // 3. Generate embeddings and store in database
      console.log('\n🧠 Step 3: Generating embeddings and storing...');
      await this.storeChunks(allChunks);

      // 4. Final statistics
      this.stats.processingTime = Date.now() - this.stats.startTime.getTime();
      console.log('\n✅ Ingestion Complete!');
      this.printStats();

      return this.stats;

    } catch (error: any) {
      console.error('❌ Ingestion failed:', error);
      this.stats.errors++;
      throw error;
    } finally {
      await this.cleanup();
    }
  }

  /**
   * Load documents from directory using LangChain loaders
   */
  private async loadDocuments(inputDir: string): Promise<Document[]> {
    if (!existsSync(inputDir)) {
      throw new Error(`Input directory does not exist: ${inputDir}`);
    }

    const loader = new DirectoryLoader(inputDir, {
      '.pdf': (path: string) => new PDFLoader(path),
      '.txt': (path: string) => new TextLoader(path),
      '.md': (path: string) => new TextLoader(path),
      '.docx': (path: string) => new TextLoader(path), // Would need DocxLoader for real DOCX
    });

    const documents = await loader.load();
    
    // Enhance metadata with file information
    for (const doc of documents) {
      const filePath = doc.metadata.source;
      const fileName = basename(filePath);
      const fileContent = doc.pageContent;
      
      doc.metadata = {
        ...doc.metadata,
        fileName,
        fileType: extname(fileName).slice(1),
        fileSize: Buffer.byteLength(fileContent, 'utf8'),
        documentHash: createHash('sha256').update(fileContent).digest('hex'),
        processedAt: new Date().toISOString()
      };
    }

    return documents;
  }

  /**
   * Process a batch of documents
   */
  private async processBatch(
    documents: Document[], 
    includeAI: boolean, 
    skipExisting: boolean
  ): Promise<Document[]> {
    const allChunks: Document[] = [];

    for (const document of documents) {
      try {
        // Check if document already processed
        if (skipExisting) {
          const exists = await this.checkDocumentExists(document.metadata.documentHash);
          if (exists) {
            console.log(`⏩ Skipping ${document.metadata.fileName} (already processed)`);
            continue;
          }
        }

        console.log(`📝 Processing: ${document.metadata.fileName}`);

        // Split document into chunks
        const chunks = await this.textSplitter.splitDocuments([document]);
        
        // Enhance each chunk with AI analysis
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          
          // Add chunk-specific metadata
          chunk.metadata = {
            ...chunk.metadata,
            chunkIndex: i,
            totalChunks: chunks.length
          } as EnhancedDocumentMetadata;

          if (includeAI) {
            await this.enhanceChunkWithAI(chunk);
          }

          allChunks.push(chunk);
        }

        console.log(`  ✅ Created ${chunks.length} chunks`);

      } catch (error: any) {
        console.error(`❌ Error processing ${document.metadata.fileName}:`, error);
        this.stats.errors++;
      }
    }

    return allChunks;
  }

  /**
   * Enhance chunk with AI-powered legal analysis
   */
  private async enhanceChunkWithAI(chunk: Document): Promise<void> {
    try {
      const content = chunk.pageContent;

      // 1. Generate legal summary
      const summaryPrompt = `As a legal expert, provide a concise 2-sentence summary of this legal text, focusing on key legal concepts:

${content}

Summary:`;

      const summaryResponse = await this.llm.invoke(summaryPrompt);
      chunk.metadata.aiSummary = summaryResponse.content;

      // 2. Extract legal entities and key information
      const entitiesPrompt = `Extract key legal information from this text in JSON format:
{
  "parties": ["list of parties/entities"],
  "dates": ["important dates"],
  "legal_concepts": ["key legal terms"],
  "jurisdiction": "applicable jurisdiction if mentioned",
  "document_type": "type of legal document"
}

Text: ${content}

JSON:`;

      const entitiesResponse = await this.llm.invoke(entitiesPrompt);
      
      try {
        const entities = JSON.parse(entitiesResponse.content as string);
        chunk.metadata.legalEntities = JSON.stringify(entities);
        chunk.metadata.keyTerms = entities.legal_concepts || [];
        chunk.metadata.jurisdiction = entities.jurisdiction;
        chunk.metadata.documentType = entities.document_type;
      } catch {
        // Fallback to simple entity extraction
        chunk.metadata.legalEntities = entitiesResponse.content;
        chunk.metadata.keyTerms = this.extractKeyTerms(content);
      }

      // 3. Assess legal risk level
      const riskPrompt = `Rate the legal risk level of this text from 1-5 (1=low, 5=high risk) and explain briefly:

${content}

Risk assessment:`;

      const riskResponse = await this.llm.invoke(riskPrompt);
      chunk.metadata.riskLevel = riskResponse.content;

      // 4. Calculate confidence score based on content quality
      chunk.metadata.confidence = this.calculateConfidenceScore(content);

      this.stats.totalTokens += this.estimateTokens(content);

    } catch (error: any) {
      console.warn(`⚠️ AI analysis failed for chunk:`, error);
      // Continue without AI enhancement
      chunk.metadata.confidence = 0.5; // Default confidence
    }
  }

  /**
   * Store chunks in PostgreSQL with vector embeddings
   */
  private async storeChunks(chunks: Document[]): Promise<void> {
    const client = await this.dbPool.connect();
    
    try {
      await client.query('BEGIN');

      // Prepare the insert statement for your unified schema
      const insertQuery = `
        INSERT INTO evidence (
          id, case_id, title, description, evidence_type, file_type,
          file_url, file_name, file_size, hash, 
          ai_summary, ai_analysis, ai_tags, metadata,
          content_embedding, title_embedding, summary_embedding,
          uploaded_by, uploaded_at
        ) VALUES (
          gen_random_uuid(), $1, $2, $3, $4, $5, 
          $6, $7, $8, $9,
          $10, $11, $12, $13,
          $14, $15, $16,
          $17, NOW()
        )`;

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        
        try {
          // Generate embeddings for different content aspects
          const [contentEmbedding, titleEmbedding, summaryEmbedding] = await Promise.all([
            this.embeddings.embedQuery(chunk.pageContent),
            this.embeddings.embedQuery(chunk.metadata.fileName || 'untitled'),
            this.embeddings.embedQuery(chunk.metadata.aiSummary || chunk.pageContent.slice(0, 500))
          ]);

          // Insert into database
          await client.query(insertQuery, [
            null, // case_id (would be set based on your business logic)
            chunk.metadata.fileName || `Chunk ${i + 1}`,
            chunk.pageContent.slice(0, 1000), // Description
            'document', // evidence_type
            chunk.metadata.fileType || 'txt', // file_type
            chunk.metadata.source, // file_url
            chunk.metadata.fileName, // file_name
            chunk.metadata.fileSize, // file_size
            chunk.metadata.documentHash, // hash
            chunk.metadata.aiSummary, // ai_summary
            chunk.metadata.legalEntities, // ai_analysis
            JSON.stringify(chunk.metadata.keyTerms || []), // ai_tags
            JSON.stringify(chunk.metadata), // metadata
            `[${contentEmbedding.join(',')}]`, // content_embedding
            `[${titleEmbedding.join(',')}]`, // title_embedding
            `[${summaryEmbedding.join(',')}]`, // summary_embedding
            'system' // uploaded_by
          ]);

          this.stats.embeddingsGenerated += 3; // We generated 3 embeddings per chunk

          if ((i + 1) % 50 === 0) {
            console.log(`💾 Stored ${i + 1}/${chunks.length} chunks`);
          }

        } catch (error: any) {
          console.error(`❌ Failed to store chunk ${i + 1}:`, error);
          this.stats.errors++;
        }
      }

      await client.query('COMMIT');
      console.log(`✅ Successfully stored ${chunks.length - this.stats.errors} chunks in database`);

    } catch (error: any) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  /**
   * Check if document already exists in database
   */
  private async checkDocumentExists(documentHash: string): Promise<boolean> {
    const result = await this.dbPool.query(
      'SELECT COUNT(*) FROM evidence WHERE hash = $1',
      [documentHash]
    );
    return parseInt(result.rows[0].count) > 0;
  }

  /**
   * Utility functions
   */
  private extractKeyTerms(text: string): string[] {
    const legalTerms = [
      'contract', 'agreement', 'liability', 'damages', 'negligence', 
      'breach', 'violation', 'compliance', 'statute', 'regulation',
      'defendant', 'plaintiff', 'evidence', 'testimony', 'witness',
      'court', 'judge', 'jury', 'appeal', 'motion', 'discovery'
    ];

    return legalTerms.filter(term => 
      text.toLowerCase().includes(term.toLowerCase())
    );
  }

  private calculateConfidenceScore(content: string): number {
    let score = 0.5; // Base score
    
    if (content.length > 500) score += 0.1;
    if (content.includes('Section') || content.includes('Article')) score += 0.1;
    if (content.match(/\d{4}/)) score += 0.1; // Contains years
    if (content.includes('hereby') || content.includes('whereas')) score += 0.1;
    if (content.match(/\$[\d,]+/)) score += 0.1; // Contains money amounts
    
    return Math.min(score, 1.0);
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4); // Rough estimation
  }

  private printStats(): void {
    console.log('\n📊 Final Processing Statistics:');
    console.log('================================');
    console.log(`📄 Documents Processed: ${this.stats.documentsProcessed.toLocaleString()}`);
    console.log(`✂️ Chunks Created: ${this.stats.chunksCreated.toLocaleString()}`);
    console.log(`🧠 Embeddings Generated: ${this.stats.embeddingsGenerated.toLocaleString()}`);
    console.log(`📈 Avg Chunks per Doc: ${this.stats.avgChunksPerDoc}`);
    console.log(`🎯 Token Usage: ${this.stats.totalTokens.toLocaleString()}`);
    console.log(`⏱️ Total Processing Time: ${(this.stats.processingTime / 1000).toFixed(1)}s`);
    console.log(`❌ Errors: ${this.stats.errors}`);
    console.log(`✅ Success Rate: ${((this.stats.chunksCreated - this.stats.errors) / this.stats.chunksCreated * 100).toFixed(1)}%`);
  }

  private async cleanup(): Promise<void> {
    await this.dbPool.end();
  }
}

/**
 * CLI Configuration
 */
const program = new Command();

program
  .name('legal-ingest')
  .description('Advanced LangChain-powered legal document ingestion pipeline')
  .version('1.0.0');

program
  .command('ingest')
  .description('Ingest legal documents from a directory')
  .argument('<input-dir>', 'Directory containing legal documents')
  .option('-r, --recursive', 'Process subdirectories recursively', true)
  .option('--no-ai', 'Skip AI analysis (faster processing)')
  .option('-b, --batch-size <number>', 'Number of documents to process per batch', '10')
  .option('--no-skip-existing', 'Process documents even if already ingested')
  .option('--dry-run', 'Show what would be processed without actually doing it')
  .action(async (inputDir: string, options: any) => {
    try {
      const processor = new LegalDocumentIngestion();
      
      await processor.ingestDocuments(inputDir, {
        recursive: options.recursive,
        includeAI: options.ai,
        batchSize: parseInt(options.batchSize),
        skipExisting: options.skipExisting,
        dryRun: options.dryRun
      });

    } catch (error: any) {
      console.error('❌ Ingestion failed:', error);
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Show database and service status')
  .action(async () => {
    try {
      // Check database connectivity
      const pool = new Pool(DB_CONFIG);
      const result = await pool.query('SELECT COUNT(*) FROM evidence');
      console.log(`📊 Evidence records in database: ${result.rows[0].count}`);
      await pool.end();

      // Check Ollama connectivity
      try {
        const response = await fetch(`${OLLAMA_BASE_URL.replace('/v1', '')}/api/tags`);
        const models = await response.json();
        console.log('🧠 Available Ollama models:', models.models?.map((m: any) => m.name).join(', ') || 'none');
      } catch {
        console.log('❌ Ollama service not available');
      }

    } catch (error: any) {
      console.error('❌ Status check failed:', error);
      process.exit(1);
    }
  });

// Run CLI
program.parse();