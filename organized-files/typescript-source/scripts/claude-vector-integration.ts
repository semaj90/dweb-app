#!/usr/bin/env tsx
import { Client } from 'pg';
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

// Database configuration
const dbConfig = {
  user: 'postgres',
  password: '123456',
  host: 'localhost',
  database: 'legal_ai_db',
  port: 5432,
};

const OLLAMA_URL = 'http://localhost:11434';
const EMBED_MODEL = 'nomic-embed-text';

interface EmbeddingResult {
  embedding: number[];
  model: string;
  timestamp: Date;
}

interface DocumentChunk {
  file: string;
  content: string;
  chunkIndex: number;
  totalChunks: number;
  tokens: number;
}

class ClaudeVectorIntegration {
  private client: Client;
  private embeddingCache: Map<string, number[]> = new Map();

  constructor() {
    this.client = new Client(dbConfig);
  }

  async connect() {
    await this.client.connect();
    console.log('🔗 Connected to PostgreSQL with pgvector');
    await this.ensureTablesExist();
  }

  async disconnect() {
    await this.client.end();
  }

  private async ensureTablesExist() {
    // Check and create documents table if needed
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        file TEXT,
        content TEXT,
        summary TEXT,
        embedding vector(768),
        chunk_index INTEGER DEFAULT 0,
        total_chunks INTEGER DEFAULT 1,
        tokens INTEGER,
        file_hash TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);

    // Create embedding cache table
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS embedding_cache (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        text_hash TEXT UNIQUE NOT NULL,
        embedding vector(768) NOT NULL,
        model VARCHAR(100) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);

    // Create indexes
    await this.client.query(`
      CREATE INDEX IF NOT EXISTS idx_documents_embedding 
      ON documents USING ivfflat (embedding vector_cosine_ops) 
      WITH (lists = 100)
    `);

    await this.client.query(`
      CREATE INDEX IF NOT EXISTS idx_documents_file_hash 
      ON documents(file_hash)
    `);
  }

  private async getEmbedding(text: string): Promise<number[]> {
    const textHash = crypto.createHash('sha256').update(text).digest('hex');

    // Check cache first
    const cached = await this.client.query(
      'SELECT embedding FROM embedding_cache WHERE text_hash = $1',
      [textHash]
    );

    if (cached.rows.length > 0) {
      console.log('📦 Using cached embedding');
      return cached.rows[0].embedding;
    }

    // Generate new embedding
    console.log('🧠 Generating new embedding...');
    const response = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: EMBED_MODEL,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Embedding generation failed: ${response.statusText}`);
    }

    const data = await response.json();
    const embedding = data.embedding;

    // Cache the embedding
    await this.client.query(
      `INSERT INTO embedding_cache (text_hash, embedding, model) 
       VALUES ($1, $2::vector, $3)
       ON CONFLICT (text_hash) DO NOTHING`,
      [textHash, `[${embedding.join(',')}]`, EMBED_MODEL]
    );

    return embedding;
  }

  async embedFile(filePath: string) {
    console.log(`📄 Processing file: ${filePath}`);
    
    const content = await fs.readFile(filePath, 'utf-8');
    const fileHash = crypto.createHash('sha256').update(content).digest('hex');
    
    // Check if file already embedded
    const existing = await this.client.query(
      'SELECT COUNT(*) FROM documents WHERE file_hash = $1',
      [fileHash]
    );

    if (parseInt(existing.rows[0].count) > 0) {
      console.log('✅ File already embedded, skipping...');
      return;
    }

    // Chunk the content (simple chunking by paragraphs or lines)
    const chunks = this.chunkContent(content, 1000); // ~1000 chars per chunk
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = await this.getEmbedding(chunk.content);
      
      await this.client.query(
        `INSERT INTO documents (file, content, embedding, chunk_index, total_chunks, tokens, file_hash)
         VALUES ($1, $2, $3::vector, $4, $5, $6, $7)`,
        [
          chunk.file,
          chunk.content,
          `[${embedding.join(',')}]`,
          chunk.chunkIndex,
          chunk.totalChunks,
          chunk.tokens,
          fileHash
        ]
      );
      
      console.log(`  ✅ Chunk ${i + 1}/${chunks.length} embedded`);
    }
  }

  private chunkContent(content: string, maxChunkSize: number): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    const lines = content.split('\n');
    let currentChunk = '';
    let chunkIndex = 0;

    for (const line of lines) {
      if (currentChunk.length + line.length > maxChunkSize && currentChunk.length > 0) {
        chunks.push({
          file: 'manual-input',
          content: currentChunk.trim(),
          chunkIndex: chunkIndex++,
          totalChunks: 0, // Will update later
          tokens: Math.floor(currentChunk.length / 4), // Rough estimate
        });
        currentChunk = '';
      }
      currentChunk += line + '\n';
    }

    if (currentChunk.trim()) {
      chunks.push({
        file: 'manual-input',
        content: currentChunk.trim(),
        chunkIndex: chunkIndex++,
        totalChunks: 0,
        tokens: Math.floor(currentChunk.length / 4),
      });
    }

    // Update total chunks
    chunks.forEach(chunk => chunk.totalChunks = chunks.length);
    
    return chunks;
  }

  async searchSimilar(query: string, limit: number = 5) {
    const queryEmbedding = await this.getEmbedding(query);
    
    const results = await this.client.query(
      `SELECT 
        id,
        file,
        content,
        summary,
        chunk_index,
        total_chunks,
        embedding <=> $1::vector AS distance
      FROM documents
      WHERE embedding IS NOT NULL
      ORDER BY distance
      LIMIT $2`,
      [`[${queryEmbedding.join(',')}]`, limit]
    );

    return results.rows;
  }

  async generateClaudeContext(query: string, limit: number = 5) {
    console.log(`🔍 Searching for: "${query}"`);
    
    const results = await this.searchSimilar(query, limit);
    
    if (results.length === 0) {
      return {
        task: "no_results",
        prompt: query,
        context: [],
        instructions: "No relevant documents found. Please provide a general response."
      };
    }

    // Format for Claude
    const claudePayload = {
      task: "legal_rag_analysis",
      prompt: query,
      context: results.map((doc, idx) => ({
        id: doc.id,
        file: doc.file || `Document ${idx + 1}`,
        content: doc.content,
        summary: doc.summary,
        chunk_info: `Chunk ${doc.chunk_index + 1} of ${doc.total_chunks}`,
        relevance_score: (1 - doc.distance).toFixed(3),
        metadata: {
          distance: doc.distance,
          embedding_model: EMBED_MODEL,
        }
      })),
      instructions: `Analyze the provided legal documents and answer the query. 
                     Reference specific documents when making claims.
                     Highlight any contradictions or uncertainties in the sources.`,
      response_format: {
        summary: "Brief answer to the query",
        detailed_analysis: "In-depth analysis with document references",
        relevant_quotes: "Key quotes from the documents",
        recommendations: "Actionable recommendations if applicable"
      }
    };

    return claudePayload;
  }

  async embedDirectory(dirPath: string, extensions: string[] = ['.ts', '.md', '.txt', '.svelte']) {
    const files = await this.getFilesRecursively(dirPath, extensions);
    
    console.log(`📁 Found ${files.length} files to embed`);
    
    for (const file of files) {
      try {
        await this.embedFile(file);
      } catch (error) {
        console.error(`❌ Error embedding ${file}:`, error);
      }
    }
  }

  private async getFilesRecursively(dir: string, extensions: string[]): Promise<string[]> {
    const files: string[] = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
        files.push(...await this.getFilesRecursively(fullPath, extensions));
      } else if (entry.isFile() && extensions.some(ext => entry.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }

    return files;
  }
}

// CLI Interface
async function main() {
  const integration = new ClaudeVectorIntegration();
  
  try {
    await integration.connect();
    
    const command = process.argv[2];
    const args = process.argv.slice(3);

    switch (command) {
      case 'embed-file': {
        if (!args[0]) {
          console.error('❌ Please provide a file path');
          process.exit(1);
        }
        await integration.embedFile(args[0]);
        break;
      }

      case 'embed-dir': {
        const dir = args[0] || '.';
        await integration.embedDirectory(dir);
        break;
      }

      case 'search': {
        const query = args.join(' ');
        if (!query) {
          console.error('❌ Please provide a search query');
          process.exit(1);
        }
        
        const context = await integration.generateClaudeContext(query);
        console.log('\n📋 Claude-ready context:\n');
        console.log(JSON.stringify(context, null, 2));
        
        // Save to file
        const outputPath = path.join('.check-logs', `claude-context-${Date.now()}.json`);
        await fs.mkdir('.check-logs', { recursive: true });
        await fs.writeFile(outputPath, JSON.stringify(context, null, 2));
        console.log(`\n💾 Saved to: ${outputPath}`);
        break;
      }

      case 'test': {
        // Insert test documents
        console.log('📝 Inserting test legal documents...\n');
        
        const testDocs = [
          {
            content: "In contract law, the doctrine of consideration requires that each party must provide something of value. This forms the basis of a legally binding agreement.",
            file: "contract-law-basics.txt"
          },
          {
            content: "Criminal procedure mandates that evidence obtained illegally cannot be used in court. This is known as the exclusionary rule and protects constitutional rights.",
            file: "criminal-procedure.txt"
          },
          {
            content: "Tort law covers civil wrongs that cause harm or loss. Negligence is the most common tort, requiring proof of duty, breach, causation, and damages.",
            file: "tort-law-overview.txt"
          }
        ];

        for (const doc of testDocs) {
          const embedding = await integration.getEmbedding(doc.content);
          await integration.client.query(
            `INSERT INTO documents (file, content, embedding, tokens)
             VALUES ($1, $2, $3::vector, $4)`,
            [doc.file, doc.content, `[${embedding.join(',')}]`, Math.floor(doc.content.length / 4)]
          );
          console.log(`✅ Inserted: ${doc.file}`);
        }

        console.log('\n🔍 Testing search...');
        const testQuery = "What are the requirements for a valid contract?";
        const results = await integration.generateClaudeContext(testQuery);
        console.log('\nSearch results:', JSON.stringify(results, null, 2));
        break;
      }

      default:
        console.log(`
🚀 Claude + pgvector Integration CLI

Usage: npm run vector:claude [command] [args]

Commands:
  embed-file <path>     Embed a single file into pgvector
  embed-dir [path]      Embed all code files in directory (default: current)
  search <query>        Search and generate Claude context
  test                  Insert test documents and run sample search

Examples:
  npm run vector:claude embed-file src/routes/+page.svelte
  npm run vector:claude embed-dir src/lib/components
  npm run vector:claude search "How does evidence validation work?"
        `);
    }
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  } finally {
    await integration.disconnect();
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { ClaudeVectorIntegration };