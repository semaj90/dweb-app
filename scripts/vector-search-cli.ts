#!/usr/bin/env tsx
import { Client } from 'pg';
import { spawn } from 'child_process';
import { promisify } from 'util';
import readline from 'readline/promises';

// PostgreSQL connection
const dbConfig = {
  user: 'legal_admin',
  password: 'LegalAI2024!',
  host: 'localhost',
  database: 'legal_ai_db',
  port: 5432,
};

// Ollama configuration
const OLLAMA_URL = 'http://localhost:11434';
const EMBED_MODEL = 'nomic-embed-text';

interface SearchResult {
  id: string;
  content: string;
  file: string;
  summary: string;
  distance: number;
}

async function generateEmbedding(text: string): Promise<number[]> {
  const response = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: EMBED_MODEL,
      prompt: text,
    }),
  });

  const data = await response.json();
  return data.embedding;
}

async function searchSimilar(client: Client, query: string, limit: number = 5): Promise<SearchResult[]> {
  // Generate embedding for query
  console.log('🧠 Generating embedding for query...');
  const embedding = await generateEmbedding(query);
  
  // Search for similar documents
  const result = await client.query(
    `SELECT 
      id, 
      file, 
      content, 
      summary,
      embedding <=> $1::vector AS distance
    FROM documents
    WHERE embedding IS NOT NULL
    ORDER BY distance
    LIMIT $2`,
    [`[${embedding.join(',')}]`, limit]
  );

  return result.rows;
}

async function createClaudePrompt(query: string, results: SearchResult[]): Promise<string> {
  const context = results.map((r, i) => 
    `Document ${i + 1} (${r.file || 'Unknown'}):\n${r.content || r.summary || 'No content'}\n`
  ).join('\n---\n');

  return JSON.stringify({
    task: "legal_analysis",
    prompt: query,
    context: results.map(r => ({
      file: r.file,
      content: r.content,
      summary: r.summary,
      relevance_score: 1 - r.distance // Convert distance to similarity
    })),
    instructions: "Analyze the legal documents and provide insights based on the query."
  }, null, 2);
}

async function insertTestDocument(client: Client, content: string, file: string, summary: string) {
  console.log(`📄 Inserting document: ${file}...`);
  
  const embedding = await generateEmbedding(content);
  
  await client.query(
    `INSERT INTO documents (file, content, summary, embedding)
     VALUES ($1, $2, $3, $4::vector)
     ON CONFLICT (id) DO NOTHING`,
    [file, content, summary, `[${embedding.join(',')}]`]
  );
  
  console.log('✅ Document inserted!');
}

async function main() {
  const client = new Client(dbConfig);
  
  try {
    await client.connect();
    console.log('🔗 Connected to PostgreSQL');

    // Check if documents table exists and has pgvector
    const tableCheck = await client.query(`
      SELECT column_name, data_type 
      FROM information_schema.columns 
      WHERE table_name = 'documents'
    `);

    if (tableCheck.rows.length === 0) {
      console.log('📋 Creating documents table...');
      await client.query(`
        CREATE TABLE IF NOT EXISTS documents (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          file TEXT,
          content TEXT,
          summary TEXT,
          embedding vector(768),
          created_at TIMESTAMP DEFAULT NOW()
        )
      `);
      
      // Create index for vector similarity search
      await client.query(`
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
      `);
    }

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    console.log('\n🚀 Legal AI Vector Search CLI');
    console.log('Commands:');
    console.log('  search <query>  - Search for similar documents');
    console.log('  insert          - Insert a test document');
    console.log('  claude <query>  - Search and format for Claude');
    console.log('  exit            - Quit\n');

    while (true) {
      const input = await rl.question('> ');
      const [command, ...args] = input.split(' ');

      if (command === 'exit') break;

      switch (command) {
        case 'search': {
          const query = args.join(' ');
          if (!query) {
            console.log('❌ Please provide a search query');
            break;
          }

          const results = await searchSimilar(client, query);
          
          if (results.length === 0) {
            console.log('No results found. Try inserting some documents first.');
          } else {
            console.log(`\n📊 Found ${results.length} similar documents:\n`);
            results.forEach((r, i) => {
              console.log(`${i + 1}. ${r.file || 'Unknown'} (distance: ${r.distance.toFixed(4)})`);
              console.log(`   ${r.summary || r.content?.substring(0, 100) + '...'}\n`);
            });
          }
          break;
        }

        case 'insert': {
          await insertTestDocument(
            client,
            "This legal precedent establishes that contracts must be executed in good faith. The court ruled that implicit obligations exist even when not explicitly stated in the contract terms.",
            "contract-law-precedent.txt",
            "Good faith contract execution precedent"
          );

          await insertTestDocument(
            client,
            "Evidence handling procedures require maintaining chain of custody. All evidence must be logged, sealed, and tracked through each transfer to ensure admissibility in court.",
            "evidence-procedures.txt",
            "Evidence chain of custody requirements"
          );

          await insertTestDocument(
            client,
            "Criminal defense strategies often involve challenging the prosecution's evidence. Key approaches include questioning witness credibility, exposing procedural errors, and establishing reasonable doubt.",
            "defense-strategies.txt",
            "Criminal defense tactical approaches"
          );

          console.log('\n✅ Test documents inserted!');
          break;
        }

        case 'claude': {
          const query = args.join(' ');
          if (!query) {
            console.log('❌ Please provide a query for Claude');
            break;
          }

          const results = await searchSimilar(client, query);
          const claudePrompt = await createClaudePrompt(query, results);
          
          console.log('\n📋 Claude-ready JSON payload:\n');
          console.log(claudePrompt);
          
          // Copy to clipboard on Windows
          const { exec } = require('child_process');
          exec(`echo ${JSON.stringify(claudePrompt)} | clip`, (err: any) => {
            if (!err) {
              console.log('\n✅ Copied to clipboard!');
            }
          });
          break;
        }

        default:
          console.log('❌ Unknown command. Use: search, insert, claude, or exit');
      }
    }

    rl.close();
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await client.end();
  }
}

main().catch(console.error);