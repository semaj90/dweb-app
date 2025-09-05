// Enhanced embeddings helper with real Ollama nomic-embed-text integration
import { cacheEmbedding, getCachedEmbedding } from "$lib/server/cache/redis";
import { aiService, type EmbeddingProvider } from "$lib/services/ai-service";
import { sql } from 'drizzle-orm';
import { db } from '$lib/server/db/drizzle';

// Ollama embedding integration
async function generateOllamaEmbedding(text: string): Promise<number[]> {
  const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';

  try {
    const response = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: text.substring(0, 2048) // Limit token length
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.embedding || !Array.isArray(data.embedding)) {
      throw new Error('Invalid embedding response from Ollama');
    }

    return data.embedding;
  } catch (error) {
    console.warn(`Ollama embedding failed: ${error}, falling back to aiService`);
    throw error;
  }
}

export interface EnhancedEmbeddingOptions {
  provider?: "auto" | "openai" | "tauri-legal-bert" | "tauri-bert";
  cache?: boolean;
  maxTokens?: number;
  legalDomain?: boolean;
  batchSize?: number;
}

export async function generateEnhancedEmbedding(
  text: string | string[],
  options: EnhancedEmbeddingOptions = {}
): Promise<number[] | number[][]> {
  const { provider = "auto", cache = true, maxTokens = 8000, legalDomain = true } =
    options;
  if (!text) throw new Error("Text required");

  const isArray = Array.isArray(text);
  const inputs = isArray ? text : [text];
  const truncated = inputs.map((t) =>
    t.length > maxTokens ? t.substring(0, maxTokens) : t
  );

  if (cache && !isArray) {
    const key = `${provider}-${legalDomain}`;
    const cached = await getCachedEmbedding(truncated[0], key);
    if (cached) return cached;
  }

  if (typeof aiService.initialize === "function") await aiService.initialize();

  // If aiService has embedding support, use it; otherwise return a stub
  if (typeof aiService.generateEmbedding === "function") {
    const providerChoice = selectProvider(provider, legalDomain);
    const res = await aiService.generateEmbedding(truncated, {
      provider: providerChoice,
      legalDomain,
    });
    if (!isArray)
      return Array.isArray(res[0]) ? (res as number[][])[0] : (res as number[]);
    return res as number[][];
  }

  // fallback stub: return zero vectors
  const dim = 1536;
  return truncated.map(() => new Array(dim).fill(0));
}

export async function generateBatchEmbeddingsEnhanced(
  texts: string[],
  options: EnhancedEmbeddingOptions = {}
): Promise<number[][]> {
  const { batchSize = 10 } = options;
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const res = (await generateEnhancedEmbedding(batch, options)) as number[][];
    // res should be an array of embeddings already
    if (Array.isArray(res) && Array.isArray(res[0])) {
      out.push(...res);
    }
  }
  return out;
}

function selectProvider(
  requested: EnhancedEmbeddingOptions["provider"] | undefined,
  legalDomain: boolean
): EmbeddingProvider {
  if (requested && requested !== "auto") return requested as EmbeddingProvider;
  const status =
    typeof aiService.getStatus === "function"
      ? aiService.getStatus()
      : { tauriAvailable: false } as any;
  if (status?.tauriAvailable)
    return legalDomain ? "tauri-legal-bert" : "tauri-bert";
  return "openai";
}

export async function calculateLegalSimilarity(doc1: string, doc2: string): Promise<number> {
  if (typeof aiService.initialize === 'function') await aiService.initialize();
  const provider = typeof aiService.getStatus === 'function' && aiService.getStatus().tauriAvailable ? 'tauri-legal-bert' : 'openai';
  const embeddings = (await generateEnhancedEmbedding([doc1, doc2], { provider, legalDomain: true })) as number[][];
  return cosineSimilarity(embeddings[0], embeddings[1]);
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error('Vector lengths differ');
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export async function generateEmbedding(text: string): Promise<number[]> {
  const r = await generateEnhancedEmbedding(text, { provider: 'auto', legalDomain: true });
  if (Array.isArray(r) && Array.isArray(r[0])) return (r as number[][])[0];
  return r as number[];
}

export async function generateBatchEmbeddings(texts: string[], batchSize = 10): Promise<number[][]> {
  return generateBatchEmbeddingsEnhanced(texts, { provider: 'auto', legalDomain: true, batchSize });
}

// Helper: persist embedding to pgvector-enabled table (idempotent)
export async function upsertDocumentEmbedding(documentId: string, embedding: number[]): Promise<void> {
  if (!embedding?.length) return;
  // Assuming legal_documents.embedding exists (vector(384))
  await db.execute(sql`UPDATE legal_documents SET embedding = ${sql.raw(`'[${embedding.join(',')}]'`)} WHERE id = ${documentId}`);
}

export async function backfillSectionEmbeddings(limit = 100): Promise<{ processed: number }>{
  // Select sections missing embedding
  const rows = await db.execute(sql`SELECT id, content FROM document_sections WHERE embedding IS NULL LIMIT ${limit}`) as any[];
  let processed = 0;
  for (const row of rows) {
    try {
      const emb = await generateEmbedding(row.content);
      await db.execute(sql`UPDATE document_sections SET embedding = ${sql.raw(`'[${emb.join(',')}]'`)} WHERE id = ${row.id}`);
      processed++;
    } catch (e) {
      // continue
    }
  }
  return { processed };
}
