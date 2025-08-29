// High-level service for embedding_cache persistence with packed embedding support.
// Assumes there is a db client exported from ./client (adjust import if located elsewhere)
import { embeddingCache } from './schema-postgres';
import { packEmbedding } from './embedding-cache-utils';
import { eq } from 'drizzle-orm';
// Lazy import pattern to avoid circular imports if db client pulls schema
let _db: any;
async function getDb(): Promise<any> {
  if (_db) return _db;
  const mod = await import('./client');
  _db = mod.db;
  return _db;
}

export interface UpsertEmbeddingOptions {
  model: string;
  textHash: string;
  embedding: number[]; // raw float embedding
  packMethod?: 'uint8-linear' | 'int8-symmetric';
}

export async function upsertEmbedding(opts: UpsertEmbeddingOptions): Promise<any> {
  const { model, textHash, embedding, packMethod = 'int8-symmetric' } = opts;
  const db = await getDb();
  const { b64, scale, method } = packEmbedding(embedding, packMethod);
  const existing = await db.select().from(embeddingCache).where(eq(embeddingCache.textHash, textHash));
  if (existing.length) {
    await db.update(embeddingCache)
      .set({ embedding, packedEmbedding: b64, embeddingScale: scale?.toString(), model })
      .where(eq(embeddingCache.textHash, textHash));
    return { updated: true, method, scale };
  } else {
    await db.insert(embeddingCache).values({ textHash, embedding, packedEmbedding: b64, embeddingScale: scale?.toString(), model });
    return { created: true, method, scale };
  }
}

export async function getEmbedding(textHash: string): Promise<any> {
  const db = await getDb();
  const rows = await db.select().from(embeddingCache).where(eq(embeddingCache.textHash, textHash));
  return rows[0] || null;
}
