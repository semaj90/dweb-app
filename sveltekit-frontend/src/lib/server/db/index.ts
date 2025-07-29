import { db, sql, pool } from "./drizzle";
export { db, sql, pool };

// Database type detection
export const isPostgreSQL = true; // Since we're using PostgreSQL with pgvector

// Re-export all database tables and relations
export * from "./schema-postgres";

// Re-export performance optimizations (optional - may not exist)
// export { OptimizedQueries, CacheService } from '$lib/performance/optimizations';

// Database connection health check
export async function healthCheck() {
  try {
    await db.execute(sql`SELECT 1`);
    return { status: "healthy", timestamp: new Date() };
  } catch (error) {
    return { status: "unhealthy", error: error.message, timestamp: new Date() };
  }
}

// --- Context7, Bits UI, Melt UI, and Svelte 5 Integration Best Practices ---
// This file is the main DB entry point for SvelteKit/Legal AI with Context7 MCP orchestration.
// All DB, vector, and health utilities are exported here for type-safe, scalable use.

// Context7 MCP: Expose DB pool for vector store and semantic search
// (Already exported above)

// Example: Export a function to get a vector store for semantic search (Nomic embed LLM)
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";

export function getVectorStore() {
  const embeddings = new OpenAIEmbeddings({
    modelName: "nomic-embed-text",
    openAIApiKey: "N/A", // Local LLM, no key needed
    // baseURL intentionally omitted for local compatibility
  });
  return new PGVectorStore(embeddings, { pool, tableName: "vectors" });
}

// Example: Bits UI/Melt UI best practice (for Svelte 5):
// Use stores, context, and type-safe exports for all DB and UI modules.
// See README or docs for more advanced UI/agent orchestration patterns.

// --- End Context7/Legal AI DB Integration ---
