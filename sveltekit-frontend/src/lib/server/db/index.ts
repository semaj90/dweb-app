// @ts-nocheck
import { db, sql, pool } from "./drizzle";
export { db, sql, pool };

// Database type detection
export const isPostgreSQL = true; // Since we're using PostgreSQL with pgvector

// Re-export all database tables and relations from schema
export * from "./schema-postgres";

// Explicitly export tables to ensure they're available
import {
  users,
  sessions,
  cases,
  evidence,
  legalDocuments,
  caseActivities,
  statutes,
} from "./schema-postgres";

export {
  users,
  sessions,
  cases,
  evidence,
  legalDocuments,
  caseActivities,
  statutes,
};

// Re-export performance optimizations (optional - may not exist)
// export { OptimizedQueries, CacheService } from '$lib/performance/optimizations';

// Type-safe database queries helper
export function getTableByName(tableName: string) {
  const tableMap = {
    users,
    sessions,
    cases,
    evidence,
    legalDocuments,
    caseActivities,
    statutes,
  };
  
  return tableMap[tableName as keyof typeof tableMap];
}

// Database connection health check with enhanced error handling
export async function healthCheck() {
  try {
    if (!db) {
      return {
        status: "unhealthy" as const,
        error: "Database not initialized",
        timestamp: new Date(),
      };
    }
    
    // Test basic connectivity
    await db.execute(sql`SELECT 1`);
    
    // Test specific tables
    const tableTests = await Promise.allSettled([
      db.select().from(users).limit(1),
      db.select().from(sessions).limit(1),
      db.select().from(cases).limit(1),
    ]);
    
    const failedTests = tableTests.filter((result: any) => result.status === 'rejected');
    
    if (failedTests.length > 0) {
      return {
        status: "degraded" as const,
        error: `${failedTests.length} table(s) inaccessible`,
        timestamp: new Date(),
      };
    }
    
    return { 
      status: "healthy" as const, 
      timestamp: new Date(),
      tablesAccessible: tableTests.length 
    };
  } catch (error: any) {
    return { 
      status: "unhealthy" as const, 
      error: error.message, 
      timestamp: new Date() 
    };
  }
}

// --- Context7, Bits UI, Melt UI, and Svelte 5 Integration Best Practices ---
// This file is the main DB entry point for SvelteKit/Legal AI with Context7 MCP orchestration.
// All DB, vector, and health utilities are exported here for type-safe, scalable use.

// Context7 MCP: Expose DB pool for vector store and semantic search
// (Already exported above)

// Enhanced vector store with error handling
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";

export function getVectorStore() {
  try {
    const embeddings = new OpenAIEmbeddings({
      modelName: "nomic-embed-text",
      openAIApiKey: "N/A", // Local LLM, no key needed
      // baseURL intentionally omitted for local compatibility
    });
    return new PGVectorStore(embeddings, { pool, tableName: "vectors" });
  } catch (error: any) {
    console.error('Vector store initialization failed:', error);
    throw new Error(`Vector store unavailable: ${error.message}`);
  }
}

// Example: Bits UI/Melt UI best practice (for Svelte 5):
// Use stores, context, and type-safe exports for all DB and UI modules.
// See README or docs for more advanced UI/agent orchestration patterns.

// --- End Context7/Legal AI DB Integration ---
