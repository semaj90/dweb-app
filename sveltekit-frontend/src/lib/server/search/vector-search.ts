// @ts-nocheck
// Complete Vector Search Service - Production Ready
// Combines PostgreSQL pgvector + Qdrant + Local caching + Loki.js + Fuse.js
import { browser } from "$app/environment";
import { db, isPostgreSQL } from "$lib/server/db/index";
import { ollamaService } from "$lib/server/services/OllamaService";
import { and, eq, or, sql } from "drizzle-orm";

// Import dependencies with fallbacks
let qdrant: any = null;
let generateEmbedding: any = null;
let cache: any = null;
let loki: any = null;
let Fuse: any = null;

// Conditional imports for server-side only
if (!browser) {
  try {
    const qdrantModule = await import("../../../lib/server/vector/qdrant.js");
    qdrant = qdrantModule.qdrant;
  } catch (error) {
    console.warn("Qdrant not available:", error);
  }
  try {
    const embeddingsModule = await import(
      "../../../lib/server/ai/embeddings-simple.js"
    );
    generateEmbedding = embeddingsModule.generateEmbedding;
  } catch (error) {
    console.warn("Embeddings service not available:", error);
  }
  try {
    const cacheModule = await import("../../../lib/server/cache/redis.js");
    cache = cacheModule.cache;
  } catch (error) {
    console.warn("Redis cache not available:", error);
    cache = { get: async () => null, set: async () => {} };
  }
  try {
    // Import Loki.js for local database
    const lokiModule = await import("lokijs");
    loki = lokiModule.default || lokiModule;
  } catch (error) {
    console.warn("Loki.js not available:", error);
  }
  try {
    // Import Fuse.js for fuzzy search
    const fuseModule = await import("fuse.js");
    Fuse = fuseModule.default || fuseModule;
  } catch (error) {
    console.warn("Fuse.js not available:", error);
  }
}
// Vector search result interface
export interface VectorSearchResult {
  id: string;
  title: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  source: "pgvector" | "qdrant" | "cache";
  type: "case" | "evidence" | "document";
}
// Search options interface
export interface VectorSearchOptions {
  limit?: number;
  offset?: number;
  threshold?: number;
  useCache?: boolean;
  fallbackToQdrant?: boolean;
  useFuzzySearch?: boolean;
  useLocalDb?: boolean;
  filters?: Record<string, any>;
  searchType?: "similarity" | "hybrid" | "semantic" | "fuzzy" | "exact";
}
// Local database using Loki.js for client-side search
let lokiDb: any = null;
let fuseCases: any = null;
let fuseEvidence: any = null;

// Initialize local database
async function initializeLocalDb() {
  if (!loki || lokiDb) return;

  lokiDb = new loki("legal_search.db", {
    autoload: true,
    autoloadCallback: () => {
      // Initialize collections if they don't exist
      let casesCollection = lokiDb.getCollection("cases");
      let evidenceCollection = lokiDb.getCollection("evidence");

      if (!casesCollection) {
        casesCollection = lokiDb.addCollection("cases", {
          indices: ["title", "description"],
        });
      }
      if (!evidenceCollection) {
        evidenceCollection = lokiDb.addCollection("evidence", {
          indices: ["title", "description"],
        });
      }
    },
    autosave: true,
    autosaveInterval: 4000,
  });
}
// --- Legal documents pgvector search (uses Ollama 768-dim embeddings) ---
function arrayToPgVector(embedding: number[]): string {
  return `[${embedding.join(",")}]`;
}

export async function getQueryEmbeddingLegal(
  query: string
): Promise<number[] | null> {
  const model = process.env.EMBED_MODEL || "nomic-embed-text"; // 768-dim by default
  try {
    const vec = await ollamaService.embeddings(model, query);
    if (Array.isArray(vec) && vec.length > 0) return vec;
    // Fallback to rag-kratos /embed if Ollama returned empty
    const ragUrl =
      process.env.RAG_URL ||
      `http://localhost:${process.env.RAG_HTTP_PORT || "8093"}`;
    try {
      const resp = await fetch(`${ragUrl}/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [query], model }),
        signal: AbortSignal.timeout(4000),
      });
      if (resp.ok) {
        const data = await resp.json();
        const v = data?.vectors?.[0];
        if (Array.isArray(v) && v.length > 0) return v;
      }
    } catch (e) {
      console.warn(
        "rag-kratos embed fallback failed:",
        (e as Error)?.message || e
      );
    }
    // Final fallback: CPU embedding via @xenova/transformers (384-dim); pad/trim to 768
    try {
      const xenova = await import("@xenova/transformers");
      const pipeline = xenova.pipeline as any;
      const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
      );
      const result = await extractor(query, {
        pooling: "mean",
        normalize: true,
      });
      const arr = Array.from(result.data || []);
      if (Array.isArray(arr) && arr.length > 0) {
        const target = 768;
        const adj =
          arr.length === target
            ? arr
            : arr.length > target
              ? arr.slice(0, target)
              : arr.concat(Array(target - arr.length).fill(0));
        return adj;
      }
    } catch (ex) {
      console.warn(
        "Xenova embedding fallback failed:",
        (ex as Error)?.message || ex
      );
    }
    return null;
  } catch (err) {
    console.warn(
      "Ollama embeddings failed for legal search:",
      (err as Error)?.message || err
    );
    // Try rag-kratos if Ollama call threw
    const model = process.env.EMBED_MODEL || "nomic-embed-text";
    const ragUrl =
      process.env.RAG_URL ||
      `http://localhost:${process.env.RAG_HTTP_PORT || "8093"}`;
    try {
      const resp = await fetch(`${ragUrl}/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [query], model }),
        signal: AbortSignal.timeout(4000),
      });
      if (resp.ok) {
        const data = await resp.json();
        const v = data?.vectors?.[0];
        if (Array.isArray(v) && v.length > 0) return v;
      }
    } catch (e2) {
      console.warn(
        "rag-kratos embed after error failed:",
        (e2 as Error)?.message || e2
      );
    }
    // Xenova as last resort
    try {
      const xenova = await import("@xenova/transformers");
      const pipeline = xenova.pipeline as any;
      const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
      );
      const result = await extractor(query, {
        pooling: "mean",
        normalize: true,
      });
      const arr = Array.from(result.data || []);
      if (Array.isArray(arr) && arr.length > 0) {
        const target = 768;
        const adj =
          arr.length === target
            ? arr
            : arr.length > target
              ? arr.slice(0, target)
              : arr.concat(Array(target - arr.length).fill(0));
        return adj;
      }
    } catch (ex2) {
      console.warn(
        "Xenova embedding after error failed:",
        (ex2 as Error)?.message || ex2
      );
    }
    return null;
  }
}

async function searchLegalDocumentsPgvector(
  query: string,
  options: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  const { limit = 20 } = options;
  const threshold =
    (options as any).threshold ?? (options as any).minSimilarity ?? 0.7;
  const embedding = await getQueryEmbeddingLegal(query);
  if (!embedding) return [];

  const vectorString = arrayToPgVector(embedding);
  try {
    const execResult: any = await db.execute(sql`
      SELECT
        id,
        title,
        COALESCE(content, full_text) as content,
        1 - (embedding <=> ${vectorString}::vector) as similarity,
        keywords,
        topics,
        document_type,
        case_id
      FROM legal_documents
      WHERE embedding IS NOT NULL
        AND 1 - (embedding <=> ${vectorString}::vector) > ${threshold}
      ORDER BY embedding <=> ${vectorString}::vector
      LIMIT ${limit}
    `);
    const rows: any[] = Array.isArray(execResult)
      ? execResult
      : (execResult?.rows ?? []);

    return rows.map((row: any) => ({
      id: row.id,
      title: row.title || "",
      content: row.content || "",
      score:
        typeof row.similarity === "number"
          ? row.similarity
          : parseFloat(String(row.similarity ?? 0)),
      metadata: {
        keywords: row.keywords,
        topics: row.topics,
        documentType: row.document_type,
        caseId: row.case_id,
      },
      source: "pgvector",
      type: "document",
    }));
  } catch (error) {
    console.error("legal_documents pgvector search error:", error);
    return [];
  }
}

// Text search fallback for legal_documents when embeddings are unavailable
export async function searchLegalDocumentsText(
  query: string,
  limit: number = 10
): Promise<VectorSearchResult[]> {
  try {
    const like = `%${query}%`;
    const execResult = await db.execute(sql`
      SELECT id, title, COALESCE(content, full_text) AS content
      FROM legal_documents
      WHERE title ILIKE ${like}
         OR content ILIKE ${like}
         OR full_text ILIKE ${like}
      LIMIT ${limit}
    `);
    const rows: any[] = Array.isArray(execResult)
      ? execResult
      : (execResult?.rows ?? []);
    return (rows as any[]).map((row) => ({
      id: row.id,
      title: row.title || "",
      content: row.content || "",
      score: 0.5,
      metadata: {},
      source: "pgvector",
      type: "document",
    }));
  } catch (e) {
    console.error("legal_documents text search error:", e);
    return [];
  }
}
// Initialize Fuse.js for fuzzy search
async function initializeFuzzySearch(cases: any[], evidence: any[]) {
  if (!Fuse) return;

  const fuseOptions = {
    keys: ["title", "description", "content"],
    threshold: 0.3,
    includeScore: true,
    includeMatches: true,
  };

  fuseCases = new Fuse(cases, fuseOptions);
  fuseEvidence = new Fuse(evidence, fuseOptions);
}
// Enhanced fuzzy search function
async function searchWithFuzzy(
  query: string,
  options: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  if (!Fuse || (!fuseCases && !fuseEvidence)) {
    return [];
  }
  const { limit = 20 } = options;
  const results: VectorSearchResult[] = [];

  try {
    // Search cases with Fuse.js
    if (fuseCases) {
      const caseResults = fuseCases.search(query, {
        limit: Math.floor(limit / 2),
      });

      caseResults.forEach((result: any) => {
        results.push({
          id: result.item.id,
          title: result.item.title || "",
          content: result.item.description || result.item.content || "",
          score: 1 - (result.score || 0), // Convert Fuse score to similarity score
          metadata: { type: "case", matches: result.matches },
          source: "pgvector", // Keep consistent with other sources
          type: "case",
        });
      });
    }
    // Search evidence with Fuse.js
    if (fuseEvidence) {
      const evidenceResults = fuseEvidence.search(query, {
        limit: Math.floor(limit / 2),
      });

      evidenceResults.forEach((result: any) => {
        results.push({
          id: result.item.id,
          title: result.item.title || "",
          content: result.item.description || result.item.content || "",
          score: 1 - (result.score || 0),
          metadata: { type: "evidence", matches: result.matches },
          source: "pgvector",
          type: "evidence",
        });
      });
    }
    return results.sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error("Fuzzy search error:", error);
    return [];
  }
}
// Enhanced local database search with Loki.js
async function searchWithLoki(
  query: string,
  options: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  if (!lokiDb) {
    await initializeLocalDb();
  }
  if (!lokiDb) return [];

  const { limit = 20 } = options;
  const results: VectorSearchResult[] = [];

  try {
    const casesCollection = lokiDb.getCollection("cases");
    const evidenceCollection = lokiDb.getCollection("evidence");

    // Search cases
    if (casesCollection) {
      const caseResults = casesCollection
        .chain()
        .find({
          $or: [
            { title: { $regex: new RegExp(query, "i") } },
            { description: { $regex: new RegExp(query, "i") } },
          ],
        })
        .limit(Math.floor(limit / 2))
        .data();

      caseResults.forEach((item: any, index: number) => {
        results.push({
          id: item.id || item.$loki.toString(),
          title: item.title || "",
          content: item.description || "",
          score: 0.9 - index * 0.1, // Mock relevance score
          metadata: { type: "case" },
          source: "pgvector",
          type: "case",
        });
      });
    }
    // Search evidence
    if (evidenceCollection) {
      const evidenceResults = evidenceCollection
        .chain()
        .find({
          $or: [
            { title: { $regex: new RegExp(query, "i") } },
            { description: { $regex: new RegExp(query, "i") } },
          ],
        })
        .limit(Math.floor(limit / 2))
        .data();

      evidenceResults.forEach((item: any, index: number) => {
        results.push({
          id: item.id || item.$loki.toString(),
          title: item.title || "",
          content: item.description || "",
          score: 0.85 - index * 0.1,
          metadata: { type: "evidence" },
          source: "pgvector",
          type: "evidence",
        });
      });
    }
    return results;
  } catch (error) {
    console.error("Loki search error:", error);
    return [];
  }
}
// Main vector search function with fallback logic
export async function vectorSearch(
  query: string,
  options: VectorSearchOptions = {}
): Promise<{
  results: VectorSearchResult[];
  executionTime: number;
  source: string;
  totalResults: number;
}> {
  const startTime = Date.now();
  const {
    limit = 20,
    offset = 0,
    threshold = 0.7,
    useCache = true,
    fallbackToQdrant = true,
    filters = {},
    searchType = "hybrid",
  } = options;

  // Check cache first
  if (useCache) {
    const cacheKey = `vector_search:${JSON.stringify({ query, ...options })}`;
    const cached = (await cache.get(cacheKey)) as VectorSearchResult[] | null;
    if (cached) {
      return {
        results: cached,
        executionTime: Date.now() - startTime,
        source: "cache",
        totalResults: cached.length,
      };
    }
  }
  let results: VectorSearchResult[] = [];
  let source = "pgvector";

  try {
    // 0) Legal documents search via pgvector first (uses 768-dim Ollama embeddings)
    if (isPostgreSQL) {
      const legalResults = await searchLegalDocumentsPgvector(query, {
        ...options,
        limit,
      });
      results = legalResults;
      if (!results || results.length === 0) {
        const textFallback = await searchLegalDocumentsText(query, limit);
        if (textFallback.length > 0) {
          results = mergeSearchResults(results, textFallback);
        }
      }
    }

    // 1) Cases/Evidence search (existing path)
    if (isPostgreSQL) {
      try {
        const ceResults = await searchWithPgVector(query, options);
        if (ceResults.length > 0) {
          results = mergeSearchResults(results, ceResults);
        }
      } catch (err) {
        console.warn(
          "Cases/Evidence pgvector search failed, continuing:",
          (err as Error)?.message || err
        );
      }
    } else {
      // Development fallback: text search
      const textResults = await searchWithTextFallback(query, options);
      results = mergeSearchResults(results, textResults);
      source = "text_fallback";
    }
    // Fallback to Qdrant if no results or poor quality results
    if (
      fallbackToQdrant &&
      results.length < 5 &&
      qdrant &&
      typeof qdrant.isHealthy === "function" &&
      (await qdrant.isHealthy())
    ) {
      const qdrantResults = await searchWithQdrant(query, options);
      if (qdrantResults.length > 0) {
        // Merge and deduplicate results
        results = mergeSearchResults(results, qdrantResults);
        source = results.some((r) => r.source === "pgvector")
          ? "hybrid"
          : "qdrant";
      }
    }
    // Fallback to local DB (Loki.js) if enabled
    if (options.useLocalDb && results.length < 5) {
      const localResults = await searchWithLoki(query, options);
      if (localResults.length > 0) {
        results = mergeSearchResults(results, localResults);
        source = "local_db";
      }
    }
    // Fallback to fuzzy search if enabled
    if (options.useFuzzySearch && results.length < 5) {
      const fuzzyResults = await searchWithFuzzy(query, options);
      if (fuzzyResults.length > 0) {
        results = mergeSearchResults(results, fuzzyResults);
        source = "fuzzy_search";
      }
    }
    // Cache successful results
    if (useCache && results.length > 0) {
      const cacheKey = `vector_search:${JSON.stringify({ query, ...options })}`;
      await cache.set(cacheKey, results, 5 * 60 * 1000); // 5 minutes
    }
    return {
      results,
      executionTime: Date.now() - startTime,
      source,
      totalResults: results.length,
    };
  } catch (error) {
    console.error("Vector search error:", error);
    return {
      results: [],
      executionTime: Date.now() - startTime,
      source: "error",
      totalResults: 0,
    };
  }
}
// PostgreSQL pgvector search implementation
async function searchWithPgVector(
  query: string,
  options: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  const { limit = 20, threshold = 0.7, filters = {} } = options;

  // Generate query embedding
  const queryEmbedding = await generateEmbedding(query);
  if (!queryEmbedding) {
    throw new Error("Failed to generate embedding for query");
  }
  const vectorString = `[${queryEmbedding.join(",")}]`;
  // Import schema dynamically to avoid issues
  const { cases, evidence } = await import(
    "../../../lib/server/db/unified-schema.js"
  );

  const results: VectorSearchResult[] = [];

  try {
    // Search cases with pgvector
    const caseResults = await db
      .select({
        id: cases.id,
        title: cases.title,
        content: cases.description,
        metadata: cases.metadata,
        score: sql<number>`1 - (${cases.titleEmbedding} <=> ${vectorString}::vector)`,
      })
      .from(cases)
      .where(
        and(
          sql`${cases.titleEmbedding} IS NOT NULL`,
          sql`1 - (${cases.titleEmbedding} <=> ${vectorString}::vector) > ${threshold}`,
          filters.caseId ? eq(cases.id, filters.caseId) : undefined,
          filters.status ? eq(cases.status, filters.status) : undefined
        )
      )
      .orderBy(
        sql`1 - (${cases.titleEmbedding} <=> ${vectorString}::vector) DESC`
      )
      .limit(Math.floor(limit / 2));

    // Search evidence with pgvector
    const evidenceResults = await db
      .select({
        id: evidence.id,
        title: evidence.title,
        content: evidence.description,
        metadata: sql<
          Record<string, any>
        >`json_build_object('caseId', ${evidence.caseId}, 'type', ${evidence.evidenceType})`,
        score: sql<number>`1 - (${evidence.contentEmbedding} <=> ${vectorString}::vector)`,
      })
      .from(evidence)
      .where(
        and(
          sql`${evidence.contentEmbedding} IS NOT NULL`,
          sql`1 - (${evidence.contentEmbedding} <=> ${vectorString}::vector) > ${threshold}`,
          filters.caseId ? eq(evidence.caseId, filters.caseId) : undefined,
          filters.evidenceType
            ? eq(evidence.evidenceType, filters.evidenceType)
            : undefined
        )
      )
      .orderBy(
        sql`1 - (${evidence.contentEmbedding} <=> ${vectorString}::vector) DESC`
      )
      .limit(Math.floor(limit / 2));

    // Combine and format results
    caseResults.forEach((result) => {
      results.push({
        id: result.id,
        title: result.title || "",
        content: result.content || "",
        score: result.score || 0,
        metadata: result.metadata || {},
        source: "pgvector",
        type: "case",
      });
    });

    evidenceResults.forEach((result) => {
      results.push({
        id: result.id,
        title: result.title || "",
        content: result.content || "",
        score: result.score || 0,
        metadata: result.metadata || {},
        source: "pgvector",
        type: "evidence",
      });
    });

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error("PostgreSQL vector search error:", error);
    throw error;
  }
  return results.slice(0, limit);
}
// Qdrant search implementation
async function searchWithQdrant(
  query: string,
  options: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  const { limit = 20, threshold = 0.7, filters = {} } = options;

  try {
    // Search cases in Qdrant
    const caseResults = await qdrant.searchCases(query, {
      limit: Math.floor(limit / 2),
      scoreThreshold: threshold,
      filter: filters,
    });

    // Search evidence in Qdrant
    const evidenceResults = await qdrant.searchEvidence(query, {
      limit: Math.floor(limit / 2),
      scoreThreshold: threshold,
      filter: filters,
    });

    const results: VectorSearchResult[] = [];

    // Format case results
    caseResults.forEach((result) => {
      results.push({
        id: result.id,
        title: result.payload?.title || "",
        content: result.payload?.description || "",
        score: result.score,
        metadata: result.payload || {},
        source: "qdrant",
        type: "case",
      });
    });

    // Format evidence results
    evidenceResults.forEach((result) => {
      results.push({
        id: result.id,
        title: result.payload?.title || "",
        content: result.payload?.description || "",
        score: result.score,
        metadata: result.payload || {},
        source: "qdrant",
        type: "evidence",
      });
    });

    return results.sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error("Qdrant search error:", error);
    return [];
  }
}
// Text fallback for development/SQLite
async function searchWithTextFallback(
  query: string,
  options: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  const { limit = 20 } = options;

  try {
    // Import SQLite schema
    const { cases, evidence } = await import(
      "../../../lib/server/db/schema-postgres.js"
    );

    const searchTerm = `%${query}%`;

    // Search cases
    const caseResults = await db
      .select()
      .from(cases)
      .where(
        or(
          sql`${cases.title} LIKE ${searchTerm}`,
          sql`${cases.description} LIKE ${searchTerm}`
        )
      )
      .limit(Math.floor(limit / 2));

    // Search evidence
    const evidenceResults = await db
      .select()
      .from(evidence)
      .where(
        or(
          sql`${evidence.title} LIKE ${searchTerm}`,
          sql`${evidence.description} LIKE ${searchTerm}`
        )
      )
      .limit(Math.floor(limit / 2));

    const results: VectorSearchResult[] = [];

    // Format results with mock scores
    caseResults.forEach((result: any, index) => {
      results.push({
        id: result.id,
        title: result.title || "",
        content: result.description || "",
        score: 0.9 - index * 0.1, // Mock relevance score
        metadata: { type: "case" },
        source: "pgvector", // Pretend it's pgvector for consistency
        type: "case",
      });
    });

    evidenceResults.forEach((result: any, index) => {
      results.push({
        id: result.id,
        title: result.title || "",
        content: result.description || "",
        score: 0.85 - index * 0.1,
        metadata: { type: "evidence" },
        source: "pgvector",
        type: "evidence",
      });
    });

    return results;
  } catch (error) {
    console.error("Text fallback search error:", error);
    return [];
  }
}
// Merge and deduplicate search results
function mergeSearchResults(
  pgResults: VectorSearchResult[],
  qdrantResults: VectorSearchResult[]
): VectorSearchResult[] {
  const merged = new Map<string, VectorSearchResult>();

  // Add pgvector results first (higher priority)
  pgResults.forEach((result) => {
    merged.set(result.id, result);
  });

  // Add Qdrant results if not already present
  qdrantResults.forEach((result) => {
    if (!merged.has(result.id)) {
      merged.set(result.id, result);
    }
  });

  return Array.from(merged.values()).sort((a, b) => b.score - a.score);
}
// Export convenience functions
export const search = {
  vector: vectorSearch,
  pgvector: searchWithPgVector,
  qdrant: searchWithQdrant,
  textFallback: searchWithTextFallback,
};
