// @ts-nocheck
// --- Qdrant passthroughs for admin API ---
export async function getCollections() {
  const client = getQdrantClient();
  if (!client) throw new Error("Qdrant not configured");
  return client.getCollections();
}

export async function getCollection(collection: string) {
  const client = getQdrantClient();
  if (!client) throw new Error("Qdrant not configured");
  return client.getCollection(collection);
}

export async function createCollection(name: string, config: any) {
  const client = getQdrantClient();
  if (!client) throw new Error("Qdrant not configured");
  return client.createCollection(name, config);
}

export async function deleteCollection(name: string) {
  const client = getQdrantClient();
  if (!client) throw new Error("Qdrant not configured");
  return client.deleteCollection(name);
}
// Qdrant vector search service
// High-performance vector search with memory optimization
import { QdrantClient } from "@qdrant/js-client-rest";
// Use process.env instead of SvelteKit env for server-side code
// import { env } from "$env/dynamic/private";
import { generateEmbedding } from "../ai/embeddings-simple";

let qdrantClient: QdrantClient | null = null;

// Initialize Qdrant client
function getQdrantClient(): QdrantClient | null {
  if (!process.env.QDRANT_URL) {
    return null;
  }
  if (!qdrantClient) {
    qdrantClient = new QdrantClient({
      url: process.env.QDRANT_URL,
      apiKey: process.env.QDRANT_API_KEY || undefined,
    });
  }
  return qdrantClient;
}
// Collection names
const COLLECTIONS = {
  CASES: "prosecutor_cases",
  EVIDENCE: "prosecutor_evidence",
  DOCUMENTS: "prosecutor_documents",
} as const;

// Vector dimensions for text-embedding-3-small
const VECTOR_DIMENSION = 1536;

// Initialize collections if they don't exist
export async function initializeCollections(): Promise<void> {
  const client = getQdrantClient();
  if (!client) {
    console.log("Qdrant not configured, skipping collection initialization");
    return;
  }
  try {
    for (const [name, collectionName] of Object.entries(COLLECTIONS)) {
      try {
        // Check if collection exists
        await client.getCollection(collectionName);
        console.log(`✅ Qdrant collection ${collectionName} already exists`);
      } catch (error) {
        // Collection doesn't exist, create it
        await client.createCollection(collectionName, {
          vectors: {
            size: VECTOR_DIMENSION,
            distance: "Cosine",
            hnsw_config: {
              m: 16,
              ef_construct: 100,
              full_scan_threshold: 10000,
              max_indexing_threads: 1, // Memory optimization
            },
          },
          optimizers_config: {
            default_segment_number: 2,
            max_segment_size: 20000,
            indexing_threshold: 20000,
            flush_interval_sec: 5,
            max_optimization_threads: 1, // Memory optimization
          },
          quantization_config: {
            scalar: {
              type: "int8",
              quantile: 0.99,
            },
          },
        });
        console.log(`✅ Created Qdrant collection ${collectionName}`);
      }
    }
  } catch (error) {
    console.error("Failed to initialize Qdrant collections:", error);
  }
}
// Search interface
interface SearchOptions {
  limit?: number;
  offset?: number;
  filter?: any;
  withPayload?: boolean;
  scoreThreshold?: number;
}
// Search cases in Qdrant
export async function searchCases(
  query: string,
  options: SearchOptions = {}
): Promise<any[]> {
  const client = getQdrantClient();
  if (!client) {
    return [];
  }
  try {
    const queryVector = await generateEmbedding(query);
    if (!queryVector) {
      return [];
    }
    const result = await client.search(COLLECTIONS.CASES, {
      vector: queryVector,
      limit: options.limit || 20,
      offset: options.offset || 0,
      filter: options.filter,
      with_payload: options.withPayload !== false,
      score_threshold: options.scoreThreshold || 0.7,
    });

    return result.map((hit) => ({
      id: hit.id,
      score: hit.score,
      payload: hit.payload,
    }));
  } catch (error) {
    console.error("Qdrant case search failed:", error);
    return [];
  }
}
// Search evidence in Qdrant
export async function searchEvidence(
  query: string,
  options: SearchOptions = {}
): Promise<any[]> {
  const client = getQdrantClient();
  if (!client) {
    return [];
  }
  try {
    const queryVector = await generateEmbedding(query);
    if (!queryVector) {
      return [];
    }
    const result = await client.search(COLLECTIONS.EVIDENCE, {
      vector: queryVector,
      limit: options.limit || 20,
      offset: options.offset || 0,
      filter: options.filter,
      with_payload: options.withPayload !== false,
      score_threshold: options.scoreThreshold || 0.7,
    });

    return result.map((hit) => ({
      id: hit.id,
      score: hit.score,
      payload: hit.payload,
    }));
  } catch (error) {
    console.error("Qdrant evidence search failed:", error);
    return [];
  }
}
// Add or update a case in Qdrant
export async function upsertCase(
  id: string,
  embedding: number[],
  payload: any
): Promise<void> {
  const client = getQdrantClient();
  if (!client) {
    return;
  }
  try {
    await client.upsert(COLLECTIONS.CASES, {
      wait: true,
      points: [
        {
          id,
          vector: embedding,
          payload,
        },
      ],
    });
  } catch (error) {
    console.error("Failed to upsert case in Qdrant:", error);
  }
}
// Add or update evidence in Qdrant
export async function upsertEvidence(
  id: string,
  embedding: number[],
  payload: any
): Promise<void> {
  const client = getQdrantClient();
  if (!client) {
    return;
  }
  try {
    await client.upsert(COLLECTIONS.EVIDENCE, {
      wait: true,
      points: [
        {
          id,
          vector: embedding,
          payload,
        },
      ],
    });
  } catch (error) {
    console.error("Failed to upsert evidence in Qdrant:", error);
  }
}
// Delete a point from Qdrant
export async function deletePoint(
  collection: string,
  id: string
): Promise<void> {
  const client = getQdrantClient();
  if (!client) {
    return;
  }
  try {
    await client.delete(collection, {
      wait: true,
      points: [id],
    });
  } catch (error) {
    console.error("Failed to delete point from Qdrant:", error);
  }
}
// Health check
export async function isQdrantHealthy(): Promise<boolean> {
  const client = getQdrantClient();
  if (!client) {
    return false;
  }
  try {
    await client.getCollections();
    return true;
  } catch (error) {
    return false;
  }
}
// Export for easier usage
export const qdrant = {
  searchCases,
  searchEvidence,
  upsertCase,
  upsertEvidence,
  deletePoint,
  isHealthy: isQdrantHealthy,
  initializeCollections,
  collections: COLLECTIONS,
  getCollections,
  getCollection,
  createCollection,
  deleteCollection,
};
