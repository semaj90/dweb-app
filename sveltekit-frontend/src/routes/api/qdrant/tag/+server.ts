
import { json } from "@sveltejs/kit";
import { URL } from "url";
import type { RequestHandler } from './$types';


// Mock Qdrant client for development
class MockQdrantClient {
  private collections = new Map();

  async createCollection(name: string, config: any) {
    this.collections.set(name, { name, config, points: [] });
    return { status: "ok" };
  }
  async getCollections() {
    return { collections: Array.from(this.collections.values()) };
  }
  async upsert(collection: string, data: any) {
    const coll = this.collections.get(collection);
    if (!coll) throw new Error("Collection not found");

    // Mock upsert
    data.points.forEach((point: any) => {
      const existingIndex = coll.points.findIndex(
        (p: any) => p.id === point.id
      );
      if (existingIndex >= 0) {
        coll.points[existingIndex] = point;
      } else {
        coll.points.push(point);
      }
    });

    return { operation_id: "mock-op", status: "completed" };
  }
  async search(collection: string, query: any) {
    const coll = this.collections.get(collection);
    if (!coll) return [];

    // Mock search - return first few points
    return coll.points
      .slice(0, query.limit || 10)
      .map((point: any, index: number) => ({
        id: point.id,
        payload: point.payload,
        score: 0.9 - index * 0.1,
      }));
  }
  async delete(collection: string, filter: any) {
    const coll = this.collections.get(collection);
    if (!coll) return { status: "ok" };

    // Mock delete
    return { status: "ok" };
  }
  async retrieve(collection: string, query: any) {
    const coll = this.collections.get(collection);
    if (!coll) return [];

    return coll.points.filter((point: any) => query.ids.includes(point.id));
  }
  async scroll(collection: string, query: any) {
    const coll = this.collections.get(collection);
    if (!coll) return { points: [] };

    return { points: coll.points.slice(0, query.limit || 100) };
  }
  async count(collection: string, filter: any) {
    const coll = this.collections.get(collection);
    return { count: coll?.points?.length || 0 };
  }
  async getCollection(collection: string) {
    const coll = this.collections.get(collection);
    return {
      points_count: coll?.points?.length || 0,
      config: coll?.config || {},
      status: "green",
    };
  }
  async createPayloadIndex(collection: string, config: any) {
    return { status: "ok" };
  }
}
// Use mock client for development
const qdrantClient = new MockQdrantClient();

// Collection configuration
const COLLECTIONS = {
  EVIDENCE: "legal_evidence",
  TAGS: "evidence_tags",
  EMBEDDINGS: "document_embeddings",
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const session = locals.session;
    if (!session) {
      return json({ error: "Authentication required" }, { status: 401 });
    }
    const sessionId = typeof session === 'string' ? session : (session as any)?.id;
    const body = await request.json();
    const { action = "tag", ...data } = body;

    // Initialize collections if they don't exist
    await initializeCollections();

    switch (action) {
      case "tag":
        return await tagDocument(data, sessionId);

      case "search":
        return await searchDocuments(data, sessionId);

      case "batch_tag":
        return await batchTagDocuments(data, sessionId);

      case "update_tags":
        return await updateDocumentTags(data, sessionId);

      case "delete":
        return await deleteDocument(data, sessionId);

      case "get_similar":
        return await getSimilarDocuments(data, sessionId);

      case "get_stats":
        return await getCollectionStats(sessionId);

      default:
        return json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error: any) {
    console.error("Qdrant API error:", error);
    return json(
      {
        error: "Qdrant operation failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};

// Initialize Qdrant collections
async function initializeCollections(): Promise<any> {
  try {
    const collections = await qdrantClient.getCollections();
    const existingCollections =
      collections.collections?.map((c: any) => c.name) || [];

    // Create evidence collection
    if (!existingCollections.includes(COLLECTIONS.EVIDENCE)) {
      await qdrantClient.createCollection(COLLECTIONS.EVIDENCE, {
        vectors: {
          size: 1536, // OpenAI/Nomic embedding dimensions
          distance: "Cosine",
        },
        optimizers_config: {
          default_segment_number: 2,
        },
        replication_factor: 1,
      });
    }
    // Create tags collection for semantic tag search
    if (!existingCollections.includes(COLLECTIONS.TAGS)) {
      await qdrantClient.createCollection(COLLECTIONS.TAGS, {
        vectors: {
          size: 384, // Smaller embeddings for tags
          distance: "Cosine",
        },
        optimizers_config: {
          default_segment_number: 1,
        },
        replication_factor: 1,
      });
    }
    // Create document embeddings collection
    if (!existingCollections.includes(COLLECTIONS.EMBEDDINGS)) {
      await qdrantClient.createCollection(COLLECTIONS.EMBEDDINGS, {
        vectors: {
          size: 1536,
          distance: "Cosine",
        },
        optimizers_config: {
          default_segment_number: 4,
          memmap_threshold: 20000,
        },
        replication_factor: 1,
      });
    }
    // Create search indices for better performance
    await createSearchIndices();
  } catch (error: any) {
    console.warn("Collection initialization failed:", error);
  }
}
async function createSearchIndices(): Promise<any> {
  try {
    // Create payload indices for filtering
    const indexFields = [
      "evidenceType",
      "legalRelevance",
      "caseId",
      "userId",
      "tags",
    ];

    for (const collection of Object.values(COLLECTIONS)) {
      for (const field of indexFields) {
        try {
          await qdrantClient.createPayloadIndex(collection, {
            field_name: field,
            field_schema: field === "tags" ? "keyword" : "keyword",
          });
        } catch (error: any) {
          // Index might already exist, continue
        }
      }
    }
  } catch (error: any) {
    console.warn("Index creation failed:", error);
  }
}
// Tag a document with vector embeddings
async function tagDocument(data: any, userId: string): Promise<any> {
  try {
    const { id, vector, payload } = data;

    if (!id || !vector || !payload) {
      return json(
        { error: "Missing required fields: id, vector, payload" },
        { status: 400 }
      );
    }
    const documentPayload = {
      ...payload,
      userId,
      timestamp: new Date().toISOString(),
    };

    // Store in main evidence collection
    const response = await qdrantClient.upsert(COLLECTIONS.EVIDENCE, {
      wait: true,
      points: [
        {
          id,
          vector,
          payload: documentPayload,
        },
      ],
    });

    // Also create individual tag embeddings for semantic tag search
    if (payload.tags && payload.tags.length > 0) {
      await createTagEmbeddings(payload.tags, id, userId);
    }
    return json({
      success: true,
      operation_id: response.operation_id,
      status: response.status,
      message: "Document tagged successfully",
    });
  } catch (error: any) {
    throw error;
  }
}
// Create embeddings for individual tags
async function createTagEmbeddings(
  tags: string[],
  documentId: string,
  userId: string
): Promise<any> {
  try {
    const tagEmbeddings = await Promise.all(
      tags.map(async (tag) => {
        // Generate embedding for tag using Ollama
        try {
          const embeddingResponse = await fetch(
            "http://localhost:11434/api/embeddings",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model: "nomic-embed-text",
                prompt: tag,
              }),
            }
          );

          if (embeddingResponse.ok) {
            const embeddingData = await embeddingResponse.json();
            return {
              id: `${documentId}_tag_${tag.replace(/\s+/g, "_").toLowerCase()}`,
              vector: embeddingData.embedding.slice(0, 384), // Truncate for tag collection
              payload: {
                tag,
                documentId,
                userId,
                timestamp: new Date().toISOString(),
              },
            };
          }
        } catch (error: any) {
          console.warn(`Tag embedding failed for: ${tag}`, error);
        }
        return null;
      })
    );

    const validEmbeddings = tagEmbeddings.filter(Boolean);

    if (validEmbeddings.length > 0) {
      await qdrantClient.upsert(COLLECTIONS.TAGS, {
        wait: false,
        points: validEmbeddings,
      });
    }
  } catch (error: any) {
    console.warn("Tag embedding creation failed:", error);
  }
}
// Search documents using vector similarity
async function searchDocuments(data: any, userId: string): Promise<any> {
  try {
    const { vector, text, filters = {}, limit = 20, threshold = 0.7 } = data;

    let queryVector = vector;

    // If text query provided, generate embedding
    if (text && !queryVector) {
      queryVector = await generateTextEmbedding(text);
    }
    if (!queryVector) {
      return json(
        { error: "No query vector or text provided" },
        { status: 400 }
      );
    }
    // Build filter conditions
    const mustConditions = [{ key: "userId", match: { value: userId } }];

    if (filters.evidenceType?.length) {
      mustConditions.push({
        key: "evidenceType",
        match: { value: filters.evidenceType },
      });
    }
    if (filters.legalRelevance?.length) {
      mustConditions.push({
        key: "legalRelevance",
        match: { value: filters.legalRelevance },
      });
    }
    if (filters.caseId) {
      mustConditions.push({
        key: "caseId",
        match: { value: filters.caseId },
      });
    }
    if (filters.tags?.length) {
      mustConditions.push({
        key: "tags",
        match: { value: filters.tags },
      });
    }
    // Perform vector search
    const searchResult = await qdrantClient.search(COLLECTIONS.EVIDENCE, {
      vector: queryVector,
      filter: mustConditions.length > 0 ? { must: mustConditions } : undefined,
      limit,
      score_threshold: threshold,
      with_payload: true,
      with_vector: false,
    });

    // Also search tags for semantic tag suggestions
    const tagSearchResults = await qdrantClient.search(COLLECTIONS.TAGS, {
      vector: queryVector.slice(0, 384), // Truncate for tag collection
      filter: { must: [{ key: "userId", match: { value: userId } }] },
      limit: 10,
      score_threshold: 0.6,
      with_payload: true,
      with_vector: false,
    });

    return json({
      success: true,
      results: searchResult.map((result: any) => ({
        id: result.id,
        score: result.score,
        payload: result.payload,
      })),
      relatedTags: tagSearchResults.map((result: any) => ({
        tag: result.payload?.tag,
        score: result.score,
        documentId: result.payload?.documentId,
      })),
      searchStats: {
        totalResults: searchResult.length,
        maxScore: searchResult[0]?.score || 0,
        threshold,
      },
    });
  } catch (error: any) {
    throw error;
  }
}
// Batch tag multiple documents
async function batchTagDocuments(data: any, userId: string): Promise<any> {
  try {
    const { documents } = data;

    if (!Array.isArray(documents) || documents.length === 0) {
      return json({ error: "No documents provided" }, { status: 400 });
    }
    const results = [];
    const errors = [];

    // Process in batches to avoid overwhelming Qdrant
    const batchSize = 50;
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);

      try {
        const points = batch.map((doc: any) => ({
          id: doc.id,
          vector: doc.vector,
          payload: {
            ...doc.payload,
            userId,
            timestamp: new Date().toISOString(),
          },
        }));

        const response = await qdrantClient.upsert(COLLECTIONS.EVIDENCE, {
          wait: true,
          points,
        });

        results.push({
          batchIndex: Math.floor(i / batchSize),
          count: batch.length,
          operation_id: response.operation_id,
          status: response.status,
        });
      } catch (error: any) {
        errors.push({
          batchIndex: Math.floor(i / batchSize),
          error: error instanceof Error ? error.message : "Unknown error",
        });
      }
    }
    return json({
      success: true,
      processed: results.length * batchSize,
      successful: results.length,
      failed: errors.length,
      results,
      errors,
    });
  } catch (error: any) {
    throw error;
  }
}
// Update document tags
async function updateDocumentTags(data: any, userId: string): Promise<any> {
  try {
    const { documentId, tags, vector } = data;

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }
    // Get existing document
    const existing = await qdrantClient.retrieve(COLLECTIONS.EVIDENCE, {
      ids: [documentId],
      with_payload: true,
    });

    if (existing.length === 0) {
      return json({ error: "Document not found" }, { status: 404 });
    }
    // Update payload with new tags
    const updatedPayload = {
      ...existing[0].payload,
      tags: tags || existing[0].payload?.tags,
      userId,
      timestamp: new Date().toISOString(),
    };

    // Update document
    await qdrantClient.upsert(COLLECTIONS.EVIDENCE, {
      wait: true,
      points: [
        {
          id: documentId,
          vector: vector || existing[0].vector,
          payload: updatedPayload,
        },
      ],
    });

    // Update tag embeddings
    if (tags) {
      // Delete old tag embeddings
      await qdrantClient.delete(COLLECTIONS.TAGS, {
        wait: true,
        filter: {
          must: [
            { key: "documentId", match: { value: documentId } },
            { key: "userId", match: { value: userId } },
          ],
        },
      });

      // Create new tag embeddings
      await createTagEmbeddings(tags, documentId, userId);
    }
    return json({
      success: true,
      message: "Document tags updated successfully",
    });
  } catch (error: any) {
    throw error;
  }
}
// Delete document and associated tags
async function deleteDocument(data: any, userId: string): Promise<any> {
  try {
    const { documentId } = data;

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }
    // Delete from evidence collection
    await qdrantClient.delete(COLLECTIONS.EVIDENCE, {
      wait: true,
      filter: {
        must: [{ key: "userId", match: { value: userId } }],
      },
      points: [documentId],
    });

    // Delete associated tags
    await qdrantClient.delete(COLLECTIONS.TAGS, {
      wait: true,
      filter: {
        must: [
          { key: "documentId", match: { value: documentId } },
          { key: "userId", match: { value: userId } },
        ],
      },
    });

    return json({
      success: true,
      message: "Document deleted successfully",
    });
  } catch (error: any) {
    throw error;
  }
}
// Get similar documents
async function getSimilarDocuments(data: any, userId: string): Promise<any> {
  try {
    const { documentId, limit = 10 } = data;

    if (!documentId) {
      return json({ error: "Document ID required" }, { status: 400 });
    }
    // Get the source document vector
    const sourceDoc = await qdrantClient.retrieve(COLLECTIONS.EVIDENCE, {
      ids: [documentId],
      with_vector: true,
      with_payload: true,
    });

    if (sourceDoc.length === 0) {
      return json({ error: "Source document not found" }, { status: 404 });
    }
    // Search for similar documents
    const similarDocs = await qdrantClient.search(COLLECTIONS.EVIDENCE, {
      vector: sourceDoc[0].vector,
      filter: {
        must: [{ key: "userId", match: { value: userId } }],
        must_not: [{ key: "evidenceId", match: { value: documentId } }],
      },
      limit,
      score_threshold: 0.5,
      with_payload: true,
      with_vector: false,
    });

    return json({
      success: true,
      sourceDocument: {
        id: sourceDoc[0].id,
        payload: sourceDoc[0].payload,
      },
      similarDocuments: similarDocs.map((doc: any) => ({
        id: doc.id,
        score: doc.score,
        payload: doc.payload,
      })),
    });
  } catch (error: any) {
    throw error;
  }
}
// Get collection statistics
async function getCollectionStats(userId: string): Promise<any> {
  try {
    const stats: any = {};

    for (const [name, collection] of Object.entries(COLLECTIONS)) {
      try {
        const info = await qdrantClient.getCollection(collection);
        const userDocCount = await qdrantClient.count(collection, {
          filter: {
            must: [{ key: "userId", match: { value: userId } }],
          },
        });

        stats[name] = {
          totalDocuments: info.points_count || 0,
          userDocuments: userDocCount.count || 0,
          vectorSize: info.config?.params?.vectors?.size || 0,
          status: info.status || "unknown",
        };
      } catch (error: any) {
        stats[name] = { error: "Failed to get stats" };
      }
    }
    return json({
      success: true,
      collections: stats,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    throw error;
  }
}
// Helper function to generate text embedding
async function generateTextEmbedding(text: string): Promise<number[]> {
  try {
    const response = await fetch("http://localhost:11434/api/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "nomic-embed-text",
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error("Embedding generation failed");
    }
    const data = await response.json();
    return data.embedding;
  } catch (error: any) {
    console.error("Text embedding generation failed:", error);
    throw new Error("Failed to generate text embedding");
  }
}
// GET endpoint for retrieving documents
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const session = locals.session;
    if (!session) {
      return json({ error: "Authentication required" }, { status: 401 });
    }
    const sessionId = typeof session === 'string' ? session : (session as any)?.id;
    const action = url.searchParams.get("action");
    const documentId = url.searchParams.get("documentId");
    const caseId = url.searchParams.get("caseId");
    const limit = parseInt(url.searchParams.get("limit") || "20");

    switch (action) {
      case "get_document":
        if (!documentId) {
          return json({ error: "Document ID required" }, { status: 400 });
        }
        return await getDocument(documentId, sessionId);

      case "list_documents":
        return await listDocuments(sessionId, { caseId, limit });

      case "get_tags":
        return await getUserTags(sessionId);

      case "health":
        return await getHealthStatus();

      default:
        return json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error: any) {
    console.error("Qdrant GET error:", error);
    return json(
      {
        error: "Failed to retrieve data",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};

async function getDocument(documentId: string, userId: string): Promise<any> {
  const document = await qdrantClient.retrieve(COLLECTIONS.EVIDENCE, {
    ids: [documentId],
    with_payload: true,
    with_vector: false,
  });

  if (document.length === 0 || document[0].payload?.userId !== userId) {
    return json({ error: "Document not found" }, { status: 404 });
  }
  return json({
    success: true,
    document: {
      id: document[0].id,
      payload: document[0].payload,
    },
  });
}
async function listDocuments(
  userId: string,
  options: { caseId?: string; limit: number }
): Promise<any> {
  const filter = {
    must: [{ key: "userId", match: { value: userId } }],
  };

  if (options.caseId) {
    filter.must.push({ key: "caseId", match: { value: options.caseId } });
  }
  const documents = await qdrantClient.scroll(COLLECTIONS.EVIDENCE, {
    filter,
    limit: options.limit,
    with_payload: true,
    with_vector: false,
  });

  return json({
    success: true,
    documents:
      documents.points?.map((doc: any) => ({
        id: doc.id,
        payload: doc.payload,
      })) || [],
  });
}
async function getUserTags(userId: string): Promise<any> {
  const tags = await qdrantClient.scroll(COLLECTIONS.TAGS, {
    filter: {
      must: [{ key: "userId", match: { value: userId } }],
    },
    limit: 1000,
    with_payload: true,
    with_vector: false,
  });

  // Group tags by frequency
  const tagCounts = new Map<string, number>();
  tags.points?.forEach((point: any) => {
    const tag = point.payload?.tag;
    if (tag) {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    }
  });

  const sortedTags = Array.from(tagCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([tag, count]) => ({ tag, count }));

  return json({
    success: true,
    tags: sortedTags,
  });
}
async function getHealthStatus(): Promise<any> {
  try {
    const health = await qdrantClient.getCollections();
    return json({
      success: true,
      status: "healthy",
      collections: health.collections?.length || 0,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json({
      success: false,
      status: "unhealthy",
      error: error instanceof Error ? error.message : "Unknown error",
      timestamp: new Date().toISOString(),
    });
  }
}
