// TODO: Integrate QdrantService with Context7 audit/agent pipeline
// - Use this service for vector search in semantic_search-driven audit
// - Expose helper to fetch similar legal documents for audit/agent flows
// - After audit, log Qdrant search results to todo log and trigger agent actions as needed

/**
 * Example: Fetch similar documents for audit/agent pipeline
 * Usage: const similar = await qdrantService.findSimilarForAudit(vector, 5);
 */

/**
 * Find similar documents and log results for audit/agent pipeline.
 * Optionally triggers agent actions if similar docs found.
 */
export async function findSimilarForAudit(
  vector: number[],
  limit = 5,
  triggerAgent = false
) {
  const similar = await qdrantService.searchSimilar(vector, limit);
  // Log results to console (replace with file/db logging as needed)
  console.log("[Qdrant Audit] Similar documents:", similar);
  // TODO: Write to phase10-todo.log or DB
  if (triggerAgent && similar.length > 0) {
    // Example: trigger agent action for each similar doc (stub)
    for (const doc of similar) {
      // TODO: Replace with real agent trigger (CrewAI/Autogen, Context7)
      console.log(`[Agent Trigger] Would trigger agent for doc: ${doc.id}`);
    }
  }
  return similar;
}

// TODO: After initial test, connect this to /api/audit/semantic and agent integration for live pipeline validation
// Qdrant Service for Legal Document Vector Operations
import { QdrantClient } from "@qdrant/js-client-rest";

export interface LegalDocumentMetadata {
  documentId: string;
  filename: string;
  documentType: string;
  uploadedBy: string;
  uploadedAt: Date | string;
  jurisdiction?: string;
  practiceArea?: string;
  classification?: {
    documentType: string;
    practiceArea: string;
    jurisdiction: string;
    confidentialityLevel: string;
    tags: string[];
  };
  extractedData?: {
    parties?: string[];
    dates?: string[];
    amounts?: string[];
    legalCitations?: string[];
    keyTerms?: string[];
  };
  fileMetadata: {
    size: number;
    mimeType: string;
    pageCount?: number;
    wordCount?: number;
    language?: string;
  };
  [key: string]: unknown;
}

export interface QdrantServiceConfig {
  url: string;
  collectionName: string;
  vectorSize: number;
  apiKey?: string;
}

export class QdrantService {
  private client: QdrantClient;
  private collectionName: string;
  private vectorSize: number;

  constructor(config: QdrantServiceConfig) {
    this.client = new QdrantClient({
      url: config.url,
      apiKey: config.apiKey,
    });
    this.collectionName = config.collectionName;
    this.vectorSize = config.vectorSize;
  }

  async ensureCollection(): Promise<void> {
    try {
      await this.client.getCollection(this.collectionName);
    } catch (error) {
      // Collection doesn't exist, create it
      await this.client.createCollection(this.collectionName, {
        vectors: {
          size: this.vectorSize,
          distance: "Cosine",
        },
      });
    }
  }

  async upsertPoints(
    points: Array<{
      id: string;
      vector: number[];
      payload: LegalDocumentMetadata;
    }>
  ): Promise<void> {
    await this.ensureCollection();
    await this.client.upsert(this.collectionName, {
      wait: true,
      points: points,
    });
  }

  async searchSimilar(
    vector: number[],
    limit: number = 10,
    filter?: Record<string, any>
  ): Promise<
    Array<{ id: string; score: number; payload: LegalDocumentMetadata }>
  > {
    await this.ensureCollection();

    const searchResult = await this.client.search(this.collectionName, {
      vector,
      limit,
      filter,
      with_payload: true,
      score_threshold: 0.5,
    });

    return searchResult.map((result) => ({
      id: result.id as string,
      score: result.score,
      payload: result.payload as LegalDocumentMetadata,
    }));
  }

  async deletePoints(ids: string[]): Promise<void> {
    await this.client.delete(this.collectionName, {
      wait: true,
      points: ids,
    });
  }

  async getCollectionInfo() {
    try {
      return await this.client.getCollection(this.collectionName);
    } catch (error) {
      return null;
    }
  }
}

// Export singleton instance
export const qdrantService = new QdrantService({
  url: process.env.QDRANT_URL || "http://localhost:6333",
  collectionName: "legal_documents",
  vectorSize: 768,
  apiKey: process.env.QDRANT_API_KEY,
});
