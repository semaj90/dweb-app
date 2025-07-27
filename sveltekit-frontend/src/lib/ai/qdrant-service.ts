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
  ): Promise<Array<{ id: string; score: number; payload: LegalDocumentMetadata }>> {
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