import { Redis } from "ioredis";
import type { OCRResult } from "./ocr-processor";

export interface EmbeddingVector {
  id: string;
  vector: number[];
  metadata: {
    document_id: string;
    case_id: string;
    content_type: "case" | "document" | "evidence" | "precedent";
    text_chunk: string;
    confidence: number;
    timestamp: Date;
    jurisdiction?: string;
    case_type?: string;
    user_id?: string;
  };
}

export interface SearchResult {
  id: string;
  score: number;
  metadata: EmbeddingVector["metadata"];
  highlights: string[];
  legal_relevance_score: number;
  prosecution_score: number;
  final_rank: number;
}

export interface RankingWeights {
  similarity_weight: number;
  legal_relevance_weight: number;
  prosecution_weight: number;
  recency_weight: number;
  confidence_weight: number;
  jurisdiction_boost: number;
}

export class EnhancedVectorEmbeddingService {
  private redis: Redis;
  private embeddingModel: string = "nomic-embed-text";
  private vectorDimension: number = 768;
  private chunkSize: number = 512;
  private chunkOverlap: number = 50;

  private defaultRankingWeights: RankingWeights = {
    similarity_weight: 0.35,
    legal_relevance_weight: 0.25,
    prosecution_weight: 0.2,
    recency_weight: 0.1,
    confidence_weight: 0.05,
    jurisdiction_boost: 0.05,
  };

  constructor() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || "localhost",
      port: parseInt(process.env.REDIS_PORT || "6379"),
      password: process.env.REDIS_PASSWORD || undefined,
      db: 0,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
    } as any); // Type assertion to avoid strict Redis options check

    this.initializeService();
  }

  private async initializeService(): Promise<void> {
    try {
      await this.redis.ping();
      console.log("✅ Redis connected for vector embeddings");

      // Initialize vector indexes if they don't exist
      await this.createVectorIndexes();
    } catch (error) {
      console.error("❌ Failed to initialize vector embedding service:", error);
    }
  }

  private async createVectorIndexes(): Promise<void> {
    try {
      // Create vector index for legal documents
      const indexExists = await this.redis.call("FT._LIST");

      if (
        !Array.isArray(indexExists) ||
        !indexExists.includes("legal_vectors")
      ) {
        await this.redis.call(
          "FT.CREATE",
          "legal_vectors",
          "ON",
          "HASH",
          "PREFIX",
          "1",
          "vec:",
          "SCHEMA",
          "vector",
          "VECTOR",
          "FLAT",
          "6",
          "TYPE",
          "FLOAT32",
          "DIM",
          this.vectorDimension.toString(),
          "DISTANCE_METRIC",
          "COSINE",
          "case_id",
          "TEXT",
          "content_type",
          "TEXT",
          "jurisdiction",
          "TEXT",
          "case_type",
          "TEXT",
          "confidence",
          "NUMERIC",
          "timestamp",
          "NUMERIC"
        );
        console.log("✅ Created legal_vectors index");
      }
    } catch (error) {
      console.warn(
        "⚠️ Vector index creation failed (may already exist):",
        error
      );
    }
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch("http://localhost:11434/api/embeddings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: this.embeddingModel,
          prompt: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const data = await response.json();
      return data.embedding;
    } catch (error) {
      console.error("❌ Embedding generation failed:", error);
      throw error;
    }
  }

  private chunkText(text: string): string[] {
    const chunks: string[] = [];
    const words = text.split(/\\s+/);

    for (let i = 0; i < words.length; i += this.chunkSize - this.chunkOverlap) {
      const chunk = words.slice(i, i + this.chunkSize).join(" ");
      if (chunk.trim().length > 0) {
        chunks.push(chunk.trim());
      }
    }

    return chunks;
  }

  async processAndStoreDocument(
    documentId: string,
    caseId: string,
    ocrResult: OCRResult,
    contentType: EmbeddingVector["metadata"]["content_type"] = "document",
    userId?: string
  ): Promise<string[]> {
    const storedIds: string[] = [];

    try {
      console.log(`📄 Processing document ${documentId} for embeddings...`);

      // Chunk the document text
      const textChunks = this.chunkText(ocrResult.text);

      for (let i = 0; i < textChunks.length; i++) {
        const chunk = textChunks[i];
        const vectorId = `${documentId}_chunk_${i}`;

        // Generate embedding
        const embedding = await this.generateEmbedding(chunk);

        // Calculate legal relevance score
        const legalRelevanceScore = this.calculateLegalRelevanceScore(chunk);

        // Create embedding vector
        const embeddingVector: EmbeddingVector = {
          id: vectorId,
          vector: embedding,
          metadata: {
            document_id: documentId,
            case_id: caseId,
            content_type: contentType,
            text_chunk: chunk,
            confidence: ocrResult.confidence,
            timestamp: new Date(),
            user_id: userId,
          },
        };

        // Store in Redis
        await this.storeEmbeddingVector(embeddingVector, legalRelevanceScore);
        storedIds.push(vectorId);

        console.log(
          `✅ Stored chunk ${i + 1}/${textChunks.length} for document ${documentId}`
        );
      }

      // Store document-level metadata
      await this.redis.hset(`doc:${documentId}`, {
        case_id: caseId,
        chunk_count: textChunks.length,
        total_confidence: ocrResult.confidence,
        processing_time: ocrResult.processing_time,
        created_at: new Date().toISOString(),
      });

      console.log(
        `🎉 Completed processing document ${documentId}: ${storedIds.length} chunks stored`
      );
      return storedIds;
    } catch (error) {
      console.error(`❌ Failed to process document ${documentId}:`, error);
      throw error;
    }
  }

  private async storeEmbeddingVector(
    embeddingVector: EmbeddingVector,
    legalRelevanceScore: number
  ): Promise<void> {
    const vectorKey = `vec:${embeddingVector.id}`;

    await this.redis.hset(vectorKey, {
      vector: Buffer.from(new Float32Array(embeddingVector.vector).buffer),
      case_id: embeddingVector.metadata.case_id,
      document_id: embeddingVector.metadata.document_id,
      content_type: embeddingVector.metadata.content_type,
      text_chunk: embeddingVector.metadata.text_chunk,
      confidence: embeddingVector.metadata.confidence,
      timestamp: embeddingVector.metadata.timestamp.getTime(),
      jurisdiction: embeddingVector.metadata.jurisdiction || "",
      case_type: embeddingVector.metadata.case_type || "",
      user_id: embeddingVector.metadata.user_id || "",
      legal_relevance_score: legalRelevanceScore,
    });
  }

  private calculateLegalRelevanceScore(text: string): number {
    const legalTerms = [
      "contract",
      "agreement",
      "liability",
      "damages",
      "evidence",
      "testimony",
      "precedent",
      "statute",
      "regulation",
      "jurisdiction",
      "court",
      "judge",
      "plaintiff",
      "defendant",
      "prosecution",
      "defense",
      "verdict",
      "appeal",
      "criminal",
      "civil",
      "constitutional",
      "federal",
      "state",
      "local",
      "felony",
      "misdemeanor",
      "violation",
      "breach",
      "negligence",
      "fraud",
    ];

    const legalPhrases = [
      "beyond reasonable doubt",
      "preponderance of evidence",
      "due process",
      "probable cause",
      "burden of proof",
      "chain of custody",
      "expert witness",
      "legal precedent",
      "statutory interpretation",
      "constitutional rights",
    ];

    let score = 0;
    const lowerText = text.toLowerCase();

    // Count legal terms (weighted)
    legalTerms.forEach((term) => {
      const matches = (lowerText.match(new RegExp(term, "g")) || []).length;
      score += matches * 2;
    });

    // Count legal phrases (higher weight)
    legalPhrases.forEach((phrase) => {
      const matches = (lowerText.match(new RegExp(phrase, "g")) || []).length;
      score += matches * 5;
    });

    // Normalize to 0-100 scale
    const normalizedScore = Math.min(
      100,
      (score / text.split(/\\s+/).length) * 1000
    );
    return normalizedScore;
  }

  private calculateProsecutionScore(text: string, caseType?: string): number {
    const prosecutionTerms = [
      "evidence",
      "witness",
      "crime",
      "victim",
      "suspect",
      "investigation",
      "forensic",
      "alibi",
      "motive",
      "opportunity",
      "intent",
      "guilty",
      "charges",
      "indictment",
      "conviction",
      "sentence",
      "penalty",
    ];

    const evidenceQualityTerms = [
      "DNA",
      "fingerprint",
      "surveillance",
      "confession",
      "corroboration",
      "chain of custody",
      "forensic analysis",
      "expert testimony",
    ];

    let score = 0;
    const lowerText = text.toLowerCase();

    // Base prosecution relevance
    prosecutionTerms.forEach((term) => {
      const matches = (lowerText.match(new RegExp(term, "g")) || []).length;
      score += matches * 1.5;
    });

    // Evidence quality boost
    evidenceQualityTerms.forEach((term) => {
      const matches = (lowerText.match(new RegExp(term, "g")) || []).length;
      score += matches * 3;
    });

    // Case type multiplier
    if (caseType) {
      const criminalTypes = [
        "criminal",
        "felony",
        "misdemeanor",
        "assault",
        "fraud",
        "theft",
      ];
      if (criminalTypes.some((type) => caseType.toLowerCase().includes(type))) {
        score *= 1.2;
      }
    }

    // Normalize to 0-95 scale (leaving room for AI adjustment)
    const normalizedScore = Math.min(
      95,
      (score / text.split(/\\s+/).length) * 800
    );
    return normalizedScore;
  }

  async searchSimilar(
    queryText: string,
    options: {
      limit?: number;
      case_type?: string;
      jurisdiction?: string;
      content_type?: string;
      user_id?: string;
      weights?: Partial<RankingWeights>;
    } = {}
  ): Promise<SearchResult[]> {
    try {
      console.log(`🔍 Searching for: "${queryText.substring(0, 100)}..."`);

      const {
        limit = 10,
        case_type,
        jurisdiction,
        content_type,
        user_id,
        weights,
      } = options;

      const rankingWeights = { ...this.defaultRankingWeights, ...weights };

      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(queryText);

      // Build search query
      let searchQuery = "*";
      const filters: string[] = [];

      if (case_type) filters.push(`@case_type:${case_type}`);
      if (jurisdiction) filters.push(`@jurisdiction:${jurisdiction}`);
      if (content_type) filters.push(`@content_type:${content_type}`);
      if (user_id) filters.push(`@user_id:${user_id}`);

      if (filters.length > 0) {
        searchQuery = filters.join(" ");
      }

      // Perform vector search
      const searchResults = (await this.redis.call(
        "FT.SEARCH",
        "legal_vectors",
        searchQuery,
        "PARAMS",
        "2",
        "query_vector",
        Buffer.from(new Float32Array(queryEmbedding).buffer),
        "SORTBY",
        "__vector_score",
        "LIMIT",
        "0",
        (limit * 2).toString(),
        "RETURN",
        "10",
        "case_id",
        "document_id",
        "content_type",
        "text_chunk",
        "confidence",
        "legal_relevance_score",
        "timestamp",
        "jurisdiction",
        "case_type",
        "__vector_score"
      )) as any[];

      // Process and rank results
      const results: SearchResult[] = [];

      for (let i = 1; i < searchResults.length; i += 2) {
        const fields = searchResults[i + 1];
        const vectorScore = parseFloat(
          fields[fields.indexOf("__vector_score") + 1]
        );

        const metadata = {
          document_id: fields[fields.indexOf("document_id") + 1],
          case_id: fields[fields.indexOf("case_id") + 1],
          content_type: fields[fields.indexOf("content_type") + 1],
          text_chunk: fields[fields.indexOf("text_chunk") + 1],
          confidence: parseFloat(fields[fields.indexOf("confidence") + 1]),
          timestamp: new Date(
            parseInt(fields[fields.indexOf("timestamp") + 1])
          ),
          jurisdiction: fields[fields.indexOf("jurisdiction") + 1] || undefined,
          case_type: fields[fields.indexOf("case_type") + 1] || undefined,
        };

        const legalRelevanceScore = parseFloat(
          fields[fields.indexOf("legal_relevance_score") + 1]
        );
        const prosecutionScore = this.calculateProsecutionScore(
          metadata.text_chunk,
          metadata.case_type
        );

        // Calculate final ranking score
        const finalRank = this.calculateFinalRank({
          similarity_score: 1 - vectorScore, // Convert distance to similarity
          legal_relevance_score: legalRelevanceScore,
          prosecution_score: prosecutionScore,
          confidence: metadata.confidence,
          timestamp: metadata.timestamp,
          jurisdiction_match: jurisdiction === metadata.jurisdiction,
          weights: rankingWeights,
        });

        results.push({
          id: searchResults[i],
          score: 1 - vectorScore,
          metadata,
          highlights: this.extractHighlights(metadata.text_chunk, queryText),
          legal_relevance_score: legalRelevanceScore,
          prosecution_score: prosecutionScore,
          final_rank: finalRank,
        });
      }

      // Sort by final rank and limit results
      const rankedResults = results
        .sort((a, b) => b.final_rank - a.final_rank)
        .slice(0, limit);

      console.log(`✅ Found ${rankedResults.length} ranked results`);
      return rankedResults;
    } catch (error) {
      console.error("❌ Vector search failed:", error);
      throw error;
    }
  }

  private calculateFinalRank(params: {
    similarity_score: number;
    legal_relevance_score: number;
    prosecution_score: number;
    confidence: number;
    timestamp: Date;
    jurisdiction_match: boolean;
    weights: RankingWeights;
  }): number {
    const {
      similarity_score,
      legal_relevance_score,
      prosecution_score,
      confidence,
      timestamp,
      jurisdiction_match,
      weights,
    } = params;

    // Normalize scores to 0-1 range
    const normalizedSimilarity = Math.max(0, Math.min(1, similarity_score));
    const normalizedLegalRelevance = legal_relevance_score / 100;
    const normalizedProsecution = prosecution_score / 100;
    const normalizedConfidence = confidence / 100;

    // Recency score (newer documents get higher scores)
    const daysSinceCreation =
      (Date.now() - timestamp.getTime()) / (1000 * 60 * 60 * 24);
    const normalizedRecency = Math.max(0, 1 - daysSinceCreation / 365); // Decay over 1 year

    // Calculate weighted score
    let finalScore =
      normalizedSimilarity * weights.similarity_weight +
      normalizedLegalRelevance * weights.legal_relevance_weight +
      normalizedProsecution * weights.prosecution_weight +
      normalizedRecency * weights.recency_weight +
      normalizedConfidence * weights.confidence_weight;

    // Apply jurisdiction boost
    if (jurisdiction_match) {
      finalScore += weights.jurisdiction_boost;
    }

    // Ensure score stays in 0-1 range
    return Math.max(0, Math.min(1, finalScore));
  }

  private extractHighlights(
    text: string,
    query: string,
    maxHighlights: number = 3
  ): string[] {
    const highlights: string[] = [];
    const queryTerms = query.toLowerCase().split(/\\s+/);
    const sentences = text.split(/[.!?]+/);

    for (const sentence of sentences) {
      const lowerSentence = sentence.toLowerCase();
      const matchCount = queryTerms.filter((term) =>
        lowerSentence.includes(term)
      ).length;

      if (matchCount > 0 && highlights.length < maxHighlights) {
        highlights.push(sentence.trim());
      }
    }

    return highlights;
  }

  async getCachedEmbedding(text: string): Promise<number[] | null> {
    try {
      const cacheKey = `embedding:${Buffer.from(text).toString("base64")}`;
      const cached = await this.redis.get(cacheKey);

      if (cached) {
        return JSON.parse(cached);
      }

      return null;
    } catch (error) {
      console.warn("⚠️ Cache retrieval failed:", error);
      return null;
    }
  }

  async cacheEmbedding(
    text: string,
    embedding: number[],
    ttl: number = 7200
  ): Promise<void> {
    try {
      const cacheKey = `embedding:${Buffer.from(text).toString("base64")}`;
      await this.redis.setex(cacheKey, ttl, JSON.stringify(embedding));
    } catch (error) {
      console.warn("⚠️ Cache storage failed:", error);
    }
  }

  async getProcessingStats(): Promise<{
    total_vectors: number;
    total_documents: number;
    total_cases: number;
    cache_hit_rate: number;
    avg_processing_time: number;
  }> {
    try {
      const [vectorCount, docCount] = await Promise.all([
        this.redis.call("FT.INFO", "legal_vectors"),
        this.redis.keys("doc:*"),
      ]);

      return {
        total_vectors: parseInt((vectorCount as any)[7]) || 0,
        total_documents: docCount.length,
        total_cases: 0, // TODO: Implement case counting
        cache_hit_rate: 0.85, // TODO: Implement actual cache hit rate tracking
        avg_processing_time: 2500, // TODO: Implement actual timing tracking
      };
    } catch (error) {
      console.error("❌ Failed to get processing stats:", error);
      return {
        total_vectors: 0,
        total_documents: 0,
        total_cases: 0,
        cache_hit_rate: 0,
        avg_processing_time: 0,
      };
    }
  }

  async cleanup(): Promise<void> {
    await this.redis.disconnect();
  }
}

// Export singleton instance
export const vectorEmbeddingService = new EnhancedVectorEmbeddingService();
