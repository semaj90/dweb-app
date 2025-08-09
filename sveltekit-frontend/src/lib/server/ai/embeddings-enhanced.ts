// @ts-nocheck
// Enhanced Embeddings Service with Local + Cloud Integration
// Combines local Tauri/Rust embeddings with cloud fallbacks
// Use process.env for server-side environment variables
import { cacheEmbedding, getCachedEmbedding } from "$lib/server/cache/redis";
import { aiService, type EmbeddingProvider } from "$lib/services/ai-service";

interface EnhancedEmbeddingOptions {
  provider?: "auto" | "openai" | "tauri-legal-bert" | "tauri-bert";
  cache?: boolean;
  maxTokens?: number;
  legalDomain?: boolean;
  batchSize?: number;
}
export async function generateEnhancedEmbedding(
  text: string | string[],
  options: EnhancedEmbeddingOptions = {},
): Promise<number[] | number[][]> {
  const {
    provider = "auto",
    cache = true,
    maxTokens = 8000,
    legalDomain = true,
    batchSize = 10,
  } = options;

  if (!text) {
    throw new Error("Text is required for embedding generation");
  }
  const isArray = Array.isArray(text);
  const inputs = isArray ? text : [text];
  const truncatedInputs = inputs.map((t) =>
    t.length > maxTokens ? t.substring(0, maxTokens) : t,
  );

  // Check cache for single inputs
  if (cache && !isArray) {
    const cacheKey = `${provider}-${legalDomain}`;
    const cachedEmbedding = await getCachedEmbedding(
      truncatedInputs[0],
      cacheKey,
    );
    if (cachedEmbedding) {
      return cachedEmbedding;
    }
  }
  let result: number[] | number[][];

  try {
    // Initialize AI service
    await aiService.initialize();

    // Determine provider
    const selectedProvider = selectProvider(provider, legalDomain);

    // Generate embeddings
    result = await aiService.generateEmbedding(truncatedInputs, {
      provider: selectedProvider,
      legalDomain,
      batchSize,
    });

    // Cache single results
    if (
      cache &&
      !isArray &&
      Array.isArray(result) &&
      !Array.isArray(result[0])
    ) {
      const cacheKey = `${selectedProvider}-${legalDomain}`;
      await cacheEmbedding(truncatedInputs[0], result as number[], cacheKey);
    }
    return isArray ? result : (result as number[][])[0];
  } catch (error) {
    console.error("Enhanced embedding generation failed:", error);

    // Fallback to simple OpenAI if local fails
    if (
      provider === "auto" ||
      provider === "tauri-legal-bert" ||
      provider === "tauri-bert"
    ) {
      console.log("Falling back to OpenAI embeddings");
      return generateOpenAIEmbeddings(truncatedInputs, isArray);
    }
    throw error;
  }
}
// Batch embedding generation with progress tracking
export async function generateBatchEmbeddingsEnhanced(
  texts: string[],
  options: EnhancedEmbeddingOptions = {},
  onProgress?: (completed: number, total: number) => void,
): Promise<number[][]> {
  const { batchSize = 10 } = options;
  const results: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    try {
      const batchResult = (await generateEnhancedEmbedding(
        batch,
        options,
      )) as number[][];
      results.push(...batchResult);

      if (onProgress) {
        onProgress(Math.min(i + batchSize, texts.length), texts.length);
      }
    } catch (error) {
      console.error(`Batch ${i}-${i + batchSize} failed:`, error);
      // Add empty embeddings for failed items
      for (let j = 0; j < batch.length; j++) {
        results.push([]);
      }
    }
    // Small delay between batches to avoid rate limits
    if (i + batchSize < texts.length) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  return results;
}
// Provider selection logic
function selectProvider(
  requested: EnhancedEmbeddingOptions["provider"],
  legalDomain: boolean,
): EmbeddingProvider {
  if (requested && requested !== "auto") {
    return requested;
  }
  // Auto-select based on availability and domain
  const status = aiService.getStatus();

  if (status.tauriAvailable) {
    if (legalDomain) {
      return "tauri-legal-bert"; // Prefer legal-BERT for legal domain
    }
    return "tauri-bert"; // Use general BERT for other domains
  }
  return "openai"; // Fallback to cloud
}
// OpenAI fallback implementation
async function generateOpenAIEmbeddings(
  texts: string[],
  returnArray: boolean,
): Promise<number[] | number[][]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured");
  }
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "text-embedding-ada-002",
      input: texts,
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.statusText}`);
  }
  const data = await response.json();
  const embeddings = data.data.map((item: any) => item.embedding);

  return returnArray ? embeddings : embeddings[0];
}
// Legal document-specific embedding with metadata
export async function generateLegalEmbedding(
  documentText: string,
  metadata: {
    documentType?: "contract" | "case_law" | "statute" | "brief" | "other";
    jurisdiction?: string;
    subject?: string[];
  } = {},
): Promise<{
  embedding: number[];
  metadata: any;
  confidence: number;
  classification?: any;
}> {
  await aiService.initialize();

  // Generate embedding with legal context
  const embedding = (await generateEnhancedEmbedding(documentText, {
    provider: "tauri-legal-bert",
    legalDomain: true,
    maxTokens: 2000, // Limit for legal documents
  })) as number[];

  let classification: {
    classification: any;
    keyEntities: string[];
    similarity: number;
    summary: string;
    riskAssessment: string;
  } | null = null;
  let confidence = 0.8; // Default confidence

  // Add legal classification if available
  if (aiService.getStatus().tauriAvailable) {
    try {
      classification = await aiService.analyzeLegalDocument(documentText);
      confidence = classification.similarity || 0.8;
    } catch (error) {
      console.warn("Legal classification failed:", error);
    }
  }
  return {
    embedding,
    metadata: {
      ...metadata,
      generatedAt: new Date().toISOString(),
      provider: "tauri-legal-bert",
      documentLength: documentText.length,
    },
    confidence,
    classification,
  };
}
// Similarity calculation between legal documents
export async function calculateLegalSimilarity(
  doc1: string,
  doc2: string,
): Promise<number> {
  await aiService.initialize();

  if (aiService.getStatus().tauriAvailable) {
    // Use local legal similarity calculation via embeddings
    const embeddings = (await generateEnhancedEmbedding([doc1, doc2], {
      provider: "tauri-legal-bert",
      legalDomain: true,
    })) as number[][];

    return cosineSimilarity(embeddings[0], embeddings[1]);
  } else {
    // Fallback to embedding comparison
    const embeddings = (await generateEnhancedEmbedding([doc1, doc2], {
      provider: "openai",
      legalDomain: true,
    })) as number[][];

    return cosineSimilarity(embeddings[0], embeddings[1]);
  }
}
// Cosine similarity calculation
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have same length for similarity calculation");
  }
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
// Backward compatibility export
export async function generateEmbedding(
  text: string,
  model?: string,
): Promise<number[]> {
  const result = await generateEnhancedEmbedding(text, {
    provider: "auto",
    legalDomain: true,
  });

  return Array.isArray(result) && Array.isArray(result[0])
    ? (result[0] as number[])
    : (result as number[]);
}
export async function generateBatchEmbeddings(
  texts: string[],
  model?: string,
  batchSize: number = 10,
): Promise<number[][]> {
  return generateBatchEmbeddingsEnhanced(texts, {
    provider: "auto",
    legalDomain: true,
    batchSize,
  });
}
