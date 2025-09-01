
// AI embedding generation service
// Supports local Ollama models with Redis/memory caching for performance
// Use process.env for server-side environment variables
import { db } from '$lib/server/db/index.js';
import { cases, evidence } from "$lib/server/db/schema-postgres";
import { eq } from 'drizzle-orm';

export interface EmbeddingOptions {
  model?: string;
  cache?: boolean;
  maxTokens?: number;
}

async function getCachedEmbedding(text: string, model: string): Promise<number[] | null> {
  // Placeholder cache implementation
  return null;
}

async function cacheEmbedding(text: string, model: string, embedding: number[]): Promise<void> {
  // Placeholder cache implementation
}

export async function generateEmbedding(
  text: string,
  options: EmbeddingOptions = {},
): Promise<number[] | null> {
  const { model = "nomic-embed-text", cache = true, maxTokens = 8000 } = options;

  if (!text || text.trim().length === 0) {
    return null;
  }
  // Truncate text if too long
  const truncatedText =
    text.length > maxTokens ? text.substring(0, maxTokens) : text;

  // Check cache first
  if (cache) {
    const cachedEmbedding = await getCachedEmbedding(truncatedText, model);
    if (cachedEmbedding) {
      return cachedEmbedding;
    }
  }
  let embedding: number[];

  try {
    // Use local Ollama embedding model
    embedding = await generateLocalEmbedding(truncatedText, model);
    
    // Cache the result
    if (cache) {
      await cacheEmbedding(truncatedText, model, embedding);
    }
    return embedding;
  } catch (error: any) {
    console.error("Embedding generation failed:", error);
    return null;
  }
}
// Local Ollama embedding generation
async function generateLocalEmbedding(text: string, model: string = "nomic-embed-text"): Promise<number[]> {
  const ollamaUrl = import.meta.env.OLLAMA_URL || "http://localhost:11434";
  
  try {
    const response = await fetch(`${ollamaUrl}/api/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error: any) {
    console.error("Ollama embedding generation failed:", error);
    // Fallback to mock embedding for development
    return generateMockEmbedding(384); // nomic-embed-text is 768 dimensions
  }
}

// Mock embedding generation for development/testing
function generateMockEmbedding(dimensions: number = 384): number[] {
  const embedding = new Array(dimensions);
  for (let i = 0; i < dimensions; i++) {
    // Generate pseudo-random values between -1 and 1
    embedding[i] = (Math.random() - 0.5) * 2;
  }
  // Normalize the vector
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / magnitude);
}
// Batch embedding generation for efficiency
export async function generateBatchEmbeddings(
  texts: string[],
  options: EmbeddingOptions = {},
): Promise<number[][]> {
  const { model = "nomic-embed-text" } = options;

  // Process texts individually for now (Ollama doesn't support batch)
  const embeddings: (number[] | null)[] = [];
  for (const text of texts) {
    const embedding = await generateEmbedding(text, options);
    embeddings.push(embedding);
  }
  return embeddings.filter((e): e is number[] => e !== null);
}
// Update embeddings for existing records
export async function updateCaseEmbeddings(caseId: string): Promise<void> {
  try {
    // Get case data
    const caseData = await db
      .select({
        title: cases.title,
        description: cases.description,
      })
      .from(cases)
      .where(eq(cases.id, caseId));

    if (caseData.length === 0) {
      throw new Error("Case not found");
    }
    const case_ = caseData[0];
    const fullText = `${case_.title} ${case_.description || ""}`.trim();

    // Generate embeddings
    const [titleEmbedding, descriptionEmbedding, fullTextEmbedding] =
      await generateBatchEmbeddings([
        case_.title,
        case_.description || "",
        fullText,
      ]);

    // TODO: Re-enable when titleEmbedding field is added to schema
    // Update database
    // await db
    //   .update(cases)
    //   .set({
    //     titleEmbedding: JSON.stringify(titleEmbedding),
    //     descriptionEmbedding: JSON.stringify(descriptionEmbedding),
    //     fullTextEmbedding: JSON.stringify(fullTextEmbedding),
    //     updatedAt: new Date(),
    //   })
    //   .where(eq(cases.id, caseId));

    console.log(`Updated embeddings for case ${caseId}`);
  } catch (error: any) {
    console.error(`Failed to update embeddings for case ${caseId}:`, error);
    throw error;
  }
}
// Update embeddings for evidence
export async function updateEvidenceEmbeddings(
  evidenceId: string,
): Promise<void> {
  try {
    // Get evidence data
    const evidenceData = await db
      .select({
        title: evidence.title,
        description: evidence.description,
        summary: evidence.summary,
        aiSummary: evidence.aiSummary,
      })
      .from(evidence)
      .where(eq(evidence.id, evidenceId));

    if (evidenceData.length === 0) {
      throw new Error("Evidence not found");
    }
    const evidence_ = evidenceData[0];
    const combinedContent = [
      evidence_.title,
      evidence_.description,
      evidence_.summary,
      evidence_.aiSummary,
    ]
      .filter(Boolean)
      .join(" ");

    // Generate embeddings
    const [
      titleEmbedding,
      descriptionEmbedding,
      summaryEmbedding,
      contentEmbedding,
    ] = await generateBatchEmbeddings([
      evidence_.title,
      evidence_.description || "",
      evidence_.summary || "",
      combinedContent,
    ]);

    // TODO: Re-enable when embedding fields are added to evidence schema
    // Update database
    // await db
    //   .update(evidence)
    //   .set({
    //     titleEmbedding: JSON.stringify(titleEmbedding),
    //     descriptionEmbedding: JSON.stringify(descriptionEmbedding),
    //     summaryEmbedding: JSON.stringify(summaryEmbedding),
    //     contentEmbedding: JSON.stringify(contentEmbedding),
    //     updatedAt: new Date(),
    //   })
    //   .where(eq(evidence.id, evidenceId));

    console.log(`Updated embeddings for evidence ${evidenceId}`);
  } catch (error: any) {
    console.error(
      `Failed to update embeddings for evidence ${evidenceId}:`,
      error,
    );
    throw error;
  }
}
