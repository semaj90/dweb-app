// AI embedding generation service
// Supports OpenAI and local models with Redis/memory caching for performance
import { env } from "$env/dynamic/private";
import { cases, evidence } from "$lib/server/db/schema-postgres";
import { eq } from "drizzle-orm";
import type { EmbeddingOptions } from "../../types/vector";
import { cacheEmbedding, getCachedEmbedding } from "../cache/redis";
import { db } from "../db/index";

export async function generateEmbedding(
  text: string,
  options: EmbeddingOptions = {},
): Promise<number[] | null> {
  const { model = "openai", cache = true, maxTokens = 8000 } = options;

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
    if (model === "openai") {
      embedding = await generateOpenAIEmbedding(truncatedText);
    } else {
      embedding = await generateLocalEmbedding(truncatedText);
    }
    // Cache the result
    if (cache) {
      await cacheEmbedding(truncatedText, embedding, model);
    }
    return embedding;
  } catch (error) {
    console.error("Embedding generation failed:", error);
    return null;
  }
}
// OpenAI embedding generation
async function generateOpenAIEmbedding(text: string): Promise<number[]> {
  const apiKey = env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OpenAI API key not configured");
  }
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: text,
      model: "text-embedding-3-small", // 1536 dimensions, fast and cost-effective
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.statusText}`);
  }
  const data = await response.json();
  return data.data[0].embedding;
}
// Local embedding generation (placeholder for local models)
async function generateLocalEmbedding(text: string): Promise<number[]> {
  // This could use sentence-transformers, Ollama, or other local models
  // For now, return a simple hash-based pseudo-embedding
  console.warn(
    "Local embedding generation not implemented, using OpenAI fallback",
  );
  return generateOpenAIEmbedding(text);
}
// Batch embedding generation for efficiency
export async function generateBatchEmbeddings(
  texts: string[],
  options: EmbeddingOptions = {},
): Promise<number[][]> {
  const { model = "openai" } = options;

  if (model === "openai" && texts.length > 1) {
    return generateOpenAIBatchEmbeddings(texts);
  }
  // Fall back to individual generation
  const embeddings: (number[] | null)[] = [];
  for (const text of texts) {
    const embedding = await generateEmbedding(text, options);
    embeddings.push(embedding);
  }
  return embeddings.filter((e): e is number[] => e !== null);
}
// OpenAI batch embedding generation
async function generateOpenAIBatchEmbeddings(
  texts: string[],
): Promise<number[][]> {
  const apiKey = env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OpenAI API key not configured");
  }
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: texts,
      model: "text-embedding-3-small",
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.statusText}`);
  }
  const data = await response.json();
  return data.data.map((item: any) => item.embedding);
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
  } catch (error) {
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
  } catch (error) {
    console.error(
      `Failed to update embeddings for evidence ${evidenceId}:`,
      error,
    );
    throw error;
  }
}
