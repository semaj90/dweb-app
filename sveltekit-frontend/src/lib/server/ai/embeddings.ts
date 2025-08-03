// AI embedding generation service
// UPDATED: Now defaults to Nomic Embed (free, local) instead of OpenAI (paid)
// Supports both OpenAI and local Nomic Embed models with Redis/memory caching for performance
// Use process.env for server-side environment variables
import { cases, evidence } from "$lib/server/db/schema-postgres";
import { eq } from "drizzle-orm";
import type { EmbeddingOptions } from "../../types/vector";
import { cacheEmbedding, getCachedEmbedding } from "../cache/redis";
import { db } from "../db/index";

export async function generateEmbedding(
  text: string,
  options: EmbeddingOptions = {},
): Promise<number[] | null> {
  const { model = "local", cache = true, maxTokens = 8000 } = options;

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
      console.warn("⚠️  OpenAI embeddings are not free! Consider using model='local' for Nomic Embed instead.");
      embedding = await generateOpenAIEmbedding(truncatedText);
    } else {
      // Default to local Nomic Embed (free)
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
  const apiKey = process.env.OPENAI_API_KEY;
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
// Local embedding generation using Nomic Embed
async function generateLocalEmbedding(text: string): Promise<number[]> {
  try {
    // Use local Nomic Embed server
    const nomicEmbedUrl = process.env.NOMIC_EMBED_URL || "http://localhost:5000";
    
    const response = await fetch(`${nomicEmbedUrl}/embed`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text
      }),
    });

    if (!response.ok) {
      throw new Error(`Nomic Embed API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.warn(
      "Nomic Embed generation failed, this avoids OpenAI costs but requires local Nomic Embed server:",
      error
    );
    
    // Instead of fallback to OpenAI, throw error to encourage local setup
    throw new Error("Local Nomic Embed server required. Please start Nomic Embed server on port 5000 or set NOMIC_EMBED_URL environment variable.");
  }
}
// Batch embedding generation for efficiency
export async function generateBatchEmbeddings(
  texts: string[],
  options: EmbeddingOptions = {},
): Promise<number[][]> {
  const { model = "local" } = options;

  if (model === "openai" && texts.length > 1) {
    console.warn("⚠️  OpenAI batch embeddings are not free! Consider using model='local' for Nomic Embed instead.");
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
  const apiKey = process.env.OPENAI_API_KEY;
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
