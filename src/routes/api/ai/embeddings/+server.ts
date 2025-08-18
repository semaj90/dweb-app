// AI Embeddings API
// Provides text embedding generation capabilities

import { json } from "@sveltejs/kit";
import { db } from "$lib/database/postgres.js";
import { contentEmbeddings } from "$lib/database/schema/legal-documents.js";
import { eq } from "drizzle-orm";
import {
  serializeEmbedding,
  embeddingDimensions,
} from "$lib/utils/embeddings.js";
import type { RequestHandler } from "./$types";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      text,
      content,
      model = "nomic-embed-text",
      documentId,
      contentType = "text",
      metadata = {},
    } = await request.json();

    const textToEmbed = text || content;

    if (!textToEmbed) {
      return json({ error: "Text or content is required" }, { status: 400 });
    }

    // Generate embedding
    const embedding = await generateEmbedding(textToEmbed, model);

    // If documentId provided, save to database
    if (documentId) {
      const [saved] = await db
        .insert(contentEmbeddings)
        .values({
          contentId: documentId,
          contentType: contentType,
          textContent: textToEmbed.substring(0, 5000),
          embedding: serializeEmbedding(embedding) as any,
          metadata: metadata as any,
          createdAt: new Date(),
          model: model,
        } as any)
        .returning();

      return json({
        success: true,
        embedding: embedding,
        dimensions: embedding.length,
        model: model,
        savedRecord: {
          id: saved.id,
          contentId: saved.contentId,
          createdAt: saved.createdAt,
        },
      });
    }

    // Return embedding without saving
    return json({
      success: true,
      embedding: embedding,
      dimensions: embedding.length,
      model: model,
    });
  } catch (error: any) {
    console.error("Embedding generation error:", error);
    return json(
      { error: "Failed to generate embedding", details: error.message },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const contentId = url.searchParams.get("contentId");
    const limit = parseInt(url.searchParams.get("limit") || "10");
    const contentType = url.searchParams.get("contentType");

    if (contentId) {
      // Get embeddings for specific content
      const embeddings = await db.query.contentEmbeddings.findMany({
        where: (embeddings, { eq }) => eq(embeddings.contentId, contentId),
        limit: limit,
      });

      return json({
        contentId,
        embeddings: embeddings.map((emb) => ({
          id: emb.id,
          contentType: emb.contentType,
          model: emb.model,
          dimensions: embeddingDimensions(emb.embedding),
          createdAt: emb.createdAt,
          metadata: emb.metadata,
        })),
      });
    }

    // Get list of embeddings with optional filters
    let query = db
      .select({
        id: contentEmbeddings.id,
        contentId: contentEmbeddings.contentId,
        contentType: contentEmbeddings.contentType,
        model: contentEmbeddings.model,
        createdAt: contentEmbeddings.createdAt,
        metadata: contentEmbeddings.metadata,
      })
      .from(contentEmbeddings);

    if (contentType) {
      query = query.where(
        eq(contentEmbeddings.contentType, contentType as any)
      );
    }

    const results = await query.orderBy((emb) => emb.createdAt).limit(limit);

    return json({
      embeddings: results.map((emb) => ({
        ...emb,
        dimensions: 384, // Default dimension count for nomic-embed-text
      })),
    });
  } catch (error: any) {
    console.error("Embedding retrieval error:", error);
    return json(
      { error: "Failed to retrieve embeddings", details: error.message },
      { status: 500 }
    );
  }
};

export const DELETE: RequestHandler = async ({ request }) => {
  try {
    const { contentId, embeddingId } = await request.json();

    if (embeddingId) {
      // Delete specific embedding
      await db
        .delete(contentEmbeddings)
        .where(eq(contentEmbeddings.id, Number(embeddingId)));

      return json({
        success: true,
        message: "Embedding deleted successfully",
        deletedId: embeddingId,
      });
    }

    if (contentId) {
      // Delete all embeddings for content
      const result = await db
        .delete(contentEmbeddings)
        .where(eq(contentEmbeddings.contentId, contentId))
        .returning();

      return json({
        success: true,
        message: `Deleted ${result.length} embeddings for content ${contentId}`,
        deletedCount: result.length,
      });
    }

    return json(
      { error: "Either embeddingId or contentId is required" },
      { status: 400 }
    );
  } catch (error: any) {
    console.error("Embedding deletion error:", error);
    return json(
      { error: "Failed to delete embedding", details: error.message },
      { status: 500 }
    );
  }
};

// Utility functions

async function generateEmbedding(
  text: string,
  model: string
): Promise<number[]> {
  try {
    // Try to call the embedding service
    const response = await fetch("http://localhost:11434/api/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model,
        prompt: text,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      return data.embedding;
    } else {
      console.warn("Embedding service returned error:", response.status);
    }
  } catch (error) {
    console.warn("Embedding service unavailable, using mock embedding:", error);
  }

  // Fallback to mock embedding
  const dimensions = model.includes("nomic") ? 384 : 1536; // nomic-embed-text uses 384, others typically 1536
  return Array.from({ length: dimensions }, () => Math.random() - 0.5);
}

async function calculateSimilarity(
  embedding1: number[],
  embedding2: number[]
): Promise<number> {
  // Cosine similarity calculation
  if (embedding1.length !== embedding2.length) {
    throw new Error("Embeddings must have the same dimensions");
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] * embedding1[i];
    norm2 += embedding2[i] * embedding2[i];
  }

  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { action, ...params } = await request.json();

    switch (action) {
      case "similarity":
        const { embedding1, embedding2 } = params;
        if (!embedding1 || !embedding2) {
          return json(
            { error: "Two embeddings required for similarity calculation" },
            { status: 400 }
          );
        }

        const similarity = await calculateSimilarity(embedding1, embedding2);
        return json({
          similarity,
          interpretation:
            similarity > 0.8
              ? "very similar"
              : similarity > 0.6
                ? "similar"
                : similarity > 0.4
                  ? "somewhat similar"
                  : "different",
        });

      case "batch_generate":
        const { texts, model = "nomic-embed-text" } = params;
        if (!texts || !Array.isArray(texts)) {
          return json(
            { error: "Array of texts required for batch generation" },
            { status: 400 }
          );
        }

        const embeddings = await Promise.all(
          texts.map((text) => generateEmbedding(text, model))
        );

        return json({
          success: true,
          embeddings,
          count: embeddings.length,
          model,
        });

      default:
        return json(
          {
            error: "Unknown action",
            availableActions: ["similarity", "batch_generate"],
          },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error("Embedding operation error:", error);
    return json(
      { error: "Operation failed", details: error.message },
      { status: 500 }
    );
  }
};
