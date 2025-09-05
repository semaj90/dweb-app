// @ts-nocheck
// src/routes/api/embeddings/generate/+server.ts
import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { vectorService } from "$lib/server/vector/vectorService.js";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { text, metadata } = await request.json();

    if (!text || typeof text !== "string") {
      throw error(400, "Text parameter is required and must be a string");
    }

    // Generate embedding using the vector service
    const embedding = await vectorService.generateEmbedding(text);

    // Return the embedding with metadata
    return json({
      success: true,
      embedding,
      metadata: metadata || {},
      model: "nomic-embed-text",
      dimensions: embedding.length,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error("Embedding generation error:", err);

    // Handle specific error types
    if (err instanceof Error && err.message.includes("Ollama")) {
      throw error(503, "Ollama embedding service unavailable");
    }

    throw error(
      500,
      `Failed to generate embedding: ${err instanceof Error ? err.message : "Unknown error"}`,
    );
  }
};

export const GET: RequestHandler = async () => {
  return json({
    message: "Embedding generation endpoint",
    methods: ["POST"],
    example: {
      text: "Legal document content to embed",
      metadata: { type: "case", id: "case-123" },
    },
  });
};
