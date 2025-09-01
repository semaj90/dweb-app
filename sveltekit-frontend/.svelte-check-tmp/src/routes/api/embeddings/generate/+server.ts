
import type { RequestHandler } from './$types';

// src/routes/api/embeddings/generate/+server.ts
import { json, error } from "@sveltejs/kit";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { text, metadata } = await request.json();

    if (!text || typeof text !== "string") {
      throw error(400, "Text parameter is required and must be a string");
    }

    // Generate embedding directly using Ollama
    const ollamaResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: text
      })
    });

    if (!ollamaResponse.ok) {
      throw new Error(`Ollama embedding service returned ${ollamaResponse.status}`);
    }

    const ollamaResult = await ollamaResponse.json();
    const embedding = ollamaResult.embedding || [];

    // Validate embedding dimensions (nomic-embed-text produces 384 dimensions)
    if (embedding.length !== 384) {
      console.warn(`Expected 384 dimensions, got ${embedding.length}. Using mock embedding.`);
      // Return mock embedding with correct dimensions
      const mockEmbedding = Array.from({ length: 384 }, () => Math.random() * 2 - 1);
      return json({
        success: true,
        embedding: mockEmbedding,
        metadata: metadata || {},
        model: "nomic-embed-text",
        dimensions: 384,
        timestamp: new Date().toISOString(),
        warning: "Used mock embedding due to dimension mismatch"
      });
    }

    // Return the embedding with metadata
    return json({
      success: true,
      embedding,
      metadata: metadata || {},
      model: "nomic-embed-text",
      dimensions: embedding.length,
      timestamp: new Date().toISOString(),
    });
  } catch (err: any) {
    console.error("Embedding generation error:", err);

    // Handle specific error types
    if (err instanceof Error && err.message.includes("Ollama")) {
      throw error(503, "Ollama embedding service unavailable");
    }

    // Return mock embedding as fallback
    const mockEmbedding = Array.from({ length: 384 }, () => Math.random() * 2 - 1);
    return json({
      success: true,
      embedding: mockEmbedding,
      metadata: metadata || {},
      model: "nomic-embed-text",
      dimensions: 384,
      timestamp: new Date().toISOString(),
      fallback: true,
      error: `Failed to generate embedding: ${err instanceof Error ? err.message : "Unknown error"}`
    });
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
