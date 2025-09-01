/// <reference types="vite/client" />
import type { EmbeddingResponse } from "$lib/types/ollama";
import type { RequestHandler } from './$types';

/**
 * Ollama Embeddings API Endpoint
 * Handles text embedding generation for legal documents
 */


const OLLAMA_BASE_URL = import.meta.env.OLLAMA_URL || 'http://localhost:11434';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      text,
      model = 'nomic-embed-text:latest',
      normalize = true,
      truncate = true,
    } = await request.json();

    if (!text) {
      return json({ error: 'Text is required' }, { status: 400 });
    }

    // Truncate text if too long (embedding models have token limits)
    const truncatedText = truncate ? text.substring(0, 2000) : text;

    const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model.replace(':latest', ''),
        prompt: truncatedText,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embeddings API error: ${response.status} ${response.statusText}`);
    }

    const data: EmbeddingResponse = await response.json();

    let embedding = data.embedding;

    // Normalize embedding if requested
    if (normalize && embedding) {
      const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
      if (norm > 0) {
        embedding = embedding.map(val => val / norm);
      }
    }

    return json({
      success: true,
      embedding,
      dimensions: embedding?.length || 0,
      model,
      metadata: {
        originalTextLength: text.length,
        truncatedTextLength: truncatedText.length,
        normalized: normalize,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: any) {
    console.error('Embeddings API error:', error);
    return json(
      {
        success: false,
        error: 'Failed to generate embeddings',
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
};