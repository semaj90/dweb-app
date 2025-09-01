import type { RequestHandler } from './$types';
import { db } from '$lib/server/database';
import { document_chunks, vectors, cases, evidence } from '$lib/server/db/schema-postgres';
import { json, error } from '@sveltejs/kit';
import { sql } from 'drizzle-orm';

// Generate query embedding using Ollama
async function generateQueryEmbedding(query: string): Promise<number[]> {
  try {
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text:latest',
        prompt: query
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Validate embedding dimension
    if (!result.embedding || !Array.isArray(result.embedding)) {
      throw new Error('Invalid embedding format from Ollama');
    }
    
    return result.embedding;
  } catch (err) {
    console.error('Query embedding generation failed:', err);
    throw new Error('Failed to generate query embedding');
  }
}

// Generate RAG response using Gemma3 legal model
async function generateRAGResponse(query: string, context: any[]): Promise<string> {
  try {
    const contextText = context.map(c => `${c.chunk_text}`).join('\n\n');
    
    const prompt = `Based on the following legal context, provide a comprehensive response to the query.

Context:
${contextText}

Query: ${query}

Response:`;

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 1000
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama generation error: ${response.statusText}`);
    }

    const result = await response.json();
    return result.response;
  } catch (err) {
    console.error('RAG response generation failed:', err);
    throw new Error('Failed to generate RAG response');
  }
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, limit = 5, threshold = 0.7, includeRAGResponse = true } = await request.json();

    if (!query) {
      return error(400, 'Missing required field: query');
    }

    // Generate query embedding
    const queryEmbedding = await generateQueryEmbedding(query);
    
    if (!queryEmbedding || queryEmbedding.length !== 768) {
      throw new Error(`Invalid query embedding dimension: expected 768, got ${queryEmbedding?.length}`);
    }

    // Perform vector similarity search using pgvector
    const embeddingStr = `[${queryEmbedding.join(',')}]`;
    
    const similarChunks = await db
      .select({
        id: document_chunks.id,
        chunk_text: document_chunks.chunk_text,
        chunk_sequence: document_chunks.chunk_sequence,
        evidence_id: document_chunks.evidence_id,
        embedding: document_chunks.embedding,
        similarity: sql<number>`1 - (embedding <=> ${embeddingStr}::vector)`.as('similarity')
      })
      .from(document_chunks)
      .where(sql`1 - (embedding <=> ${embeddingStr}::vector) > ${threshold}`)
      .orderBy(sql`embedding <=> ${embeddingStr}::vector`)
      .limit(limit);

    let ragResponse = null;
    
    if (includeRAGResponse && similarChunks.length > 0) {
      ragResponse = await generateRAGResponse(query, similarChunks);
    }

    // Enhance results with entity information
    const enhancedResults = await Promise.all(
      similarChunks.map(async (chunk) => {
        let entityInfo = null;
        
        if (chunk.evidence_id) {
          const evidenceResult = await db
            .select({
              id: evidence.id,
              name: evidence.name,
              case_id: evidence.case_id
            })
            .from(evidence)
            .where(sql`${evidence.id} = ${chunk.evidence_id}`)
            .limit(1);
          
          if (evidenceResult.length > 0) {
            entityInfo = { type: 'evidence', ...evidenceResult[0] };
          }
        }
        
        return {
          ...chunk,
          similarity: Math.round(chunk.similarity * 1000) / 1000, // Round to 3 decimal places
          entityInfo
        };
      })
    );

    return json({
      success: true,
      query,
      results: enhancedResults,
      ragResponse,
      metadata: {
        resultCount: similarChunks.length,
        threshold,
        embeddingModel: 'nomic-embed-text:latest',
        ragModel: includeRAGResponse ? 'gemma3-legal:latest' : null,
        searchTime: Date.now()
      }
    });

  } catch (err) {
    console.error('Vector search error:', err);
    return error(500, `Search failed: ${err.message}`);
  }
};