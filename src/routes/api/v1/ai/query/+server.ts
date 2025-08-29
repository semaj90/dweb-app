import { json } from '@sveltejs/kit';
import { z } from 'zod';
import { db } from '$lib/db/client';
import { 
  documentChunks, 
  documents, 
  cases,
  ragQueries,
  ragQueryResults,
  getDocumentChunksWithSimilarity 
} from '$lib/db/schema/rag-integration';
import { eq, and } from 'drizzle-orm';
import { randomUUID } from 'crypto';

const querySchema = z.object({
  query: z.string().min(1).max(2000),
  caseId: z.string().uuid().optional(),
  maxResults: z.number().min(1).max(20).default(5),
  minSimilarity: z.number().min(0).max(1).default(0.7),
  model: z.string().default('gemma3-legal'),
  maxTokens: z.number().min(50).max(4000).default(2000),
  temperature: z.number().min(0).max(2).default(0.1)
});

// Ollama client configuration
const OLLAMA_BASE_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const EMBEDDING_MODEL = 'nomic-embed-text';
const DEFAULT_LLM_MODEL = 'gemma3-legal';

export async function POST({ request }): Promise<any> {
  const startTime = Date.now();
  
  try {
    // Parse and validate request
    const body = await request.json();
    const { 
      query, 
      caseId, 
      maxResults, 
      minSimilarity, 
      model, 
      maxTokens, 
      temperature 
    } = querySchema.parse(body);

    // Step 1: Generate query embedding
    const queryEmbedding = await generateEmbedding(query);
    if (!queryEmbedding) {
      return json({ error: 'Failed to generate query embedding' }, { status: 500 });
    }

    // Step 2: Vector similarity search
    let similarChunks = [];
    
    if (caseId) {
      // Search within specific case
      const [caseRecord] = await db
        .select()
        .from(cases)
        .where(eq(cases.uuid, caseId))
        .limit(1);

      if (!caseRecord) {
        return json({ error: 'Case not found' }, { status: 404 });
      }

      // Custom query for case-specific search
      const caseChunks = await db.execute(`
        SELECT 
          dc.id, dc.uuid, dc.content, dc.word_count, dc.metadata, dc.created_at,
          d.id as document_id, d.uuid as document_uuid, d.filename, d.original_name,
          c.id as case_id, c.uuid as case_uuid, c.title as case_title,
          1 - (dc.embedding <=> $1::vector) as similarity_score
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        JOIN cases c ON d.case_id = c.id
        WHERE c.id = $2
          AND dc.embedding IS NOT NULL
          AND 1 - (dc.embedding <=> $1::vector) > $3
        ORDER BY dc.embedding <=> $1::vector
        LIMIT $4
      `, [JSON.stringify(queryEmbedding), caseRecord.id, minSimilarity, maxResults]);

      similarChunks = caseChunks.rows;
    } else {
      // Global search across all cases
      const globalChunks = await getDocumentChunksWithSimilarity(
        db, 
        queryEmbedding, 
        minSimilarity, 
        maxResults
      );
      similarChunks = globalChunks.rows;
    }

    // Step 3: Prepare context for LLM
    const context = similarChunks
      .map((chunk, index) => {
        return `[Source ${index + 1}: ${chunk.filename || chunk.original_name}]\n${chunk.content}\n`;
      })
      .join('\n---\n\n');

    // Step 4: Generate response with Gemma3
    const llmResponse = await queryGemma3({
      query,
      context,
      model,
      maxTokens,
      temperature
    });

    if (!llmResponse) {
      return json({ error: 'Failed to generate AI response' }, { status: 500 });
    }

    // Step 5: Store query and results for analytics
    const queryId = randomUUID();
    const processingTime = Date.now() - startTime;
    
    const [savedQuery] = await db
      .insert(ragQueries)
      .values({
        uuid: queryId,
        caseId: caseId ? (await db.select().from(cases).where(eq(cases.uuid, caseId)).limit(1))[0]?.id : null,
        query,
        queryEmbedding: JSON.stringify(queryEmbedding),
        response: llmResponse.response,
        model,
        tokensUsed: llmResponse.tokensUsed || 0,
        processingTimeMs: processingTime,
        similarityThreshold: minSimilarity,
        resultsCount: similarChunks.length
      })
      .returning();

    // Save query results for source tracking
    for (let i = 0; i < similarChunks.length; i++) {
      const chunk = similarChunks[i];
      await db
        .insert(ragQueryResults)
        .values({
          queryId: savedQuery.id,
          chunkId: chunk.id,
          similarityScore: chunk.similarity_score,
          rank: i + 1,
          used: true
        });
    }

    // Step 6: Format response
    const response = {
      answer: llmResponse.response,
      sources: similarChunks.map((chunk, index) => ({
        id: chunk.uuid || chunk.id.toString(),
        title: chunk.original_name || chunk.filename || `Document ${chunk.document_id}`,
        excerpt: chunk.content.substring(0, 200) + (chunk.content.length > 200 ? '...' : ''),
        similarity: Math.round(chunk.similarity_score * 100) / 100,
        caseId: chunk.case_uuid,
        documentId: chunk.document_uuid,
        chunkId: chunk.uuid,
        rank: index + 1
      })),
      processingTime,
      model,
      tokensUsed: llmResponse.tokensUsed || 0,
      queryId,
      metadata: {
        embeddingModel: EMBEDDING_MODEL,
        similarityThreshold: minSimilarity,
        maxResults,
        contextLength: context.length,
        temperature
      }
    };

    return json(response);

  } catch (error: any) {
    console.error('RAG query error:', error);
    
    if (error instanceof z.ZodError) {
      return json({ error: 'Invalid request data', details: error.errors }, { status: 400 });
    }
    
    return json({ 
      error: 'Internal server error',
      message: error.message,
      processingTime: Date.now() - startTime
    }, { status: 500 });
  }
}

// Generate embedding using Ollama nomic-embed-text
async function generateEmbedding(text: string): Promise<number[] | null> {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        prompt: text
      })
    });

    if (!response.ok) {
      console.error('Embedding generation failed:', response.status, response.statusText);
      return null;
    }

    const data = await response.json();
    return data.embedding;

  } catch (error: any) {
    console.error('Embedding generation error:', error);
    return null;
  }
}

// Query Gemma3 with context
async function queryGemma3({
  query,
  context,
  model,
  maxTokens,
  temperature
}: {
  query: string;
  context: string;
  model: string;
  maxTokens: number;
  temperature: number;
}): Promise<any> {
  try {
    // Construct legal-focused prompt
    const prompt = `You are a legal AI assistant with access to case documents and legal materials. Based on the provided context, answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly.

CONTEXT:
${context}

USER QUESTION: ${query}

LEGAL ANALYSIS:`;

    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: {
          num_predict: maxTokens,
          temperature,
          top_p: 0.9,
          top_k: 40,
          repeat_penalty: 1.1
        }
      })
    });

    if (!response.ok) {
      console.error('LLM generation failed:', response.status, response.statusText);
      return null;
    }

    const data = await response.json();
    
    return {
      response: data.response,
      tokensUsed: data.eval_count || 0,
      model: data.model || model,
      done: data.done
    };

  } catch (error: any) {
    console.error('LLM query error:', error);
    return null;
  }
}

// Streaming version for real-time responses
export async function GET({ url }): Promise<any> {
  const query = url.searchParams.get('query');
  const caseId = url.searchParams.get('caseId');
  const model = url.searchParams.get('model') || DEFAULT_LLM_MODEL;

  if (!query) {
    return json({ error: 'Query parameter required' }, { status: 400 });
  }

  try {
    // Generate embedding and get context (same as POST)
    const queryEmbedding = await generateEmbedding(query);
    if (!queryEmbedding) {
      return json({ error: 'Failed to generate query embedding' }, { status: 500 });
    }

    let similarChunks = [];
    if (caseId) {
      const [caseRecord] = await db
        .select()
        .from(cases)
        .where(eq(cases.uuid, caseId))
        .limit(1);

      if (caseRecord) {
        const caseChunks = await db.execute(`
          SELECT dc.*, d.filename, d.original_name,
                 1 - (dc.embedding <=> $1::vector) as similarity_score
          FROM document_chunks dc
          JOIN documents d ON dc.document_id = d.id
          JOIN cases c ON d.case_id = c.id
          WHERE c.id = $2 AND dc.embedding IS NOT NULL
            AND 1 - (dc.embedding <=> $1::vector) > 0.7
          ORDER BY dc.embedding <=> $1::vector
          LIMIT 5
        `, [JSON.stringify(queryEmbedding), caseRecord.id]);
        
        similarChunks = caseChunks.rows;
      }
    } else {
      const globalChunks = await getDocumentChunksWithSimilarity(db, queryEmbedding, 0.7, 5);
      similarChunks = globalChunks.rows;
    }

    const context = similarChunks
      .map((chunk, index) => `[Source ${index + 1}]\n${chunk.content}`)
      .join('\n\n---\n\n');

    // Stream response from Ollama
    const prompt = `Based on the following context, answer the question concisely:

CONTEXT:
${context}

QUESTION: ${query}

ANSWER:`;

    const ollamaResponse = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
        stream: true
      })
    });

    if (!ollamaResponse.ok) {
      return json({ error: 'Streaming request failed' }, { status: 500 });
    }

    // Return streaming response
    return new Response(ollamaResponse.body, {
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    });

  } catch (error: any) {
    console.error('Streaming query error:', error);
    return json({ error: 'Streaming failed' }, { status: 500 });
  }
}