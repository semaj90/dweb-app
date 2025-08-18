    console.log(`✅ Stored embedding in Qdrant for evidence: ${evidenceId}`);
    
  } catch (error) {
    console.error(`❌ Failed to store embedding in Qdrant:`, error);
    // Don't throw - continue with other storage methods
  }
}

async function storeInPgVector(evidenceId: string, embedding: EmbeddingResult): Promise<void> {
  try {
    const db = await import('../../sveltekit-frontend/src/lib/server/db.js');
    
    // Store in PostgreSQL with pgvector extension
    await db.db.execute(`
      INSERT INTO evidence_vectors (
        evidence_id, 
        model, 
        dimensions, 
        vector, 
        metadata,
        created_at
      ) VALUES (
        $1, $2, $3, $4::vector, $5, NOW()
      )
      ON CONFLICT (evidence_id, model) 
      DO UPDATE SET 
        vector = EXCLUDED.vector,
        metadata = EXCLUDED.metadata,
        updated_at = NOW()
    `, [
      evidenceId,
      embedding.model,
      embedding.dim,
      `[${embedding.vector.join(',')}]`, // pgvector format
      JSON.stringify(embedding.metadata)
    ]);
    
    console.log(`✅ Stored embedding in pgvector for evidence: ${evidenceId}`);
    
  } catch (error) {
    console.error(`❌ Failed to store embedding in pgvector:`, error);
    // Don't throw - this is secondary storage
  }
}

// Similarity search functions
export async function findSimilarEvidences(
  queryVector: number[], 
  limit: number = 5,
  threshold: number = 0.7
): Promise<Array<{ evidenceId: string; score: number; metadata: any }>> {
  try {
    const collectionName = process.env.QDRANT_COLLECTION || 'evidence_embeddings';
    
    const searchResult = await qdrantClient.search(collectionName, {
      vector: queryVector,
      limit,
      score_threshold: threshold,
      with_payload: true
    });
    
    return searchResult.map(point => ({
      evidenceId: point.payload?.evidence_id as string,
      score: point.score || 0,
      metadata: point.payload
    }));
    
  } catch (error) {
    console.error('❌ Similarity search failed:', error);
    return [];
  }
}

export async function searchByText(
  queryText: string,
  limit: number = 5,
  threshold: number = 0.7
): Promise<Array<{ evidenceId: string; score: number; metadata: any }>> {
  try {
    // Generate embedding for query text
    const queryEmbedding = await generateEmbeddings({
      evidenceId: 'query',
      textOverride: queryText
    });
    
    return await findSimilarEvidences(queryEmbedding.vector, limit, threshold);
    
  } catch (error) {
    console.error('❌ Text search failed:', error);
    return [];
  }
}

// Health check for embedding service
export async function checkEmbeddingHealth(): Promise<{
  ollama: boolean;
  qdrant: boolean;
  pgvector: boolean;
}> {
  const health = {
    ollama: false,
    qdrant: false,
    pgvector: false
  };
  
  // Check Ollama
  try {
    const ollamaUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
    const response = await fetch(`${ollamaUrl}/api/tags`);
    health.ollama = response.ok;
  } catch (error) {
    health.ollama = false;
  }
  
  // Check Qdrant
  try {
    const collections = await qdrantClient.getCollections();
    health.qdrant = Array.isArray(collections.collections);
  } catch (error) {
    health.qdrant = false;
  }
  
  // Check pgvector
  try {
    const db = await import('../../sveltekit-frontend/src/lib/server/db.js');
    await db.db.execute('SELECT 1');
    health.pgvector = true;
  } catch (error) {
    health.pgvector = false;
  }
  
  return health;
}

// Get embedding capabilities
export function getEmbeddingCapabilities(): {
  models: string[];
  dimensions: Record<string, number>;
  providers: string[];
} {
  return {
    models: [
      'nomic-embed-text',
      'all-minilm',
      'fallback:tfidf'
    ],
    dimensions: {
      'nomic-embed-text': 768,
      'all-minilm': 384,
      'fallback:tfidf': 384
    },
    providers: [
      'ollama',
      'webgpu',
      'fallback'
    ]
  };
}
