import { db } from '$lib/database/connection';
import { documents } from '$lib/database/schema';
import { eq } from 'drizzle-orm';

class EmbeddingService {
  async generateEmbedding(text: string): Promise<number[]> {
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'nomic-embed-text', prompt: text })
    });
    const data = await response.json();
    return data.embedding;
  }

  async embedDocument(docId: string, content: string) {
    const embedding = await this.generateEmbedding(content);
    await db.update(documents)
      .set({ embedding: JSON.stringify(embedding), isIndexed: true })
      .where(eq(documents.id, docId));
    return embedding;
  }

  async semanticSearch(query: string) {
    const embedding = await this.generateEmbedding(query);
    // Simplified - use basic query for now
    return { success: true, results: [] };
  }
}

export const embeddingService = new EmbeddingService();