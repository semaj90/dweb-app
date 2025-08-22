
/**
 * AI Auto-Tagging Service
 * GPU-accelerated document analysis with Ollama + nomic-embed
 * Stores embeddings in PostgreSQL pgvector, auto-tags with gemma3-legal
 */

import { ollamaService } from "./ollamaService";
// TODO: Fix import - // Orphaned content: import {  import { evidence } from "$lib/db/schema";
import {
  eq,
  sql
} from "drizzle-orm";
import type { Evidence } from '$lib/types/legal-types';

interface EvidenceMetadata {
  aiTags: string[];
  entities: ExtractedEntity[];
  summary: string;
  confidence: number;
  relationships: DocumentRelationship[];
  autoTaggedAt: string;
}

export interface AutoTaggingResult {
  tags: string[];
  entities: ExtractedEntity[];
  summary: string;
  confidence: number;
  embedding: number[];
  relationships: DocumentRelationship[];
}

export interface ExtractedEntity {
  type: 'person' | 'organization' | 'location' | 'date' | 'legal_term' | 'case_number';
  text: string;
  confidence: number;
  position: { start: number; end: number };
}

export interface DocumentRelationship {
  type: 'references' | 'contradicts' | 'supports' | 'similar_to';
  targetId: string;
  confidence: number;
  description: string;
}

class AIAutoTaggingService {
  private ollamaEndpoint = 'http://localhost:11434';
  
  /**
   * Auto-tag document with AI analysis
   */
  async autoTagDocument(
    documentId: string,
    content: string,
    documentType: string
  ): Promise<AutoTaggingResult> {
    try {
      // 1. Generate embedding with nomic-embed-text (GPU accelerated)
      const embedding = await this.generateEmbedding(content);
      
      // 2. Analyze content with gemma3-legal
      const analysis = await this.analyzeContent(content, documentType);
      
      // 3. Find similar documents using pgvector
      const similarDocs = await this.findSimilarDocuments(embedding, documentId);
      
      // 4. Extract relationships
      const relationships = await this.extractRelationships(content, similarDocs);
      
      // 5. Update database with tags and embeddings
      await this.updateDocumentTags(documentId, {
        tags: analysis.tags,
        entities: analysis.entities,
        summary: analysis.summary,
        confidence: analysis.confidence,
        embedding,
        relationships
      });
      
      return {
        tags: analysis.tags,
        entities: analysis.entities,
        summary: analysis.summary,
        confidence: analysis.confidence,
        embedding,
        relationships
      };
      
    } catch (error) {
      console.error('Auto-tagging failed:', error);
      throw new Error(`Auto-tagging failed: ${error.message}`);
    }
  }
  
  /**
   * Generate embedding using nomic-embed-text (GPU accelerated)
   * Falls back to Qdrant server-side embedding if Ollama is unavailable
   */
  public async generateEmbedding(text: string): Promise<number[]> {
    try {
      // Primary: Try Ollama first (local GPU acceleration)
      const response = await fetch(`${this.ollamaEndpoint}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: text.substring(0, 8192) // Limit to model context
        })
      });
      
      if (!response.ok) {
        throw new Error(`Ollama embedding failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      return result.embedding;
      
    } catch (ollamaError) {
      console.warn('Ollama embedding failed, trying server-side Qdrant:', ollamaError);
      
      try {
        // Fallback: Use server-side Qdrant embedding
        if (typeof window === 'undefined') {
          // Server-side: Import and use Qdrant directly
          const { fetchEmbedding } = await import('$lib/server/qdrant');
          return await fetchEmbedding(text);
        } else {
          // Client-side: Make API call to server-side embedding
          const response = await fetch('/api/embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text.substring(0, 8192) })
          });
          
          if (!response.ok) {
            throw new Error(`Server embedding failed: ${response.statusText}`);
          }
          
          const result = await response.json();
          return result.embedding;
        }
      } catch (fallbackError) {
        console.error('All embedding methods failed:', fallbackError);
        // Return zero vector as ultimate fallback
        return new Array(768).fill(0);
      }
    }
  }
  
  /**
   * Analyze content with gemma3-legal for tags and entities
   */
  private async analyzeContent(content: string, documentType: string) {
    const prompt = `Analyze this ${documentType} legal document and provide:
1. Relevant tags (max 10)
2. Key entities with types and confidence
3. Brief summary (max 200 words)
4. Overall confidence score (0-1)

Document:
${content.substring(0, 4000)}

Return JSON format:
{
  "tags": ["contract", "liability", "employment"],
  "entities": [
    {"type": "person", "text": "John Doe", "confidence": 0.95, "position": {"start": 10, "end": 18}},
    {"type": "organization", "text": "Acme Corp", "confidence": 0.90, "position": {"start": 25, "end": 34}}
  ],
  "summary": "Employment contract between...",
  "confidence": 0.87
}`;

    const response = await fetch(`${this.ollamaEndpoint}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal',
        prompt,
        format: 'json',
        stream: false
      })
    });
    
    if (!response.ok) {
      throw new Error(`Content analysis failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    try {
      return JSON.parse(result.response);
    } catch (parseError) {
      // Fallback to basic analysis
      return {
        tags: [documentType, 'auto-generated'],
        entities: [],
        summary: 'AI analysis completed with basic tagging.',
        confidence: 0.5
      };
    }
  }
  
  /**
   * Find similar documents using pgvector cosine similarity
   */
  private async findSimilarDocuments(embedding: number[], excludeId: string, limit = 5) {
    const embeddingStr = `[${embedding.join(',')}]`;
    
    try {
      // Raw SQL query for pgvector similarity
      const result = await db.execute(sql.raw(`
        SELECT id, title, 1 - (embedding <=> '${embeddingStr}'::vector) as similarity
        FROM evidence 
        WHERE id != '${excludeId}' AND embedding IS NOT NULL
        ORDER BY embedding <=> '${embeddingStr}'::vector
        LIMIT ${limit}
      `));
      return result.rows.map((row: any) => ({
        id: row.id,
        title: row.title,
        similarity: row.similarity
      }));
    } catch (error) {
      console.warn('Similar document search failed:', error);
      return [];
    }
  }
  
  /**
   * Extract relationships between documents
   */
  private async extractRelationships(
    content: string,
    similarDocs: any[]
  ): Promise<DocumentRelationship[]> {
    if (similarDocs.length === 0) return [];
    
    const relationships: DocumentRelationship[] = [];
    
    for (const doc of similarDocs.slice(0, 3)) { // Limit to top 3
      if (doc.similarity > 0.8) {
        relationships.push({
          type: 'similar_to',
          targetId: doc.id,
          confidence: doc.similarity,
          description: `High similarity to "${doc.title}"`
        });
      }
    }
    
    return relationships;
  }
  
  /**
   * Update document with auto-generated tags and metadata
   */
  private async updateDocumentTags(documentId: string, result: AutoTaggingResult) {
    const metadata: EvidenceMetadata = {
      aiTags: result.tags,
      entities: result.entities,
      summary: result.summary,
      confidence: result.confidence,
      relationships: result.relationships,
      autoTaggedAt: new Date().toISOString()
    };
    
    await db.update(evidence)
      .set({
        tags: result.tags,
        summary: result.summary,
        // metadata: metadata, // Uncomment when schema supports metadata
        embedding: `[${result.embedding.join(',')}]` as any, // pgvector format
        updatedAt: new Date()
      })
      .where(eq(evidence.id, documentId));
  }
  
  /**
   * Batch auto-tag multiple documents
   */
  async batchAutoTag(documents: Array<{id: string, content: string, type: string}>) {
    const results = [];
    
    for (const doc of documents) {
      try {
        const result = await this.autoTagDocument(doc.id, doc.content, doc.type);
        results.push({ id: doc.id, success: true, result });
      } catch (error) {
        results.push({ id: doc.id, success: false, error: error.message });
      }
    }
    
    return results;
  }
  
  /**
   * Search documents using semantic similarity
   */
  async semanticSearch(query: string, limit = 10) {
    const queryEmbedding = await this.generateEmbedding(query);
    const embeddingStr = `[${queryEmbedding.join(',')}]`;
    
    try {
      const result = await db.execute(sql.raw(`
        SELECT 
          id, title, description, tags, summary,
          1 - (embedding <=> '${embeddingStr}'::vector) as similarity
        FROM evidence 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> '${embeddingStr}'::vector
        LIMIT ${limit}
      `));
      return result.rows.map((row: any) => ({
        id: row.id,
        title: row.title,
        description: row.description,
        tags: row.tags,
        summary: row.summary,
        similarity: row.similarity
      }));
    } catch (error) {
      console.error('Semantic search failed:', error);
      return [];
    }
  }
}

// Export singleton instance
export const aiAutoTaggingService = new AIAutoTaggingService();