// Nomic Embeddings Integration for Legal AI
// Phase 8: AI-Aware Browser States with Vector Intelligence

import { ollamaService } from '$lib/services/ollama-service';

export interface EmbeddingResult {
  embedding: number[];
  dimension: number;
  model: string;
  metadata?: Record<string, any>;
}

export interface DocumentChunk {
  id: string;
  text: string;
  metadata: {
    source?: string;
    page?: number;
    type?: 'contract' | 'evidence' | 'case_law' | 'regulation';
    jurisdiction?: string;
    date?: string;
  };
}

export class NomicEmbeddingsService {
  private readonly MODEL_NAME = 'nomic-embed-text';
  private readonly DIMENSION = 768;
  
  async embed(text: string): Promise<EmbeddingResult> {
    try {
      // Use Ollama service for Nomic embeddings
      const embedding = await ollamaService.generateEmbedding(text, this.MODEL_NAME);
      
      return {
        embedding,
        dimension: this.DIMENSION,
        model: this.MODEL_NAME,
        metadata: {
          length: text.length,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      console.error('Nomic embedding generation failed:', error);
      throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    // Process in smaller batches to avoid memory issues
    const BATCH_SIZE = 10;
    const results: EmbeddingResult[] = [];
    
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const batchPromises = batch.map(text => this.embed(text));
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    }
    
    return results;
  }

  async embedDocuments(documents: DocumentChunk[]): Promise<Array<DocumentChunk & { embedding: number[] }>> {
    const embeddings = await this.embedBatch(documents.map(doc => doc.text));
    
    return documents.map((doc, index) => ({
      ...doc,
      embedding: embeddings[index].embedding
    }));
  }

  // Phase 8: AI-Aware reranking with custom scoring
  rerank(results: Array<{ score: number; [key: string]: any }>, context: {
    userIntent?: string;
    timeOfDay?: string;
    focusedElement?: string;
    caseId?: string;
  }): Array<{ score: number; [key: string]: any }> {
    return results.map(result => {
      let adjustedScore = result.score;
      
      // Context-aware scoring adjustments
      if (context.userIntent && result.metadata?.intent === context.userIntent) {
        adjustedScore += 0.2;
      }
      
      if (context.caseId && result.metadata?.caseId === context.caseId) {
        adjustedScore += 0.15;
      }
      
      // Time-based relevance (legal deadlines, court hours)
      if (context.timeOfDay === 'business_hours' && result.metadata?.type === 'case_law') {
        adjustedScore += 0.1;
      }
      
      return {
        ...result,
        score: adjustedScore,
        originalScore: result.score,
        rerankingApplied: true
      };
    }).sort((a, b) => b.score - a.score);
  }

  // Phase 8: Matrix LOD-aware embedding with GPU considerations
  async embedWithLOD(text: string, lodLevel: 'low' | 'mid' | 'high' = 'mid'): Promise<EmbeddingResult> {
    // Adjust text processing based on LOD level for GPU performance
    let processedText = text;
    
    switch (lodLevel) {
      case 'low':
        // Truncate for low-detail requirements
        processedText = text.substring(0, 512);
        break;
      case 'high':
        // Full text with enhanced context
        processedText = text; // No truncation
        break;
      default:
        // Medium detail - balance between performance and accuracy
        processedText = text.substring(0, 2048);
    }
    
    const result = await this.embed(processedText);
    
    return {
      ...result,
      metadata: {
        ...result.metadata,
        lodLevel,
        originalLength: text.length,
        processedLength: processedText.length
      }
    };
  }
}

// Singleton instance for global use
export const nomicEmbeddings = new NomicEmbeddingsService();

// Export for backward compatibility
export default nomicEmbeddings;