// Enhanced Legal NLP Service using Transformers.js
// Implements sentence transformers for YoRHa Legal AI Platform

import { pipeline, env } from '@xenova/transformers';

// Configure for local execution
env.allowLocalModels = false;
env.useBrowserCache = true;

interface EmbeddingResult {
  data: Float32Array;
  dimensions: number;
  model: string;
}

interface SimilarityResult {
  text: string;
  score: number;
  index: number;
}

class LegalNLPService {
  private model: any = null;
  private isInitialized = false;
  private modelName = 'Xenova/all-MiniLM-L6-v2'; // 384 dimensions, fast inference

  async initialize(): Promise<void> {
    if (!this.model && !this.isInitialized) {
      console.log('🔧 Initializing Legal NLP Service with sentence transformers...');
      try {
        this.model = await pipeline('feature-extraction', this.modelName);
        this.isInitialized = true;
        console.log('✅ Legal NLP Service initialized successfully');
      } catch (error) {
        console.error('❌ Failed to initialize Legal NLP Service:', error);
        throw error;
      }
    }
  }

  async embedText(text: string): Promise<EmbeddingResult> {
    await this.initialize();
    
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const result = await this.model(text, { 
      pooling: 'mean', 
      normalize: true 
    });

    return {
      data: result.data,
      dimensions: result.data.length,
      model: this.modelName
    };
  }

  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    await this.initialize();
    
    const embeddings = await Promise.all(
      texts.map(text => this.embedText(text))
    );
    
    return embeddings;
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async similaritySearch(
    query: string, 
    documents: string[], 
    threshold = 0.5
  ): Promise<SimilarityResult[]> {
    const [queryEmbedding, ...docEmbeddings] = await this.embedBatch([query, ...documents]);
    
    const results = documents.map((doc, i) => ({
      text: doc,
      score: this.cosineSimilarity(queryEmbedding.data, docEmbeddings[i].data),
      index: i
    }));
    
    return results
      .filter(result => result.score >= threshold)
      .sort((a, b) => b.score - a.score);
  }

  // Legal-specific analysis methods
  async analyzeLegalDocument(content: string): Promise<{
    summary: string;
    keywords: string[];
    sentiment: 'positive' | 'negative' | 'neutral';
    complexity: 'low' | 'medium' | 'high';
    legalDomain: string[];
  }> {
    // Simplified analysis - would be enhanced with legal-specific models
    const sentences = this.splitIntoSentences(content);
    const embedding = await this.embedText(content);
    
    // Basic keyword extraction (would use more sophisticated methods in production)
    const keywords = this.extractKeywords(content);
    
    // Estimate complexity based on sentence length and legal terminology
    const complexity = this.estimateComplexity(content);
    
    // Detect legal domains
    const legalDomain = this.detectLegalDomains(content);
    
    return {
      summary: sentences.slice(0, 2).join(' '), // Simple summary
      keywords,
      sentiment: 'neutral', // Legal documents are typically neutral
      complexity,
      legalDomain
    };
  }

  private splitIntoSentences(text: string): string[] {
    // Enhanced sentence splitting for legal documents
    return text.split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  private extractKeywords(text: string): string[] {
    // Legal terminology extraction
    const legalTerms = [
      'contract', 'liability', 'damages', 'breach', 'statute', 'regulation',
      'precedent', 'jurisdiction', 'plaintiff', 'defendant', 'evidence',
      'testimony', 'discovery', 'motion', 'ruling', 'appeal', 'injunction'
    ];
    
    const words = text.toLowerCase().split(/\W+/);
    return legalTerms.filter(term => words.includes(term));
  }

  private estimateComplexity(text: string): 'low' | 'medium' | 'high' {
    const avgSentenceLength = text.split(/[.!?]+/).reduce((sum, s) => sum + s.length, 0) / text.split(/[.!?]+/).length;
    const legalTermCount = this.extractKeywords(text).length;
    
    if (avgSentenceLength > 100 || legalTermCount > 10) return 'high';
    if (avgSentenceLength > 50 || legalTermCount > 5) return 'medium';
    return 'low';
  }

  private detectLegalDomains(text: string): string[] {
    const domains = {
      'corporate': ['corporation', 'securities', 'shareholder', 'board', 'merger'],
      'employment': ['employee', 'discrimination', 'harassment', 'termination'],
      'intellectual_property': ['patent', 'trademark', 'copyright', 'trade secret'],
      'criminal': ['criminal', 'felony', 'misdemeanor', 'prosecution'],
      'civil_rights': ['discrimination', 'civil rights', 'equal protection']
    };
    
    const lowerText = text.toLowerCase();
    const detectedDomains: string[] = [];
    
    Object.entries(domains).forEach(([domain, terms]) => {
      if (terms.some(term => lowerText.includes(term))) {
        detectedDomains.push(domain);
      }
    });
    
    return detectedDomains;
  }

  // Chunk text for embedding storage
  chunkText(text: string, maxChunkSize = 500, overlap = 50): string[] {
    const sentences = this.splitIntoSentences(text);
    const chunks: string[] = [];
    let currentChunk = '';
    
    for (const sentence of sentences) {
      if (currentChunk.length + sentence.length > maxChunkSize && currentChunk) {
        chunks.push(currentChunk.trim());
        // Start new chunk with overlap
        const words = currentChunk.split(' ');
        currentChunk = words.slice(-overlap / 10).join(' ') + ' ' + sentence;
      } else {
        currentChunk += (currentChunk ? ' ' : '') + sentence;
      }
    }
    
    if (currentChunk) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }
}

// Export singleton instance
export const legalNLP = new LegalNLPService();

// Export types
export type { EmbeddingResult, SimilarityResult };

// Export class for testing
export { LegalNLPService };