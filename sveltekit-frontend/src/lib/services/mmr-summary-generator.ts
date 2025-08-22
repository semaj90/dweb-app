/**
 * MMR-based Summary Generator
 * Generates summaries using Maximal Marginal Relevance to balance relevance and diversity
 */

import type { LegalDocument } from './types/legal';

interface SummaryRequest {
  documents: LegalDocument[];
  query?: string;
  maxLength?: number;
  diversityLambda?: number;
}

interface SummaryResult {
  summary: string;
  selectedSentences?: string[];
  relevanceScores?: number[];
  diversityScores?: number[];
  metadata: {
    documentsProcessed?: number;
    processingTime: number;
    lambda?: number;
    sentenceCount?: number;
    sourceDocuments?: number;
    averageRelevance?: number;
    averageDiversity?: number;
    mmrConfig?: unknown;
    sources?: unknown[];
    confidence?: number;
  };
}

interface MMRConfig {
  lambda: number; // Balance between relevance and diversity (0.0 to 1.0)
  maxSummaryLength: number; // Maximum characters in summary
  maxSentences: number; // Maximum sentences to include
  diversityThreshold: number; // Minimum similarity threshold for diversity
  relevanceWeight: number; // Weight for relevance scoring
}

interface ScoredSentence {
  text: string;
  relevanceScore: number;
  diversityScore: number;
  mmrScore: number;
  position: number;
  embeddings?: number[];
}

export class MMRSummaryGenerator {
  private config: MMRConfig;
  private embeddingCache: Map<string, number[]> = new Map();

  constructor(config: Partial<MMRConfig> = {}) {
    this.config = {
      lambda: 0.7, // Favor relevance slightly over diversity
      maxSummaryLength: 500,
      maxSentences: 5,
      diversityThreshold: 0.8,
      relevanceWeight: 1.0,
      ...config,
    };
  }

  async generateSummary(
    documents: LegalDocument[],
    query: string,
    request: SummaryRequest
  ): Promise<SummaryResult> {
    const startTime = Date.now();

    try {
      // 1. Extract and score sentences from all documents
      const sentences = await this.extractSentences(documents);

      // 2. Compute relevance scores against query
      const scoredSentences = await this.scoreRelevance(sentences, query);

      // 3. Apply MMR algorithm to select diverse, relevant sentences
      const selectedSentences = this.selectWithMMR(scoredSentences);

      // 4. Construct final summary
      const summary = this.constructSummary(selectedSentences, request);

      return {
        summary: summary.text,
        metadata: {
          processingTime: Date.now() - startTime,
          sentenceCount: selectedSentences.length,
          sourceDocuments: documents.length,
          averageRelevance: this.calculateAverageScore(selectedSentences, 'relevance'),
          averageDiversity: this.calculateAverageScore(selectedSentences, 'diversity'),
          mmrConfig: this.config,
          sources: this.extractSources(selectedSentences, documents),
          confidence: this.calculateConfidence(selectedSentences),
          ...summary.metadata,
        },
      };
    } catch (error) {
      console.error('[MMR] Summary generation failed:', error);
      throw new Error(`MMR summary generation failed: ${error.message}`);
    }
  }

  private async extractSentences(documents: LegalDocument[]): Promise<string[]> {
    const sentences: string[] = [];

    for (const doc of documents) {
      try {
        // Use enhanced sentence splitter for legal text
  const { splitSentencesEnhanced } = await import('$text/enhanced-sentence-splitter');
        const docSentences = splitSentencesEnhanced(doc.content);

        // Add position metadata as sentence prefix
        docSentences.forEach((sentence, index) => {
          sentences.push(`${doc.id}:${index}:${sentence}`);
        });
      } catch (error) {
        console.warn('[MMR] Fallback to basic splitting for doc:', doc.id);
        // Fallback to basic sentence splitting
        const basicSentences = doc.content.split(/[.!?]+/).filter((s) => s.trim().length > 10);
        basicSentences.forEach((sentence, index) => {
          sentences.push(`${doc.id}:${index}:${sentence.trim()}`);
        });
      }
    }

    return sentences;
  }

  private async scoreRelevance(sentences: string[], query: string): Promise<ScoredSentence[]> {
    const scored: ScoredSentence[] = [];

    for (let i = 0; i < sentences.length; i++) {
      const [docId, position, ...textParts] = sentences[i].split(':');
      const text = textParts.join(':');

      if (text.trim().length < 20) continue; // Skip very short sentences

      const relevanceScore = await this.calculateRelevanceScore(text, query);

      scored.push({
        text: text.trim(),
        relevanceScore,
        diversityScore: 0, // Will be calculated during MMR
        mmrScore: 0,
        position: parseInt(position) || i,
        embeddings: await this.getEmbeddings(text),
      });
    }

    // Sort by relevance initially
    return scored.sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  private selectWithMMR(candidates: ScoredSentence[]): ScoredSentence[] {
    const selected: ScoredSentence[] = [];
    const remaining = [...candidates];

    // Select the most relevant sentence first
    if (remaining.length > 0) {
      const first = remaining.shift()!;
      first.mmrScore = first.relevanceScore;
      selected.push(first);
    }

    // Apply MMR for subsequent selections
    while (selected.length < this.config.maxSentences && remaining.length > 0) {
      let bestIndex = -1;
      let bestScore = -Infinity;

      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i];

        // Calculate diversity score (minimum similarity to already selected)
        const diversityScore = this.calculateDiversityScore(candidate, selected);

        // MMR score: λ * relevance + (1-λ) * diversity
        const mmrScore =
          this.config.lambda * candidate.relevanceScore + (1 - this.config.lambda) * diversityScore;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIndex = i;
        }
      }

      if (bestIndex >= 0) {
        const selected_sentence = remaining.splice(bestIndex, 1)[0];
        selected_sentence.mmrScore = bestScore;
        selected_sentence.diversityScore = this.calculateDiversityScore(
          selected_sentence,
          selected
        );
        selected.push(selected_sentence);
      } else {
        break; // No more suitable candidates
      }
    }

    return selected;
  }

  private calculateDiversityScore(candidate: ScoredSentence, selected: ScoredSentence[]): number {
    if (selected.length === 0) return 1.0;

    let minSimilarity = Infinity;

    for (const sentence of selected) {
      const similarity = this.calculateSimilarity(candidate, sentence);
      minSimilarity = Math.min(minSimilarity, similarity);
    }

    // Convert similarity to diversity (lower similarity = higher diversity)
    return 1.0 - minSimilarity;
  }

  private calculateSimilarity(sent1: ScoredSentence, sent2: ScoredSentence): number {
    // Use embeddings if available
    if (sent1.embeddings && sent2.embeddings) {
      return this.cosineSimilarity(sent1.embeddings, sent2.embeddings);
    }

    // Fallback to lexical similarity
    return this.lexicalSimilarity(sent1.text, sent2.text);
  }

  private cosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length) return 0;

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    if (norm1 === 0 || norm2 === 0) return 0;

    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  private lexicalSimilarity(text1: string, text2: string): number {
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter((word) => words2.has(word)));
    const union = new Set([...words1, ...words2]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private async calculateRelevanceScore(sentence: string, query: string): Promise<number> {
    // Simple relevance based on keyword overlap and TF-IDF concepts
    const queryTerms = query.toLowerCase().split(/\s+/);
    const sentenceTerms = sentence.toLowerCase().split(/\s+/);

    let score = 0;
    const sentenceLength = sentenceTerms.length;

    for (const term of queryTerms) {
      const termFreq = sentenceTerms.filter((st) => st.includes(term)).length;
      if (termFreq > 0) {
        // TF-IDF inspired scoring
        const tf = termFreq / sentenceLength;
        const idf = Math.log(sentenceLength / termFreq);
        score += tf * idf * this.config.relevanceWeight;
      }
    }

    // Normalize by query length
    return score / queryTerms.length;
  }

  private async getEmbeddings(text: string): Promise<number[] | undefined> {
    if (this.embeddingCache.has(text)) {
      return this.embeddingCache.get(text);
    }

    try {
      // Try to get embeddings from Ollama
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: text,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const embeddings = data.embedding || data.embeddings;
        if (embeddings) {
          this.embeddingCache.set(text, embeddings);
          return embeddings;
        }
      }
    } catch (error) {
      console.warn('[MMR] Embedding generation failed, using lexical similarity');
    }

    return undefined;
  }

  private constructSummary(
    sentences: ScoredSentence[],
    request: SummaryRequest
  ): { text: string; metadata: any } {
    // Sort by original position to maintain logical flow
    const ordered = sentences.sort((a, b) => a.position - b.position);

    let summary = '';
    let currentLength = 0;

    for (const sentence of ordered) {
      const addition = (summary ? ' ' : '') + sentence.text;

      if (currentLength + addition.length > this.config.maxSummaryLength) {
        // Try to fit a truncated version
        const remaining = this.config.maxSummaryLength - currentLength;
        if (remaining > 50) {
          // Only if we have meaningful space
          summary += addition.substring(0, remaining - 3) + '...';
        }
        break;
      }

      summary += addition;
      currentLength += addition.length;
    }

    return {
      text: summary,
      metadata: {
        actualLength: summary.length,
        sentencesUsed: ordered.length,
        truncated: currentLength >= this.config.maxSummaryLength,
      },
    };
  }

  private calculateAverageScore(
    sentences: ScoredSentence[],
    scoreType: 'relevance' | 'diversity'
  ): number {
    if (sentences.length === 0) return 0;

    const sum = sentences.reduce((acc, sent) => {
      return acc + (scoreType === 'relevance' ? sent.relevanceScore : sent.diversityScore);
    }, 0);

    return sum / sentences.length;
  }

  private extractSources(sentences: ScoredSentence[], documents: LegalDocument[]): string[] {
    const docIds = new Set<string>();

    // This would need to be enhanced to track document IDs through the pipeline
    // For now, return all source documents
    return documents.map((doc) => doc.title || doc.id);
  }

  private calculateConfidence(sentences: ScoredSentence[]): number {
    if (sentences.length === 0) return 0;

    // Confidence based on average MMR score and sentence count
    const avgMmrScore = sentences.reduce((acc, sent) => acc + sent.mmrScore, 0) / sentences.length;
    const lengthFactor = Math.min(sentences.length / this.config.maxSentences, 1.0);

    return avgMmrScore * lengthFactor;
  }
}

// Convenience function for quick MMR summarization
export async function generateMMRSummary(
  documents: LegalDocument[],
  query: string,
  config: Partial<MMRConfig> = {}
): Promise<SummaryResult> {
  const generator = new MMRSummaryGenerator(config);

  const request: SummaryRequest = {
    // 'type' prop removed; align with SummaryRequest interface sans extraneous fields
    query,
    maxLength: config.maxSummaryLength || 500,
    format: 'paragraph',
  } as any;

  return generator.generateSummary(documents, query, request);
}

// Integration test helper
export async function testMMRSummaryGeneration(): Promise<boolean> {
  try {
    const mockDocuments: LegalDocument[] = [
      {
        id: 'doc1',
        title: 'Contract Law Basics',
        content:
          'A contract is a legally binding agreement. It requires offer, acceptance, and consideration. Contracts can be written or oral. Written contracts are generally preferred for important agreements.',
        type: 'legal',
        createdAt: new Date(),
        updatedAt: new Date(),
      },
      {
        id: 'doc2',
        title: 'Employment Law',
        content:
          'Employment contracts govern the relationship between employers and employees. They specify wages, benefits, and working conditions. Termination clauses are important considerations.',
        type: 'legal',
        createdAt: new Date(),
        updatedAt: new Date(),
      },
    ];

    const result = await generateMMRSummary(mockDocuments, 'contract requirements and terms', {
      maxSentences: 3,
      maxSummaryLength: 200,
    });

    const isValid =
      result.summary.length > 0 &&
      result.summary.length <= 200 &&
      result.metadata.sentenceCount <= 3;

    console.log('[test] MMR summary generation:', isValid ? 'PASS' : 'FAIL');
    console.log('[test] Result:', {
      summary: result.summary.substring(0, 100) + '...',
      length: result.summary.length,
      sentences: result.metadata.sentenceCount,
    });

    return isValid;
  } catch (error) {
    console.error('[test] MMR summary generation failed:', error);
    return false;
  }
}
