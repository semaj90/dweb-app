// @ts-nocheck
/**
 * AI Summarization Service with Embeddings Generation
 * Comprehensive document processing and analysis pipeline
 */
import { ollamaCudaService } from "./ollama-cuda-service";
import { db, {,   evidence,,   embeddingCache,,   cases, } from
// Orphaned content: import { eq, sql, and, desc, isNotNull
import type { AiAnalysisResult } from "$lib/schemas/file-upload";
// Orphaned content: import {

import { SystemMessage, HumanMessage } from "@langchain/core/messages";

export interface SummarizationOptions {
  maxLength?: number;
  style?:
    | "bullet_points"
    | "paragraph"
    | "executive_summary"
    | "technical"
    | "legal";
  includeKeywords?: boolean;
  includeEntities?: boolean;
  includeSentiment?: boolean;
  includeCategories?: boolean;
  language?: string;
  confidenceThreshold?: number;
  useCache?: boolean;
}

export interface DocumentChunk {
  id: string;
  content: string;
  startIndex: number;
  endIndex: number;
  tokenCount: number;
  embedding?: number[];
  importance?: number;
}

export interface SummarizationResult {
  summary: string;
  keyPoints: string[];
  entities: Array<{
    name: string;
    type: string;
    confidence: number;
    mentions: number;
  }>;
  keywords: string[];
  categories: string[];
  sentiment?: {
    score: number;
    label: "positive" | "negative" | "neutral";
  };
  confidence: number;
  processingTime: number;
  model: string;
  chunks: DocumentChunk[];
  embedding?: number[];
  wordCount: number;
  readingTime: number; // in minutes
}

export interface BatchSummarizationResult {
  results: Array<{
    documentId: string;
    success: boolean;
    result?: SummarizationResult;
    error?: string;
  }>;
  totalProcessed: number;
  totalSuccess: number;
  totalFailures: number;
  processingTime: number;
}

class AISummarizationService {
  private static instance: AISummarizationService;
  private maxChunkSize = 4000; // Maximum characters per chunk
  private chunkOverlap = 200; // Overlap between chunks
  private cache = new Map<string, SummarizationResult>();

  private constructor() {}

  public static getInstance(): AISummarizationService {
    if (!AISummarizationService.instance) {
      AISummarizationService.instance = new AISummarizationService();
    }
    return AISummarizationService.instance;
  }

  /**
   * Summarize a document with comprehensive analysis
   */
  public async summarizeDocument(
    content: string,
    options: SummarizationOptions = {}
  ): Promise<SummarizationResult> {
    const startTime = Date.now();

    try {
      // Set defaults
      const opts: Required<SummarizationOptions> = {
        maxLength: options.maxLength || 500,
        style: options.style || "paragraph",
        includeKeywords: options.includeKeywords ?? true,
        includeEntities: options.includeEntities ?? true,
        includeSentiment: options.includeSentiment ?? false,
        includeCategories: options.includeCategories ?? true,
        language: options.language || "en",
        confidenceThreshold: options.confidenceThreshold || 0.7,
        useCache: options.useCache ?? true,
      };

      // Check cache first
      const cacheKey = this.generateCacheKey(content, opts);
      if (opts.useCache && this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey)!;
      }

      // Calculate basic metrics
      const wordCount = this.countWords(content);
      const readingTime = Math.ceil(wordCount / 200); // Assuming 200 WPM reading speed

      // Split content into manageable chunks
      const chunks = await this.chunkDocument(content);

      // Generate embeddings for chunks
      const chunksWithEmbeddings = await this.generateChunkEmbeddings(chunks);

      // Optimize Ollama for legal analysis
      await ollamaCudaService.optimizeForUseCase("legal-analysis");

      // Generate comprehensive summary
      const summaryPrompt = this.buildSummaryPrompt(
        content.substring(0, 8000),
        opts
      );
      const summaryResponse = await ollamaCudaService.chatCompletion(
        [
          new SystemMessage(
            "You are a legal AI assistant specializing in document analysis and summarization."
          ),
          new HumanMessage(summaryPrompt),
        ],
        {
          temperature: 0.3,
          maxTokens: 2000,
        }
      );

      // Parse AI response
      const analysis = this.parseAIResponse(summaryResponse);

      // Generate document embedding
      const documentEmbedding = await this.generateDocumentEmbedding(content);

      // Build result
      const result: SummarizationResult = {
        summary: analysis.summary,
        keyPoints: analysis.keyPoints,
        entities: analysis.entities,
        keywords: analysis.keywords,
        categories: analysis.categories,
        sentiment: opts.includeSentiment ? analysis.sentiment : undefined,
        confidence: analysis.confidence,
        processingTime: Date.now() - startTime,
        model: ollamaCudaService.currentModel,
        chunks: chunksWithEmbeddings,
        embedding: documentEmbedding,
        wordCount,
        readingTime,
      };

      // Cache result
      if (opts.useCache) {
        this.cache.set(cacheKey, result);
      }

      return result;
    } catch (error) {
      console.error("Document summarization failed:", error);
      throw new Error(
        `Summarization failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  /**
   * Batch summarize multiple documents
   */
  public async batchSummarize(
    documents: Array<{ id: string; content: string }>,
    options: SummarizationOptions = {}
  ): Promise<BatchSummarizationResult> {
    const startTime = Date.now();
    const results: BatchSummarizationResult["results"] = [];

    let totalSuccess = 0;
    let totalFailures = 0;

    for (const doc of documents) {
      try {
        const result = await this.summarizeDocument(doc.content, {
          ...options,
          useCache: true, // Enable caching for batch operations
        });

        results.push({
          documentId: doc.id,
          success: true,
          result,
        });
        totalSuccess++;

        // Brief pause to prevent overwhelming the AI service
        await new Promise((resolve) => setTimeout(resolve, 100));
      } catch (error) {
        results.push({
          documentId: doc.id,
          success: false,
          error: error instanceof Error ? error.message : "Unknown error",
        });
        totalFailures++;
      }
    }

    return {
      results,
      totalProcessed: documents.length,
      totalSuccess,
      totalFailures,
      processingTime: Date.now() - startTime,
    };
  }

  /**
   * Summarize evidence from database
   */
  public async summarizeEvidence(
    evidenceId: string,
    options: SummarizationOptions = {}
  ): Promise<SummarizationResult> {
    try {
      // Get evidence from database
      const evidenceRecord = await db
        .select()
        .from(evidence)
        .where(eq(evidence.id, evidenceId))
        .limit(1);

      if (evidenceRecord.length === 0) {
        throw new Error("Evidence not found");
      }

      const record = evidenceRecord[0];

      // Extract content based on file type
      let content = "";
      if (
        record.aiAnalysis &&
        typeof record.aiAnalysis === "object" &&
        "ocrText" in record.aiAnalysis
      ) {
        content = record.aiAnalysis.ocrText as string;
      } else if (record.summary) {
        content = record.summary;
      } else if (record.description) {
        content = record.description;
      } else {
        content = `${record.title}\n\nFile: ${record.fileName}\nType: ${record.mimeType}`;
      }

      if (!content.trim()) {
        throw new Error("No extractable content from evidence");
      }

      // Perform summarization
      const result = await this.summarizeDocument(content, options);

      // Update evidence record with AI summary
      await db
        .update(evidence)
        .set({
          aiSummary: result.summary,
          aiAnalysis: {
            ...(typeof record.aiAnalysis === "object" &&
            record.aiAnalysis !== null
              ? record.aiAnalysis
              : {}),
            summary: result.summary,
            keyPoints: result.keyPoints,
            entities: result.entities,
            keywords: result.keywords,
            categories: result.categories,
            confidence: result.confidence,
            processingTime: result.processingTime,
            model: result.model,
            updatedAt: new Date().toISOString(),
          },
          contentEmbedding: result.embedding,
          updatedAt: new Date(),
        })
        .where(eq(evidence.id, evidenceId));

      return result;
    } catch (error) {
      console.error(`Failed to summarize evidence ${evidenceId}:`, error);
      throw error;
    }
  }

  /**
   * Generate case summary from all evidence
   */
  public async summarizeCase(
    caseId: string,
    options: SummarizationOptions = {}
  ): Promise<SummarizationResult> {
    try {
      // Get case and all its evidence
      const caseRecord = await db
        .select()
        .from(cases)
        .where(eq(cases.id, caseId))
        .limit(1);

      if (caseRecord.length === 0) {
        throw new Error("Case not found");
      }

      const caseData = caseRecord[0];

      // Get all evidence for this case
      const evidenceRecords = await db
        .select()
        .from(evidence)
        .where(eq(evidence.caseId, caseId))
        .orderBy(desc(evidence.createdAt));

      // Combine all content
      let combinedContent = `Case: ${caseData.title}\n`;
      combinedContent += `Description: ${caseData.description}\n`;
      combinedContent += `Category: ${caseData.category}\n`;
      combinedContent += `Priority: ${caseData.priority}\n\n`;

      combinedContent += `Evidence Summary:\n`;
      evidenceRecords.forEach((ev, index) => {
        combinedContent += `${index + 1}. ${ev.title}\n`;
        if (ev.aiSummary) {
          combinedContent += `   Summary: ${ev.aiSummary}\n`;
        }
        if (ev.description) {
          combinedContent += `   Description: ${ev.description}\n`;
        }
        combinedContent += `   Type: ${ev.evidenceType}\n\n`;
      });

      // Perform comprehensive case summarization
      const result = await this.summarizeDocument(combinedContent, {
        ...options,
        style: "executive_summary",
        maxLength: 800,
      });

      // Update case record with AI summary
      await db
        .update(cases)
        .set({
          aiSummary: result.summary,
          contentEmbedding: result.embedding,
          updatedAt: new Date(),
        })
        .where(eq(cases.id, caseId));

      return result;
    } catch (error) {
      console.error(`Failed to summarize case ${caseId}:`, error);
      throw error;
    }
  }

  /**
   * Find similar documents using embeddings
   */
  public async findSimilarDocuments(
    documentId: string,
    threshold: number = 0.8,
    limit: number = 10
  ): Promise<
    Array<{
      documentId: string;
      similarity: number;
      title: string;
      type: string;
      summary?: string;
    }>
  > {
    try {
      // Get document embedding
      const doc = await db
        .select()
        .from(evidence)
        .where(eq(evidence.id, documentId))
        .limit(1);

      if (doc.length === 0 || !doc[0].contentEmbedding) {
        throw new Error("Document not found or no embedding available");
      }

      const queryEmbedding = doc[0].contentEmbedding;

      // Find similar documents using cosine similarity
      const similarDocs = await db
        .select({
          id: evidence.id,
          title: evidence.title,
          evidenceType: evidence.evidenceType,
          aiSummary: evidence.aiSummary,
          similarity: sql<number>`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding})`,
        })
        .from(evidence)
        .where(
          and(
            isNotNull(evidence.contentEmbedding),
            sql`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding}) > ${threshold}`
          )
        )
        .orderBy(
          sql`1 - (${evidence.contentEmbedding} <=> ${queryEmbedding}) DESC`
        )
        .limit(limit);

      return similarDocs.map((doc) => ({
        documentId: doc.id,
        similarity: doc.similarity,
        title: doc.title,
        type: doc.evidenceType,
        summary: doc.aiSummary || undefined,
      }));
    } catch (error) {
      console.error("Failed to find similar documents:", error);
      throw error;
    }
  }

  /**
   * Generate summary statistics for a case
   */
  public async getCaseSummaryStats(caseId: string): Promise<{
    totalEvidence: number;
    processedEvidence: number;
    avgConfidence: number;
    mostCommonCategories: string[];
    totalWordCount: number;
    avgReadingTime: number;
  }> {
    try {
      const evidenceRecords = await db
        .select()
        .from(evidence)
        .where(eq(evidence.caseId, caseId));

      const totalEvidence = evidenceRecords.length;
      const processedEvidence = evidenceRecords.filter(
        (e) => e.aiAnalysis && Object.keys(e.aiAnalysis).length > 0
      ).length;

      // Calculate average confidence
      const confidenceScores = evidenceRecords
        .map((e) => (e.aiAnalysis as any)?.confidence)
        .filter((c) => typeof c === "number");
      const avgConfidence =
        confidenceScores.length > 0
          ? confidenceScores.reduce((sum, c) => sum + c, 0) /
            confidenceScores.length
          : 0;

      // Get most common categories
      const allCategories = evidenceRecords
        .flatMap((e) => (e.aiAnalysis as any)?.categories || [])
        .filter((c) => typeof c === "string");

      const categoryCount = allCategories.reduce(
        (acc, cat) => {
          acc[cat] = (acc[cat] || 0) + 1;
          return acc;
        },
        {} as Record<string, number>
      );

      const mostCommonCategories = Object.entries(categoryCount)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([cat]) => cat);

      // Estimate word counts and reading time
      const totalWordCount = evidenceRecords.reduce((sum, e) => {
        const content = e.aiSummary || e.description || "";
        return sum + this.countWords(content);
      }, 0);

      const avgReadingTime = Math.ceil(totalWordCount / 200);

      return {
        totalEvidence,
        processedEvidence,
        avgConfidence,
        mostCommonCategories,
        totalWordCount,
        avgReadingTime,
      };
    } catch (error) {
      console.error("Failed to get case summary stats:", error);
      throw error;
    }
  }

  // Private helper methods

  private chunkDocument(content: string): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    const sentences = content
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 0);

    let currentChunk = "";
    let startIndex = 0;
    let chunkId = 0;

    for (const sentence of sentences) {
      const trimmedSentence = sentence.trim();
      if (!trimmedSentence) continue;

      const potentialChunk =
        currentChunk + (currentChunk ? ". " : "") + trimmedSentence;

      if (
        potentialChunk.length > this.maxChunkSize &&
        currentChunk.length > 0
      ) {
        // Save current chunk
        chunks.push({
          id: `chunk_${chunkId++}`,
          content: currentChunk,
          startIndex,
          endIndex: startIndex + currentChunk.length,
          tokenCount: this.estimateTokenCount(currentChunk),
        });

        // Start new chunk with overlap
        const overlapText = this.getOverlapText(
          currentChunk,
          this.chunkOverlap
        );
        currentChunk = overlapText + trimmedSentence;
        startIndex = startIndex + currentChunk.length - overlapText.length;
      } else {
        currentChunk = potentialChunk;
      }
    }

    // Add final chunk
    if (currentChunk.trim()) {
      chunks.push({
        id: `chunk_${chunkId}`,
        content: currentChunk,
        startIndex,
        endIndex: startIndex + currentChunk.length,
        tokenCount: this.estimateTokenCount(currentChunk),
      });
    }

    return chunks;
  }

  private async generateChunkEmbeddings(
    chunks: DocumentChunk[]
  ): Promise<DocumentChunk[]> {
    const chunksWithEmbeddings: DocumentChunk[] = [];

    for (const chunk of chunks) {
      try {
        const embedding = await ollamaCudaService.generateEmbedding(
          chunk.content
        );
        chunksWithEmbeddings.push({
          ...chunk,
          embedding,
        });
      } catch (error) {
        console.warn(
          `Failed to generate embedding for chunk ${chunk.id}:`,
          error
        );
        chunksWithEmbeddings.push(chunk);
      }
    }

    return chunksWithEmbeddings;
  }

  private async generateDocumentEmbedding(content: string): Promise<number[]> {
    try {
      // Use first 4000 characters for document-level embedding
      const truncatedContent = content.substring(0, 4000);
      return await ollamaCudaService.generateEmbedding(truncatedContent);
    } catch (error) {
      console.error("Failed to generate document embedding:", error);
      throw error;
    }
  }

  private buildSummaryPrompt(
    content: string,
    options: Required<SummarizationOptions>
  ): string {
    let prompt = `Analyze and summarize the following legal document. `;

    switch (options.style) {
      case "bullet_points":
        prompt += `Provide a summary in bullet point format. `;
        break;
      case "executive_summary":
        prompt += `Provide an executive summary suitable for legal professionals. `;
        break;
      case "technical":
        prompt += `Provide a technical analysis with detailed insights. `;
        break;
      case "legal":
        prompt += `Provide a legal analysis focusing on legal implications and precedents. `;
        break;
      default:
        prompt += `Provide a comprehensive paragraph summary. `;
    }

    prompt += `Maximum length: ${options.maxLength} words.\n\n`;

    const requestedAnalysis = [];
    if (options.includeKeywords)
      requestedAnalysis.push("key terms and keywords");
    if (options.includeEntities)
      requestedAnalysis.push(
        "important entities (people, organizations, locations)"
      );
    if (options.includeCategories)
      requestedAnalysis.push("legal categories and classifications");
    if (options.includeSentiment) requestedAnalysis.push("sentiment analysis");

    if (requestedAnalysis.length > 0) {
      prompt += `Additionally, identify: ${requestedAnalysis.join(", ")}.\n\n`;
    }

    prompt += `Document content:\n${content}\n\n`;
    prompt += `Provide your response as a JSON object with the following structure:
{
  "summary": "Your summary here",
  "keyPoints": ["key point 1", "key point 2"],
  "entities": [{"name": "Entity Name", "type": "person|organization|location|other", "confidence": 0.9, "mentions": 3}],
  "keywords": ["keyword1", "keyword2"],
  "categories": ["category1", "category2"],
  ${options.includeSentiment ? '"sentiment": {"score": 0.5, "label": "neutral"},' : ""}
  "confidence": 0.85
}`;

    return prompt;
  }

  private parseAIResponse(response: string): any {
    try {
      // Try to parse as JSON
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }

      // Fallback parsing
      return {
        summary: response.substring(0, 500),
        keyPoints: [],
        entities: [],
        keywords: [],
        categories: [],
        confidence: 0.5,
      };
    } catch (error) {
      console.warn("Failed to parse AI response:", error);
      return {
        summary: response.substring(0, 500),
        keyPoints: [],
        entities: [],
        keywords: [],
        categories: [],
        confidence: 0.5,
      };
    }
  }

  private generateCacheKey(
    content: string,
    options: Required<SummarizationOptions>
  ): string {
    const contentHash = createHash("md5").update(content).digest("hex");
    const optionsHash = createHash("md5")
      .update(JSON.stringify(options))
      .digest("hex");
    return `summary_${contentHash}_${optionsHash}`;
  }

  private countWords(text: string): number {
    return text
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0).length;
  }

  private estimateTokenCount(text: string): number {
    // Rough estimation: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  }

  private getOverlapText(text: string, overlapLength: number): string {
    if (text.length <= overlapLength) return text;

    // Try to find a sentence boundary for clean overlap
    const lastPart = text.substring(text.length - overlapLength);
    const sentenceMatch = lastPart.match(/[.!?]\s+(.*)$/);

    if (sentenceMatch) {
      return sentenceMatch[1];
    }

    return lastPart;
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }
}

// Export singleton instance
export const aiSummarizationService = AISummarizationService.getInstance();
export default aiSummarizationService;
