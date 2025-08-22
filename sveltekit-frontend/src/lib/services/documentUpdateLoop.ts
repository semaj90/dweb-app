
// Document Update Loop Service
// Auto re-embed and re-rank on document changes with intelligent diff detection

import { db } from "$lib/server/database";
import {
  documents, 
  documentVectors, 
  caseSummaryVectors,
  evidenceVectors,
  queryVectors
} from "$lib/db/schema";
import { eq, sql, and, desc } from "drizzle-orm";

import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import crypto from 'crypto';

// ============================================================================
// CONFIGURATION & TYPES
// ============================================================================

interface DocumentChange {
  documentId: string;
  changeType: 'content' | 'metadata' | 'analysis';
  oldContent?: string;
  newContent: string;
  changeHash: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  affectedChunks?: number[];
}

interface ReembedResult {
  documentId: string;
  chunksUpdated: number;
  chunksAdded: number;
  chunksRemoved: number;
  similarityImpact: number;
  processingTime: number;
  rerankedQueries: number;
}

interface RerankingJob {
  queryId: string;
  query: string;
  originalResults: Array<{id: string, score: number}>;
  newResults: Array<{id: string, score: number}>;
  improvement: number;
}

// ============================================================================
// DOCUMENT UPDATE DETECTION
// ============================================================================

export class DocumentUpdateLoop {
  private embeddings: OllamaEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private updateQueue: DocumentChange[] = [];
  private isProcessing: boolean = false;

  constructor() {
    this.embeddings = new OllamaEmbeddings({
      baseUrl: 'http://localhost:11434',
      model: 'nomic-embed-text',
    });

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
  }

  // ============================================================================
  // CHANGE DETECTION
  // ============================================================================

  async detectDocumentChanges(documentId: string, newContent: string): Promise<DocumentChange | null> {
    try {
      // Get current document
      const [currentDoc] = await db
        .select()
        .from(documents)
        .where(eq(documents.id, documentId))
        .limit(1);

      if (!currentDoc) {
        throw new Error(`Document ${documentId} not found`);
      }

      const oldContent = currentDoc.extractedText || '';
      
      // Quick hash comparison
      const oldHash = crypto.createHash('sha256').update(oldContent).digest('hex');
      const newHash = crypto.createHash('sha256').update(newContent).digest('hex');

      if (oldHash === newHash) {
        return null; // No changes detected
      }

      // Calculate content similarity to determine priority
      const oldEmbedding = await this.embeddings.embedQuery(oldContent.substring(0, 1000));
      const newEmbedding = await this.embeddings.embedQuery(newContent.substring(0, 1000));
      
      const similarity = this.cosineSimilarity(oldEmbedding, newEmbedding);
      const priority = this.calculateChangePriority(similarity, oldContent.length, newContent.length);

      // Detect affected chunks
      const affectedChunks = await this.detectAffectedChunks(documentId, oldContent, newContent);

      const change: DocumentChange = {
        documentId,
        changeType: 'content',
        oldContent,
        newContent,
        changeHash: newHash,
        priority,
        affectedChunks
      };

      console.log(`📝 Document change detected: ${documentId} (priority: ${priority}, similarity: ${similarity.toFixed(3)})`);
      
      return change;

    } catch (error) {
      console.error('❌ Change detection failed:', error);
      throw error;
    }
  }

  private calculateChangePriority(similarity: number, oldLength: number, newLength: number): 'low' | 'medium' | 'high' | 'critical' {
    const lengthChange = Math.abs(newLength - oldLength) / Math.max(oldLength, 1);
    
    if (similarity < 0.3 || lengthChange > 0.5) return 'critical';
    if (similarity < 0.6 || lengthChange > 0.3) return 'high';
    if (similarity < 0.8 || lengthChange > 0.1) return 'medium';
    return 'low';
  }

  private async detectAffectedChunks(documentId: string, oldContent: string, newContent: string): Promise<number[]> {
    try {
      // Get existing chunks
      const existingChunks = await db
        .select({
          chunkIndex: documentVectors.chunkIndex,
          content: documentVectors.content
        })
        .from(documentVectors)
        .where(eq(documentVectors.documentId, documentId))
        .orderBy(documentVectors.chunkIndex);

      // Split new content
      const newChunks = await this.textSplitter.splitText(newContent);
      
      const affectedChunks: number[] = [];

      // Compare chunks to find differences
      for (let i = 0; i < Math.max(existingChunks.length, newChunks.length); i++) {
        const oldChunk = existingChunks[i]?.content || '';
        const newChunk = newChunks[i] || '';
        
        if (oldChunk !== newChunk) {
          affectedChunks.push(i);
        }
      }

      return affectedChunks;

    } catch (error) {
      console.warn('Failed to detect affected chunks:', error);
      return []; // Return empty array on error
    }
  }

  // ============================================================================
  // RE-EMBEDDING PIPELINE
  // ============================================================================

  async reembedDocument(change: DocumentChange): Promise<ReembedResult> {
    const startTime = Date.now();
    console.log(`🔄 Re-embedding document: ${change.documentId}`);

    try {
      // Delete existing vectors
      await db
        .delete(documentVectors)
        .where(eq(documentVectors.documentId, change.documentId));

      // Split new content into chunks
      const chunks = await this.textSplitter.splitText(change.newContent);
      
      // Generate embeddings for all chunks
      const embeddings = await Promise.all(
        chunks.map((chunk: any) => this.embeddings.embedQuery(chunk))
      );

      // Store new vectors
      const vectorRecords = chunks.map((chunk, index) => ({
        documentId: change.documentId,
        chunkIndex: index,
        content: chunk,
        embedding: embeddings[index],
        metadata: {
          reembeddedAt: new Date().toISOString(),
          changeHash: change.changeHash,
          chunkSize: chunk.length,
          priority: change.priority
        }
      }));

      await db.insert(documentVectors).values(vectorRecords);

      // Update document record
      await db
        .update(documents)
        .set({
          extractedText: change.newContent,
          updatedAt: new Date(),
          analysis: {
            lastReembedded: new Date().toISOString(),
            chunksCount: chunks.length,
            changeHash: change.changeHash,
            priority: change.priority
          }
        })
        .where(eq(documents.id, change.documentId));

      const processingTime = Date.now() - startTime;

      const result: ReembedResult = {
        documentId: change.documentId,
        chunksUpdated: chunks.length,
        chunksAdded: Math.max(0, chunks.length - (change.affectedChunks?.length || 0)),
        chunksRemoved: Math.max(0, (change.affectedChunks?.length || 0) - chunks.length),
        similarityImpact: 0, // Will be calculated in re-ranking
        processingTime,
        rerankedQueries: 0 // Will be updated in re-ranking
      };

      console.log(`✅ Re-embedding complete: ${chunks.length} chunks in ${processingTime}ms`);
      
      return result;

    } catch (error) {
      console.error('❌ Re-embedding failed:', error);
      throw error;
    }
  }

  // ============================================================================
  // INTELLIGENT RE-RANKING
  // ============================================================================

  async rerankAffectedQueries(documentId: string): Promise<RerankingJob[]> {
    console.log(`🏆 Re-ranking queries affected by document: ${documentId}`);

    try {
      // Find recent queries that returned this document
      const affectedQueries = await db
        .select({
          id: queryVectors.id,
          userId: queryVectors.userId,
          query: queryVectors.query,
          embedding: queryVectors.embedding,
          clickedResults: queryVectors.clickedResults
        })
        .from(queryVectors)
        .where(
          and(
            sql`${queryVectors.createdAt} > NOW() - INTERVAL '7 days'`,
            sql`${queryVectors.clickedResults}::jsonb ? ${documentId}`
          )
        )
        .orderBy(desc(queryVectors.createdAt))
        .limit(20);

      const rerankingJobs: RerankingJob[] = [];

      for (const queryRecord of affectedQueries) {
        const job = await this.rerankSingleQuery(queryRecord, documentId);
        if (job) {
          rerankingJobs.push(job);
        }
      }

      console.log(`✅ Re-ranked ${rerankingJobs.length} affected queries`);
      
      return rerankingJobs;

    } catch (error) {
      console.error('❌ Re-ranking failed:', error);
      return [];
    }
  }

  private async rerankSingleQuery(queryRecord: any, changedDocumentId: string): Promise<RerankingJob | null> {
    try {
      const queryEmbedding = queryRecord.embedding;
      
      // Get original results (from clicked results)
      const originalResults = (queryRecord.clickedResults as unknown[]) || [];

      // Perform new search with updated embeddings
      const newSearchResults = await db
        .select({
          id: documentVectors.documentId,
          similarity: sql<number>`1 - (${documentVectors.embedding} <=> ${queryEmbedding})`
        })
        .from(documentVectors)
        .where(sql`1 - (${documentVectors.embedding} <=> ${queryEmbedding}) > 0.5`)
        .orderBy(sql`${documentVectors.embedding} <=> ${queryEmbedding}`)
        .limit(10);

      const newResults = newSearchResults.map((r: any) => ({
        id: r.id,
        score: r.similarity
      }));

      // Calculate improvement (simplified metric)
      const changedDocResult = newResults.find((r: any) => r.id === changedDocumentId);
      const originalDocResult = originalResults.find((r: any) => r.id === changedDocumentId);
      
      const improvement = changedDocResult && originalDocResult 
        ? changedDocResult.score - (originalDocResult.score || 0)
        : 0;

      return {
        queryId: queryRecord.id,
        query: queryRecord.query,
        originalResults,
        newResults,
        improvement
      };

    } catch (error) {
      console.warn('Failed to re-rank single query:', error);
      return null;
    }
  }

  // ============================================================================
  // QUEUE PROCESSING
  // ============================================================================

  async queueDocumentUpdate(documentId: string, newContent: string): Promise<void> {
    const change = await this.detectDocumentChanges(documentId, newContent);
    
    if (change) {
      this.updateQueue.push(change);
      this.processQueue(); // Don't await - process in background
    }
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessing || this.updateQueue.length === 0) {
      return;
    }

    this.isProcessing = true;
    console.log(`🔄 Processing ${this.updateQueue.length} document updates`);

    try {
      // Sort by priority
      this.updateQueue.sort((a, b) => {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      });

      while (this.updateQueue.length > 0) {
        const change = this.updateQueue.shift();
        if (!change) continue;

        try {
          // Re-embed document
          const reembedResult = await this.reembedDocument(change);
          
          // Re-rank affected queries
          const rerankingJobs = await this.rerankAffectedQueries(change.documentId);
          
          reembedResult.rerankedQueries = rerankingJobs.length;
          reembedResult.similarityImpact = rerankingJobs.reduce((sum, job) => sum + job.improvement, 0) / rerankingJobs.length;

          console.log(`✅ Document update complete: ${change.documentId}`, {
            chunksUpdated: reembedResult.chunksUpdated,
            queriesReranked: reembedResult.rerankedQueries,
            avgImprovement: reembedResult.similarityImpact?.toFixed(3)
          });

        } catch (error) {
          console.error(`❌ Failed to process document update: ${change.documentId}`, error);
        }
      }

    } finally {
      this.isProcessing = false;
    }
  }

  // ============================================================================
  // UTILITIES
  // ============================================================================

  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  // ============================================================================
  // PUBLIC API
  // ============================================================================

  async getQueueStatus() {
    return {
      queued: this.updateQueue.length,
      processing: this.isProcessing,
      priorities: this.updateQueue.reduce((acc, change) => {
        acc[change.priority] = (acc[change.priority] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    };
  }

  async forceReembedDocument(documentId: string): Promise<ReembedResult> {
    const [document] = await db
      .select()
      .from(documents)
      .where(eq(documents.id, documentId))
      .limit(1);

    if (!document) {
      throw new Error(`Document ${documentId} not found`);
    }

    const change: DocumentChange = {
      documentId,
      changeType: 'content',
      newContent: document.extractedText || '',
      changeHash: crypto.createHash('sha256').update(document.extractedText || '').digest('hex'),
      priority: 'high'
    };

    return await this.reembedDocument(change);
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const documentUpdateLoop = new DocumentUpdateLoop();